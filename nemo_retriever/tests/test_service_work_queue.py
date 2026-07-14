# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import (
    AuthConfig,
    LoggingConfig,
    MCPConfig,
    PipelinePoolConfig,
    ServiceConfig,
    WorkQueueConfig,
)
from nemo_retriever.service.services.pipeline_pool import PoolType, WorkItem, _Pool
from nemo_retriever.service.services.work_queue import (
    StaleLease,
    WorkBroker,
    WorkQueueFull,
)


def _config(tmp_path, **updates) -> WorkQueueConfig:
    values = {
        "spool_directory": str(tmp_path),
        "spool_limit_bytes": 1024,
        "claim_timeout_s": 0.02,
        "lease_ttl_s": 0.2,
        "heartbeat_interval_s": 0.05,
        "max_delivery_attempts": 3,
    }
    values.update(updates)
    return WorkQueueConfig(**values)


async def _enqueue(broker: WorkBroker, work_id: str, payload: bytes = b"payload"):
    return await broker.enqueue(
        PoolType.BATCH,
        work_id=work_id,
        job_id="job",
        payload=payload,
        filename=f"{work_id}.pdf",
        retain_results=False,
        pipeline_spec=None,
        trace_context={"traceparent": "00-abc"},
    )


@pytest.mark.anyio
async def test_fifo_spool_integrity_and_ack_cleanup(tmp_path):
    broker = WorkBroker(_config(tmp_path), PipelinePoolConfig(batch_queue_size=2))
    await broker.start()
    try:
        first = await _enqueue(broker, "first", b"one")
        await _enqueue(broker, "second", b"two")
        assert first.spool_path.read_bytes() == b"one"

        claim1 = await broker.claim(
            PoolType.BATCH, worker_uid="pod-a", worker_ip="10.0.0.1"
        )
        claim2 = await broker.claim(
            PoolType.BATCH, worker_uid="pod-b", worker_ip="10.0.0.2"
        )
        assert [claim1.work_id, claim2.work_id] == ["first", "second"]

        lease = claim1.lease
        assert lease is not None
        await broker.acknowledge(claim1.work_id, lease.lease_id, lease.generation)
        assert not first.spool_path.exists()
    finally:
        await broker.shutdown()


@pytest.mark.anyio
async def test_item_and_shared_byte_limits_leave_no_partial_spool(tmp_path):
    broker = WorkBroker(
        _config(tmp_path, spool_limit_bytes=5),
        PipelinePoolConfig(realtime_queue_size=1, batch_queue_size=1),
    )
    await broker.start()
    try:
        await _enqueue(broker, "one", b"1234")
        with pytest.raises(WorkQueueFull):
            await _enqueue(broker, "item-full", b"x")
        with pytest.raises(WorkQueueFull):
            await broker.enqueue(
                PoolType.REALTIME,
                work_id="byte-full",
                job_id="job",
                payload=b"xx",
                filename=None,
                retain_results=False,
                pipeline_spec=None,
                trace_context=None,
            )
        assert sorted(path.name for path in tmp_path.glob("*.payload")) == [
            "one.payload"
        ]
    finally:
        await broker.shutdown()


@pytest.mark.anyio
async def test_heartbeat_release_expiry_and_stale_generation(tmp_path):
    broker = WorkBroker(_config(tmp_path), PipelinePoolConfig(batch_queue_size=2))
    await broker.start()
    try:
        await _enqueue(broker, "work")
        first = await broker.claim(
            PoolType.BATCH, worker_uid="pod-a", worker_ip="10.0.0.1"
        )
        lease1 = first.lease
        assert lease1 is not None
        old_expiry = lease1.expires_at
        await broker.heartbeat("work", lease1.lease_id, lease1.generation)
        assert lease1.expires_at >= old_expiry

        await broker.release("work", lease1.lease_id, lease1.generation)
        second = await broker.claim(
            PoolType.BATCH, worker_uid="pod-b", worker_ip="10.0.0.2"
        )
        lease2 = second.lease
        assert lease2 is not None and lease2.generation == lease1.generation + 1
        with pytest.raises(StaleLease):
            broker.validate_callback("work", lease1.lease_id, lease1.generation)

        lease2.expires_at = time.monotonic() - 1
        broker._expire_locked(PoolType.BATCH)
        third = await broker.claim(
            PoolType.BATCH, worker_uid="pod-c", worker_ip="10.0.0.3"
        )
        assert third.delivery_attempt == 3
    finally:
        await broker.shutdown()


@pytest.mark.anyio
async def test_three_expired_deliveries_exhaust_and_delete(tmp_path):
    broker = WorkBroker(_config(tmp_path), PipelinePoolConfig(batch_queue_size=1))
    await broker.start()
    try:
        record = await _enqueue(broker, "work")
        for attempt in range(3):
            claimed = await broker.claim(
                PoolType.BATCH, worker_uid=f"pod-{attempt}", worker_ip="10.0.0.1"
            )
            assert claimed is not None and claimed.lease is not None
            claimed.lease.expires_at = time.monotonic() - 1
            broker._expire_locked(PoolType.BATCH)
        assert not broker.has_record("work")
        assert not record.spool_path.exists()
    finally:
        await broker.shutdown()


class _BlockingPullClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(heartbeat_interval_s=60.0)
        self.claim_count = 0
        self._blocked = asyncio.Event()

    async def claim(self):
        self.claim_count += 1
        if self.claim_count <= 2:
            return WorkItem(id=f"work-{self.claim_count}", payload=b"x")
        await self._blocked.wait()

    async def close(self):
        return None


@pytest.mark.anyio
async def test_execution_slots_make_at_most_one_claim_each():
    client = _BlockingPullClient()
    processing = asyncio.Event()
    started = 0

    async def work(_item):
        nonlocal started
        started += 1
        if started == 2:
            processing.set()
        await asyncio.Event().wait()

    pool = _Pool(
        "batch", num_workers=2, max_queue_size=99, work_fn=work, pull_client=client
    )
    pool.start()
    try:
        await asyncio.wait_for(processing.wait(), timeout=1)
        assert client.claim_count == 2
        assert pool.queue_depth == 0
    finally:
        await pool.shutdown()


def test_gateway_upload_claim_payload_and_callback_lifecycle(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(results_dir))
    config = ServiceConfig(
        mode="gateway",
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        mcp=MCPConfig(enabled=False),
        pipeline=PipelinePoolConfig(realtime_queue_size=2, batch_queue_size=2),
        work_queue=_config(tmp_path / "spool", gateway_url="http://testserver"),
    )

    with TestClient(create_app(config)) as client:
        created = client.post("/v1/ingest/job", json={"expected_documents": 1})
        assert created.status_code == 201
        job_id = created.json()["job_id"]

        accepted = client.post(
            f"/v1/ingest/job/{job_id}/whole",
            files={"file": ("document.txt", b"hello gateway", "text/plain")},
            data={"metadata": "{}"},
        )
        assert accepted.status_code == 202
        document_id = accepted.json()["document_id"]

        pending = client.get(f"/v1/ingest/job/{job_id}/document/{document_id}")
        assert pending.status_code == 202
        assert pending.json()["status"] == "pending"

        claim_response = client.post(
            "/v1/internal/work/claim",
            json={"pool": "batch", "worker_uid": "pod-uid"},
        )
        assert claim_response.status_code == 200
        claim = claim_response.json()
        assert claim["work_id"] == document_id
        assert claim["delivery_attempt"] == 1

        processing = client.get(f"/v1/ingest/job/{job_id}/document/{document_id}")
        assert processing.json()["status"] == "processing"

        payload = client.get(
            f"/v1/internal/work/{document_id}/payload",
            headers={
                "X-Work-Lease-Id": claim["lease_id"],
                "X-Work-Lease-Generation": str(claim["lease_generation"]),
            },
        )
        assert payload.content == b"hello gateway"

        callback = client.post(
            "/v1/internal/job-callback",
            json={
                "id": document_id,
                "status": "completed",
                "result_rows": 0,
                "lease_id": claim["lease_id"],
                "lease_generation": claim["lease_generation"],
            },
        )
        assert callback.status_code == 200
        assert not (tmp_path / "spool" / f"{document_id}.payload").exists()

        stale = client.post(
            "/v1/internal/job-callback",
            json={
                "id": document_id,
                "status": "completed",
                "lease_id": claim["lease_id"],
                "lease_generation": claim["lease_generation"],
            },
        )
        assert stale.status_code == 409


def test_internal_work_endpoints_require_configured_service_auth(tmp_path):
    config = ServiceConfig(
        mode="gateway",
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        mcp=MCPConfig(enabled=False),
        auth=AuthConfig(api_token="secret"),
        work_queue=_config(tmp_path / "spool", gateway_url="http://testserver"),
    )
    with TestClient(create_app(config)) as client:
        body = {"pool": "batch", "worker_uid": "pod-uid"}
        assert client.post("/v1/internal/work/claim", json=body).status_code == 401
        assert (
            client.post(
                "/v1/internal/work/claim",
                json=body,
                headers={"Authorization": "Bearer secret"},
            ).status_code
            == 204
        )
