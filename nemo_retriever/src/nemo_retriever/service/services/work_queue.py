# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable gateway-owned FIFO work broker and split-worker pull client."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from nemo_retriever.service.config import PipelinePoolConfig, WorkQueueConfig
from nemo_retriever.service.services.durable_state import DurableStateStore
from nemo_retriever.service.services.pipeline_pool import PoolType, WorkItem
from nemo_retriever.service.services.prometheus import (
    POOL_CLAIM_LATENCY,
    POOL_HEARTBEAT_FAILURES,
    WORK_QUEUE_ACTIVE_LEASES,
    WORK_QUEUE_BYTES,
    WORK_QUEUE_CLAIMS,
    WORK_QUEUE_DEMAND,
    WORK_QUEUE_EXHAUSTED,
    WORK_QUEUE_EXPIRATIONS,
    WORK_QUEUE_ITEMS,
    WORK_QUEUE_MAX_ACTIVE_LEASES,
    WORK_QUEUE_REQUEUES,
    WORK_QUEUE_STALE_CALLBACKS,
    WORK_QUEUE_WAIT,
)

logger = logging.getLogger(__name__)


class WorkQueueFull(RuntimeError):
    """Admission would exceed a pool item limit or the shared spool limit."""


class StaleLease(RuntimeError):
    """The supplied lease no longer owns this work item."""


@dataclass
class WorkLease:
    lease_id: str
    generation: int
    worker_uid: str
    worker_ip: str
    expires_at: float


@dataclass
class WorkRecord:
    work_id: str
    job_id: str
    pool: PoolType
    filename: str | None
    spool_path: Path
    payload_size: int
    payload_sha256: str
    retain_results: bool
    pipeline_spec: dict[str, Any] | None
    trace_context: dict[str, str]
    enqueued_at: float
    delivery_attempt: int = 0
    generation: int = 0
    lease: WorkLease | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    fifo_sequence: int = 0


class WorkBroker:
    """FIFO broker with durable metadata and atomically spooled payloads."""

    def __init__(self, config: WorkQueueConfig, pools: PipelinePoolConfig) -> None:
        self.config = config
        self._limits = {
            PoolType.REALTIME: pools.realtime_queue_size,
            PoolType.BATCH: pools.batch_queue_size,
        }
        self._lease_caps = {
            PoolType.REALTIME: config.max_active_leases_realtime,
            PoolType.BATCH: config.max_active_leases_batch,
        }
        self._queues: dict[PoolType, deque[str]] = {
            PoolType.REALTIME: deque(),
            PoolType.BATCH: deque(),
        }
        self._records: dict[str, WorkRecord] = {}
        self._conditions = {pool: asyncio.Condition() for pool in PoolType}
        self._admission_lock = asyncio.Lock()
        self._spool_bytes = 0
        self._next_sequence = 1
        self._expiry_task: asyncio.Task[None] | None = None
        self._running = False
        self._spool = Path(config.spool_directory)
        self._state = DurableStateStore(self._spool) if config.persistence_enabled else None

    async def start(self) -> None:
        self._spool.mkdir(parents=True, exist_ok=True)
        if self._state is not None:
            self._state.open()
            from nemo_retriever.service.services.job_tracker import get_job_tracker

            tracker = get_job_tracker()
            if tracker is not None:
                tracker.set_state_store(self._state)
                await self._restore()
            elif self._state.load_work():
                raise RuntimeError("Durable work exists but the job tracker is unavailable")
        else:
            for orphan in self._spool.glob("*.payload"):
                try:
                    await asyncio.to_thread(orphan.unlink)
                except FileNotFoundError:
                    pass
        self._running = True
        self._expiry_task = asyncio.create_task(self._expiry_loop())
        self._publish_metrics()

    async def shutdown(self) -> None:
        """Stop claims and checkpoint state without discarding accepted work."""
        self._running = False
        for condition in self._conditions.values():
            async with condition:
                condition.notify_all()
        if self._expiry_task is not None:
            self._expiry_task.cancel()
            await asyncio.gather(self._expiry_task, return_exceptions=True)
            self._expiry_task = None
        if self._state is not None:
            await asyncio.to_thread(self._state.close)
        self._publish_metrics()

    def _active(self, pool: PoolType) -> int:
        return sum(record.lease is not None for record in self._records.values() if record.pool is pool)

    def _publish_metrics(self) -> None:
        for pool in PoolType:
            queued = len(self._queues[pool])
            active = self._active(pool)
            WORK_QUEUE_ITEMS.labels(pool=pool.value).set(queued)
            WORK_QUEUE_BYTES.labels(pool=pool.value).set(
                sum(
                    record.payload_size
                    for record in self._records.values()
                    if record.pool is pool and record.lease is None
                )
            )
            WORK_QUEUE_ACTIVE_LEASES.labels(pool=pool.value).set(active)
            WORK_QUEUE_DEMAND.labels(pool=pool.value).set(queued + active)
            WORK_QUEUE_MAX_ACTIVE_LEASES.labels(pool=pool.value).set(self._lease_caps[pool])

    @staticmethod
    def _write_spool(path: Path, payload: bytes) -> None:
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def _record_data(self, record: WorkRecord) -> dict[str, Any]:
        lease = record.lease
        return {
            "work_id": record.work_id,
            "job_id": record.job_id,
            "pool": record.pool.value,
            "filename": record.filename,
            "spool_path": str(record.spool_path),
            "payload_size": record.payload_size,
            "payload_sha256": record.payload_sha256,
            "retain_results": record.retain_results,
            "pipeline_spec": record.pipeline_spec,
            "trace_context": record.trace_context,
            "enqueued_at": record.enqueued_at,
            "delivery_attempt": record.delivery_attempt,
            "generation": record.generation,
            "extra": record.extra,
            "fifo_sequence": record.fifo_sequence,
            "lease": (
                None
                if lease is None
                else {
                    "lease_id": lease.lease_id,
                    "generation": lease.generation,
                    "worker_uid": lease.worker_uid,
                    "worker_ip": lease.worker_ip,
                    "expires_at": time.time() + max(0.0, lease.expires_at - time.monotonic()),
                }
            ),
        }

    def _save(self, record: WorkRecord) -> None:
        if self._state is not None:
            self._state.save_work(
                record.work_id,
                record.pool.value,
                record.fifo_sequence,
                self._record_data(record),
            )

    def _delete_state(self, work_id: str) -> None:
        if self._state is not None:
            self._state.delete_work(work_id)

    def _decode_record(self, raw: dict[str, Any], sequence: int) -> WorkRecord:
        lease_raw = raw.get("lease")
        lease = None
        if lease_raw is not None:
            lease = WorkLease(
                lease_id=lease_raw["lease_id"],
                generation=int(lease_raw["generation"]),
                worker_uid=lease_raw["worker_uid"],
                worker_ip=lease_raw["worker_ip"],
                expires_at=time.monotonic(),
            )
        return WorkRecord(
            work_id=raw["work_id"],
            job_id=raw["job_id"],
            pool=PoolType(raw["pool"]),
            filename=raw.get("filename"),
            spool_path=Path(raw["spool_path"]),
            payload_size=int(raw["payload_size"]),
            payload_sha256=raw["payload_sha256"],
            retain_results=bool(raw.get("retain_results")),
            pipeline_spec=raw.get("pipeline_spec"),
            trace_context=dict(raw.get("trace_context") or {}),
            enqueued_at=float(raw.get("enqueued_at", time.time())),
            delivery_attempt=int(raw.get("delivery_attempt", 0)),
            generation=int(raw.get("generation", 0)),
            lease=lease,
            extra=dict(raw.get("extra") or {}),
            fifo_sequence=sequence,
        )

    @staticmethod
    def _payload_valid(record: WorkRecord) -> bool:
        try:
            if record.spool_path.stat().st_size != record.payload_size:
                return False
            digest = hashlib.sha256()
            with record.spool_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest() == record.payload_sha256
        except OSError:
            return False

    async def _restore(self) -> None:
        assert self._state is not None
        from nemo_retriever.service.services.job_tracker import (
            DocumentStatus,
            get_job_tracker,
        )

        tracker = get_job_tracker()
        assert tracker is not None
        referenced: set[Path] = set()
        restored_ids: set[str] = set()
        for sequence, raw in self._state.load_work():
            record = self._decode_record(raw, sequence)
            self._next_sequence = max(self._next_sequence, sequence + 1)
            document = tracker.get_document(record.work_id)
            if document is None or document.status in (
                DocumentStatus.COMPLETED,
                DocumentStatus.FAILED,
            ):
                self._delete_state(record.work_id)
                try:
                    record.spool_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if not await asyncio.to_thread(self._payload_valid, record):
                tracker.mark_failed(record.work_id, "Restored work payload is missing or corrupt")
                self._delete_state(record.work_id)
                try:
                    record.spool_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            self._records[record.work_id] = record
            self._spool_bytes += record.payload_size
            restored_ids.add(record.work_id)
            referenced.add(record.spool_path.resolve())
            if record.lease is not None:
                record.lease = None
                if self._requeue_or_exhaust_locked(record, "recovery", front=False):
                    continue
            elif record.delivery_attempt >= self.config.max_delivery_attempts:
                self._exhaust_locked(record)
                continue
            else:
                self._queues[record.pool].append(record.work_id)
                self._save(record)

        for document in tracker.all_documents():
            if document.status is DocumentStatus.PENDING and document.id not in restored_ids:
                tracker.unregister_pending(document.id)
            elif document.status is DocumentStatus.PROCESSING and document.id not in restored_ids:
                tracker.mark_failed(document.id, "Accepted work metadata is missing")

        for payload in self._spool.glob("*.payload"):
            if payload.resolve() not in referenced:
                try:
                    await asyncio.to_thread(payload.unlink)
                except FileNotFoundError:
                    pass

    async def enqueue(
        self,
        pool: PoolType,
        *,
        work_id: str,
        job_id: str,
        payload: bytes,
        filename: str | None,
        retain_results: bool,
        pipeline_spec: dict[str, Any] | None,
        trace_context: Mapping[str, str] | None,
        extra: Mapping[str, Any] | None = None,
    ) -> WorkRecord:
        """Fsync payload and commit metadata before reporting admission."""
        condition = self._conditions[pool]
        async with condition:
            async with self._admission_lock:
                if not self._running:
                    raise WorkQueueFull("gateway work broker is shutting down")
                if work_id in self._records:
                    raise ValueError(f"Duplicate work id {work_id!r}")
                if len(self._queues[pool]) >= self._limits[pool]:
                    raise WorkQueueFull(f"{pool.value} work queue is at capacity")
                if self._spool_bytes + len(payload) > self.config.spool_limit_bytes:
                    raise WorkQueueFull("gateway work spool byte limit exceeded")

                spool_path = self._spool / f"{work_id}.payload"
                try:
                    await asyncio.to_thread(self._write_spool, spool_path, payload)
                    record = WorkRecord(
                        work_id=work_id,
                        job_id=job_id,
                        pool=pool,
                        filename=filename,
                        spool_path=spool_path,
                        payload_size=len(payload),
                        payload_sha256=hashlib.sha256(payload).hexdigest(),
                        retain_results=retain_results,
                        pipeline_spec=pipeline_spec,
                        trace_context=dict(trace_context or {}),
                        enqueued_at=time.time(),
                        extra=dict(extra or {}),
                        fifo_sequence=self._next_sequence,
                    )
                    self._save(record)
                except Exception:
                    try:
                        await asyncio.to_thread(spool_path.unlink)
                    except FileNotFoundError:
                        pass
                    from nemo_retriever.service.services.job_tracker import get_job_tracker

                    tracker = get_job_tracker()
                    if tracker is not None:
                        tracker.unregister_pending(work_id)
                    raise
                self._next_sequence += 1
                self._records[work_id] = record
                self._queues[pool].append(work_id)
                self._spool_bytes += len(payload)
            self._publish_metrics()
            condition.notify(1)
            return record

    async def claim(self, pool: PoolType, *, worker_uid: str, worker_ip: str) -> WorkRecord | None:
        deadline = time.monotonic() + self.config.claim_timeout_s
        condition = self._conditions[pool]
        async with condition:
            while self._running:
                self._expire_locked(pool)
                if self._active(pool) < self._lease_caps[pool]:
                    while self._queues[pool]:
                        work_id = self._queues[pool].popleft()
                        record = self._records.get(work_id)
                        if record is None or record.lease is not None:
                            continue
                        record.delivery_attempt += 1
                        record.generation += 1
                        record.lease = WorkLease(
                            lease_id=uuid.uuid4().hex,
                            generation=record.generation,
                            worker_uid=worker_uid,
                            worker_ip=worker_ip,
                            expires_at=time.monotonic() + self.config.lease_ttl_s,
                        )
                        self._save(record)
                        WORK_QUEUE_CLAIMS.labels(pool=pool.value, worker_uid=worker_uid).inc()
                        WORK_QUEUE_WAIT.labels(pool=pool.value).observe(max(0.0, time.time() - record.enqueued_at))
                        from nemo_retriever.service.services.job_tracker import get_job_tracker

                        tracker = get_job_tracker()
                        if tracker is not None:
                            tracker.mark_processing(record.work_id)
                        self._publish_metrics()
                        return record
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    await asyncio.wait_for(condition.wait(), timeout=remaining)
                except TimeoutError:
                    return None
        return None

    def _current(self, work_id: str, lease_id: str, generation: int) -> WorkRecord:
        record = self._records.get(work_id)
        lease = record.lease if record is not None else None
        if lease is None or lease.lease_id != lease_id or lease.generation != generation:
            raise StaleLease(f"Lease for work {work_id!r} has been superseded")
        return record

    async def heartbeat(self, work_id: str, lease_id: str, generation: int) -> None:
        record = self._current(work_id, lease_id, generation)
        assert record.lease is not None
        record.lease.expires_at = time.monotonic() + self.config.lease_ttl_s
        self._save(record)

    async def payload_path(self, work_id: str, lease_id: str, generation: int) -> Path:
        record = self._current(work_id, lease_id, generation)
        if not record.spool_path.is_file():
            condition = self._conditions[record.pool]
            async with condition:
                self._requeue_or_exhaust_locked(record, "payload_fetch")
                condition.notify_all()
            raise StaleLease(f"Payload for work {work_id!r} is unavailable")
        return record.spool_path

    async def release(self, work_id: str, lease_id: str, generation: int, *, reason: str = "release") -> None:
        record = self._current(work_id, lease_id, generation)
        condition = self._conditions[record.pool]
        async with condition:
            self._requeue_or_exhaust_locked(record, reason)
            self._publish_metrics()
            condition.notify_all()

    def validate_callback(self, work_id: str, lease_id: str, generation: int) -> WorkRecord:
        try:
            return self._current(work_id, lease_id, generation)
        except StaleLease:
            record = self._records.get(work_id)
            WORK_QUEUE_STALE_CALLBACKS.labels(pool=record.pool.value if record else "unknown").inc()
            raise

    def has_record(self, work_id: str) -> bool:
        return work_id in self._records

    async def acknowledge(self, work_id: str, lease_id: str, generation: int) -> None:
        record = self._current(work_id, lease_id, generation)
        condition = self._conditions[record.pool]
        async with condition:
            self._delete_state(work_id)
            self._records.pop(work_id, None)
            self._spool_bytes -= record.payload_size
            try:
                await asyncio.to_thread(record.spool_path.unlink)
            except FileNotFoundError:
                pass
            self._publish_metrics()
            condition.notify_all()

    def _exhaust_locked(self, record: WorkRecord) -> None:
        from nemo_retriever.service.services.job_tracker import get_job_tracker

        tracker = get_job_tracker()
        if tracker is not None:
            tracker.mark_failed(record.work_id, "Work delivery attempts exhausted")
        self._delete_state(record.work_id)
        self._records.pop(record.work_id, None)
        self._spool_bytes -= record.payload_size
        try:
            record.spool_path.unlink()
        except FileNotFoundError:
            pass
        WORK_QUEUE_EXHAUSTED.labels(pool=record.pool.value).inc()

    def _requeue_or_exhaust_locked(self, record: WorkRecord, reason: str, *, front: bool = True) -> bool:
        record.lease = None
        if record.delivery_attempt >= self.config.max_delivery_attempts:
            self._exhaust_locked(record)
            return True
        if front:
            self._queues[record.pool].appendleft(record.work_id)
        else:
            self._queues[record.pool].append(record.work_id)
        self._save(record)
        WORK_QUEUE_REQUEUES.labels(pool=record.pool.value, reason=reason).inc()
        return False

    def _expire_locked(self, pool: PoolType) -> None:
        now = time.monotonic()
        changed = False
        for record in list(self._records.values()):
            if record.pool is not pool or record.lease is None or record.lease.expires_at > now:
                continue
            WORK_QUEUE_EXPIRATIONS.labels(pool=pool.value).inc()
            self._requeue_or_exhaust_locked(record, "expiry")
            changed = True
        if changed:
            self._publish_metrics()

    async def _expiry_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(min(1.0, self.config.lease_ttl_s / 4))
                for pool, condition in self._conditions.items():
                    async with condition:
                        before_active = self._active(pool)
                        before_queued = len(self._queues[pool])
                        self._expire_locked(pool)
                        if self._active(pool) != before_active or len(self._queues[pool]) != before_queued:
                            condition.notify_all()
        except asyncio.CancelledError:
            pass

    def claim_payload(self, record: WorkRecord, *, base_url: str) -> dict[str, Any]:
        assert record.lease is not None
        lease = record.lease
        return {
            "work_id": record.work_id,
            "job_id": record.job_id,
            "pool": record.pool.value,
            "lease_id": lease.lease_id,
            "lease_generation": lease.generation,
            "delivery_attempt": record.delivery_attempt,
            "filename": record.filename,
            "payload_url": f"{base_url.rstrip('/')}/v1/internal/work/{quote(record.work_id, safe='')}/payload",
            "payload_size": record.payload_size,
            "payload_sha256": record.payload_sha256,
            "retain_results": record.retain_results,
            "pipeline_spec": record.pipeline_spec,
            "trace_context": record.trace_context,
        }


class GatewayWorkClient:
    """One client per worker pod; each execution coroutine claims directly."""

    def __init__(self, config: WorkQueueConfig, *, pool: PoolType, headers: Mapping[str, str]) -> None:
        self.config = config
        self.pool = pool
        self.headers = dict(headers)
        self.worker_uid = os.environ.get("POD_UID") or f"local-{os.getpid()}"
        self._client: Any | None = None

    async def _http(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.config.gateway_url.rstrip("/"),
                headers=self.headers,
                timeout=self.config.claim_timeout_s + 10,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def claim(self) -> WorkItem | None:
        client = await self._http()
        t0 = time.monotonic()
        response = await client.post(
            "/v1/internal/work/claim",
            json={"pool": self.pool.value, "worker_uid": self.worker_uid},
        )
        POOL_CLAIM_LATENCY.labels(pool=self.pool.value).observe(time.monotonic() - t0)
        if response.status_code == 204:
            return None
        response.raise_for_status()
        claim = response.json()
        lease_headers = {
            "X-Work-Lease-Id": claim["lease_id"],
            "X-Work-Lease-Generation": str(claim["lease_generation"]),
        }
        payload_response = await client.get(claim["payload_url"], headers=lease_headers)
        if payload_response.status_code != 200:
            await self.release(claim, reason="payload_fetch")
            payload_response.raise_for_status()
        payload = payload_response.content
        if len(payload) != claim["payload_size"] or hashlib.sha256(payload).hexdigest() != claim["payload_sha256"]:
            await self.release(claim, reason="hash_mismatch")
            raise RuntimeError(f"Gateway payload integrity check failed for {claim['work_id']}")
        return WorkItem(
            id=claim["work_id"],
            payload=payload,
            filename=claim.get("filename"),
            callback_url=f"{self.config.gateway_url.rstrip('/')}/v1/internal/job-callback",
            callback_headers=self.headers,
            job_id=claim.get("job_id"),
            retain_results=bool(claim.get("retain_results")),
            pipeline_spec=claim.get("pipeline_spec"),
            trace_context=claim.get("trace_context") or {},
            lease_id=claim["lease_id"],
            lease_generation=claim["lease_generation"],
            delivery_attempt=claim["delivery_attempt"],
            worker_uid=self.worker_uid,
        )

    async def heartbeat(self, item: WorkItem) -> bool:
        client = await self._http()
        response = await client.post(
            f"/v1/internal/work/{quote(item.id, safe='')}/heartbeat",
            json={"lease_id": item.lease_id, "lease_generation": item.lease_generation},
        )
        if response.status_code != 200:
            POOL_HEARTBEAT_FAILURES.labels(pool=self.pool.value).inc()
            return False
        return True

    async def release(self, claim: Mapping[str, Any], *, reason: str = "release") -> None:
        client = await self._http()
        await client.post(
            f"/v1/internal/work/{quote(str(claim['work_id']), safe='')}/release",
            json={
                "lease_id": claim["lease_id"],
                "lease_generation": claim["lease_generation"],
                "reason": reason,
            },
        )


_instance: WorkBroker | None = None


async def init_work_broker(config: WorkQueueConfig, pools: PipelinePoolConfig) -> WorkBroker:
    global _instance
    _instance = WorkBroker(config, pools)
    await _instance.start()
    return _instance


def get_work_broker() -> WorkBroker | None:
    return _instance


async def shutdown_work_broker() -> None:
    global _instance
    if _instance is not None:
        await _instance.shutdown()
        _instance = None
