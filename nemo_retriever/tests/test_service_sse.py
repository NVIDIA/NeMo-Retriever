# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""J4 — per-job SSE routing — HTTP-surface tests.

We assert the route shape and error responses for
``GET /v1/ingest/job/{job_id}/events``:

* unknown job → ``404`` (validated before the stream opens),
* the legacy firehose ``GET /v1/ingest/events`` is removed (404),
* the route is registered (``openapi.json``-style listing) so a
  refactor that drops it would fail loudly.

The per-job filtering semantics of the underlying ``EventBus`` are
already covered in :mod:`test_service_job_tracker` — see
``test_real_event_bus_filters_by_job_id`` and
``test_real_event_bus_firehose_subscriber_sees_everything``.

Reading a *live* SSE response with ``fastapi.TestClient`` blocks on
the server-side keepalive timer (see the 30-second
``wait_for(queue.get(), ...)`` in ``ingest_job_events``). End-to-end
streaming is exercised manually with ``curl`` against a real pod;
unit-testing it here would require an async-native client and a
non-trivial event-loop wrapper, which is out of scope for J4.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import (
    PipelineOverridesConfig,
    PipelinePoolConfig,
    ServiceConfig,
)
from nemo_retriever.service.services.pipeline_pool import WorkItem


@pytest.fixture
def captured_items() -> list[WorkItem]:
    return []


@pytest.fixture
def app_with_stub_pool(monkeypatch: pytest.MonkeyPatch, captured_items: list[WorkItem]):
    """Standalone app whose pools record items instead of running pipelines."""

    async def _stub_work(item: WorkItem) -> tuple[int, list[dict[str, Any]]]:
        captured_items.append(item)
        return 1, [{"id": item.id, "stub": True}]

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        lambda _c: _stub_work,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        lambda _c: _stub_work,
    )

    cfg = ServiceConfig(
        mode="standalone",
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        pipeline_overrides=PipelineOverridesConfig(),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client


def test_per_job_sse_route_404_when_job_missing(app_with_stub_pool: TestClient) -> None:
    """GET on an unknown job_id must return 404 *before* opening the stream.

    This is the only safe way to assert this with ``TestClient`` —
    the 404 short-circuits before the StreamingResponse generator
    runs, so no keepalive timer is at play.
    """
    resp = app_with_stub_pool.get("/v1/ingest/job/does-not-exist/events")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_legacy_firehose_route_is_removed(app_with_stub_pool: TestClient) -> None:
    """``GET /v1/ingest/events`` was deleted in J4 — it should now 404."""
    resp = app_with_stub_pool.get("/v1/ingest/events")
    assert resp.status_code == 404


def test_per_job_sse_route_is_registered(app_with_stub_pool: TestClient) -> None:
    """The per-job SSE route shows up in the OpenAPI schema."""
    schema = app_with_stub_pool.get("/openapi.json").json()
    assert "/v1/ingest/job/{job_id}/events" in schema["paths"], sorted(schema["paths"])
    methods = schema["paths"]["/v1/ingest/job/{job_id}/events"]
    assert "get" in methods


def test_legacy_firehose_route_is_not_registered(app_with_stub_pool: TestClient) -> None:
    """The schema should not advertise the removed firehose endpoint."""
    schema = app_with_stub_pool.get("/openapi.json").json()
    assert "/v1/ingest/events" not in schema["paths"], sorted(schema["paths"])
