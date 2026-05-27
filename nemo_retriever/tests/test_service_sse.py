# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""J4 — per-job SSE routing — HTTP-surface tests.

We assert the route shape and error responses for
``GET /v1/ingest/job/{job_id}/events``:

* unknown job → ``404`` (validated before the stream opens),
* the legacy firehose ``GET /v1/ingest/events`` is removed and now
  returns ``410 Gone`` with a migration body that names the
  replacement route — so an older SDK build hitting a new service
  fails with an actionable error instead of an empty result,
* the legacy single-shot ``POST /v1/ingest`` route is similarly
  surfaced with ``410 Gone`` + migration body,
* the per-job route is registered in OpenAPI (``openapi.json``-style
  listing) so a refactor that drops it would fail loudly,
* the legacy 410 stubs are **hidden** from OpenAPI so they aren't
  advertised as supported endpoints.

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


def test_legacy_firehose_route_returns_410_with_migration_body(
    app_with_stub_pool: TestClient,
) -> None:
    """``GET /v1/ingest/events`` returns ``410 Gone`` with a migration hint.

    Replaces the older "should now 404" contract.  The default FastAPI
    404 had no body, which meant older SDK builds calling the firehose
    surfaced a generic "no documents completed" failure with no clue
    that the route had moved.  We now return an explicit ``410 Gone``
    naming the replacement route (``/v1/ingest/job/{job_id}/events``)
    and the underlying cause (SDK / service version mismatch).
    """
    resp = app_with_stub_pool.get("/v1/ingest/events")
    assert resp.status_code == 410, resp.text
    detail = resp.json().get("detail", "")
    # Body must name the replacement route so operators can act on it.
    assert "/v1/ingest/job/{job_id}/events" in detail, detail
    # And it must surface the actual cause — SDK / service version
    # mismatch — so this isn't mistaken for a generic transient failure.
    assert "SDK" in detail and "service" in detail, detail


def test_legacy_ingest_upload_route_returns_410_with_migration_body(
    app_with_stub_pool: TestClient,
) -> None:
    """``POST /v1/ingest`` (legacy single-shot upload) returns ``410 Gone``.

    Older SDK builds upload through this path.  Without an explicit
    handler FastAPI returns a body-less 404 and the SDK surfaces an
    empty result — the customer-facing regression captured in the
    26.05-RC2 release-integration report.  The 410 body must name the
    replacement pair (``/v1/ingest/job`` + ``/v1/ingest/job/{job_id}/document``).
    """
    # Body is intentionally empty — the route should reject the request
    # on path alone, before any multipart parsing.
    resp = app_with_stub_pool.post("/v1/ingest")
    assert resp.status_code == 410, resp.text
    detail = resp.json().get("detail", "")
    assert "/v1/ingest/job" in detail, detail
    assert "/v1/ingest/job/{job_id}/document" in detail, detail
    assert "SDK" in detail and "service" in detail, detail


def test_per_job_sse_route_is_registered(app_with_stub_pool: TestClient) -> None:
    """The per-job SSE route shows up in the OpenAPI schema."""
    schema = app_with_stub_pool.get("/openapi.json").json()
    assert "/v1/ingest/job/{job_id}/events" in schema["paths"], sorted(schema["paths"])
    methods = schema["paths"]["/v1/ingest/job/{job_id}/events"]
    assert "get" in methods


def test_legacy_routes_are_not_registered_in_openapi(
    app_with_stub_pool: TestClient,
) -> None:
    """Schema must not advertise the removed firehose or legacy upload routes.

    Both legacy stubs (``GET /v1/ingest/events`` and ``POST /v1/ingest``)
    are registered with ``include_in_schema=False`` so they exist for
    error-handling purposes but do not show up as supported endpoints
    in clients generated from ``/openapi.json``.
    """
    schema = app_with_stub_pool.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/v1/ingest/events" not in paths, sorted(paths)
    # ``/v1/ingest`` is also reserved for the 410 stub and must not be
    # exposed as a real upload route to schema-generated clients.
    assert "/v1/ingest" not in paths, sorted(paths)
