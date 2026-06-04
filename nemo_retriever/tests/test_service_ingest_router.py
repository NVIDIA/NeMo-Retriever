# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Phase 1 tests for the ``/v1/ingest`` router.

We use a FastAPI ``TestClient`` against a standalone service whose pipeline
pool is initialised with a stub work-fn that records ``WorkItem`` instances.
This lets us assert:

* the validated ``PipelineSpec`` is attached to the work-item;
* policy denials produce HTTP 403 / 501 responses;
* ``/v1/ingest/pipeline-config`` exposes the ``allowed_overrides`` block.

The stub work-fn never imports ``nemo_retriever.graph_ingestor`` so the test
runs without any GPU / Ray dependencies.
"""

from __future__ import annotations

import json
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
from .conftest import create_test_job


@pytest.fixture
def captured_items() -> list[WorkItem]:
    return []


@pytest.fixture
def app_with_stub_pool(monkeypatch: pytest.MonkeyPatch, captured_items: list[WorkItem]):
    """Build a standalone-mode app whose pools record items instead of running pipelines."""

    async def _stub_work(item: WorkItem) -> tuple[int, list[dict[str, Any]]]:
        captured_items.append(item)
        return 1, [{"id": item.id, "stub": True}]

    def _stub_realtime(_config: ServiceConfig):
        return _stub_work

    def _stub_batch(_config: ServiceConfig):
        return _stub_work

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        _stub_realtime,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        _stub_batch,
    )

    cfg = ServiceConfig(
        mode="standalone",
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        pipeline_overrides=PipelineOverridesConfig(),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client


def _make_pdf_bytes() -> bytes:
    """Return a 1-byte non-PDF payload — the worker is stubbed so content doesn't matter."""
    return b"%PDF-1.4\n%stub\n"


def test_ingest_without_spec_falls_back_to_legacy_pipeline(
    app_with_stub_pool: TestClient, captured_items: list[WorkItem]
) -> None:
    job_id = create_test_job(app_with_stub_pool)
    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert "document_id" in body
    assert body["job_id"] == job_id

    # Wait briefly for the async worker loop to consume the queued item.
    import time as _time

    deadline = _time.monotonic() + 5.0
    while not captured_items and _time.monotonic() < deadline:
        _time.sleep(0.05)

    assert len(captured_items) == 1
    item = captured_items[0]
    assert item.pipeline_spec is None  # legacy path
    assert item.job_id == job_id


def test_ingest_with_valid_spec_attaches_to_work_item(
    app_with_stub_pool: TestClient, captured_items: list[WorkItem]
) -> None:
    job_id = create_test_job(app_with_stub_pool)
    metadata = {
        "pipeline": {
            "extraction_mode": "pdf",
            "extract_params": {"extract_text": False, "dpi": 300},
            "stage_order": ["extract"],
        }
    }
    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 202, resp.text

    import time as _time

    deadline = _time.monotonic() + 5.0
    while not captured_items and _time.monotonic() < deadline:
        _time.sleep(0.05)

    assert len(captured_items) == 1
    item = captured_items[0]
    assert item.pipeline_spec is not None
    assert item.pipeline_spec["extraction_mode"] == "pdf"
    assert item.pipeline_spec["extract_params"]["dpi"] == 300
    assert item.pipeline_spec["stage_order"] == ["extract"]


def test_ingest_rejects_trust_sensitive_override(app_with_stub_pool: TestClient) -> None:
    job_id = create_test_job(app_with_stub_pool)
    metadata = {"pipeline": {"extract_params": {"page_elements_invoke_url": "http://attacker/"}}}
    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 403, resp.text
    assert "trust-sensitive" in resp.json()["detail"]


def test_ingest_rejects_caption_when_endpoint_not_configured(app_with_stub_pool: TestClient) -> None:
    """Without ``nim_endpoints.caption_invoke_url``, caption overrides are 403."""
    job_id = create_test_job(app_with_stub_pool)
    metadata = {"pipeline": {"caption_params": {"prompt": "Describe"}}}
    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 403, resp.text
    assert "caption" in resp.json()["detail"].lower()


def test_ingest_rejects_webhook_when_sinks_disabled(app_with_stub_pool: TestClient) -> None:
    """Without ``sinks.webhook_url_prefixes`` set, the ``webhook`` stage is not allowed."""
    job_id = create_test_job(app_with_stub_pool)
    metadata = {"pipeline": {"webhook_params": {"endpoint_url": "http://x/"}, "stage_order": ["webhook"]}}
    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 403, resp.text
    detail = resp.json()["detail"].lower()
    assert "webhook" in detail
    assert "allowed_stages" in detail or "not in" in detail


def test_ingest_rejects_webhook_params_without_stage_when_sinks_disabled(
    app_with_stub_pool: TestClient,
) -> None:
    """Bare ``webhook_params`` (no stage_order entry) still fails the sink allowlist check."""
    job_id = create_test_job(app_with_stub_pool)
    metadata = {"pipeline": {"webhook_params": {"endpoint_url": "http://x/"}}}
    resp = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 403, resp.text
    assert "disabled" in resp.json()["detail"].lower()


def test_create_job_returns_201_and_aggregate_fields(app_with_stub_pool: TestClient) -> None:
    """POST /v1/ingest/job opens a fresh aggregate with status=pending."""
    resp = app_with_stub_pool.post(
        "/v1/ingest/job",
        json={"expected_documents": 3, "label": "smoke"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["expected_documents"] == 3
    assert body["status"] == "pending"
    assert body["label"] == "smoke"
    assert body["job_id"]


def test_create_job_retain_results_persisted_on_aggregate(app_with_stub_pool: TestClient) -> None:
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    resp = app_with_stub_pool.post(
        "/v1/ingest/job",
        json={"expected_documents": 1, "retain_results": True},
    )
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["job_id"]
    tracker = get_job_tracker()
    assert tracker is not None
    agg = tracker.get_job(job_id)
    assert agg is not None
    assert agg.retain_results is True


def test_get_job_returns_aggregate_snapshot(app_with_stub_pool: TestClient) -> None:
    job_id = create_test_job(app_with_stub_pool, expected_documents=2)
    resp = app_with_stub_pool.get(f"/v1/ingest/job/{job_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["expected_documents"] == 2
    assert body["status"] == "pending"
    assert body["documents"] is None  # not requested
    assert body["counts"] == {} or "pending" in body["counts"]


def test_get_job_missing_returns_404(app_with_stub_pool: TestClient) -> None:
    resp = app_with_stub_pool.get("/v1/ingest/job/does-not-exist")
    assert resp.status_code == 404


def test_upload_to_missing_job_returns_404(app_with_stub_pool: TestClient) -> None:
    resp = app_with_stub_pool.post(
        "/v1/ingest/job/does-not-exist/document",
        files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )
    assert resp.status_code == 404, resp.text


def test_upload_beyond_capacity_returns_409(app_with_stub_pool: TestClient, captured_items: list[WorkItem]) -> None:
    """The (expected_documents + 1)th upload must be rejected with 409."""
    job_id = create_test_job(app_with_stub_pool, expected_documents=1)
    first = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("a.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )
    assert first.status_code == 202, first.text

    second = app_with_stub_pool.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("b.pdf", _make_pdf_bytes(), "application/pdf")},
        data={"metadata": "{}"},
    )
    assert second.status_code == 409, second.text


def test_pipeline_config_endpoint_reports_allowed_overrides(
    app_with_stub_pool: TestClient,
) -> None:
    resp = app_with_stub_pool.get("/v1/ingest/pipeline-config")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "allowed_overrides" in body
    assert body["allowed_overrides"]["mode"] == "allow_list"
    assert "dpi" in body["allowed_overrides"]["allowed_extract_keys"]
    assert "ocr_invoke_url" in body["allowed_overrides"]["denied_key_substrings"]
