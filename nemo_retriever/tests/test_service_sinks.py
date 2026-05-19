# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 tests for remote sink stages (store, webhook, vdb_upload).

Three layers:

* the client-side ``ServiceIngestor`` rejects local paths early and
  emits the right ``_pipeline_spec`` shape;
* the policy / sink allowlist accepts, denies, or wildcards URLs based
  on ``SinksConfig``;
* the worker-side ``_build_graph_ingestor_from_spec`` actually wires
  the corresponding stages onto the underlying ``GraphIngestor``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nemo_retriever.params import StoreParams, VdbUploadParams, WebhookParams
from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import (
    PipelineOverridesConfig,
    PipelinePoolConfig,
    ServiceConfig,
    SinksConfig,
)
from nemo_retriever.service.models.pipeline_spec import PipelineSpec
from nemo_retriever.service.policy import (
    PolicyError,
    SinkUrlAllowlist,
    validate_pipeline_spec,
)
from nemo_retriever.service.services.pipeline_pool import WorkItem
from nemo_retriever.service.services.pipeline_executor import _build_graph_ingestor_from_spec
from nemo_retriever.service_ingestor import ServiceIngestor


# ----------------------------------------------------------------------
# Client: ServiceIngestor fluent → spec dict
# ----------------------------------------------------------------------


def test_store_with_remote_uri_populates_spec() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.store(StoreParams(storage_uri="s3://my-bucket/imgs", image_format="png"))
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["store_params"]["storage_uri"] == "s3://my-bucket/imgs"
    assert payload["store_params"]["image_format"] == "png"
    assert "store" in payload["stage_order"]


def test_store_rejects_local_path_on_client() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="local path"):
        ing.store(StoreParams(storage_uri="/var/data/imgs"))


def test_webhook_requires_full_url() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="endpoint_url"):
        ing.webhook(WebhookParams(endpoint_url=None))
    with pytest.raises(ValueError, match="http"):
        ing.webhook(WebhookParams(endpoint_url="example.com"))


def test_webhook_populates_spec() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.webhook(WebhookParams(endpoint_url="https://hooks.example.com/jobs"))
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["webhook_params"]["endpoint_url"] == "https://hooks.example.com/jobs"
    assert "webhook" in payload["stage_order"]


def test_vdb_upload_remote_uri_populates_spec() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.vdb_upload(VdbUploadParams(vdb_op="lancedb", vdb_kwargs={"lancedb_uri": "s3://corpus/lancedb"}))
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["vdb_upload_params"]["vdb_op"] == "lancedb"
    assert payload["vdb_upload_params"]["vdb_kwargs"] == {"lancedb_uri": "s3://corpus/lancedb"}
    # vdb_upload doesn't enter stage_order (matches GraphIngestor behaviour).
    assert "vdb_upload" not in payload["stage_order"]


def test_vdb_upload_rejects_local_lancedb_path() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="local path"):
        ing.vdb_upload(VdbUploadParams(vdb_op="lancedb", vdb_kwargs={"lancedb_uri": "./lancedb"}))


def test_vdb_upload_with_sidecar_requires_local_file(tmp_path) -> None:
    """Phase 2 (no sidecar) path: a missing local sidecar file should fail clearly.

    The actual sidecar upload + id substitution lives in
    ``test_service_sidecar.py``; this test just confirms the bare-metal
    error path when the caller points at a non-existent file.
    """
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(FileNotFoundError, match="not found"):
        ing.vdb_upload(
            VdbUploadParams(
                meta_dataframe=str(tmp_path / "does-not-exist.csv"),
                meta_source_field="source",
                meta_fields=["title"],
                vdb_kwargs={"lancedb_uri": "s3://corpus/lancedb"},
            )
        )


# ----------------------------------------------------------------------
# Policy: SinkUrlAllowlist accept / reject
# ----------------------------------------------------------------------


def test_sink_allowlist_disabled_by_default() -> None:
    allowlist = SinkUrlAllowlist()
    assert not allowlist.has_any_sink_enabled()
    with pytest.raises(PolicyError):
        allowlist.check_storage_uri("s3://x/")
    with pytest.raises(PolicyError):
        allowlist.check_webhook_url("https://hooks.example.com/x")


def test_storage_scheme_allowlist_admits_match() -> None:
    allowlist = SinkUrlAllowlist(storage_uri_schemes=["s3://", "gs://"])
    allowlist.check_storage_uri("s3://b/k")
    allowlist.check_storage_uri("gs://b/k")
    with pytest.raises(PolicyError, match="scheme"):
        allowlist.check_storage_uri("azure://b/k")


def test_storage_scheme_wildcard() -> None:
    allowlist = SinkUrlAllowlist(storage_uri_schemes=["*"])
    allowlist.check_storage_uri("any://destination/path")


def test_webhook_prefix_allowlist() -> None:
    allowlist = SinkUrlAllowlist(webhook_url_prefixes=["https://hooks.example.com/"])
    allowlist.check_webhook_url("https://hooks.example.com/jobs/abc")
    with pytest.raises(PolicyError, match="prefix"):
        allowlist.check_webhook_url("https://attacker.example.com/leak")


def test_vdb_uri_allowlist() -> None:
    allowlist = SinkUrlAllowlist(vdb_uri_schemes=["s3://"])
    allowlist.check_vdb_upload({"vdb_kwargs": {"lancedb_uri": "s3://corpus/lancedb"}})
    with pytest.raises(PolicyError, match="scheme"):
        allowlist.check_vdb_upload({"vdb_kwargs": {"lancedb_uri": "gs://corpus/lancedb"}})


def test_vdb_upload_missing_uri_rejects() -> None:
    allowlist = SinkUrlAllowlist(vdb_uri_schemes=["s3://"])
    with pytest.raises(PolicyError, match="lancedb_uri"):
        allowlist.check_vdb_upload({"vdb_kwargs": {}})


# ----------------------------------------------------------------------
# Policy: validate_pipeline_spec end-to-end
# ----------------------------------------------------------------------


def test_validate_admits_store_when_scheme_allowed() -> None:
    cfg = PipelineOverridesConfig(sinks=SinksConfig(storage_uri_schemes=["s3://"]))
    spec = PipelineSpec(
        store_params={"storage_uri": "s3://bucket/imgs", "image_format": "png"},
        stage_order=["extract", "store"],
    )
    out = validate_pipeline_spec(spec, cfg.to_policy())
    assert out is spec


def test_validate_rejects_store_when_scheme_disallowed() -> None:
    cfg = PipelineOverridesConfig(sinks=SinksConfig(storage_uri_schemes=["gs://"]))
    spec = PipelineSpec(store_params={"storage_uri": "s3://bucket/imgs"})
    with pytest.raises(PolicyError):
        validate_pipeline_spec(spec, cfg.to_policy())


def test_validate_admits_webhook_when_prefix_allowed() -> None:
    cfg = PipelineOverridesConfig(sinks=SinksConfig(webhook_url_prefixes=["https://hooks.example.com/"]))
    spec = PipelineSpec(
        webhook_params={"endpoint_url": "https://hooks.example.com/notify"},
        stage_order=["extract", "webhook"],
    )
    out = validate_pipeline_spec(spec, cfg.to_policy())
    assert out is spec


def test_validate_rejects_webhook_with_disallowed_prefix() -> None:
    cfg = PipelineOverridesConfig(sinks=SinksConfig(webhook_url_prefixes=["https://hooks.example.com/"]))
    spec = PipelineSpec(webhook_params={"endpoint_url": "https://attacker.evil/leak"})
    with pytest.raises(PolicyError, match="prefix"):
        validate_pipeline_spec(spec, cfg.to_policy())


def test_validate_admits_vdb_upload_when_scheme_allowed() -> None:
    cfg = PipelineOverridesConfig(sinks=SinksConfig(vdb_uri_schemes=["s3://"]))
    spec = PipelineSpec(vdb_upload_params={"vdb_op": "lancedb", "vdb_kwargs": {"lancedb_uri": "s3://corpus/lancedb"}})
    out = validate_pipeline_spec(spec, cfg.to_policy())
    assert out is spec


def test_validate_rejects_vdb_kwargs_storage_options_nested() -> None:
    cfg = PipelineOverridesConfig(sinks=SinksConfig(vdb_uri_schemes=["s3://"]))
    spec = PipelineSpec(
        vdb_upload_params={
            "vdb_op": "lancedb",
            "vdb_kwargs": {
                "lancedb_uri": "s3://corpus/lancedb",
                "storage_options": {"endpoint_url": "https://attacker.example/"},
            },
        }
    )
    with pytest.raises(PolicyError, match="allow_list"):
        validate_pipeline_spec(spec, cfg.to_policy())


def test_validate_rejects_vdb_kwargs_endpoint_override() -> None:
    cfg = PipelineOverridesConfig(sinks=SinksConfig(vdb_uri_schemes=["s3://"]))
    spec = PipelineSpec(
        vdb_upload_params={
            "vdb_op": "lancedb",
            "vdb_kwargs": {
                "lancedb_uri": "s3://allowed-bucket/",
                "endpoint_override": "https://attacker.example/",
            },
        }
    )
    with pytest.raises(PolicyError, match="trust-sensitive"):
        validate_pipeline_spec(spec, cfg.to_policy())


def test_validate_rejects_storage_options_by_default() -> None:
    cfg = PipelineOverridesConfig(sinks=SinksConfig(storage_uri_schemes=["s3://"]))
    spec = PipelineSpec(
        store_params={
            "storage_uri": "s3://bucket/imgs",
            "storage_options": {"anon": True},
        }
    )
    with pytest.raises(PolicyError, match="allow_list"):
        validate_pipeline_spec(spec, cfg.to_policy())


def test_validate_rejects_nested_storage_options_endpoint_when_enabled() -> None:
    cfg = PipelineOverridesConfig(
        sinks=SinksConfig(storage_uri_schemes=["s3://"]),
        extra_store_keys=["storage_options"],
    )
    spec = PipelineSpec(
        store_params={
            "storage_uri": "s3://bucket/imgs",
            "storage_options": {"endpoint_url": "https://attacker.example/"},
        }
    )
    with pytest.raises(PolicyError, match="trust-sensitive"):
        validate_pipeline_spec(spec, cfg.to_policy())


def test_validate_admits_vdb_kwargs_shape_keys() -> None:
    cfg = PipelineOverridesConfig(sinks=SinksConfig(vdb_uri_schemes=["s3://"]))
    spec = PipelineSpec(
        vdb_upload_params={
            "vdb_op": "lancedb",
            "vdb_kwargs": {
                "lancedb_uri": "s3://corpus/lancedb",
                "table_name": "docs",
                "overwrite": True,
            },
        }
    )
    out = validate_pipeline_spec(spec, cfg.to_policy())
    assert out is spec


# ----------------------------------------------------------------------
# Worker: _build_graph_ingestor_from_spec wires the stages
# ----------------------------------------------------------------------


def test_worker_wires_store_stage_from_spec() -> None:
    spec = {
        "extraction_mode": "pdf",
        "extract_params": {"dpi": 200},
        "store_params": {"storage_uri": "s3://bucket/imgs", "image_format": "png"},
        "stage_order": ["extract", "store"],
    }
    ingestor, mode, has_vdb = _build_graph_ingestor_from_spec(
        "doc.pdf", b"%PDF-1.4", base_extract={}, base_embed=None, spec=spec
    )
    assert mode == "pdf"
    assert has_vdb is False
    assert ingestor._store_params is not None
    assert ingestor._store_params.storage_uri == "s3://bucket/imgs"


def test_worker_wires_webhook_stage_from_spec() -> None:
    spec = {
        "webhook_params": {"endpoint_url": "https://hooks.example.com/x", "columns": ["text"]},
        "stage_order": ["extract", "webhook"],
    }
    ingestor, _mode, _has_vdb = _build_graph_ingestor_from_spec(
        "doc.pdf", b"%PDF-1.4", base_extract={}, base_embed=None, spec=spec
    )
    assert ingestor._webhook_params is not None
    assert ingestor._webhook_params.endpoint_url == "https://hooks.example.com/x"


def test_worker_wires_vdb_upload_and_flags_skip() -> None:
    spec = {
        "vdb_upload_params": {"vdb_op": "lancedb", "vdb_kwargs": {"lancedb_uri": "s3://corpus/lancedb"}},
    }
    ingestor, _mode, has_vdb = _build_graph_ingestor_from_spec(
        "doc.pdf", b"%PDF-1.4", base_extract={}, base_embed=None, spec=spec
    )
    assert has_vdb is True
    assert ingestor._vdb_upload_params is not None
    assert ingestor._vdb_upload_params.vdb_op == "lancedb"


def test_worker_auto_appends_store_without_stage_order_entry() -> None:
    """``store_params`` alone (no ``stage_order=["store"]``) still wires the stage."""
    spec = {"store_params": {"storage_uri": "s3://bucket/imgs"}}
    ingestor, _mode, _has_vdb = _build_graph_ingestor_from_spec(
        "doc.pdf", b"%PDF-1.4", base_extract={}, base_embed=None, spec=spec
    )
    assert ingestor._store_params is not None


# ----------------------------------------------------------------------
# Router integration: TestClient round-trip with sinks enabled
# ----------------------------------------------------------------------


@pytest.fixture
def captured_items() -> list[WorkItem]:
    return []


@pytest.fixture
def app_with_sinks(monkeypatch: pytest.MonkeyPatch, captured_items: list[WorkItem]):
    """Standalone-mode app with sinks enabled and a stub work-fn."""

    async def _stub_work(item: WorkItem) -> tuple[int, list[dict[str, Any]]]:
        captured_items.append(item)
        return 1, []

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        lambda _cfg: _stub_work,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        lambda _cfg: _stub_work,
    )

    cfg = ServiceConfig(
        mode="standalone",
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        pipeline_overrides=PipelineOverridesConfig(
            sinks=SinksConfig(
                storage_uri_schemes=["s3://"],
                webhook_url_prefixes=["https://hooks.example.com/"],
                vdb_uri_schemes=["s3://"],
            )
        ),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client


def _wait_for_captures(captured_items: list[WorkItem], count: int = 1, timeout_s: float = 5.0) -> None:
    import time as _time

    deadline = _time.monotonic() + timeout_s
    while len(captured_items) < count and _time.monotonic() < deadline:
        _time.sleep(0.05)


def test_router_accepts_store_with_allowed_scheme(app_with_sinks: TestClient, captured_items: list[WorkItem]) -> None:
    from .conftest import create_test_job

    metadata = {
        "pipeline": {
            "store_params": {"storage_uri": "s3://bucket/imgs"},
            "stage_order": ["extract", "store"],
        }
    }
    job_id = create_test_job(app_with_sinks)
    resp = app_with_sinks.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", b"%PDF-1.4\n%stub\n", "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 202, resp.text
    _wait_for_captures(captured_items)
    assert len(captured_items) == 1
    item = captured_items[0]
    assert item.pipeline_spec is not None
    assert item.pipeline_spec["store_params"]["storage_uri"] == "s3://bucket/imgs"


def test_router_rejects_store_with_disallowed_scheme(app_with_sinks: TestClient) -> None:
    from .conftest import create_test_job

    metadata = {"pipeline": {"store_params": {"storage_uri": "gs://bucket/imgs"}}}
    job_id = create_test_job(app_with_sinks)
    resp = app_with_sinks.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", b"%PDF", "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 403, resp.text
    assert "scheme" in resp.json()["detail"].lower()


def test_router_rejects_webhook_with_disallowed_prefix(app_with_sinks: TestClient) -> None:
    from .conftest import create_test_job

    metadata = {"pipeline": {"webhook_params": {"endpoint_url": "https://attacker.evil/leak"}}}
    job_id = create_test_job(app_with_sinks)
    resp = app_with_sinks.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", b"%PDF", "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 403, resp.text
    assert "prefix" in resp.json()["detail"].lower()


def test_pipeline_config_reports_sinks(app_with_sinks: TestClient) -> None:
    resp = app_with_sinks.get("/v1/ingest/pipeline-config")
    assert resp.status_code == 200
    body = resp.json()
    sinks = body["allowed_overrides"]["sinks"]
    assert sinks["storage_uri_schemes"] == ["s3://"]
    assert sinks["webhook_url_prefixes"] == ["https://hooks.example.com/"]
    assert sinks["vdb_uri_schemes"] == ["s3://"]
    # store / webhook stages are now in the allowed list
    assert "store" in body["allowed_overrides"]["allowed_stages"]
    assert "webhook" in body["allowed_overrides"]["allowed_stages"]
