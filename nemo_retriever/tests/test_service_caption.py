# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 tests for the ``caption`` stage in service run_mode.

Captioning is the canonical example of "trust the operator, not the
client": the remote VLM endpoint URL + API key + model name are
configured by the service operator (``nim_endpoints.caption_invoke_url``
in ``retriever-service.yaml``). Clients may submit behavioural knobs —
``prompt``, ``system_prompt``, ``batch_size``, sampling params — but
*never* redirect the destination.

The CPU-only worker pod cannot honor local-execution params (``device``,
``hf_cache_dir``, ``tensor_parallel_size``, ``gpu_memory_utilization``)
either, so we surface that mismatch immediately on the client and the
server.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nemo_retriever.common.params import CaptionParams
from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import (
    NimEndpointsConfig,
    PipelineOverridesConfig,
    PipelinePoolConfig,
    ServiceConfig,
)
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.common.policy import (
    PolicyError,
    PipelineOverridesPolicy,
    validate_pipeline_spec,
)
from nemo_retriever.service.services.pipeline_executor import (
    _TRUST_OWNED_CAPTION_KEYS,
    _build_graph_ingestor_from_spec,
    _merge_server_owned,
    build_caption_params,
)
from nemo_retriever.service.services.pipeline_pool import WorkItem
from nemo_retriever.service.service_ingestor import ServiceIngestor


# ----------------------------------------------------------------------
# Client: ServiceIngestor.caption() shapes the spec
# ----------------------------------------------------------------------


def test_caption_populates_spec_with_behaviour_knobs_only() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.caption(
        CaptionParams(
            prompt="Caption succinctly.",
            system_prompt="/think",
            batch_size=4,
            caption_infographics=True,
        )
    )
    payload = ing._pipeline_payload()
    assert payload is not None
    caption = payload["caption_params"]
    assert caption["prompt"] == "Caption succinctly."
    assert caption["system_prompt"] == "/think"
    assert caption["batch_size"] == 4
    assert caption["caption_infographics"] is True
    # Trust-sensitive fields must never appear in the wire payload.
    assert "endpoint_url" not in caption
    assert "api_key" not in caption
    assert "model_name" not in caption
    # caption is in stage_order.
    assert "caption" in payload["stage_order"]


def test_caption_rejects_endpoint_url_via_kwargs() -> None:
    """kwargs are unambiguous caller intent; the client rejects immediately."""
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="server-owned"):
        ing.caption(endpoint_url="http://attacker.evil/v1")


def test_caption_rejects_api_key_via_kwargs() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="server-owned"):
        ing.caption(api_key="leaked-secret")


def test_caption_rejects_model_name_via_kwargs() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="server-owned"):
        ing.caption(model_name="evil/model")


def test_caption_rejects_local_execution_keys_via_kwargs() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="local in-process GPU"):
        ing.caption(device="cuda:0")
    with pytest.raises(ValueError, match="local in-process GPU"):
        ing.caption(tensor_parallel_size=4)


def test_caption_silently_strips_class_default_api_key_and_model_name() -> None:
    """Defaults from CaptionParams (env-injected api_key, default model_name)
    shouldn't trip the client check — they're not caller intent. The
    server-owned merge in the worker reapplies the operator's true values.
    """
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.caption(CaptionParams(prompt="just a prompt"))
    payload = ing._pipeline_payload()
    assert payload is not None
    caption = payload["caption_params"]
    assert "api_key" not in caption
    assert "endpoint_url" not in caption
    assert "model_name" not in caption


# ----------------------------------------------------------------------
# Policy: caption_enabled gates the stage
# ----------------------------------------------------------------------


def test_policy_caption_disabled_by_default() -> None:
    policy = PipelineOverridesPolicy()
    assert policy.caption_enabled is False
    assert "caption" not in policy.allowed_stages


def test_policy_caption_enabled_opens_stage() -> None:
    policy = PipelineOverridesPolicy(caption_enabled=True)
    assert policy.caption_enabled is True
    assert "caption" in policy.allowed_stages


def test_validate_admits_caption_when_endpoint_configured() -> None:
    cfg = PipelineOverridesConfig()
    policy = cfg.to_policy(caption_enabled=True)
    spec = PipelineSpec(
        caption_params={"prompt": "Describe", "batch_size": 8},
        stage_order=["extract", "caption"],
    )
    out = validate_pipeline_spec(spec, policy)
    assert out is spec


def test_validate_rejects_caption_when_endpoint_absent() -> None:
    cfg = PipelineOverridesConfig()
    policy = cfg.to_policy(caption_enabled=False)
    spec = PipelineSpec(caption_params={"prompt": "Describe"})
    with pytest.raises(PolicyError, match="endpoint"):
        validate_pipeline_spec(spec, policy)


def test_validate_rejects_local_execution_keys_even_when_caption_enabled() -> None:
    cfg = PipelineOverridesConfig()
    policy = cfg.to_policy(caption_enabled=True)
    spec = PipelineSpec(caption_params={"device": "cuda:0", "prompt": "x"})
    with pytest.raises(PolicyError, match="local in-process GPU"):
        validate_pipeline_spec(spec, policy)


def test_validate_rejects_trust_sensitive_caption_override() -> None:
    cfg = PipelineOverridesConfig()
    policy = cfg.to_policy(caption_enabled=True)
    spec = PipelineSpec(caption_params={"endpoint_url": "http://x/v1"})
    with pytest.raises(PolicyError, match="trust-sensitive"):
        validate_pipeline_spec(spec, policy)


def test_validate_rejects_unknown_caption_key_in_allow_list_mode() -> None:
    cfg = PipelineOverridesConfig()
    policy = cfg.to_policy(caption_enabled=True)
    spec = PipelineSpec(caption_params={"vendor_secret_knob": 42})
    with pytest.raises(PolicyError, match="allow_list"):
        validate_pipeline_spec(spec, policy)


def test_validate_admits_extra_caption_key_when_operator_widens() -> None:
    cfg = PipelineOverridesConfig(extra_caption_keys=["vendor_secret_knob"])
    policy = cfg.to_policy(caption_enabled=True)
    spec = PipelineSpec(caption_params={"vendor_secret_knob": 42, "prompt": "x"})
    out = validate_pipeline_spec(spec, policy)
    assert out is spec


# ----------------------------------------------------------------------
# Worker merge: server-owned caption keys always win
# ----------------------------------------------------------------------


def test_merge_preserves_server_caption_endpoint_and_model() -> None:
    base = {
        "endpoint_url": "http://server/v1/caption",
        "api_key": "server-secret",
        "model_name": "nvidia/nv-vlm",
    }
    override = {
        "endpoint_url": "http://attacker.evil",
        "api_key": "leaked",
        "model_name": "evil/model",
        "prompt": "Custom prompt",
        "batch_size": 16,
    }
    merged = _merge_server_owned(base, override, _TRUST_OWNED_CAPTION_KEYS)
    assert merged["endpoint_url"] == "http://server/v1/caption"
    assert merged["api_key"] == "server-secret"
    assert merged["model_name"] == "nvidia/nv-vlm"
    assert merged["prompt"] == "Custom prompt"
    assert merged["batch_size"] == 16


# ----------------------------------------------------------------------
# Worker: _build_graph_ingestor_from_spec wires the caption stage
# ----------------------------------------------------------------------


def test_worker_wires_caption_with_merged_params() -> None:
    base_caption = {
        "endpoint_url": "http://server/v1/caption",
        "api_key": "k",
        "model_name": "nvidia/vlm",
    }
    spec = {
        "extraction_mode": "pdf",
        "caption_params": {"prompt": "Caption tightly.", "batch_size": 2},
        "stage_order": ["extract", "caption"],
    }
    ingestor, _mode, _has_vdb = _build_graph_ingestor_from_spec(
        "doc.pdf",
        b"%PDF",
        base_extract={},
        base_embed=None,
        spec=spec,
        base_caption=base_caption,
    )
    assert ingestor._caption_params is not None
    cp = ingestor._caption_params
    assert cp.endpoint_url == "http://server/v1/caption"
    assert cp.model_name == "nvidia/vlm"
    assert cp.prompt == "Caption tightly."
    assert cp.batch_size == 2


def test_worker_auto_appends_caption_without_stage_order_entry() -> None:
    base_caption = {"endpoint_url": "http://server/v1/caption", "api_key": "k"}
    spec = {"caption_params": {"prompt": "anything"}}
    ingestor, _mode, _has_vdb = _build_graph_ingestor_from_spec(
        "doc.pdf",
        b"%PDF",
        base_extract={},
        base_embed=None,
        spec=spec,
        base_caption=base_caption,
    )
    assert ingestor._caption_params is not None
    assert ingestor._caption_params.prompt == "anything"


def test_worker_raises_when_caption_spec_without_base() -> None:
    """If the policy slips through but no base endpoint is configured, fail loud."""
    spec = {"caption_params": {"prompt": "x"}}
    with pytest.raises(RuntimeError, match="no caption endpoint"):
        _build_graph_ingestor_from_spec(
            "doc.pdf", b"%PDF", base_extract={}, base_embed=None, spec=spec, base_caption=None
        )


def test_build_caption_params_returns_none_without_endpoint() -> None:
    nim = NimEndpointsConfig()
    assert build_caption_params(nim) is None


def test_build_caption_params_from_nim_config() -> None:
    nim = NimEndpointsConfig(
        caption_invoke_url="http://server/v1/caption",
        caption_model_name="nvidia/nv-vlm",
        api_key="k",
    )
    cp = build_caption_params(nim)
    assert cp is not None
    assert cp.endpoint_url == "http://server/v1/caption"
    assert cp.model_name == "nvidia/nv-vlm"
    assert cp.api_key == "k"


# ----------------------------------------------------------------------
# Router integration: TestClient round-trip with caption enabled
# ----------------------------------------------------------------------


@pytest.fixture
def captured_items() -> list[WorkItem]:
    return []


@pytest.fixture
def app_with_caption(monkeypatch: pytest.MonkeyPatch, captured_items: list[WorkItem]):
    """Standalone-mode app with caption endpoint configured and a stub work-fn."""

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
        nim_endpoints=NimEndpointsConfig(
            caption_invoke_url="http://caption.svc/v1",
            caption_model_name="nvidia/nv-vlm",
            api_key="server-secret",
        ),
        pipeline_overrides=PipelineOverridesConfig(),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client


def _wait_for_captures(captured_items: list[WorkItem], count: int = 1, timeout_s: float = 5.0) -> None:
    import time as _time

    deadline = _time.monotonic() + timeout_s
    while len(captured_items) < count and _time.monotonic() < deadline:
        _time.sleep(0.05)


def test_router_admits_caption_when_endpoint_configured(
    app_with_caption: TestClient, captured_items: list[WorkItem]
) -> None:
    from .conftest import create_test_job

    metadata = {
        "pipeline": {
            "caption_params": {"prompt": "Describe", "batch_size": 8},
            "stage_order": ["extract", "caption"],
        }
    }
    job_id = create_test_job(app_with_caption)
    resp = app_with_caption.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", b"%PDF-1.4\n%stub\n", "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 202, resp.text
    _wait_for_captures(captured_items)
    assert len(captured_items) == 1
    item = captured_items[0]
    assert item.pipeline_spec is not None
    assert item.pipeline_spec["caption_params"]["prompt"] == "Describe"
    assert item.pipeline_spec["caption_params"]["batch_size"] == 8


def test_router_rejects_caption_endpoint_override(app_with_caption: TestClient) -> None:
    """Even with the stage enabled, clients cannot redirect the endpoint."""
    from .conftest import create_test_job

    metadata = {"pipeline": {"caption_params": {"endpoint_url": "http://attacker.evil/v1"}}}
    job_id = create_test_job(app_with_caption)
    resp = app_with_caption.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", b"%PDF", "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 403, resp.text
    assert "trust-sensitive" in resp.json()["detail"].lower()


def test_router_rejects_local_execution_keys_with_caption_enabled(
    app_with_caption: TestClient,
) -> None:
    from .conftest import create_test_job

    metadata = {"pipeline": {"caption_params": {"device": "cuda:0", "prompt": "x"}}}
    job_id = create_test_job(app_with_caption)
    resp = app_with_caption.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", b"%PDF", "application/pdf")},
        data={"metadata": json.dumps(metadata)},
    )
    assert resp.status_code == 403, resp.text
    assert "local in-process gpu" in resp.json()["detail"].lower()


def test_pipeline_config_reports_caption_enabled(app_with_caption: TestClient) -> None:
    resp = app_with_caption.get("/v1/ingest/pipeline-config")
    assert resp.status_code == 200
    body = resp.json()["allowed_overrides"]
    assert body["caption_enabled"] is True
    assert "caption" in body["allowed_stages"]
    assert "prompt" in body["allowed_caption_keys"]
    assert "batch_size" in body["allowed_caption_keys"]
