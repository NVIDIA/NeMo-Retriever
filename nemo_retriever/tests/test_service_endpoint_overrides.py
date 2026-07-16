# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request model-endpoint overrides for service run_mode.

Covers the opt-in ``pipeline_overrides.endpoint_overrides`` channel across
every layer:

* schema — :class:`EndpointOverrides` emptiness + :class:`PipelineSpec` wiring;
* config — ``EndpointOverridesConfig`` → ``EndpointOverridePolicy`` plumbing;
* policy — accept when the operator opted in, reject otherwise, prefix
  allowlist enforcement, client-supplied caption endpoints unlocking the
  caption stage;
* worker — the base embed / caption dicts are retargeted and win the
  server-owned merge;
* client — ``ServiceIngestor.embed(...)`` / ``.caption(...)`` route model
  endpoints into the spec's ``endpoint_overrides`` channel; and
* answer — ``POST /v1/answer`` honors a per-request LLM override only when
  enabled, and reranks retrieval hits (server default or per-request
  rerank endpoint override) before answer generation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nemo_retriever.common.params import EmbedParams
from nemo_retriever.common.policy import PolicyError, validate_pipeline_spec
from nemo_retriever.common.schemas.pipeline_spec import EndpointOverrides, PipelineSpec
from nemo_retriever.models.llm.types import GenerationResult
from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import (
    EndpointOverridesConfig,
    LLMConfig,
    LoggingConfig,
    NimEndpointsConfig,
    PipelineOverridesConfig,
    PipelinePoolConfig,
    RerankConfig,
    ServiceConfig,
    VectorDbConfig,
)
from nemo_retriever.service.service_ingestor import ServiceIngestor
from nemo_retriever.service.services.pipeline_executor import (
    _apply_caption_endpoint_override,
    _apply_embed_endpoint_override,
    _build_graph_ingestor_from_spec,
)


@pytest.fixture(autouse=True)
def _no_remote_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------


def test_endpoint_overrides_is_empty() -> None:
    assert EndpointOverrides().is_empty()
    assert not EndpointOverrides(embed_invoke_url="http://x/embed").is_empty()
    assert not EndpointOverrides(caption_model_name="my-vlm").is_empty()


def test_pipeline_spec_with_endpoint_overrides_is_not_empty() -> None:
    spec = PipelineSpec(endpoint_overrides=EndpointOverrides(embed_invoke_url="http://x/embed"))
    assert not spec.is_empty()
    # An all-None overrides block does not make the spec non-empty.
    assert PipelineSpec(endpoint_overrides=EndpointOverrides()).is_empty()


# ----------------------------------------------------------------------
# Config → policy plumbing
# ----------------------------------------------------------------------


def test_config_defaults_disable_all_endpoint_overrides() -> None:
    cfg = EndpointOverridesConfig()
    assert (cfg.embed, cfg.caption, cfg.llm, cfg.rerank) == (False, False, False, False)
    assert cfg.allowed_url_prefixes == []


def test_to_policy_carries_endpoint_override_flags() -> None:
    cfg = PipelineOverridesConfig(
        endpoint_overrides=EndpointOverridesConfig(
            embed=True, caption=True, rerank=True, allowed_url_prefixes=["https://"]
        )
    )
    policy = cfg.to_policy()
    assert policy.endpoint_overrides.embed is True
    assert policy.endpoint_overrides.caption is True
    assert policy.endpoint_overrides.llm is False
    assert policy.endpoint_overrides.rerank is True
    assert policy.endpoint_overrides.allowed_url_prefixes == ["https://"]
    described = policy.describe()["endpoint_overrides"]
    assert described["embed"] is True
    assert described["rerank"] is True
    assert described["allowed_url_prefixes"] == ["https://"]


def test_check_rerank_policy_accept_and_reject() -> None:
    from nemo_retriever.common.policy import EndpointOverridePolicy

    disabled = EndpointOverridePolicy()
    with pytest.raises(PolicyError) as exc:
        disabled.check_rerank(url="http://x/rerank")
    assert exc.value.status_code == 403

    enabled = EndpointOverridePolicy(rerank=True, allowed_url_prefixes=["https://"])
    enabled.check_rerank(url="https://ok/rerank")  # no raise
    with pytest.raises(PolicyError):
        enabled.check_rerank(url="http://blocked/rerank")
    assert enabled.any_enabled() is True


# ----------------------------------------------------------------------
# Policy: accept / reject
# ----------------------------------------------------------------------


def test_policy_rejects_endpoint_override_when_disabled() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(endpoint_overrides=EndpointOverrides(embed_invoke_url="http://x/embed"))
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, policy)
    assert exc.value.status_code == 403
    assert "endpoint_overrides" in exc.value.detail


def test_policy_accepts_embed_override_when_enabled() -> None:
    cfg = PipelineOverridesConfig(endpoint_overrides=EndpointOverridesConfig(embed=True))
    spec = PipelineSpec(endpoint_overrides=EndpointOverrides(embed_invoke_url="http://x/embed", embed_model_name="m"))
    assert validate_pipeline_spec(spec, cfg.to_policy()) is spec


def test_policy_rejects_caption_override_when_only_embed_enabled() -> None:
    cfg = PipelineOverridesConfig(endpoint_overrides=EndpointOverridesConfig(embed=True))
    spec = PipelineSpec(endpoint_overrides=EndpointOverrides(caption_invoke_url="http://x/vlm"))
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, cfg.to_policy())
    assert exc.value.status_code == 403


def test_policy_enforces_url_prefix_allowlist() -> None:
    cfg = PipelineOverridesConfig(
        endpoint_overrides=EndpointOverridesConfig(embed=True, allowed_url_prefixes=["https://trusted/"])
    )
    ok = PipelineSpec(endpoint_overrides=EndpointOverrides(embed_invoke_url="https://trusted/embed"))
    assert validate_pipeline_spec(ok, cfg.to_policy()) is ok

    bad = PipelineSpec(endpoint_overrides=EndpointOverrides(embed_invoke_url="http://evil/embed"))
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(bad, cfg.to_policy())
    assert exc.value.status_code == 403


def test_policy_rejects_bare_api_key_without_endpoint() -> None:
    cfg = PipelineOverridesConfig(endpoint_overrides=EndpointOverridesConfig(embed=True))
    spec = PipelineSpec(endpoint_overrides=EndpointOverrides(api_key="leaked"))
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, cfg.to_policy())
    assert exc.value.status_code == 400


def test_endpoint_only_override_allowed_under_reject_mode() -> None:
    """endpoint_overrides is an independent gate, so mode='reject' does not block it."""
    cfg = PipelineOverridesConfig(mode="reject", endpoint_overrides=EndpointOverridesConfig(embed=True))
    spec = PipelineSpec(endpoint_overrides=EndpointOverrides(embed_invoke_url="http://x/embed"))
    assert validate_pipeline_spec(spec, cfg.to_policy()) is spec


def test_client_caption_endpoint_unlocks_caption_stage() -> None:
    """A client-supplied VLM endpoint enables caption even without a server NIM."""
    cfg = PipelineOverridesConfig(endpoint_overrides=EndpointOverridesConfig(caption=True))
    spec = PipelineSpec(
        endpoint_overrides=EndpointOverrides(caption_invoke_url="http://x/vlm"),
        caption_params={"prompt": "Describe"},
        stage_order=["extract", "caption"],
    )
    # caption_enabled=False on the server, but the client brought its own endpoint.
    out = validate_pipeline_spec(spec, cfg.to_policy(caption_enabled=False))
    assert out is spec


def test_client_caption_params_without_endpoint_still_rejected_when_server_has_none() -> None:
    cfg = PipelineOverridesConfig(endpoint_overrides=EndpointOverridesConfig(caption=True))
    spec = PipelineSpec(caption_params={"prompt": "Describe"}, stage_order=["extract", "caption"])
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, cfg.to_policy(caption_enabled=False))
    assert exc.value.status_code == 403


# ----------------------------------------------------------------------
# Worker merge
# ----------------------------------------------------------------------


def test_apply_embed_endpoint_override_retargets_base() -> None:
    base = {"embed_invoke_url": "http://server/embed", "model_name": "server-m", "api_key": "server-key"}
    ov = {"embed_invoke_url": "http://client/embed", "embed_model_name": "client-m", "api_key": "client-key"}
    out = _apply_embed_endpoint_override(base, ov)
    assert out["embed_invoke_url"] == "http://client/embed"
    assert out["model_name"] == "client-m"
    assert out["embed_model_name"] == "client-m"
    assert out["api_key"] == "client-key"
    # The base dict is not mutated in place.
    assert base["embed_invoke_url"] == "http://server/embed"


def test_apply_embed_endpoint_override_noop_without_embed_fields() -> None:
    base = {"embed_invoke_url": "http://server/embed"}
    assert _apply_embed_endpoint_override(base, {"caption_invoke_url": "http://x/vlm"}) is base


def test_apply_caption_endpoint_override_creates_base_when_none() -> None:
    ov = {"caption_invoke_url": "http://client/vlm", "caption_model_name": "vlm-x", "api_key": "k"}
    out = _apply_caption_endpoint_override(None, ov)
    assert out == {"endpoint_url": "http://client/vlm", "model_name": "vlm-x", "api_key": "k"}


def test_build_graph_ingestor_applies_embed_endpoint_override() -> None:
    base_embed = {"embed_invoke_url": "http://server/embed", "model_name": "server-m", "api_key": "server-key"}
    spec = {
        "extraction_mode": "auto",
        "stage_order": ["extract", "embed"],
        "endpoint_overrides": {"embed_invoke_url": "http://client/embed", "embed_model_name": "client-m"},
    }
    ingestor, _mode, _ = _build_graph_ingestor_from_spec(
        "doc.pdf",
        b"%PDF-1.4 stub",
        {},
        base_embed,
        spec,
    )
    assert ingestor._embed_params is not None
    assert ingestor._embed_params.embed_invoke_url == "http://client/embed"
    assert ingestor._embed_params.model_name == "client-m"
    # Server key is preserved because the client did not override it.
    assert ingestor._embed_params.api_key == "server-key"


def test_build_graph_ingestor_applies_caption_endpoint_override_without_server_endpoint() -> None:
    spec = {
        "extraction_mode": "auto",
        "stage_order": ["extract", "caption"],
        "caption_params": {"prompt": "Describe the figure"},
        "endpoint_overrides": {"caption_invoke_url": "http://client/vlm", "caption_model_name": "vlm-x"},
    }
    ingestor, _mode, _ = _build_graph_ingestor_from_spec(
        "doc.pdf",
        b"%PDF-1.4 stub",
        {},
        None,
        spec,
        base_caption=None,
    )
    assert ingestor._caption_params is not None
    assert ingestor._caption_params.endpoint_url == "http://client/vlm"
    assert ingestor._caption_params.model_name == "vlm-x"
    assert ingestor._caption_params.prompt == "Describe the figure"


def test_build_graph_ingestor_client_cannot_override_via_embed_params() -> None:
    """The denylist still protects the ordinary embed_params path."""
    base_embed = {"embed_invoke_url": "http://server/embed", "api_key": "server-key"}
    spec = {
        "extraction_mode": "auto",
        "stage_order": ["extract", "embed"],
        # Even if a malicious embed_params slipped past validation, the
        # server-owned merge restores the endpoint.
        "embed_params": {"embed_invoke_url": "http://attacker/", "inference_batch_size": 8},
    }
    ingestor, _mode, _ = _build_graph_ingestor_from_spec("doc.pdf", b"%PDF-1.4 stub", {}, base_embed, spec)
    assert ingestor._embed_params.embed_invoke_url == "http://server/embed"


# ----------------------------------------------------------------------
# Client SDK
# ----------------------------------------------------------------------


def test_embed_routes_endpoint_fields_to_overrides() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.embed(
        embed_invoke_url="http://client/embed",
        embed_model_name="client-embed",
        embed_model_provider_prefix="openai",
        inference_batch_size=64,
    )
    payload = ing._pipeline_payload()
    assert payload is not None
    ov = payload["endpoint_overrides"]
    assert ov["embed_invoke_url"] == "http://client/embed"
    assert ov["embed_model_name"] == "client-embed"
    assert ov["embed_model_provider_prefix"] == "openai"
    # Shape knobs stay in embed_params; the endpoint fields never leak there.
    assert payload["embed_params"]["inference_batch_size"] == 64
    assert "embed_invoke_url" not in payload["embed_params"]
    assert "embed" in payload["stage_order"]
    # Round-trips through the wire schema.
    assert PipelineSpec.model_validate(payload).endpoint_overrides.embed_invoke_url == "http://client/embed"


def test_embed_via_embed_params_model_routes_endpoint() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.embed(EmbedParams(embed_invoke_url="http://client/embed", inference_batch_size=8))
    payload = ing._pipeline_payload()
    assert payload["endpoint_overrides"]["embed_invoke_url"] == "http://client/embed"
    assert payload["embed_params"]["inference_batch_size"] == 8


def test_embed_without_endpoint_sets_no_overrides() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.embed(inference_batch_size=32)
    payload = ing._pipeline_payload()
    assert "endpoint_overrides" not in payload
    assert payload["embed_params"]["inference_batch_size"] == 32


def test_caption_routes_endpoint_fields_to_overrides() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.caption(endpoint_url="http://client/vlm", model_name="vlm-x", prompt="Describe")
    payload = ing._pipeline_payload()
    ov = payload["endpoint_overrides"]
    assert ov["caption_invoke_url"] == "http://client/vlm"
    assert ov["caption_model_name"] == "vlm-x"
    # Behavioural knobs stay in caption_params; endpoint/model do not leak.
    assert payload["caption_params"]["prompt"] == "Describe"
    assert "endpoint_url" not in payload["caption_params"]
    assert "model_name" not in payload["caption_params"]


def test_caption_without_endpoint_sets_no_overrides() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.caption(prompt="Describe")
    payload = ing._pipeline_payload()
    assert "endpoint_overrides" not in payload
    assert payload["caption_params"]["prompt"] == "Describe"


def test_caption_rejects_local_execution_keys() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="local"):
        ing.caption(device="cuda:0")


def test_embed_and_caption_endpoint_overrides_merge() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.embed(embed_invoke_url="http://client/embed").caption(endpoint_url="http://client/vlm")
    ov = ing._pipeline_payload()["endpoint_overrides"]
    assert ov["embed_invoke_url"] == "http://client/embed"
    assert ov["caption_invoke_url"] == "http://client/vlm"


# ----------------------------------------------------------------------
# Answer path: LLM endpoint override
# ----------------------------------------------------------------------


class _AnswerFakeResponse:
    status_code = 200
    content = json.dumps({"results": [{"hits": [{"text": "context"}]}]}).encode()

    def json(self) -> dict[str, Any]:
        return json.loads(self.content.decode())


class _AnswerFakeAsyncClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, **kwargs) -> _AnswerFakeResponse:
        return _AnswerFakeResponse()


def _make_answer_app(monkeypatch: pytest.MonkeyPatch, tmp_path, *, llm_override: bool, prefixes=None):
    async def _stub_work(_item):
        return 0, []

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        lambda _config: _stub_work,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        lambda _config: _stub_work,
    )
    cfg = ServiceConfig(
        mode="standalone",
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(enabled=True, vectordb_url="http://vectordb:7671"),
        llm=LLMConfig(
            enabled=True,
            model="server/model",
            api_base="http://server-llm:8000/v1",
            api_key="server-key",
            max_tokens=128,
        ),
        pipeline_overrides=PipelineOverridesConfig(
            endpoint_overrides=EndpointOverridesConfig(llm=llm_override, allowed_url_prefixes=prefixes or [])
        ),
    )
    return create_app(cfg)


def test_answer_llm_override_rejected_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    app = _make_answer_app(monkeypatch, tmp_path, llm_override=False)
    monkeypatch.setattr("httpx.AsyncClient", _AnswerFakeAsyncClient)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/answer",
            json={"query": "q", "llm_model": "client/model", "llm_api_base": "http://client-llm:8000/v1"},
        )
    assert resp.status_code == 403
    assert "LLM endpoint overrides are disabled" in resp.json()["detail"]


def test_answer_llm_override_applied_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    app = _make_answer_app(monkeypatch, tmp_path, llm_override=True)
    monkeypatch.setattr("httpx.AsyncClient", _AnswerFakeAsyncClient)

    fake_llm = SimpleNamespace(
        generate=lambda query, chunks, *, reasoning_enabled=None: GenerationResult(
            answer="ok", latency_s=0.1, model="client/model"
        )
    )
    with patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs", return_value=fake_llm) as from_kwargs:
        with TestClient(app) as client:
            resp = client.post(
                "/v1/answer",
                json={
                    "query": "q",
                    "llm_model": "client/model",
                    "llm_api_base": "http://client-llm:8000/v1",
                    "llm_api_key": "client-key",
                },
            )
    assert resp.status_code == 200, resp.text
    kwargs = from_kwargs.call_args.kwargs
    assert kwargs["model"] == "client/model"
    assert kwargs["api_base"] == "http://client-llm:8000/v1"
    assert kwargs["api_key"] == "client-key"


def test_answer_llm_override_prefix_allowlist_enforced(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    app = _make_answer_app(monkeypatch, tmp_path, llm_override=True, prefixes=["https://"])
    monkeypatch.setattr("httpx.AsyncClient", _AnswerFakeAsyncClient)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/answer",
            json={"query": "q", "llm_api_base": "http://client-llm:8000/v1"},
        )
    assert resp.status_code == 403
    assert "does not match any allowed prefix" in resp.json()["detail"]


# ----------------------------------------------------------------------
# Answer path: reranking (server default + per-request endpoint override)
# ----------------------------------------------------------------------


class _MultiHitResponse:
    status_code = 200
    content = json.dumps({"results": [{"hits": [{"text": f"c{i}"} for i in range(8)]}]}).encode()

    def json(self) -> dict[str, Any]:
        return json.loads(self.content.decode())


class _CapturingAsyncClient:
    """Records the JSON body posted to the vectordb so tests can assert over-fetch."""

    last_json: dict[str, Any] | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, **kwargs) -> _MultiHitResponse:
        _CapturingAsyncClient.last_json = kwargs.get("json")
        return _MultiHitResponse()


def _make_rerank_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    rerank_override: bool = False,
    rerank_enabled: bool = False,
    server_rerank_url: str | None = None,
    server_rerank_model: str | None = None,
    refine_factor: int = 4,
    prefixes=None,
):
    async def _stub_work(_item):
        return 0, []

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        lambda _config: _stub_work,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        lambda _config: _stub_work,
    )
    cfg = ServiceConfig(
        mode="standalone",
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(enabled=True, vectordb_url="http://vectordb:7671"),
        llm=LLMConfig(enabled=True, model="server/model", api_base="http://server-llm:8000/v1", api_key="server-key"),
        nim_endpoints=NimEndpointsConfig(
            rerank_invoke_url=server_rerank_url,
            rerank_model_name=server_rerank_model,
            api_key="server-key",
        ),
        rerank=RerankConfig(enabled=rerank_enabled, refine_factor=refine_factor),
        pipeline_overrides=PipelineOverridesConfig(
            endpoint_overrides=EndpointOverridesConfig(rerank=rerank_override, allowed_url_prefixes=prefixes or [])
        ),
    )
    return create_app(cfg)


def _stub_llm(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Patch the answer LLM; return a list that captures the chunks it receives."""
    captured: list[list[str]] = []

    def _generate(query, chunks, *, reasoning_enabled=None):
        captured.append(list(chunks))
        return GenerationResult(answer="ok", latency_s=0.1, model="server/model")

    fake_llm = SimpleNamespace(generate=_generate)
    monkeypatch.setattr(
        "nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs",
        lambda **kwargs: fake_llm,
    )
    return captured


def test_answer_reranks_with_server_endpoint_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    app = _make_rerank_app(
        monkeypatch,
        tmp_path,
        rerank_enabled=True,
        server_rerank_url="http://rerank.svc/v1",
        server_rerank_model="server/rerank",
        refine_factor=3,
    )
    monkeypatch.setattr("httpx.AsyncClient", _CapturingAsyncClient)
    _stub_llm(monkeypatch)

    calls: dict[str, Any] = {}

    def _fake_rerank_hits(query, hits, **kwargs):
        calls["query"] = query
        calls["kwargs"] = kwargs
        calls["n_hits"] = len(hits)
        return list(reversed(hits))[: kwargs.get("top_n")]

    monkeypatch.setattr("nemo_retriever.operators.rerank.rerank_hits", _fake_rerank_hits)

    with TestClient(app) as client:
        resp = client.post("/v1/answer", json={"query": "q", "top_k": 2})
    assert resp.status_code == 200, resp.text
    # Over-fetch: top_k(2) * refine_factor(3) candidates requested from vectordb.
    assert _CapturingAsyncClient.last_json == {"query": "q", "top_k": 6}
    assert calls["kwargs"]["rerank_invoke_url"] == "http://rerank.svc/v1"
    assert calls["kwargs"]["model_name"] == "server/rerank"
    assert calls["kwargs"]["api_key"] == "server-key"
    assert calls["kwargs"]["top_n"] == 2


def test_answer_no_rerank_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    app = _make_rerank_app(monkeypatch, tmp_path, rerank_enabled=False, server_rerank_url="http://rerank.svc/v1")
    monkeypatch.setattr("httpx.AsyncClient", _CapturingAsyncClient)
    _stub_llm(monkeypatch)

    def _boom(*a, **k):
        raise AssertionError("rerank_hits should not be called when disabled")

    monkeypatch.setattr("nemo_retriever.operators.rerank.rerank_hits", _boom)

    with TestClient(app) as client:
        resp = client.post("/v1/answer", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200, resp.text
    # No over-fetch when not reranking.
    assert _CapturingAsyncClient.last_json == {"query": "q", "top_k": 5}


def test_answer_rerank_override_rejected_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    app = _make_rerank_app(monkeypatch, tmp_path, rerank_override=False)
    monkeypatch.setattr("httpx.AsyncClient", _CapturingAsyncClient)
    with TestClient(app) as client:
        resp = client.post("/v1/answer", json={"query": "q", "rerank_invoke_url": "http://client/rerank"})
    assert resp.status_code == 403
    assert "rerank endpoint overrides are disabled" in resp.json()["detail"]


def test_answer_rerank_override_applied_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    app = _make_rerank_app(
        monkeypatch,
        tmp_path,
        rerank_override=True,
        server_rerank_url="http://rerank.svc/v1",
    )
    monkeypatch.setattr("httpx.AsyncClient", _CapturingAsyncClient)
    _stub_llm(monkeypatch)

    calls: dict[str, Any] = {}

    def _fake_rerank_hits(query, hits, **kwargs):
        calls["kwargs"] = kwargs
        return hits[: kwargs.get("top_n")]

    monkeypatch.setattr("nemo_retriever.operators.rerank.rerank_hits", _fake_rerank_hits)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/answer",
            json={
                "query": "q",
                "top_k": 3,
                "rerank_invoke_url": "http://client/rerank",
                "rerank_model_name": "client/rerank",
                "rerank_api_key": "client-key",
            },
        )
    assert resp.status_code == 200, resp.text
    assert calls["kwargs"]["rerank_invoke_url"] == "http://client/rerank"
    assert calls["kwargs"]["model_name"] == "client/rerank"
    assert calls["kwargs"]["api_key"] == "client-key"


def test_answer_rerank_override_prefix_allowlist_enforced(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    app = _make_rerank_app(monkeypatch, tmp_path, rerank_override=True, prefixes=["https://"])
    monkeypatch.setattr("httpx.AsyncClient", _CapturingAsyncClient)
    with TestClient(app) as client:
        resp = client.post("/v1/answer", json={"query": "q", "rerank_invoke_url": "http://client/rerank"})
    assert resp.status_code == 403
    assert "does not match any allowed prefix" in resp.json()["detail"]


def test_answer_rerank_requested_without_endpoint_returns_400(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    app = _make_rerank_app(monkeypatch, tmp_path, rerank_enabled=False, server_rerank_url=None)
    monkeypatch.setattr("httpx.AsyncClient", _CapturingAsyncClient)
    _stub_llm(monkeypatch)
    with TestClient(app) as client:
        resp = client.post("/v1/answer", json={"query": "q", "rerank": True})
    assert resp.status_code == 400
    assert "no rerank endpoint" in resp.json()["detail"]
