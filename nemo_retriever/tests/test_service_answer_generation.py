# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-mode answer generation over the VectorDB query endpoint."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nemo_retriever.models.llm.types import GenerationResult, JudgeResult
from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import LLMConfig, LoggingConfig, PipelinePoolConfig, ServiceConfig, VectorDbConfig


def test_llm_config_defaults_to_reasoning_enabled_for_external_provider_safety() -> None:
    assert LLMConfig().reasoning_enabled is True


def test_llm_config_allows_empty_model_when_disabled_for_helm_default() -> None:
    assert LLMConfig(enabled=False, model="").model == ""


def test_llm_config_rejects_empty_model_when_enabled() -> None:
    with pytest.raises(ValueError, match="llm.model must be set when llm.enabled is true"):
        LLMConfig(enabled=True, model="", api_base="http://external-llm:8000/v1")


@pytest.fixture
def app_with_answer_config(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Standalone service app with workers stubbed and LLM answering enabled."""

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
            model="openai/nvidia/llama-3.3-nemotron-super-49b-v1.5",
            api_base="http://llama-3-3-nemotron-super-49b-v1-5:8000/v1",
            api_key="not-needed",
            max_tokens=128,
            timeout=180.0,
            reasoning_enabled=False,
        ),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client


def test_answer_retrieves_from_vectordb_and_generates_with_configured_llm(
    app_with_answer_config: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    class _FakeResponse:
        status_code = 200
        content = json.dumps(
            {
                "results": [
                    {
                        "hits": [
                            {"text": "Super-49B is the answer generator.", "source": "doc.pdf", "page_number": 1},
                            {"text": "NRL queries LanceDB before generation.", "source": "doc.pdf", "page_number": 2},
                        ]
                    }
                ]
            }
        ).encode()

        def json(self) -> dict[str, Any]:
            return json.loads(self.content.decode())

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            requests.append({"url": url, **kwargs})
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)

    fake_llm = SimpleNamespace(
        generate=lambda query, chunks, *, reasoning_enabled=None: GenerationResult(
            answer=f"{query}: {len(chunks)} chunks",
            latency_s=0.25,
            model="openai/nvidia/llama-3.3-nemotron-super-49b-v1.5",
        )
    )

    with patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs", return_value=fake_llm) as from_kwargs:
        resp = app_with_answer_config.post(
            "/v1/answer",
            json={"query": "What generates answers?", "top_k": 2, "include_chunks": True, "include_metadata": True},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["query"] == "What generates answers?"
    assert body["answer"] == "What generates answers?: 2 chunks"
    assert body["chunk_count"] == 2
    assert body["chunks"] == ["Super-49B is the answer generator.", "NRL queries LanceDB before generation."]
    assert body["metadata"] == [
        {"source": "doc.pdf", "page_number": 1},
        {"source": "doc.pdf", "page_number": 2},
    ]

    assert requests == [
        {
            "url": "http://vectordb:7671/v1/query",
            "json": {"query": "What generates answers?", "top_k": 2},
        }
    ]
    from_kwargs.assert_called_once_with(
        model="openai/nvidia/llama-3.3-nemotron-super-49b-v1.5",
        api_base="http://llama-3-3-nemotron-super-49b-v1-5:8000/v1",
        api_key="not-needed",
        temperature=0.0,
        top_p=None,
        max_tokens=128,
        extra_params={},
        num_retries=3,
        timeout=180.0,
        rag_system_prompt=None,
        rag_system_prompt_prefix=None,
        reasoning_enabled=False,
    )


def test_answer_preserves_vectordb_error_content_type(
    app_with_answer_config: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        status_code = 429
        content = b"rate limited"
        headers = {"content-type": "text/plain; charset=utf-8"}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)

    with patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs") as from_kwargs:
        resp = app_with_answer_config.post("/v1/answer", json={"query": "q"})

    assert resp.status_code == 429
    assert resp.text == "rate limited"
    assert resp.headers["content-type"] == "text/plain; charset=utf-8"
    from_kwargs.assert_not_called()


def test_answer_response_fields_respect_chunk_and_metadata_flags(
    app_with_answer_config: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        status_code = 200
        content = json.dumps(
            {
                "results": [
                    {
                        "hits": [
                            {
                                "text": "citation context",
                                "source": "doc.pdf",
                                "page_number": 3,
                            }
                        ]
                    }
                ]
            }
        ).encode()

        def json(self) -> dict[str, Any]:
            return json.loads(self.content.decode())

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
    seen: dict[str, Any] = {}

    def _generate(query, chunks, *, reasoning_enabled=None):
        seen["chunks"] = chunks
        return GenerationResult(answer="answer", latency_s=0.1, model="m")

    fake_llm = SimpleNamespace(generate=_generate)

    with patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs", return_value=fake_llm):
        resp = app_with_answer_config.post("/v1/answer", json={"query": "q", "include_metadata": True})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["chunk_count"] == 1
    assert body["chunks"] is None
    assert body["metadata"] == [{"source": "doc.pdf", "page_number": 3}]
    assert seen["chunks"] == ["citation context"]

    resp = app_with_answer_config.post("/v1/answer", json={"query": "q", "include_chunks": True})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["chunk_count"] == 1
    assert body["chunks"] == ["citation context"]
    assert body["metadata"] is None


def test_answer_preserves_present_empty_text_hit_before_fallbacks(
    app_with_answer_config: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        status_code = 200
        content = json.dumps({"results": [{"hits": [{"text": "", "content": "fallback content"}]}]}).encode()

        def json(self) -> dict[str, Any]:
            return json.loads(self.content.decode())

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
    seen: dict[str, Any] = {}

    def _generate(query, chunks, *, reasoning_enabled=None):
        seen["chunks"] = chunks
        return GenerationResult(answer="answer", latency_s=0.1, model="m")

    fake_llm = SimpleNamespace(generate=_generate)

    with patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs", return_value=fake_llm):
        resp = app_with_answer_config.post("/v1/answer", json={"query": "q", "include_chunks": True})

    assert resp.status_code == 200, resp.text
    assert seen["chunks"] == [""]
    assert resp.json()["chunks"] == [""]


def test_answer_returns_404_when_llm_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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

    app = create_app(
        ServiceConfig(
            mode="standalone",
            logging=LoggingConfig(file=str(tmp_path / "service.log")),
            pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
            vectordb=VectorDbConfig(enabled=True, vectordb_url="http://vectordb:7671"),
            llm=LLMConfig(enabled=False),
        )
    )

    with TestClient(app) as client:
        resp = client.post("/v1/answer", json={"query": "q"})

    assert resp.status_code == 404
    assert "LLM answer generation is not enabled" in resp.json()["detail"]


def test_answer_returns_404_when_vectordb_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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

    app = create_app(
        ServiceConfig(
            mode="standalone",
            logging=LoggingConfig(file=str(tmp_path / "service.log")),
            pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
            vectordb=VectorDbConfig(enabled=False),
            llm=LLMConfig(enabled=True),
        )
    )

    with TestClient(app) as client:
        resp = client.post("/v1/answer", json={"query": "q"})

    assert resp.status_code == 404
    assert "VectorDB is not enabled" in resp.json()["detail"]


def test_answer_forwards_request_reasoning_override(
    app_with_answer_config: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        status_code = 200
        content = json.dumps({"results": [{"hits": [{"text": "context"}]}]}).encode()

        def json(self) -> dict[str, Any]:
            return json.loads(self.content.decode())

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
    seen: dict[str, Any] = {}

    def _generate(query, chunks, *, reasoning_enabled=None):
        seen["reasoning_enabled"] = reasoning_enabled
        return GenerationResult(answer="ok", latency_s=0.1, model="m")

    fake_llm = SimpleNamespace(generate=_generate)

    with patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs", return_value=fake_llm):
        resp = app_with_answer_config.post("/v1/answer", json={"query": "q", "reasoning_enabled": True})

    assert resp.status_code == 200, resp.text
    assert seen["reasoning_enabled"] is True


def test_answer_scores_reference_without_judge(
    app_with_answer_config: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        status_code = 200
        content = json.dumps(
            {
                "results": [
                    {
                        "hits": [
                            {
                                "text": "RAG retrieves passages and feeds them to a generator.",
                                "source": "doc.pdf",
                            }
                        ]
                    }
                ]
            }
        ).encode()

        def json(self) -> dict[str, Any]:
            return json.loads(self.content.decode())

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
    fake_llm = SimpleNamespace(
        generate=lambda query, chunks: GenerationResult(
            answer="RAG retrieves passages and feeds them to a generator.",
            latency_s=0.1,
            model="m",
        )
    )

    with (
        patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs", return_value=fake_llm),
        patch("nemo_retriever.models.llm.clients.LLMJudge.from_kwargs") as judge_from_kwargs,
    ):
        resp = app_with_answer_config.post(
            "/v1/answer",
            json={
                "query": "What is RAG?",
                "reference": "RAG retrieves passages and feeds them to a generator.",
                "include_chunks": True,
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["chunk_count"] == 1
    assert body["chunks"] == ["RAG retrieves passages and feeds them to a generator."]
    assert body["metadata"] is None
    assert body["answer_in_context"] is True
    assert body["token_f1"] == pytest.approx(1.0, abs=1e-6)
    assert body["exact_match"] is True
    assert body["judge_score"] is None
    judge_from_kwargs.assert_not_called()


def test_answer_scores_with_opt_in_judge(
    app_with_answer_config: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        status_code = 200
        content = json.dumps({"results": [{"hits": [{"text": "context"}]}]}).encode()

        def json(self) -> dict[str, Any]:
            return json.loads(self.content.decode())

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
    fake_llm = SimpleNamespace(
        generate=lambda query, chunks: GenerationResult(answer="expected answer", latency_s=0.1, model="m")
    )
    fake_judge = SimpleNamespace(
        judge=lambda query, reference, candidate: JudgeResult(score=1.0, reasoning="", error=None)
    )

    with (
        patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs", return_value=fake_llm),
        patch("nemo_retriever.models.llm.clients.LLMJudge.from_kwargs", return_value=fake_judge) as judge_from_kwargs,
    ):
        resp = app_with_answer_config.post(
            "/v1/answer",
            json={"query": "q", "reference": "expected answer", "judge": True},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["judge_score"] == 1.0
    assert body["judge_error"] is None
    assert body["failure_mode"] == "correct"
    assert body["chunks"] is None
    assert body["metadata"] is None
    judge_from_kwargs.assert_called_once_with(
        model="openai/nvidia/llama-3.3-nemotron-super-49b-v1.5",
        api_base="http://llama-3-3-nemotron-super-49b-v1-5:8000/v1",
        api_key="not-needed",
        extra_params={},
        num_retries=3,
        timeout=180.0,
    )


def test_answer_judge_requires_reference(app_with_answer_config: TestClient) -> None:
    resp = app_with_answer_config.post("/v1/answer", json={"query": "q", "judge": True})

    assert resp.status_code == 422
    assert "judge requires reference" in resp.text


def test_answer_returns_502_when_llm_generation_fails(
    app_with_answer_config: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        status_code = 200
        content = json.dumps({"results": [{"hits": [{"text": "context"}]}]}).encode()

        def json(self) -> dict[str, Any]:
            return json.loads(self.content.decode())

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
    fake_llm = SimpleNamespace(
        generate=lambda query, chunks, *, reasoning_enabled=None: GenerationResult(
            answer="",
            latency_s=0.0,
            model="openai/nvidia/llama-3.3-nemotron-super-49b-v1.5",
            error="connection refused",
        )
    )

    with patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs", return_value=fake_llm):
        resp = app_with_answer_config.post("/v1/answer", json={"query": "q"})

    assert resp.status_code == 502
    assert resp.json()["detail"] == "LLM answer generation failed: connection refused"


def test_service_start_reads_llm_api_key_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from typer.testing import CliRunner

    from nemo_retriever.service.cli import app as service_cli_app

    config_path = tmp_path / "retriever-service.yaml"
    config_path.write_text(
        "mode: standalone\n" "llm:\n" "  enabled: true\n" "  api_base: http://external-llm:8000/v1\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}
    application = object()

    def _fake_create_app(config: ServiceConfig) -> object:
        captured["config"] = config
        return application

    def _fake_uvicorn_run(app: object, *, host: str, port: int, log_level: str) -> None:
        captured["uvicorn"] = {"app": app, "host": host, "port": port, "log_level": log_level}

    monkeypatch.setattr("nemo_retriever.service.app.create_app", _fake_create_app)
    monkeypatch.setattr("uvicorn.run", _fake_uvicorn_run)

    result = CliRunner().invoke(
        service_cli_app,
        ["start", "--config", str(config_path)],
        env={"NEMO_RETRIEVER_LLM_API_KEY": "secret-from-env"},
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].llm.api_key == "secret-from-env"
    assert captured["uvicorn"]["app"] is application
