# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import PropertyMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import nemo_retriever.service.vectordb_app as vectordb_module
from nemo_retriever.service.app import create_app
from nemo_retriever.service.agentic_query import build_agentic_query_request
from nemo_retriever.service.config import (
    AgenticConfig,
    LoggingConfig,
    PipelinePoolConfig,
    ServiceConfig,
    VectorDbConfig,
)
from nemo_retriever.service.query_schema import (
    AgenticQueryRequest,
    AgenticQueryResponse,
    AgenticQueryResult,
)
from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app


def test_agentic_service_config_requires_remote_model_and_endpoint() -> None:
    with pytest.raises(ValidationError, match="agentic.invoke_url"):
        AgenticConfig(enabled=True, llm_model="model")
    with pytest.raises(ValidationError, match="agentic.llm_model"):
        AgenticConfig(
            enabled=True,
            invoke_url="https://llm.example/v1/chat/completions",
        )


def test_build_agentic_query_request_maps_server_owned_configuration() -> None:
    request = build_agentic_query_request(
        AgenticQueryRequest(query="revenue trend", top_k=3),
        config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
            backend_top_k=25,
            react_max_steps=7,
        ),
        lancedb_uri="/indexes/finance",
        table_name="finance",
        embed_endpoint="https://embed.example/v1/embeddings",
        embed_model="embed-model",
        embed_model_provider_prefix="openai",
        embed_api_key="embed-key",
    )

    assert request.query == "revenue trend"
    assert request.retrieval.top_k == 3
    assert request.storage.lancedb_uri == "/indexes/finance"
    assert request.storage.table_name == "finance"
    assert request.embed.embed_invoke_url == "https://embed.example/v1/embeddings"
    assert request.embed.embed_model_name == "embed-model"
    assert request.embed.embed_model_provider_prefix == "openai"
    assert request.embed.embed_api_key == "embed-key"
    assert request.agentic.enabled is True
    assert request.agentic.llm_model == "model"
    assert request.agentic.invoke_url == "https://llm.example/v1/chat/completions"
    assert request.agentic.backend_top_k == 25
    assert request.agentic.react_max_steps == 7


def test_agentic_query_endpoint_is_disabled_by_default(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
    )

    with TestClient(app) as client:
        response = client.post("/v1/agentic/query", json={"query": "q"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Agentic retrieval is not enabled."


def test_agentic_query_endpoint_runs_shared_workflow(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="finance",
        embed_endpoint="https://embed.example/v1/embeddings",
        embed_model="embed-model",
        agentic_config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
        ),
    )
    expected = AgenticQueryResponse(
        results=[
            AgenticQueryResult(
                rank=1,
                doc_id="report.pdf",
                result_source="selection_agent",
            )
        ]
    )

    with (
        patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True),
        patch.object(vectordb_module, "run_agentic_query", return_value=expected) as run_query,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/agentic/query",
            json={"query": "revenue trend", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "rank": 1,
                "doc_id": "report.pdf",
                "result_source": "selection_agent",
            }
        ]
    }
    request = run_query.call_args.args[0]
    assert request == AgenticQueryRequest(query="revenue trend", top_k=3)
    assert run_query.call_args.kwargs["lancedb_uri"] == str(tmp_path)
    assert run_query.call_args.kwargs["table_name"] == "finance"
    assert run_query.call_args.kwargs["embed_api_key"] == ""


def test_agentic_query_rejects_top_k_above_backend_depth(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
        agentic_config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
            backend_top_k=5,
        ),
    )

    with (
        patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/agentic/query",
            json={"query": "revenue trend", "top_k": 6},
        )

    assert response.status_code == 422
    assert "cannot exceed" in response.json()["detail"]


def test_service_proxies_agentic_query_to_vectordb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
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
    config = ServiceConfig(
        mode="standalone",
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(
            enabled=True,
            vectordb_url="http://vectordb:7671",
        ),
        agentic=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
            request_timeout_s=321.0,
        ),
    )
    seen: dict[str, object] = {}

    class _FakeResponse:
        status_code = 200
        content = json.dumps({"results": []}).encode()

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            seen["timeout"] = kwargs["timeout"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            seen["url"] = url
            seen["body"] = json.loads(kwargs["content"])
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)

    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/agentic/query",
            json={"query": "revenue trend", "top_k": 3},
        )

    assert response.status_code == 200
    assert response.json() == {"results": []}
    assert seen == {
        "timeout": 321.0,
        "url": "http://vectordb:7671/v1/agentic/query",
        "body": {"query": "revenue trend", "top_k": 3},
    }
