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
from nemo_retriever.service.agentic_query import (
    agentic_ranked_to_hits,
    build_agentic_query_request,
    run_agentic_query,
)
from nemo_retriever.service.config import (
    AgenticConfig,
    AuthConfig,
    LoggingConfig,
    PipelinePoolConfig,
    ServiceConfig,
    VectorDbConfig,
)
from nemo_retriever.service.query_schema import (
    AgenticQueryResponse,
    MAX_AGENTIC_QUERY_CHARS,
    QueryRequest,
    QueryResult,
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


def test_query_request_agentic_requires_hits_format() -> None:
    with pytest.raises(ValidationError, match="single query string"):
        QueryRequest(query=["a", "b"], agentic=True)
    with pytest.raises(ValidationError, match="non-empty"):
        QueryRequest(query="   ", agentic=True)
    with pytest.raises(ValidationError, match="format='hits'"):
        QueryRequest(query="q", agentic=True, format="evidence")


def test_build_agentic_query_request_maps_server_owned_configuration() -> None:
    request = build_agentic_query_request(
        query="revenue trend",
        top_k=3,
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


def test_run_agentic_query_includes_provider_usage() -> None:
    from nemo_retriever.query.workflow import AgenticQueryDocumentsResult

    usage = {
        "input_tokens": 12,
        "cache_tokens": 4,
        "output_tokens": 5,
        "total_tokens": 17,
        "stages": {"main_agent": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}},
    }
    workflow_result = AgenticQueryDocumentsResult(
        hits=[{"doc_id": "report_7", "rank": 1, "result_source": "final_results"}],
        usage=usage,
    )

    with patch(
        "nemo_retriever.service.agentic_query.agentic_query_documents_with_metadata",
        return_value=workflow_result,
    ):
        response = run_agentic_query(
            query="revenue trend",
            top_k=1,
            config=AgenticConfig(
                enabled=True,
                llm_model="model",
                invoke_url="https://llm.example/v1/chat/completions",
            ),
            lancedb_uri="/indexes/finance",
            table_name="finance",
            embed_endpoint="https://embed.example/v1/embeddings",
            embed_model="embed-model",
            embed_model_provider_prefix=None,
            embed_api_key="",
        )

    assert response.usage is not None
    assert response.usage.model_dump() == usage


def test_agentic_ranked_to_hits_keeps_rehydrated_classic_fields() -> None:
    hits = agentic_ranked_to_hits(
        [
            {
                "text": "revenue grew 4%",
                "metadata": {"type": "text"},
                "source": "/indexes/report.pdf",
                "source_id": "/indexes/report.pdf",
                "page_number": 7,
                "_score": 0.42,
                "rank": 1,
                "doc_id": "report_7",
                "result_source": "selection_agent",
            }
        ]
    )
    assert hits == [
        {
            "text": "revenue grew 4%",
            "metadata": {"type": "text", "rank": 1, "result_source": "selection_agent"},
            "source": "/indexes/report.pdf",
            "source_id": "/indexes/report.pdf",
            "page_number": 7,
            "_score": 0.42,
            "doc_id": "report_7",
            "rank": 1,
            "result_source": "selection_agent",
        }
    ]


def test_agentic_ranked_to_hits_falls_back_to_doc_id_without_rehydrated_metadata() -> None:
    hits = agentic_ranked_to_hits([{"rank": 1, "doc_id": "report.pdf", "result_source": "final_results"}])
    assert hits == [
        {
            "text": None,
            "metadata": {"rank": 1, "result_source": "final_results"},
            "source": "report.pdf",
            "source_id": None,
            "path": None,
            "page_number": None,
            "pdf_basename": None,
            "pdf_page": None,
            "doc_id": "report.pdf",
            "rank": 1,
            "result_source": "final_results",
        }
    ]


def test_agentic_ranked_to_hits_rejects_blank_doc_id() -> None:
    with pytest.raises(ValueError, match="missing a non-empty doc_id"):
        agentic_ranked_to_hits([{"rank": 1, "doc_id": "", "result_source": "selection_agent"}])


def test_agentic_query_flag_rejected_when_disabled(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
    )

    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "q", "agentic": True})

    assert response.status_code == 400
    assert "not enabled" in response.json()["detail"]


def test_agentic_query_rejects_collection_target(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
        agentic_config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/query",
            json={"query": "q", "agentic": True, "collection_name": "workspace"},
        )

    assert response.status_code == 501


def test_agentic_true_runs_react_workflow_on_v1_query(tmp_path) -> None:
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
            QueryResult(
                hits=agentic_ranked_to_hits(
                    [
                        {
                            "text": "revenue grew 4%",
                            "metadata": {"type": "text"},
                            "source": "/indexes/report.pdf",
                            "page_number": 7,
                            "rank": 1,
                            "doc_id": "report_7",
                            "result_source": "selection_agent",
                        }
                    ]
                )
            )
        ],
        query_mode="agentic",
        usage={
            "input_tokens": 120,
            "cache_tokens": 40,
            "output_tokens": 30,
            "total_tokens": 150,
            "stages": {},
        },
    )

    with (
        patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True),
        patch.object(vectordb_module, "run_agentic_query", return_value=expected) as run_query,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/query",
            json={"query": "revenue trend", "top_k": 3, "agentic": True},
        )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "hits": [
                    {
                        "text": "revenue grew 4%",
                        "metadata": {"type": "text", "rank": 1, "result_source": "selection_agent"},
                        "source": "/indexes/report.pdf",
                        "page_number": 7,
                        "doc_id": "report_7",
                        "rank": 1,
                        "result_source": "selection_agent",
                    }
                ]
            }
        ],
        "query_mode": "agentic",
        "usage": {
            "input_tokens": 120,
            "cache_tokens": 40,
            "output_tokens": 30,
            "total_tokens": 150,
            "stages": {},
        },
    }
    assert run_query.call_args.kwargs["query"] == "revenue trend"
    assert run_query.call_args.kwargs["top_k"] == 3
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
            "/v1/query",
            json={"query": "revenue trend", "top_k": 6, "agentic": True},
        )

    assert response.status_code == 422
    assert "cannot exceed" in response.json()["detail"]


def test_agentic_query_rejects_query_above_length_limit(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
        agentic_config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
        ),
    )

    with (
        patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True),
        patch.object(vectordb_module, "run_agentic_query") as run_query,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/query",
            json={"query": "x" * (MAX_AGENTIC_QUERY_CHARS + 1), "agentic": True},
        )

    assert response.status_code == 422
    run_query.assert_not_called()


def test_agentic_query_slots_are_bounded_and_released_by_the_worker(tmp_path) -> None:
    """Capacity follows the worker thread, not the caller: a saturated pool sheds
    load with 503 instead of queueing behind non-cancellable ReAct work, and a
    completed query returns its slot."""
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="https://embed.example/v1/embeddings",
        agentic_config=AgenticConfig(
            enabled=True,
            llm_model="model",
            invoke_url="https://llm.example/v1/chat/completions",
        ),
    )
    expected = AgenticQueryResponse(results=[QueryResult(hits=[])])

    with (
        patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True),
        patch.object(vectordb_module, "run_agentic_query", return_value=expected),
        TestClient(app) as client,
    ):
        slots = app.state.agentic_slots
        assert slots is not None

        for _ in range(vectordb_module.MAX_CONCURRENT_AGENTIC_QUERIES):
            assert slots.acquire(blocking=False) is True

        busy = client.post("/v1/query", json={"query": "revenue trend", "agentic": True})

        assert busy.status_code == 503
        assert busy.headers["Retry-After"] == "30"
        query_semaphore = app.state.vectordb_state.query_semaphore
        assert query_semaphore.locked() is False

        slots.release()
        accepted = client.post("/v1/query", json={"query": "revenue trend", "agentic": True})

        assert accepted.status_code == 200
        assert slots.acquire(blocking=False) is True


def test_gateway_proxies_agentic_flag_to_vectordb(
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
        auth=AuthConfig(allow_unscoped_dev=True),
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
        content = json.dumps({"results": [{"hits": []}]}).encode()

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
            "/v1/query",
            json={"query": "revenue trend", "top_k": 3, "agentic": True},
        )

    assert response.status_code == 200
    assert response.json() == {"results": [{"hits": []}]}
    assert seen == {
        "timeout": 321.0,
        "url": "http://vectordb:7671/v1/query",
        "body": {"query": "revenue trend", "top_k": 3, "agentic": True},
    }


def test_service_rejects_agentic_flag_when_not_configured(
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
        auth=AuthConfig(allow_unscoped_dev=True),
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(enabled=True, vectordb_url="http://vectordb:7671"),
    )

    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/query",
            json={"query": "revenue trend", "agentic": True},
        )

    assert response.status_code == 400
    assert "not enabled" in response.json()["detail"]
