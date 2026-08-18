# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import (
    AuthConfig,
    LoggingConfig,
    NimEndpointsConfig,
    PipelinePoolConfig,
    ServiceConfig,
    VectorDbConfig,
)


def _configure_noop_workers(monkeypatch):
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


def test_reranked_query_uses_main_service_orchestration(monkeypatch, tmp_path) -> None:
    _configure_noop_workers(monkeypatch)
    seen: dict[str, object] = {}

    class _Response:
        status_code = 200
        content = json.dumps(
            {
                "results": [
                    {
                        "hits": [
                            {"text": "first", "source": "a"},
                            {"text": "second", "source": "b"},
                            {"text": "third", "source": "c"},
                        ]
                    }
                ]
            }
        ).encode()

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            seen["timeout"] = kwargs["timeout"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, **kwargs):
            seen["url"] = url
            seen["body"] = json.loads(kwargs["content"])
            return _Response()

    def _rerank(query, hits, **kwargs):
        seen["rerank"] = {"query": query, "kwargs": kwargs}
        return [{"_rerank_score": 0.9, **hits[2]}, {"_rerank_score": 0.8, **hits[1]}]

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    monkeypatch.setattr("nemo_retriever.operators.rerank.rerank_hits", _rerank)
    config = ServiceConfig(
        mode="standalone",
        auth=AuthConfig(allow_unscoped_dev=True),
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(enabled=True, vectordb_url="http://vectordb:7671"),
        nim_endpoints=NimEndpointsConfig(
            rerank_invoke_url="http://reranker:8080",
            rerank_model_name="rerank-model",
            api_key="server-secret",
        ),
    )

    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/query",
            json={"query": "revenue", "top_k": 2, "rerank": True, "rerank_top_k": 3},
        )

    assert response.status_code == 200
    assert response.json()["results"][0]["hits"] == [
        {"_rerank_score": 0.9, "text": "third", "source": "c"},
        {"_rerank_score": 0.8, "text": "second", "source": "b"},
    ]
    assert seen["url"] == "http://vectordb:7671/v1/query"
    assert seen["body"] == {"query": "revenue", "top_k": 3}
    assert seen["rerank"] == {
        "query": "revenue",
        "kwargs": {
            "rerank_invoke_url": "http://reranker:8080",
            "model_name": "rerank-model",
            "api_key": "server-secret",
            "top_n": 2,
        },
    }


def test_reranked_query_defaults_remote_model_to_vl(monkeypatch, tmp_path) -> None:
    _configure_noop_workers(monkeypatch)
    seen: dict[str, object] = {}

    class _Response:
        status_code = 200
        content = json.dumps({"results": [{"hits": [{"text": "first"}]}]}).encode()

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, **kwargs):
            return _Response()

    def _rerank(query, hits, **kwargs):
        seen["kwargs"] = kwargs
        return hits

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    monkeypatch.setattr("nemo_retriever.operators.rerank.rerank_hits", _rerank)
    config = ServiceConfig(
        mode="standalone",
        auth=AuthConfig(allow_unscoped_dev=True),
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(enabled=True, vectordb_url="http://vectordb:7671"),
        nim_endpoints=NimEndpointsConfig(rerank_invoke_url="http://reranker:8080"),
    )

    with TestClient(create_app(config)) as client:
        response = client.post("/v1/query", json={"query": "revenue", "top_k": 1, "rerank": True})

    assert response.status_code == 200
    assert seen["kwargs"]["model_name"] == "nvidia/llama-nemotron-rerank-vl-1b-v2"


def test_reranked_query_requires_main_service_reranker(monkeypatch, tmp_path) -> None:
    _configure_noop_workers(monkeypatch)
    config = ServiceConfig(
        mode="standalone",
        auth=AuthConfig(allow_unscoped_dev=True),
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(enabled=True, vectordb_url="http://vectordb:7671"),
    )

    with TestClient(create_app(config)) as client:
        response = client.post("/v1/query", json={"query": "revenue", "rerank": True})

    assert response.status_code == 400
    assert "rerank_invoke_url" in response.json()["detail"]


def test_reranked_query_uses_lazy_local_main_service_model(monkeypatch, tmp_path) -> None:
    _configure_noop_workers(monkeypatch)
    seen: dict[str, object] = {}
    local_model = object()

    class _Response:
        status_code = 200
        content = json.dumps({"results": [{"hits": [{"text": "first"}, {"text": "second"}]}]}).encode()

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, **kwargs):
            seen["body"] = json.loads(kwargs["content"])
            return _Response()

    def _create_local_reranker(model_name, **kwargs):
        seen["factory"] = {"model_name": model_name, "kwargs": kwargs}
        return local_model

    def _rerank(query, hits, **kwargs):
        seen["rerank"] = {"query": query, "kwargs": kwargs}
        return [{"_rerank_score": 0.9, **hits[1]}]

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    monkeypatch.setattr("nemo_retriever.models.create_local_reranker", _create_local_reranker)
    monkeypatch.setattr("nemo_retriever.operators.rerank.rerank_hits", _rerank)
    config = ServiceConfig(
        mode="standalone",
        auth=AuthConfig(allow_unscoped_dev=True),
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(enabled=True, vectordb_url="http://vectordb:7671"),
        local_models={
            "enabled": False,
            "device": "cuda:0",
            "hf_cache_dir": "/models",
            "rerank": {
                "enabled": True,
                "model_name": "local-reranker",
                "backend": "hf",
                "gpu_memory_utilization": 0.6,
                "max_length": 256,
                "batch_size": 4,
            },
        },
    )

    with TestClient(create_app(config)) as client:
        response = client.post("/v1/query", json={"query": "revenue", "top_k": 1, "rerank": True})

    assert response.status_code == 200
    assert seen["body"] == {"query": "revenue", "top_k": 50}
    assert seen["factory"] == {
        "model_name": "local-reranker",
        "kwargs": {
            "backend": "hf",
            "device": "cuda:0",
            "hf_cache_dir": "/models",
            "gpu_memory_utilization": 0.6,
        },
    }
    assert seen["rerank"] == {
        "query": "revenue",
        "kwargs": {
            "model": local_model,
            "model_name": "local-reranker",
            "max_length": 256,
            "batch_size": 4,
            "top_n": 1,
        },
    }


def test_false_rerank_string_uses_normal_query_path(monkeypatch, tmp_path) -> None:
    _configure_noop_workers(monkeypatch)
    seen: dict[str, object] = {}

    class _Response:
        status_code = 200
        content = json.dumps({"results": [{"hits": [{"text": "first"}]}]}).encode()

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, **kwargs):
            seen["body"] = json.loads(kwargs["content"])
            return _Response()

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    config = ServiceConfig(
        mode="standalone",
        auth=AuthConfig(allow_unscoped_dev=True),
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(enabled=True, vectordb_url="http://vectordb:7671"),
    )

    with TestClient(create_app(config)) as client:
        response = client.post("/v1/query", json={"query": "revenue", "rerank": "false"})

    assert response.status_code == 200
    assert seen["body"] == {"query": "revenue", "rerank": "false"}
