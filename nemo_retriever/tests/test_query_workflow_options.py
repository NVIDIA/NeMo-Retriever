# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import nemo_retriever.query.workflow as query_workflow
from nemo_retriever.query.options import (
    QueryEmbedOptions,
    QueryRerankOptions,
    QueryRequest,
    QueryRetrievalOptions,
    QueryStorageOptions,
)


def test_query_request_builds_retriever_kwargs_without_rerank(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(query_workflow, "Retriever", FakeRetriever)
    request = QueryRequest(
        query="deployment?",
        retrieval=QueryRetrievalOptions(top_k=3),
        storage=QueryStorageOptions(lancedb_uri="/tmp/lancedb", table_name="docs"),
    )

    assert query_workflow.query_documents(request) == []
    assert retriever_calls == [
        {
            "top_k": 3,
            "vdb_kwargs": {"uri": "/tmp/lancedb", "table_name": "docs"},
        }
    ]


def test_query_request_builds_retriever_kwargs_with_embed_and_remote_rerank(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(query_workflow, "Retriever", FakeRetriever)
    request = QueryRequest(
        query="deployment?",
        embed=QueryEmbedOptions(
            embed_invoke_url="http://embed:8000/v1/embeddings",
            embed_model_name="nvidia/llama-nemotron-embed-1b-v2",
        ),
        rerank=QueryRerankOptions(
            enabled=True,
            reranker_invoke_url="http://rerank:8000/v1/ranking",
        ),
    )

    assert query_workflow.query_documents(request) == []
    assert retriever_calls == [
        {
            "top_k": 10,
            "vdb_kwargs": {"uri": "lancedb", "table_name": "nemo-retriever"},
            "embed_kwargs": {
                "embed_invoke_url": "http://embed:8000/v1/embeddings",
                "embedding_endpoint": "http://embed:8000/v1/embeddings",
                "model_name": "nvidia/llama-nemotron-embed-1b-v2",
                "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
            },
            "rerank": True,
            "rerank_kwargs": {
                "rerank_invoke_url": "http://rerank:8000/v1/ranking",
                "api_key": "nvapi-test",
            },
        }
    ]


def test_query_documents_uses_typed_request(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []
    query_calls: list[tuple[str, dict[str, Any]]] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
            query_calls.append((query, kwargs))
            return [{"text": "passage", "source": "doc.pdf", "page_number": 1}]

    monkeypatch.setattr(query_workflow, "Retriever", FakeRetriever)

    request = QueryRequest(
        query="deployment?",
        retrieval=QueryRetrievalOptions(
            top_k=1,
            candidate_k=3,
            page_dedup=True,
            content_types="text,table",
        ),
    )

    assert query_workflow.query_documents(request) == [{"text": "passage", "source": "doc.pdf", "page_number": 1}]
    assert retriever_calls == [{"top_k": 1, "vdb_kwargs": {"uri": "lancedb", "table_name": "nemo-retriever"}}]
    assert query_calls == [
        (
            "deployment?",
            {
                "candidate_k": 3,
                "page_dedup": True,
                "content_types": "text,table",
            },
        )
    ]
