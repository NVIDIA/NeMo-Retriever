# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nemo_retriever.vdb.records import normalize_retrieval_results
from nemo_retriever.vdb.vector_search import search_vdb_with_vectors, supports_vector_search_vdb


class _FakeLanceSearch:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits
        self.selected_fields: list[str] | None = None
        self.limit_value: int | None = None

    def select(self, fields: list[str]) -> "_FakeLanceSearch":
        self.selected_fields = fields
        return self

    def limit(self, value: int) -> "_FakeLanceSearch":
        self.limit_value = value
        return self

    def refine_factor(self, value: int) -> "_FakeLanceSearch":
        return self

    def nprobes(self, value: int) -> "_FakeLanceSearch":
        return self

    def to_list(self) -> list[dict[str, Any]]:
        return self.hits


class _FakeLanceTable:
    def __init__(self) -> None:
        self.schema = [
            SimpleNamespace(name="text"),
            SimpleNamespace(name="metadata"),
            SimpleNamespace(name="source"),
            SimpleNamespace(name="stored_image_uri"),
        ]
        self.search_calls: list[tuple[Any, dict[str, Any]]] = []
        self.last_search: _FakeLanceSearch | None = None

    def search(self, vector: Any = None, **kwargs: Any) -> _FakeLanceSearch:
        self.search_calls.append((vector, kwargs))
        self.last_search = _FakeLanceSearch(
            [
                {
                    "text": "lance chunk",
                    "metadata": '{"page_number": 4}',
                    "source": '{"source_id": "doc-lance.pdf"}',
                    "stored_image_uri": "s3://image",
                    "_distance": 0.25,
                }
            ]
        )
        return self.last_search


class _FakeMilvusHit:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return self._payload


def test_lancedb_vector_search_consumes_precomputed_vectors_and_normalizes() -> None:
    table = _FakeLanceTable()

    raw = search_vdb_with_vectors(
        vdb_op="lancedb",
        query_vectors=[[0.1, 0.2, 0.3]],
        query_texts=["what is indexed?"],
        top_k=3,
        vdb_kwargs={"table": table, "vector_column_name": "vector"},
    )
    result = normalize_retrieval_results(raw)

    assert table.search_calls[0][1] == {"vector_column_name": "vector"}
    assert table.last_search is not None
    assert table.last_search.limit_value == 3
    assert table.last_search.selected_fields == ["text", "metadata", "source", "stored_image_uri"]
    assert result == [
        [
            {
                "text": "lance chunk",
                "metadata": '{"page_number": 4}',
                "source": "doc-lance.pdf",
                "source_id": "doc-lance.pdf",
                "path": "doc-lance.pdf",
                "page_number": 4,
                "pdf_basename": "doc-lance",
                "pdf_page": "doc-lance_4",
                "stored_image_uri": "s3://image",
                "_distance": 0.25,
            }
        ]
    ]


def test_milvus_vector_search_reuses_client_search_helper_with_precomputed_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pymilvus")
    from nv_ingest_client.util.vdb import milvus as client_milvus

    captured: dict[str, Any] = {}

    def fake_dense_retrieval(
        queries: list[str],
        collection_name: str,
        client: Any,
        dense_model: Any,
        top_k: int,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        captured.update(
            {
                "queries": queries,
                "collection_name": collection_name,
                "client": client,
                "top_k": top_k,
                "kwargs": kwargs,
                "vectors": dense_model([queries[0]]),
            }
        )
        return [
            [
                _FakeMilvusHit(
                    {
                        "id": 1,
                        "distance": 0.1,
                        "entity": {
                            "text": "milvus chunk",
                            "source": {"source_id": "doc-milvus.pdf"},
                            "content_metadata": {"page_number": 9},
                        },
                    }
                )
            ]
        ]

    monkeypatch.setattr(client_milvus, "dense_retrieval", fake_dense_retrieval)

    fake_client = object()
    raw = search_vdb_with_vectors(
        vdb_op="milvus",
        query_vectors=[[0.4, 0.5, 0.6]],
        query_texts=["what is indexed?"],
        top_k=7,
        vdb_kwargs={
            "collection_name": "docs",
            "client": fake_client,
            "dense_field": "vector",
        },
    )
    result = normalize_retrieval_results(raw)

    assert captured["queries"] == ["what is indexed?"]
    assert captured["collection_name"] == "docs"
    assert captured["client"] is fake_client
    assert captured["top_k"] == 7
    assert captured["vectors"] == [[0.4, 0.5, 0.6]]
    assert captured["kwargs"]["dense_field"] == "vector"
    assert result[0][0]["text"] == "milvus chunk"
    assert result[0][0]["pdf_page"] == "doc-milvus_9"


def test_vector_search_support_check_only_handles_known_ops() -> None:
    assert supports_vector_search_vdb("lancedb")
    assert supports_vector_search_vdb("milvus")
    assert not supports_vector_search_vdb("custom")
    assert search_vdb_with_vectors(
        vdb_op="custom",
        query_vectors=[],
        query_texts=[],
        top_k=1,
        vdb_kwargs={},
    ) is None
