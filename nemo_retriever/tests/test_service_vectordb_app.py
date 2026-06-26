# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastapi.testclient import TestClient

import nemo_retriever.service.vectordb_app as vectordb_app_module
from nemo_retriever.query.filters import build_query_where_clause
from nemo_retriever.query.options import QueryFilterOptions
from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app


def test_query_empty_index_returns_422(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="test_table",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )
    with TestClient(app) as client:
        resp = client.post("/v1/query", json={"query": "hello", "top_k": 3})

    assert resp.status_code == 422
    assert "No data has been ingested yet" in resp.json()["detail"]


def test_query_rejects_negative_page_filter(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="test_table",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )
    with TestClient(app) as client:
        resp = client.post("/v1/query", json={"query": "hello", "top_k": 3, "page_number": -1})

    assert resp.status_code == 422


def test_vectordb_search_applies_where_before_limit() -> None:
    calls: list[tuple[str, object]] = []

    class FakeQuery:
        def where(self, where: str) -> "FakeQuery":
            calls.append(("where", where))
            return self

        def limit(self, top_k: int) -> "FakeQuery":
            calls.append(("limit", top_k))
            return self

        def to_list(self) -> list[dict[str, object]]:
            return [{"text": "match", "metadata": "{}", "source": "{}"}]

    class FakeTable:
        def search(self, vector: list[float]) -> FakeQuery:
            calls.append(("search", vector))
            return FakeQuery()

    class FakeDB:
        def open_table(self, table_name: str) -> FakeTable:
            calls.append(("open_table", table_name))
            return FakeTable()

    state = object.__new__(VectorDBState)
    state._table_exists = True
    state._db = FakeDB()
    state.table_name = "docs"

    result = state.search([[1.0, 0.0]], 2, "metadata LIKE '%\"page_number\":3%'")

    assert result[0][0]["text"] == "match"
    assert calls == [
        ("open_table", "docs"),
        ("search", [1.0, 0.0]),
        ("where", "metadata LIKE '%\"page_number\":3%'"),
        ("limit", 2),
    ]


def test_query_route_accepts_filters_and_applies_where(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, object]] = []

    class FakeState:
        table_exists = True
        embed_endpoint = "http://embed.example/v1/embeddings"

        def total_rows(self) -> int:
            return 1

        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            calls.append(("embed", texts))
            return [[1.0, 0.0]]

        def search(
            self,
            vectors: list[list[float]],
            top_k: int,
            where: str | None = None,
        ) -> list[list[dict[str, object]]]:
            calls.append(("search", {"vectors": vectors, "top_k": top_k, "where": where}))
            return [[{"text": "match", "metadata": '{"meta_a":"alpha"}', "source": "{}"}]]

    monkeypatch.setattr(vectordb_app_module, "VectorDBState", lambda **_kwargs: FakeState())

    app = vectordb_app_module.create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="test_table",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )
    with TestClient(app) as client:
        resp = client.post(
            "/v1/query",
            json={
                "query": "hello",
                "top_k": 3,
                "source_id": "docs/a.pdf",
                "page_number": 1,
                "where": 'metadata LIKE \'%"meta_a":"alpha"%\'',
            },
        )

    assert resp.status_code == 200
    assert resp.json()["results"][0]["hits"][0]["text"] == "match"
    assert calls == [
        ("embed", ["hello"]),
        (
            "search",
            {
                "vectors": [[1.0, 0.0]],
                "top_k": 3,
                "where": build_query_where_clause(
                    QueryFilterOptions(
                        source_id="docs/a.pdf",
                        page_number=1,
                        where='metadata LIKE \'%"meta_a":"alpha"%\'',
                    )
                ),
            },
        ),
    ]
