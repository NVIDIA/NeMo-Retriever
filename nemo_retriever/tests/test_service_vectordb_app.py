# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastapi.testclient import TestClient

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
