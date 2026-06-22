# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from nemo_retriever.service.vectordb_app import (
    VectorDBState,
    _tensor_to_embedding_rows,
    create_vectordb_app,
)


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


def test_query_without_embed_backend_returns_501(tmp_path) -> None:
    app = create_vectordb_app(lancedb_uri=str(tmp_path))
    with TestClient(app) as client:
        resp = client.post("/v1/query", json={"query": "hello", "top_k": 3})

    assert resp.status_code == 501
    assert "No embedding backend configured" in resp.json()["detail"]


def test_health_reports_embed_mode(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        local_embed=True,
        embed_model="nvidia/llama-nemotron-embed-1b-v2",
    )
    with TestClient(app) as client:
        resp = client.get("/v1/health")

    assert resp.status_code == 200
    assert resp.json()["embed_mode"] == "local"


def test_tensor_to_embedding_rows_handles_batch() -> None:
    tensor = MagicMock()
    tensor.detach.return_value = tensor
    tensor.cpu.return_value = tensor
    tensor.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
    assert _tensor_to_embedding_rows(tensor) == [[0.1, 0.2], [0.3, 0.4]]


def test_vector_db_state_local_embed_queries() -> None:
    mock_embedder = MagicMock()
    tensor = MagicMock()
    tensor.detach.return_value = tensor
    tensor.cpu.return_value = tensor
    tensor.tolist.return_value = [[1.0, 2.0]]
    mock_embedder.embed_queries.return_value = tensor

    state = VectorDBState(
        lancedb_uri="/tmp/unused",
        table_name="t",
        embed_endpoint="",
        embed_model="nvidia/llama-nemotron-embed-1b-v2",
        embed_api_key="",
        local_embed=True,
        local_embed_backend="hf",
    )

    with patch("nemo_retriever.models.create_local_embedder", return_value=mock_embedder):
        vectors = state.embed_queries(["hello"])

    assert vectors == [[1.0, 2.0]]
    mock_embedder.embed_queries.assert_called_once_with(["hello"])


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
