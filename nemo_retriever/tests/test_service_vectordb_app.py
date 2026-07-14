# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

from fastapi.testclient import TestClient

from nemo_retriever.service.vectordb_app import (
    VectorDBState,
    _embed_queries_remote,
    _strategies_for_retrieval_mode,
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


def test_health_reports_effective_retrieval_mode_none_without_table(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="http://embed.example/v1/embeddings",
    )
    with TestClient(app) as client:
        resp = client.get("/v1/health")

    assert resp.status_code == 200
    body = resp.json()
    assert "retrieval_mode" not in body
    assert body["effective_retrieval_mode"] is None


def test_health_stays_ok_when_mode_resolution_errors(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        embed_endpoint="http://embed.example/v1/embeddings",
    )
    with TestClient(app) as client:
        with patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True), patch.object(
            VectorDBState,
            "resolve_effective_retrieval_mode",
            side_effect=OSError("transient I/O error"),
        ):
            resp = client.get("/v1/health")

    # Health backs k8s probes; a mode-resolution failure must not 500.
    assert resp.status_code == 200
    assert resp.json()["effective_retrieval_mode"] == "unknown"


def test_strategies_for_retrieval_mode() -> None:
    assert _strategies_for_retrieval_mode("dense") == ["dense"]
    assert _strategies_for_retrieval_mode("hybrid") == ["hybrid"]


def test_write_rows_creates_then_appends_table(tmp_path) -> None:
    state = VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
    )
    assert state.table_exists is False

    row = {
        "vector": [1.0, 0.0, 0.0, 0.0],
        "text": "seed",
        "pdf_page": "p1",
        "filename": "f.pdf",
        "pdf_basename": "f.pdf",
        "page_number": 1,
        "source": "f.pdf",
        "source_id": "f.pdf",
        "path": "/f.pdf",
        "metadata": "{}",
        "stored_image_uri": "",
        "content_type": "text",
        "bbox_xyxy_norm": "",
    }
    assert state.write_rows([row]) == 1
    assert state.table_exists is True
    assert state.total_rows() == 1

    # A second write appends rather than overwriting the existing table.
    assert state.write_rows([dict(row, text="second")]) == 1
    assert state.total_rows() == 2


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


def test_remote_embed_queries_delegates_model_prefix(monkeypatch) -> None:
    calls = {}

    def fake_infer_microservice(data, **kwargs):
        calls["data"] = data
        calls.update(kwargs)
        return [[0.1, 0.2]]

    monkeypatch.setattr("nemo_retriever.models.nim.util.infer_microservice", fake_infer_microservice)

    vectors = _embed_queries_remote(
        ["hello"],
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_endpoint="https://litellm.example.com/v1/embeddings",
        embed_api_key="k",
        embed_model_provider_prefix="nvidia",
    )

    assert vectors == [[0.1, 0.2]]
    assert calls["data"] == ["hello"]
    assert calls["model_name"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert calls["model_provider_prefix"] == "nvidia"
    assert calls["embedding_endpoint"] == "https://litellm.example.com/v1/embeddings"


_CANNED_HITS = [
    {
        "text": "Revenue grew 12% year over year.",
        "pdf_basename": "10k_2023.pdf",
        "page_number": 12,
        "content_type": "text",
        "_score": 0.91,
        "metadata": {},
    }
]


def _query_app(tmp_path):
    return create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="t",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )


def test_query_evidence_format_returns_evidence_coverage(tmp_path) -> None:
    app = _query_app(tmp_path)
    with patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True), patch.object(
        VectorDBState, "embed_queries", return_value=[[0.1, 0.2]]
    ), patch.object(VectorDBState, "search", return_value=([_CANNED_HITS], ["dense"])):
        with TestClient(app) as client:
            resp = client.post("/v1/query", json={"query": "revenue", "top_k": 5, "format": "evidence"})

    assert resp.status_code == 200
    body = resp.json()
    assert list(body) == ["results"]
    assert len(body["results"]) == 1
    item = body["results"][0]
    assert set(item) == {"evidence", "coverage"}

    ev = item["evidence"][0]
    assert ev["source"] == "10k_2023"
    assert ev["citation"] == "10k_2023 p.12"
    assert ev["locator"] == {"kind": "page", "value": 12}
    assert ev["modality"] == "text"
    assert ev["fidelity"] == "verbatim"
    assert ev["score"] == 0.91

    coverage = item["coverage"]
    assert coverage["strategies_used"] == ["dense"]
    assert coverage["n_docs_seen"] == 1
    assert coverage["thin_spots"] == ["single source"]


def test_query_hybrid_evidence_reports_hybrid_strategy(tmp_path) -> None:
    # A table whose capabilities resolve to hybrid (vector + FTS) reports hybrid.
    app = _query_app(tmp_path)
    with patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True), patch.object(
        VectorDBState, "embed_queries", return_value=[[0.1, 0.2]]
    ), patch.object(VectorDBState, "search", return_value=([_CANNED_HITS], ["hybrid"])):
        with TestClient(app) as client:
            resp = client.post("/v1/query", json={"query": "revenue", "top_k": 5, "format": "evidence"})

    assert resp.status_code == 200
    assert resp.json()["results"][0]["coverage"]["strategies_used"] == ["hybrid"]


def test_query_unqueryable_table_returns_422(tmp_path) -> None:
    # An unqueryable table (e.g. FTS-only / no vector column) surfaces as 422.
    app = _query_app(tmp_path)
    with patch.object(VectorDBState, "table_exists", new_callable=PropertyMock, return_value=True), patch.object(
        VectorDBState, "embed_queries", return_value=[[0.1, 0.2]]
    ), patch.object(
        VectorDBState,
        "search",
        side_effect=ValueError(
            "LanceDB table 't' at '" + str(tmp_path) + "' has an FTS index but no vector "
            "column; sparse-only retrieval is not supported by the VectorDB service."
        ),
    ):
        with TestClient(app) as client:
            resp = client.post("/v1/query", json={"query": "revenue", "top_k": 5})

    assert resp.status_code == 422
    assert "sparse-only retrieval is not supported" in resp.json()["detail"]


def test_search_hybrid_delegates_to_lancedb_wrapper(tmp_path) -> None:
    state = VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="docs",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
    )
    state._table_exists = True

    mock_caps = MagicMock()
    mock_caps.has_vector = True
    mock_caps.has_fts = True
    mock_caps.retrieval_mode = "hybrid"
    mock_caps.vector_column = "vector"

    mock_vdb = MagicMock()
    mock_vdb.retrieval.return_value = [[{"text": "hit", "_score": 0.5}]]

    with patch.object(state, "_table_capabilities", return_value=mock_caps), patch(
        "nemo_retriever.common.vdb.lancedb.LanceDB",
        return_value=mock_vdb,
    ):
        hits, strategies = state.search([[0.1, 0.2]], ["revenue"], top_k=3)

    assert strategies == ["hybrid"]
    assert hits[0][0]["text"] == "hit"
    mock_vdb.retrieval.assert_called_once()
    call_kwargs = mock_vdb.retrieval.call_args.kwargs
    assert call_kwargs["top_k"] == 3
    assert call_kwargs["hybrid"] is True
    assert call_kwargs["query_texts"] == ["revenue"]
