# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi.testclient import TestClient

import nemo_retriever.service.vectordb_app as vectordb_module
from nemo_retriever.service.vectordb_app import (
    VectorDBState,
    _embed_queries_remote,
    _tensor_to_embedding_rows,
    create_vectordb_app,
)


@pytest.mark.parametrize(
    ("extra_args", "expected_key"),
    [([], "env-key"), (["--embed-api-key", "explicit-key"], "explicit-key")],
)
def test_main_resolves_remote_embed_api_key(monkeypatch, extra_args, expected_key) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    monkeypatch.setattr(sys, "argv", ["vectordb_app", *extra_args])
    create_app = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(vectordb_module, "create_vectordb_app", create_app)
    monkeypatch.setattr(vectordb_module.uvicorn, "run", MagicMock())

    vectordb_module.main()

    assert create_app.call_args.kwargs["embed_api_key"] == expected_key


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
    ), patch.object(VectorDBState, "search", return_value=[_CANNED_HITS]):
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
