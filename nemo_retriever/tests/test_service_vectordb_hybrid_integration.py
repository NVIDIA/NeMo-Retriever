# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app

_DIM = 4
_ROW = {
    "vector": [1.0, 0.0, 0.0, 0.0],
    "pdf_page": "10k_2023_12",
    "filename": "10k_2023.pdf",
    "pdf_basename": "10k_2023.pdf",
    "page_number": 12,
    "source": "10k_2023.pdf",
    "source_id": "10k_2023.pdf",
    "path": "/data/10k_2023.pdf",
    "text": "Revenue grew 12% year over year.",
    "metadata": json.dumps({"page_number": 12, "type": "text"}),
    "stored_image_uri": "",
    "content_type": "text",
    "bbox_xyxy_norm": "",
}


@pytest.mark.integration
def test_write_rows_builds_fts_index_for_hybrid_mode(tmp_path) -> None:
    state = VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
        retrieval_mode="hybrid",
    )
    written = state.write_rows([_ROW])
    assert written == 1

    caps = state._table_capabilities()
    assert caps is not None
    assert caps.has_vector
    assert caps.has_fts
    assert state.resolve_effective_retrieval_mode() == "hybrid"


@pytest.mark.integration
def test_write_rows_dense_mode_skips_fts_index(tmp_path) -> None:
    state = VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
        retrieval_mode="dense",
    )
    written = state.write_rows([_ROW])
    assert written == 1

    caps = state._table_capabilities()
    assert caps is not None
    assert caps.has_vector
    assert not caps.has_fts
    assert state.resolve_effective_retrieval_mode() == "dense"


@pytest.mark.integration
def test_auto_mode_resolves_hybrid_when_fts_present(tmp_path) -> None:
    # Seed with hybrid so the table has both a vector column and an FTS index.
    seed = VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
        retrieval_mode="hybrid",
    )
    seed.write_rows([_ROW])

    auto = VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
        retrieval_mode="auto",
    )
    assert auto.resolve_effective_retrieval_mode() == "hybrid"


@pytest.mark.integration
def test_auto_mode_resolves_dense_when_no_fts(tmp_path) -> None:
    # Seed with dense so the table has a vector column but no FTS index.
    seed = VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
        retrieval_mode="dense",
    )
    seed.write_rows([_ROW])

    auto = VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
        retrieval_mode="auto",
    )
    assert auto.resolve_effective_retrieval_mode() == "dense"


@pytest.mark.integration
def test_hybrid_mode_builds_fts_on_existing_dense_table(tmp_path) -> None:
    seed = VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
        retrieval_mode="dense",
    )
    seed.write_rows([_ROW])

    hybrid = VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
        retrieval_mode="hybrid",
    )
    assert hybrid.resolve_effective_retrieval_mode() == "hybrid"


@pytest.mark.integration
def test_query_hybrid_end_to_end_over_real_lancedb(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        retrieval_mode="hybrid",
    )

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0, 0.0, 0.0]]):
        with TestClient(app) as client:
            write = client.post("/internal/vectordb/write", json={"rows": [_ROW]})
            assert write.status_code == 200, write.text

            resp = client.post(
                "/v1/query",
                json={"query": "revenue", "top_k": 5, "format": "evidence"},
            )

    assert resp.status_code == 200, resp.text
    coverage = resp.json()["results"][0]["coverage"]
    assert coverage["strategies_used"] == ["hybrid"]


@pytest.mark.integration
def test_query_auto_end_to_end_selects_hybrid_over_real_lancedb(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        retrieval_mode="auto",
    )

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0, 0.0, 0.0]]):
        with TestClient(app) as client:
            write = client.post("/internal/vectordb/write", json={"rows": [_ROW]})
            assert write.status_code == 200, write.text

            resp = client.post(
                "/v1/query",
                json={"query": "revenue", "top_k": 5, "format": "evidence"},
            )

    assert resp.status_code == 200, resp.text
    coverage = resp.json()["results"][0]["coverage"]
    # auto detects the FTS index built during the hybrid-capable write and upgrades to hybrid.
    assert coverage["strategies_used"] == ["hybrid"]
