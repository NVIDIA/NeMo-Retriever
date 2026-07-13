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
def test_fts_build_failure_on_create_preserves_rows_on_retry(tmp_path) -> None:
    state = VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
        retrieval_mode="hybrid",
    )

    # First write persists the rows but the FTS build blows up before
    # `_table_exists` is published.
    with patch.object(VectorDBState, "_ensure_hybrid_indexes", side_effect=RuntimeError("disk full")):
        with pytest.raises(RuntimeError, match="disk full"):
            state.write_rows([_ROW])

    assert state._table_exists is False
    assert state._table_present_on_disk() is True

    # The retry must append to the existing table (not overwrite it) and then
    # successfully build the FTS index, leaving the original row intact.
    appended = dict(_ROW)
    appended["vector"] = [0.0, 1.0, 0.0, 0.0]
    appended["text"] = "Second batch after recovery."
    assert state.write_rows([appended]) == 1

    assert state._table_exists is True
    table = state._db.open_table("nemo_retriever")
    assert table.count_rows() == 2  # first batch preserved, not wiped
    assert state.resolve_effective_retrieval_mode() == "hybrid"


@pytest.mark.integration
def test_appended_rows_are_added_to_fts_index_in_hybrid_mode(tmp_path) -> None:
    state = VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
        retrieval_mode="hybrid",
    )
    assert state.write_rows([_ROW]) == 1

    appended = dict(_ROW)
    appended["vector"] = [0.0, 1.0, 0.0, 0.0]
    appended["text"] = "Zephyr quarterly guidance mentions unicorn synergy."

    # The append must run the incremental index refresh (optimize), not a full
    # FTS rebuild, so the write lock is held only for the new fragment. The spy
    # records the call while still executing the real optimize.
    original_refresh = VectorDBState._refresh_indexes
    refresh_calls: list[bool] = []

    def _spy(self, table):
        refresh_calls.append(True)
        return original_refresh(self, table)

    with patch.object(VectorDBState, "_refresh_indexes", _spy):
        assert state.write_rows([appended]) == 1
    assert refresh_calls, "append in hybrid mode must incrementally refresh indexes"

    # The appended row must be discoverable through lexical (FTS) search.
    table = state._db.open_table("nemo_retriever")
    hits = table.search("unicorn", query_type="fts", fts_columns="text").limit(5).to_list()
    assert any("unicorn synergy" in hit["text"] for hit in hits)


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
