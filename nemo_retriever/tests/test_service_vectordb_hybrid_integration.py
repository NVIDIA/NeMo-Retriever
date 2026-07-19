# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import timedelta
from unittest.mock import patch

import lancedb
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


def _state(tmp_path) -> VectorDBState:
    return VectorDBState(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        embed_api_key="",
    )


def _prebuild_fts_index(uri: str, table_name: str) -> None:
    """Simulate an ingestion pipeline that wrote a table with a BM25/FTS index.

    The VectorDB service itself never builds FTS; the query path only detects
    an index that was created at ingestion time.
    """
    table = lancedb.connect(uri).open_table(table_name)
    table.create_fts_index("text", replace=True)
    for stub in table.list_indices():
        if "text" in stub.name.lower() or "fts" in stub.name.lower():
            table.wait_for_index([stub.name], timeout=timedelta(seconds=600))


@pytest.mark.integration
def test_write_rows_persists_rows_without_building_fts(tmp_path) -> None:
    state = _state(tmp_path)
    assert state.write_rows([_ROW]) == 1

    caps = state._table_capabilities()
    assert caps is not None
    assert caps.has_vector
    # The service must not build an FTS index on write; the table stays dense.
    assert not caps.has_fts
    assert state.resolve_effective_retrieval_mode() == "dense"


@pytest.mark.integration
def test_append_does_not_build_or_mutate_fts(tmp_path) -> None:
    state = _state(tmp_path)
    assert state.write_rows([_ROW]) == 1

    appended = dict(_ROW)
    appended["vector"] = [0.0, 1.0, 0.0, 0.0]
    appended["text"] = "Zephyr quarterly guidance mentions unicorn synergy."
    assert state.write_rows([appended]) == 1

    table = state._db.open_table("nemo_retriever")
    assert table.count_rows() == 2
    # Still no FTS index — appends only persist rows.
    caps = state._table_capabilities()
    assert not caps.has_fts


@pytest.mark.integration
def test_auto_resolves_hybrid_when_fts_prebuilt(tmp_path) -> None:
    # Ingestion built the table with both a vector column and an FTS index.
    seed = _state(tmp_path)
    seed.write_rows([_ROW])
    _prebuild_fts_index(str(tmp_path), "nemo_retriever")

    state = _state(tmp_path)
    caps = state._table_capabilities()
    assert caps.has_vector
    assert caps.has_fts
    assert state.resolve_effective_retrieval_mode() == "hybrid"


@pytest.mark.integration
def test_auto_resolves_dense_when_no_fts(tmp_path) -> None:
    seed = _state(tmp_path)
    seed.write_rows([_ROW])

    state = _state(tmp_path)
    caps = state._table_capabilities()
    assert caps.has_vector
    assert not caps.has_fts
    assert state.resolve_effective_retrieval_mode() == "dense"


@pytest.mark.integration
def test_query_auto_selects_hybrid_when_fts_prebuilt(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0, 0.0, 0.0]]):
        with TestClient(app) as client:
            write = client.post("/internal/vectordb/write", json={"rows": [_ROW]})
            assert write.status_code == 200, write.text

            # Ingestion builds the FTS index; the query path detects it.
            _prebuild_fts_index(str(tmp_path), "nemo_retriever")

            resp = client.post(
                "/v1/query",
                json={"query": "revenue", "top_k": 5, "format": "evidence"},
            )

    assert resp.status_code == 200, resp.text
    coverage = resp.json()["results"][0]["coverage"]
    assert coverage["strategies_used"] == ["hybrid"]


@pytest.mark.integration
def test_query_auto_selects_dense_when_no_fts(tmp_path) -> None:
    app = create_vectordb_app(
        lancedb_uri=str(tmp_path),
        table_name="nemo_retriever",
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
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
    assert coverage["strategies_used"] == ["dense"]
