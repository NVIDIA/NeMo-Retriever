# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import lancedb
import pytest
from fastapi.testclient import TestClient

from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb.lancedb_capabilities import inspect_lancedb_table_object
from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app

_DIM = 4


def _record(
    *,
    vector: list[float] | None = None,
    text: str = "Revenue grew 12% year over year.",
) -> dict:
    return {
        "document_type": "text",
        "metadata": {
            "embedding": vector or [1.0, 0.0, 0.0, 0.0],
            "content": text,
            "content_metadata": {"page_number": 12, "type": "text"},
            "source_metadata": {
                "source_id": "/data/10k_2023.pdf",
                "source_name": "10k_2023.pdf",
            },
        },
    }


_RECORD = _record()


def _backend(tmp_path, *, hybrid: bool = False) -> LanceDB:
    return LanceDB(
        uri=str(tmp_path),
        table_name="nemo_retriever",
        vector_dim=_DIM,
        overwrite=False,
        build_index=False,
        hybrid=hybrid,
    )


def _write(backend: LanceDB, *records: dict) -> int:
    backend.run([list(records)])
    table = lancedb.connect(backend.uri).open_table(backend.table_name)
    return int(table.count_rows())


def _capabilities(backend: LanceDB):
    table = lancedb.connect(backend.uri).open_table(backend.table_name)
    return inspect_lancedb_table_object(table)


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
    backend = _backend(tmp_path)
    assert _write(backend, _RECORD) == 1

    caps = _capabilities(backend)
    assert caps is not None
    assert caps.has_vector
    # The service must not build an FTS index on write; the table stays dense.
    assert not caps.has_fts
    assert backend.health()["effective_retrieval_mode"] == "dense"


@pytest.mark.integration
def test_append_does_not_build_or_mutate_fts(tmp_path) -> None:
    backend = _backend(tmp_path)
    assert _write(backend, _RECORD) == 1

    appended = _record(
        vector=[0.0, 1.0, 0.0, 0.0],
        text="Zephyr quarterly guidance mentions unicorn synergy.",
    )
    assert _write(backend, appended) == 2

    table = lancedb.connect(str(tmp_path)).open_table("nemo_retriever")
    assert table.count_rows() == 2
    # Still no FTS index — appends only persist rows.
    caps = _capabilities(backend)
    assert not caps.has_fts


@pytest.mark.integration
def test_auto_resolves_hybrid_when_fts_prebuilt(tmp_path) -> None:
    # Ingestion built the table with both a vector column and an FTS index.
    seed = _backend(tmp_path, hybrid=True)
    _write(seed, _RECORD)
    _prebuild_fts_index(str(tmp_path), "nemo_retriever")

    backend = _backend(tmp_path, hybrid=True)
    caps = _capabilities(backend)
    assert caps.has_vector
    assert caps.has_fts
    assert backend.health()["effective_retrieval_mode"] == "hybrid"


@pytest.mark.integration
def test_auto_resolves_dense_when_no_fts(tmp_path) -> None:
    seed = _backend(tmp_path)
    _write(seed, _RECORD)

    backend = _backend(tmp_path)
    caps = _capabilities(backend)
    assert caps.has_vector
    assert not caps.has_fts
    assert backend.health()["effective_retrieval_mode"] == "dense"


@pytest.mark.integration
def test_query_auto_selects_hybrid_when_fts_prebuilt(tmp_path) -> None:
    backend = _backend(tmp_path, hybrid=True)
    app = create_vectordb_app(
        vdb=backend,
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0, 0.0, 0.0]]):
        with TestClient(app) as client:
            write = client.post("/internal/vectordb/write", json={"records": [[_RECORD]]})
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
    backend = _backend(tmp_path)
    app = create_vectordb_app(
        vdb=backend,
        embed_endpoint="http://embed.example/v1/embeddings",
        embed_model="nvidia/llama-nemotron-embed-vl-1b-v2",
    )

    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0, 0.0, 0.0]]):
        with TestClient(app) as client:
            write = client.post("/internal/vectordb/write", json={"records": [[_RECORD]]})
            assert write.status_code == 200, write.text

            resp = client.post(
                "/v1/query",
                json={"query": "revenue", "top_k": 5, "format": "evidence"},
            )

    assert resp.status_code == 200, resp.text
    coverage = resp.json()["results"][0]["coverage"]
    assert coverage["strategies_used"] == ["dense"]
