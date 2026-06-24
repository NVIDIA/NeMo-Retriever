# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import pytest

lancedb = pytest.importorskip("lancedb")
pa = pytest.importorskip("pyarrow")

from nemo_retriever.common.vdb.lancedb_capabilities import inspect_lancedb_table  # noqa: E402
from nemo_retriever.common.vdb.lancedb_capabilities import LanceTableCapabilities  # noqa: E402
from nemo_retriever.graph.retriever import Retriever  # noqa: E402
import nemo_retriever.graph.retriever as retriever_module  # noqa: E402


def _create_vector_table(uri: str, table_name: str, *, fts: bool = False) -> None:
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("id", pa.string()),
        ]
    )
    rows = [
        {
            "vector": [1.0, 0.0],
            "text": "alpha safety manual",
            "metadata": json.dumps({"page_number": 1, "type": "text"}),
            "source": json.dumps({"source_id": "alpha.pdf"}),
            "id": "alpha",
        }
    ]
    table = lancedb.connect(uri).create_table(table_name, data=rows, schema=schema, mode="overwrite")
    if fts:
        table.create_fts_index("text", replace=True)


def _create_sparse_table(uri: str, table_name: str) -> None:
    schema = pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("id", pa.string()),
        ]
    )
    rows = [
        {
            "text": "alpha safety manual",
            "metadata": json.dumps({"page_number": 1, "type": "text"}),
            "source": json.dumps({"source_id": "alpha.pdf"}),
            "id": "alpha",
        }
    ]
    table = lancedb.connect(uri).create_table(table_name, data=rows, schema=schema, mode="overwrite")
    table.create_fts_index("text", replace=True)


def test_detector_returns_dense_for_vector_only_table(tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "dense")

    caps = inspect_lancedb_table(uri, "dense")

    assert caps.has_vector is True
    assert caps.has_fts is False
    assert caps.vector_column == "vector"
    assert caps.text_column == "text"
    assert caps.retrieval_mode == "dense"


def test_detector_returns_hybrid_for_vector_plus_fts_table(tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "hybrid", fts=True)

    caps = inspect_lancedb_table(uri, "hybrid")

    assert caps.has_vector is True
    assert caps.has_fts is True
    assert caps.retrieval_mode == "hybrid"


def test_detector_returns_sparse_for_fts_only_table(tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_sparse_table(uri, "sparse")

    caps = inspect_lancedb_table(uri, "sparse")

    assert caps.has_vector is False
    assert caps.has_fts is True
    assert caps.vector_column is None
    assert caps.text_column == "text"
    assert caps.retrieval_mode == "sparse"


def test_sparse_query_does_not_call_embedding_graph(monkeypatch, tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_sparse_table(uri, "sparse")

    def fail_embed_graph(*_args: Any, **_kwargs: Any) -> list[list[dict[str, Any]]]:
        raise AssertionError("sparse query should not build or execute the embedding graph")

    monkeypatch.setattr(Retriever, "_execute_queries_graph", fail_embed_graph)

    hits = Retriever(vdb_kwargs={"uri": uri, "table_name": "sparse"}).query("alpha", top_k=1)

    assert hits
    assert hits[0]["text"] == "alpha safety manual"


def test_hybrid_table_query_automatically_enables_hybrid(monkeypatch, tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "hybrid", fts=True)
    calls: list[dict[str, Any]] = []

    def fake_execute(self, query_texts, **kwargs):  # noqa: ANN001
        calls.append(kwargs["vdb_call_kwargs"])
        return [[{"text": "alpha safety manual", "metadata": {"type": "text"}, "source": "alpha.pdf"}]]

    monkeypatch.setattr(Retriever, "_execute_queries_graph", fake_execute)

    Retriever(vdb_kwargs={"uri": uri, "table_name": "hybrid"}).query("alpha", top_k=1)

    assert calls == [{"hybrid": True}]


def test_existing_dense_query_behavior_is_unchanged(monkeypatch, tmp_path) -> None:
    uri = str(tmp_path / "db")
    _create_vector_table(uri, "dense")
    calls: list[dict[str, Any]] = []

    def fake_execute(self, query_texts, **kwargs):  # noqa: ANN001
        calls.append(kwargs["vdb_call_kwargs"])
        return [[{"text": "alpha safety manual", "metadata": {"type": "text"}, "source": "alpha.pdf"}]]

    monkeypatch.setattr(Retriever, "_execute_queries_graph", fake_execute)

    Retriever(vdb_kwargs={"uri": uri, "table_name": "dense"}).query("alpha", top_k=1)

    assert calls == [{}]


def test_retriever_caches_lancedb_capability_inspection(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    caps = LanceTableCapabilities(
        has_vector=True,
        has_fts=False,
        retrieval_mode="dense",
        vector_column="vector",
        text_column="text",
    )

    def fake_inspect(uri: str, table_name: str) -> LanceTableCapabilities:
        calls.append((uri, table_name))
        return caps

    monkeypatch.setattr(retriever_module, "inspect_lancedb_table", fake_inspect)

    retriever = Retriever(vdb_kwargs={"uri": "memory://db", "table_name": "docs"})

    assert retriever._resolve_lancedb_query_mode({}) == ("dense", caps, "memory://db", "docs", False)
    assert retriever._resolve_lancedb_query_mode({}) == ("dense", caps, "memory://db", "docs", False)
    assert calls == [("memory://db", "docs")]
