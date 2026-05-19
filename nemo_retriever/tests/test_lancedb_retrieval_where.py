# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LanceDB.retrieval pre-filtering (where / search_kwargs)."""

from __future__ import annotations

import json
import tempfile

import pyarrow as pa
import pytest

lancedb = pytest.importorskip("lancedb")

from nemo_retriever.vdb.lancedb import LanceDB


def _tiny_table(uri: str, *, create_fts_index: bool = False) -> None:
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
        ]
    )
    rows = [
        {
            "vector": [1.0, 0.0],
            "text": "alpha",
            "metadata": json.dumps({"doc_id": "x", "page_number": 1}),
            "source": "{}",
        },
        {
            "vector": [0.0, 1.0],
            "text": "beta",
            "metadata": json.dumps({"doc_id": "y", "page_number": 2}),
            "source": "{}",
        },
    ]
    db = lancedb.connect(uri)
    table = db.create_table("t", rows, schema=schema, mode="overwrite")
    if create_fts_index:
        table.create_fts_index("text", replace=True)


def test_retrieval_where_filters_rows() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)
    qv = [1.0, 0.0]
    unfiltered = op.retrieval([qv], top_k=10, table_path=d, table_name="t")
    assert len(unfiltered[0]) == 2
    filtered = op.retrieval([qv], top_k=10, table_path=d, table_name="t", where="text = 'alpha'")
    assert len(filtered[0]) == 1
    assert filtered[0][0]["text"] == "alpha"


def test_retrieval_filter_alias() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)
    qv = [1.0, 0.0]
    filtered = op.retrieval([qv], top_k=10, table_path=d, table_name="t", _filter="text = 'beta'")
    assert len(filtered[0]) == 1
    assert filtered[0][0]["text"] == "beta"


def test_retrieval_where_precedence_over_filter() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)
    qv = [1.0, 0.0]
    filtered = op.retrieval(
        [qv],
        top_k=10,
        table_path=d,
        table_name="t",
        where="text = 'alpha'",
        _filter="text = 'beta'",
    )
    assert len(filtered[0]) == 1
    assert filtered[0][0]["text"] == "alpha"


def test_retrieval_metadata_like_predicate() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)
    qv = [1.0, 0.0]
    pred = '%"doc_id": "x"%'
    filtered = op.retrieval([qv], top_k=10, table_path=d, table_name="t", where=f"metadata LIKE '{pred}'")
    assert len(filtered[0]) == 1
    cm = json.loads(filtered[0][0]["metadata"])
    assert cm["doc_id"] == "x"


def test_retrieval_search_kwargs_must_be_dict() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)
    with pytest.raises(TypeError, match="search_kwargs"):
        op.retrieval([[1.0, 0.0]], top_k=5, table_path=d, table_name="t", search_kwargs="bad")


def test_hybrid_retrieval_uses_query_texts() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d, create_fts_index=True)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)

    results = op.retrieval(
        [[1.0, 0.0]],
        top_k=2,
        table_path=d,
        table_name="t",
        hybrid=True,
        query_texts=["alpha"],
    )

    assert results[0]
    assert results[0][0]["text"] == "alpha"


def test_hybrid_retrieval_requires_query_texts() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d, create_fts_index=True)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)

    with pytest.raises(ValueError, match="requires query_texts"):
        op.retrieval([[1.0, 0.0]], top_k=2, table_path=d, table_name="t", hybrid=True)


def test_hybrid_retrieval_requires_query_texts_aligned_with_vectors() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d, create_fts_index=True)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)

    with pytest.raises(ValueError, match="length to match vectors length"):
        op.retrieval(
            [[1.0, 0.0]],
            top_k=2,
            table_path=d,
            table_name="t",
            hybrid=True,
            query_texts=["alpha", "beta"],
        )


def test_hybrid_retrieval_where_filters_rows() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d, create_fts_index=True)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)

    filtered = op.retrieval(
        [[1.0, 0.0]],
        top_k=10,
        table_path=d,
        table_name="t",
        hybrid=True,
        query_texts=["beta"],
        where="text = 'beta'",
    )

    assert len(filtered[0]) == 1
    assert filtered[0][0]["text"] == "beta"


def test_hybrid_retrieval_rejects_non_hybrid_query_type() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d, create_fts_index=True)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)

    with pytest.raises(ValueError, match="query_type"):
        op.retrieval(
            [[1.0, 0.0]],
            top_k=2,
            table_path=d,
            table_name="t",
            hybrid=True,
            query_texts=["alpha"],
            search_kwargs={"query_type": "vector"},
        )
