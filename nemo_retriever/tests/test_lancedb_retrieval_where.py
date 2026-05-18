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


def _tiny_table(uri: str) -> None:
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
    db.create_table("t", rows, schema=schema, mode="overwrite")


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


def test_retrieval_filter_dict_translates_to_metadata_like() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)
    qv = [1.0, 0.0]
    filtered = op.retrieval([qv], top_k=10, table_path=d, table_name="t", filter={"doc_id": "x"})
    assert len(filtered[0]) == 1
    cm = json.loads(filtered[0][0]["metadata"])
    assert cm["doc_id"] == "x"


def test_retrieval_filter_dict_multiple_keys_anded() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)
    qv = [1.0, 0.0]
    none_match = op.retrieval(
        [qv], top_k=10, table_path=d, table_name="t", filter={"doc_id": "x", "page_number_unused": "nope"}
    )
    assert len(none_match[0]) == 0


def test_retrieval_filter_empty_or_none_is_noop() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)
    qv = [1.0, 0.0]
    assert len(op.retrieval([qv], top_k=10, table_path=d, table_name="t", filter=None)[0]) == 2
    assert len(op.retrieval([qv], top_k=10, table_path=d, table_name="t", filter={})[0]) == 2
    assert len(op.retrieval([qv], top_k=10, table_path=d, table_name="t", filter={"doc_id": None})[0]) == 2


def test_retrieval_where_takes_precedence_over_filter() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)
    qv = [1.0, 0.0]
    out = op.retrieval(
        [qv],
        top_k=10,
        table_path=d,
        table_name="t",
        where="text = 'alpha'",
        filter={"doc_id": "y"},
    )
    assert len(out[0]) == 1
    assert out[0][0]["text"] == "alpha"


def test_retrieval_filter_non_string_value_raises() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)
    with pytest.raises(NotImplementedError, match="string equality"):
        op.retrieval([[1.0, 0.0]], top_k=5, table_path=d, table_name="t", filter={"page_number": 1})


def test_retrieval_filter_must_be_dict() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)
    with pytest.raises(TypeError, match="filter must be a dict"):
        op.retrieval([[1.0, 0.0]], top_k=5, table_path=d, table_name="t", filter="bad")


def test_retrieval_search_kwargs_must_be_dict() -> None:
    d = tempfile.mkdtemp()
    _tiny_table(d)
    op = LanceDB(uri=d, table_name="t", overwrite=False, vector_dim=2, validate_vector_length=False)
    with pytest.raises(TypeError, match="search_kwargs"):
        op.retrieval([[1.0, 0.0]], top_k=5, table_path=d, table_name="t", search_kwargs="bad")
