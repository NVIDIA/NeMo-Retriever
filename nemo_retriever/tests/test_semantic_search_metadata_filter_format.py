# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the metadata filter builder used by tabular semantic search.

``_build_metadata_where_clause`` emits one of two shapes depending on the
``fmt`` flag:

* ``"sql"`` (default) — the historical ``LIKE``-over-JSON predicate that
  LanceDB's ``.where()`` API accepts.
* ``"dict"`` — a flat ``{column: value | [values]}`` mapping for backends
  whose filter API consumes a dict (e.g. pgvector).

``search_semantic_index`` picks the shape by reading
``retriever.vdb_kwargs["vdb"].metadata_filter_format`` (defaulting to
``"sql"`` when no VDB instance is injected).
"""

from __future__ import annotations

from nemo_retriever.tabular_data.retrieval.data_access.semantic_search import (
    _build_metadata_where_clause,
    _metadata_filter_format,
)


# ---------------------------------------------------------------------------
# fmt="sql" — preserves the historical LIKE-over-JSON predicate
# ---------------------------------------------------------------------------


def test_no_filters_returns_none() -> None:
    assert _build_metadata_where_clause() is None
    assert _build_metadata_where_clause(labels=[], database_name=None) is None
    assert _build_metadata_where_clause(labels=None, database_name="") is None
    assert _build_metadata_where_clause(fmt="dict") is None


def test_sql_single_label_emits_like_predicate() -> None:
    out = _build_metadata_where_clause(labels=["Column"])
    assert out == """metadata LIKE '%"label":"Column"%' ESCAPE '\\'"""


def test_sql_multiple_labels_join_with_or() -> None:
    out = _build_metadata_where_clause(labels=["Column", "Table"])
    assert out == (
        """(metadata LIKE '%"label":"Column"%' ESCAPE '\\'""" """ OR metadata LIKE '%"label":"Table"%' ESCAPE '\\')"""
    )


def test_sql_label_and_database_name_combined() -> None:
    # NB: ``_escape_like`` escapes ``_`` → ``\_`` so it isn't a LIKE wildcard.
    out = _build_metadata_where_clause(labels=["Column"], database_name="dor_prod")
    assert out == (
        """metadata LIKE '%"label":"Column"%' ESCAPE '\\'"""
        """ AND metadata LIKE '%"database_name":"dor\\_prod"%' ESCAPE '\\'"""
    )


def test_sql_database_name_only() -> None:
    out = _build_metadata_where_clause(database_name="dor_prod")
    assert out == """metadata LIKE '%"database_name":"dor\\_prod"%' ESCAPE '\\'"""


# ---------------------------------------------------------------------------
# fmt="dict" — pgvector-style flat mapping
# ---------------------------------------------------------------------------


def test_dict_single_label() -> None:
    assert _build_metadata_where_clause(labels=["Column"], fmt="dict") == {"label": "Column"}


def test_dict_multiple_labels_become_list() -> None:
    assert _build_metadata_where_clause(labels=["Column", "Table"], fmt="dict") == {
        "label": ["Column", "Table"],
    }


def test_dict_label_and_database_name() -> None:
    assert _build_metadata_where_clause(labels=["Column"], database_name="dor_prod", fmt="dict") == {
        "label": "Column",
        "database_name": "dor_prod",
    }


def test_dict_database_name_only() -> None:
    assert _build_metadata_where_clause(database_name="dor_prod", fmt="dict") == {
        "database_name": "dor_prod",
    }


# ---------------------------------------------------------------------------
# _metadata_filter_format — reads the flag off the retriever's injected VDB
# ---------------------------------------------------------------------------


class _FakeVdb:
    def __init__(self, fmt: str) -> None:
        self.metadata_filter_format = fmt


class _FakeRetriever:
    def __init__(self, vdb: object | None) -> None:
        self.vdb_kwargs = {"vdb": vdb} if vdb is not None else {}


def test_metadata_filter_format_reads_from_injected_vdb() -> None:
    assert _metadata_filter_format(_FakeRetriever(_FakeVdb("dict"))) == "dict"
    assert _metadata_filter_format(_FakeRetriever(_FakeVdb("sql"))) == "sql"


def test_metadata_filter_format_defaults_to_sql() -> None:
    assert _metadata_filter_format(_FakeRetriever(None)) == "sql"
    assert _metadata_filter_format(_FakeRetriever(object())) == "sql"


def test_metadata_filter_format_unknown_falls_back_to_sql() -> None:
    assert _metadata_filter_format(_FakeRetriever(_FakeVdb("yaml"))) == "sql"
