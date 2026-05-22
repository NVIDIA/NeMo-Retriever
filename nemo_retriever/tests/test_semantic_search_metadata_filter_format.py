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


def test_no_arguments_returns_none() -> None:
    assert _build_metadata_where_clause() is None


def test_empty_labels_and_no_database_name_returns_none() -> None:
    assert _build_metadata_where_clause(labels=[], database_name=None) is None


def test_no_labels_and_empty_database_name_returns_none() -> None:
    assert _build_metadata_where_clause(labels=None, database_name="") is None


def test_no_filters_returns_none_for_dict_format() -> None:
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


def test_metadata_filter_format_reads_dict_from_injected_vdb() -> None:
    assert _metadata_filter_format(_FakeRetriever(_FakeVdb("dict"))) == "dict"


def test_metadata_filter_format_reads_sql_from_injected_vdb() -> None:
    assert _metadata_filter_format(_FakeRetriever(_FakeVdb("sql"))) == "sql"


def test_metadata_filter_format_defaults_to_sql_when_no_vdb_injected() -> None:
    assert _metadata_filter_format(_FakeRetriever(None)) == "sql"


def test_metadata_filter_format_defaults_to_sql_when_vdb_lacks_attribute() -> None:
    assert _metadata_filter_format(_FakeRetriever(object())) == "sql"


def test_metadata_filter_format_unknown_falls_back_to_sql() -> None:
    assert _metadata_filter_format(_FakeRetriever(_FakeVdb("yaml"))) == "sql"


# ---------------------------------------------------------------------------
# search_semantic_index — integration with retriever.query(vdb_kwargs=...)
# ---------------------------------------------------------------------------


from nemo_retriever.tabular_data.retrieval.data_access.semantic_search import (
    DEFAULT_FETCH_LIMIT,
    search_semantic_index,
)


class _RecordingRetriever:
    """Fake retriever that records every ``query`` invocation and returns no hits."""

    def __init__(self, vdb: object | None) -> None:
        self.vdb_kwargs = {"vdb": vdb} if vdb is not None else {}
        self.calls: list[dict] = []

    def query(self, entity: str, *, top_k: int, vdb_kwargs: dict | None) -> list[dict]:
        self.calls.append({"entity": entity, "top_k": top_k, "vdb_kwargs": vdb_kwargs})
        return []


def test_search_semantic_index_forwards_sql_where_per_label() -> None:
    retriever = _RecordingRetriever(_FakeVdb("sql"))
    search_semantic_index(retriever, "rev", label_filter=["Column"], database_name="dor_prod")

    assert len(retriever.calls) == 1
    sent = retriever.calls[0]["vdb_kwargs"]
    assert isinstance(sent, dict)
    assert set(sent.keys()) == {"where"}
    assert isinstance(sent["where"], str)
    assert """metadata LIKE '%"label":"Column"%'""" in sent["where"]
    assert """metadata LIKE '%"database_name":"dor\\_prod"%'""" in sent["where"]


def test_search_semantic_index_forwards_dict_filter_per_label() -> None:
    retriever = _RecordingRetriever(_FakeVdb("dict"))
    search_semantic_index(retriever, "rev", label_filter=["Column"], database_name="dor_prod")

    assert len(retriever.calls) == 1
    sent = retriever.calls[0]["vdb_kwargs"]
    assert sent == {"where": {"label": "Column", "database_name": "dor_prod"}}


def test_search_semantic_index_dict_runs_one_query_per_label() -> None:
    retriever = _RecordingRetriever(_FakeVdb("dict"))
    search_semantic_index(retriever, "rev", label_filter=["Column", "Table"])

    assert len(retriever.calls) == 2
    sent = [c["vdb_kwargs"] for c in retriever.calls]
    assert {"where": {"label": "Column"}} in sent
    assert {"where": {"label": "Table"}} in sent


def test_search_semantic_index_no_filter_passes_none_and_default_limit() -> None:
    retriever = _RecordingRetriever(_FakeVdb("dict"))
    search_semantic_index(retriever, "rev")

    assert len(retriever.calls) == 1
    assert retriever.calls[0]["vdb_kwargs"] is None
    assert retriever.calls[0]["top_k"] == DEFAULT_FETCH_LIMIT


def test_search_semantic_index_defaults_to_sql_when_vdb_missing() -> None:
    retriever = _RecordingRetriever(None)
    search_semantic_index(retriever, "rev", label_filter=["Column"])

    sent = retriever.calls[0]["vdb_kwargs"]
    assert isinstance(sent["where"], str)
    assert """metadata LIKE '%"label":"Column"%'""" in sent["where"]
