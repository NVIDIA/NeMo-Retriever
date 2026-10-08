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
* ``"qdrant"`` — a Qdrant payload filter passed as ``query_filter``.

``search_semantic_index`` picks the shape by reading
``retriever.vdb_kwargs["vdb"].metadata_filter_format`` (defaulting to
``"sql"`` when no VDB instance is injected).
"""

from __future__ import annotations

import pytest

from nemo_retriever.tabular_data.retrieval.data_access.semantic_search import (
    _build_metadata_where_clause,
    _hits_to_semantic_rows,
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


def test_qdrant_filters_address_metadata_payload_keys() -> None:
    assert _build_metadata_where_clause(fmt="qdrant") is None
    assert _build_metadata_where_clause(labels=["Column"], fmt="qdrant") == {
        "must": [{"key": "metadata.label", "match": {"value": "Column"}}]
    }
    assert _build_metadata_where_clause(labels=["Column", "Table"], database_name="dor_prod", fmt="qdrant") == {
        "must": [
            {"key": "metadata.label", "match": {"any": ["Column", "Table"]}},
            {"key": "metadata.database_name", "match": {"value": "dor_prod"}},
        ]
    }
    # Values are data, not query syntax, so nothing needs escaping.
    assert _build_metadata_where_clause(database_name="it's_100%", fmt="qdrant") == {
        "must": [{"key": "metadata.database_name", "match": {"value": "it's_100%"}}]
    }


class _OpRetriever:
    def __init__(self, vdb_op: str) -> None:
        self.vdb_kwargs = {"vdb_op": vdb_op, "vdb_kwargs": {}}


def test_metadata_filter_format_selects_qdrant() -> None:
    assert _metadata_filter_format(_OpRetriever("qdrant")) == "qdrant"
    assert _metadata_filter_format(_OpRetriever(" Qdrant ")) == "qdrant"
    assert _metadata_filter_format(_OpRetriever("lancedb")) == "sql"
    assert _metadata_filter_format(_FakeRetriever(_FakeVdb("qdrant"))) == "qdrant"


def test_qdrant_backend_declares_its_filter_format() -> None:
    pytest.importorskip("qdrant_client")
    from nemo_retriever.common.vdb.qdrant import Qdrant

    assert _metadata_filter_format(_FakeRetriever(Qdrant())) == "qdrant"


def test_search_semantic_index_forwards_qdrant_query_filter() -> None:
    retriever = _RecordingRetriever(_FakeVdb("qdrant"))
    search_semantic_index(retriever, "rev", label_filter=["Column"], database_name="dor_prod", per_label_k=4)

    assert retriever.calls == [
        {
            "entity": "rev",
            "top_k": 4,
            "vdb_kwargs": {
                "query_filter": {
                    "must": [
                        {"key": "metadata.label", "match": {"value": "Column"}},
                        {"key": "metadata.database_name", "match": {"value": "dor_prod"}},
                    ]
                }
            },
        }
    ]


def test_semantic_rows_rank_scores_from_every_backend_lowest_first() -> None:
    def hit(cid: str, **score: float) -> dict:
        return {"text": cid, "metadata": {"id": cid, "label": "Column"}, **score}

    rows = _hits_to_semantic_rows(
        [
            hit("distance", _distance=0.25),
            hit("similarity", _score=0.9),
            hit("hybrid", _relevance_score=0.5),
            hit("unscored"),
        ]
    )
    assert [row["score"] for row in rows] == [0.25, -0.9, -0.5, float("inf")]
    assert [row["id"] for row in sorted(rows, key=lambda row: row["score"])] == [
        "similarity",
        "hybrid",
        "distance",
        "unscored",
    ]


def test_qdrant_filter_selects_tabular_rows_in_a_real_collection(monkeypatch) -> None:
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient

    import nemo_retriever.common.vdb.qdrant as qdrant_module

    server = QdrantClient(location=":memory:")
    monkeypatch.setattr(qdrant_module, "QdrantClient", lambda **_kwargs: server)
    vdb = qdrant_module.Qdrant(collection_name="tabular", vector_dim=2)

    def record(cid: str, label: str, database: str) -> dict:
        return {
            "document_type": "text",
            "metadata": {
                "embedding": [1.0, 0.0],
                "content": cid,
                "content_metadata": {"id": cid, "label": label, "database_name": database},
                "source_metadata": {},
            },
        }

    vdb.run([[record("c1", "Column", "prod"), record("t1", "Table", "prod"), record("c2", "Column", "dev")]])
    query_filter = _build_metadata_where_clause(labels=["Column"], database_name="prod", fmt="qdrant")
    [hits] = vdb.retrieval([[1.0, 0.0]], query_filter=query_filter)
    assert [hit["metadata"]["id"] for hit in hits] == ["c1"]
    either = _build_metadata_where_clause(labels=["Column", "Table"], database_name="prod", fmt="qdrant")
    assert {hit["metadata"]["id"] for hit in vdb.retrieval([[1.0, 0.0]], query_filter=either)[0]} == {"c1", "t1"}
    rows = _hits_to_semantic_rows(hits)
    assert rows[0]["id"] == "c1" and rows[0]["score"] < 0


def test_generate_sql_retrieves_from_the_selected_vdb(monkeypatch) -> None:
    import importlib

    from nemo_retriever.tabular_data.retrieval import generate_sql

    module = importlib.import_module("nemo_retriever.tabular_data.retrieval.generate_sql")
    seen: list[dict] = []

    class _Retriever:
        def __init__(self, *, vdb_kwargs, **kwargs) -> None:
            seen.append(vdb_kwargs)

        def query(self, question):
            return []

    def _no_llm():
        raise ValueError("no LLM configured")

    monkeypatch.setattr(importlib.import_module("nemo_retriever.graph.retriever"), "Retriever", _Retriever)
    monkeypatch.setattr(module, "get_llm_client", _no_llm)
    qdrant = {"vdb_op": "qdrant", "vdb_kwargs": {"collection_name": "tables", "url": "http://q:6333"}}

    assert generate_sql("q", vdb_kwargs=qdrant) == ""
    assert generate_sql("q") == ""
    assert seen == [qdrant, {"vdb_op": "lancedb", "vdb_kwargs": {"table_name": "nemo-retriever-tabular"}}]


def test_retriever_generate_sql_uses_its_own_non_lancedb_index(monkeypatch) -> None:
    import importlib

    from nemo_retriever.graph.retriever import Retriever

    seen: list[object] = []
    module = importlib.import_module("nemo_retriever.tabular_data.retrieval")
    monkeypatch.setattr(module, "generate_sql", lambda query, vdb_kwargs=None: seen.append(vdb_kwargs) or "SELECT 1")

    qdrant = {"vdb_op": "qdrant", "vdb_kwargs": {"collection_name": "tables", "url": "http://q:6333"}}
    assert Retriever(vdb_kwargs=qdrant).generate_sql("q") == "SELECT 1"
    injected = {"vdb": object()}
    Retriever(vdb_kwargs=injected).generate_sql("q")
    Retriever().generate_sql("q")
    Retriever(vdb_kwargs={"uri": "lancedb", "table_name": "docs"}).generate_sql("q")
    Retriever(vdb_kwargs={"vdb_op": "lancedb", "vdb_kwargs": {"table_name": "docs"}}).generate_sql("q")
    assert seen == [qdrant, injected, None, None, None]


def test_caller_owned_graph_uses_its_operator_filter_format() -> None:
    from types import SimpleNamespace

    from nemo_retriever.graph.retriever import Retriever
    from nemo_retriever.operators.vdb import RetrieveVdbOperator
    from nemo_retriever.tabular_data.retrieval.data_access.semantic_search import _metadata_filter_format

    pytest.importorskip("qdrant_client")
    from nemo_retriever.common.vdb.qdrant import Qdrant

    operator = RetrieveVdbOperator(vdb=Qdrant(url="http://qdrant.test", collection_name="tables"))
    graph = SimpleNamespace(roots=[SimpleNamespace(operator=operator, children=[])])
    assert _metadata_filter_format(Retriever(graph=graph)) == "qdrant"
