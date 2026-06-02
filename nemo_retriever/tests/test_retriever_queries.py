# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the graph-based :class:`~nemo_retriever.retriever.Retriever` query surface."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_retriever.retriever import Retriever, _shape_query_hits


def _make_hits(n: int, base_score: float = 0.5) -> list[dict[str, Any]]:
    return [
        {
            "text": f"passage {i}",
            "metadata": "{}",
            "source": "doc.pdf",
            "page_number": i,
            "_distance": base_score + i * 0.01,
        }
        for i in range(n)
    ]


def _make_retriever(**overrides: Any) -> Retriever:
    defaults: dict[str, Any] = {
        "rerank": False,
        "top_k": 5,
        "vdb_kwargs": {"vdb_op": "lancedb", "vdb_kwargs": {"uri": "/tmp/r", "table_name": "t"}},
        "embed_kwargs": {"model_name": "embedder", "embed_model_name": "embedder"},
    }
    defaults.update(overrides)
    return Retriever(**defaults)


def _install_mock_graph(monkeypatch: pytest.MonkeyPatch, hits: list[list[dict[str, Any]]]) -> MagicMock:
    """Avoid constructing real LanceDB / embed operators."""
    resolved = MagicMock()
    # :meth:`Graph.execute` returns one entry per graph leaf; retrieval output is ``list[list[dict]]``.
    resolved.execute.return_value = [hits]

    graph = MagicMock()
    graph.resolve_for_local_execution.return_value = resolved

    monkeypatch.setattr(Retriever, "_build_default_graph", lambda self: graph)

    # bypass instance cache from other tests
    def fresh_get(self: Retriever, *, embed_extra: Any = None) -> MagicMock:
        graph.resolve_for_local_execution.return_value = resolved
        return graph

    monkeypatch.setattr(Retriever, "_get_graph", fresh_get)
    return resolved


class TestQueriesGraphExecution:
    def test_empty_queries_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_get = MagicMock()
        monkeypatch.setattr(Retriever, "_get_graph", mock_get)
        assert _make_retriever().queries([]) == []
        mock_get.assert_not_called()

    def test_queries_thread_top_k_and_vdb_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hit = [[{"text": "retrieved", "source": "doc.pdf", "page_number": 1}]]
        resolved = _install_mock_graph(monkeypatch, hit)
        retriever = _make_retriever(top_k=11)
        out = retriever.queries(["q"], vdb_kwargs={"where": "x"})
        assert out == hit
        resolved.execute.assert_called_once()
        _args, kw = resolved.execute.call_args
        assert kw["top_k"] == 11
        assert kw["query_texts"] == ["q"]
        assert kw["where"] == "x"
        df = _args[0]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["text"]
        assert df["text"].tolist() == ["q"]

    def test_merge_embed_params_per_call_overrides(self) -> None:
        r = _make_retriever(embed_kwargs={"model_name": "base", "embed_model_name": "base"})
        p = r._merge_embed_params({"model_name": "call"})
        assert p.model_name == "call"

    def test_rerank_inflates_retrieval_top_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resolved = _install_mock_graph(monkeypatch, [[{"text": "x"}]])
        retriever = _make_retriever(top_k=3, rerank=True, rerank_kwargs={"refine_factor": 4})
        retriever._cached_graph = None
        retriever._cache_key = None
        retriever.queries(["q"])
        assert resolved.execute.call_args.kwargs["top_k"] == 12

    def test_query_delegates_to_queries(self) -> None:
        retriever = _make_retriever()
        expected = _make_hits(2)
        with patch.object(retriever, "queries", return_value=[expected]) as mock_q:
            result = retriever.query("find", top_k=4, vdb_kwargs={"uri": "x"})
        mock_q.assert_called_once_with(
            ["find"],
            top_k=4,
            candidate_k=None,
            page_dedup=False,
            content_types=None,
            vdb_kwargs={"uri": "x"},
            embed_kwargs=None,
        )
        assert result is expected

    def test_candidate_k_widens_retrieval_before_final_truncation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hits = [[{"text": f"hit {i}", "page_number": i} for i in range(5)]]
        resolved = _install_mock_graph(monkeypatch, hits)
        out = _make_retriever(top_k=2).queries(["q"], candidate_k=5)

        assert [hit["text"] for hit in out[0]] == ["hit 0", "hit 1"]
        assert resolved.execute.call_args.kwargs["top_k"] == 5

    def test_candidate_k_must_cover_top_k(self) -> None:
        with pytest.raises(ValueError, match=r"candidate_k \(2\).*top_k \(5\)"):
            _make_retriever(top_k=5).queries(["q"], candidate_k=2)


class TestRetrieverDefaults:
    def test_default_top_k(self) -> None:
        assert Retriever().top_k == 10

    def test_rerank_disabled_by_default(self) -> None:
        assert Retriever().rerank is False

    def test_retriever_alias_is_class(self) -> None:
        from nemo_retriever.retriever import retriever

        assert retriever is Retriever


class TestQueryHitShaping:
    def test_page_dedup_preserves_first_page_hit(self) -> None:
        hits = [
            {"text": "first p1", "pdf_basename": "handbook", "page_number": 1},
            {"text": "second p1", "pdf_basename": "handbook", "page_number": 1},
            {"text": "first p2", "pdf_basename": "handbook", "page_number": 2},
        ]

        out = _shape_query_hits(hits, top_k=10, page_dedup=True)

        assert [h["text"] for h in out] == ["first p1", "first p2"]

    def test_page_dedup_does_not_strip_dotted_basename(self) -> None:
        hits = [
            {"text": "v1", "pdf_basename": "report.v1", "page_number": 1},
            {"text": "v2", "pdf_basename": "report.v2", "page_number": 1},
        ]

        out = _shape_query_hits(hits, top_k=10, page_dedup=True)

        assert [h["text"] for h in out] == ["v1", "v2"]

    def test_page_dedup_fallback_keeps_distinct_source_paths(self) -> None:
        hits = [
            {"text": "dir a", "source_id": "/dir_a/report.pdf", "page_number": 1},
            {"text": "dir b", "source_id": "/dir_b/report.pdf", "page_number": 1},
        ]

        out = _shape_query_hits(hits, top_k=10, page_dedup=True)

        assert [h["text"] for h in out] == ["dir a", "dir b"]

    def test_filters_content_types_from_dict_and_json_metadata(self) -> None:
        hits = [
            {"text": "text row", "metadata": {"type": "text"}, "page_number": 1},
            {"text": "table row", "metadata": '{"type": "table"}', "page_number": 2},
            {"text": "chart row", "metadata": {"type": "chart"}, "page_number": 3},
        ]

        out = _shape_query_hits(hits, top_k=10, content_types="text,table")

        assert [h["text"] for h in out] == ["text row", "table row"]

    def test_content_type_filter_accepts_emitted_images_alias(self) -> None:
        hits = [
            {"text": "image row", "metadata": {"type": "images"}, "page_number": 1},
            {"text": "table row", "metadata": {"type": "table"}, "page_number": 2},
            {"text": "chart row", "metadata": {"type": "chart"}, "page_number": 3},
        ]

        out = _shape_query_hits(hits, top_k=10, content_types="image")

        assert [h["text"] for h in out] == ["image row"]

    def test_uses_top_level_content_type_fallback(self) -> None:
        hits = [
            {"text": "top-level table", "metadata": "{}", "content_type": "table", "page_number": 1},
            {"text": "top-level chart", "metadata": "{}", "content_type": "chart", "page_number": 2},
        ]

        out = _shape_query_hits(hits, top_k=10, content_types="table")

        assert [h["text"] for h in out] == ["top-level table"]

    def test_uses_metadata_content_type_fallback(self) -> None:
        hits = [
            {"text": "metadata table", "metadata": {"_content_type": "table"}, "page_number": 1},
            {"text": "metadata image", "metadata": {"_content_type": "image"}, "page_number": 2},
        ]

        out = _shape_query_hits(hits, top_k=10, content_types="table")

        assert [h["text"] for h in out] == ["metadata table"]

    def test_does_not_backfill_excluded_content_types(self) -> None:
        hits = [
            {"text": "text row", "metadata": {"type": "text"}, "page_number": 1},
            {"text": "chart row 1", "metadata": {"type": "chart"}, "page_number": 2},
            {"text": "chart row 2", "metadata": {"type": "chart"}, "page_number": 3},
        ]

        out = _shape_query_hits(hits, top_k=3, content_types="text")

        assert [h["text"] for h in out] == ["text row"]

    def test_content_type_filter_excludes_untyped_hits(self) -> None:
        hits = [
            {"text": "legacy row", "metadata": {}, "page_number": 1},
            {"text": "text row", "metadata": {"type": "text"}, "page_number": 2},
        ]

        out = _shape_query_hits(hits, top_k=10, content_types="text")

        assert [h["text"] for h in out] == ["text row"]

    def test_empty_content_type_allowlist_uses_python_api_name(self) -> None:
        with pytest.raises(ValueError, match="content_types must include"):
            _shape_query_hits([], top_k=3, content_types=[])


class TestRunModeServiceRequiresHttpEmbed:
    def test_service_mode_errors_without_url(self) -> None:
        with pytest.raises(ValueError, match="run_mode='service'"):
            Retriever(run_mode="service", embed_kwargs={})._merge_embed_params()


class TestRetrieveVdbOperatorPreprocess:
    def test_dataframe_to_vectors(self) -> None:
        from nemo_retriever.vdb.operators import RetrieveVdbOperator

        df = pd.DataFrame(
            {
                "text": ["a"],
                "metadata": [{"embedding": [0.1, 0.2]}],
            }
        )
        op = RetrieveVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp", "table_name": "t"})
        vec = op.preprocess(df)
        assert vec == [[0.1, 0.2]]

    def test_dataframe_to_vectors_reads_payload_embedding_column(self) -> None:
        from nemo_retriever.vdb.operators import RetrieveVdbOperator

        df = pd.DataFrame(
            {
                "text": ["a"],
                "text_embeddings_1b_v2": [{"embedding": [0.3, 0.4]}],
            }
        )
        op = RetrieveVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp", "table_name": "t"})
        vec = op.preprocess(df)
        assert vec == [[0.3, 0.4]]

    def test_dataframe_to_vectors_reads_direct_embedding_column(self) -> None:
        from nemo_retriever.vdb.operators import RetrieveVdbOperator

        df = pd.DataFrame(
            {
                "text": ["a"],
                "text_embeddings_1b_v2": [[0.5, 0.6]],
            }
        )
        op = RetrieveVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp", "table_name": "t"})
        vec = op.preprocess(df)
        assert vec == [[0.5, 0.6]]

    def test_dataframe_to_vectors_skips_non_numeric_list_columns(self) -> None:
        from nemo_retriever.vdb.operators import RetrieveVdbOperator

        df = pd.DataFrame(
            {
                "text": ["a"],
                "tags": [["finance", "annual"]],
                "text_embeddings_1b_v2": [[0.7, 0.8]],
            }
        )
        op = RetrieveVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp", "table_name": "t"})
        vec = op.preprocess(df)
        assert vec == [[0.7, 0.8]]

    def test_dataframe_to_vectors_skips_numeric_non_embedding_list_columns(self) -> None:
        from nemo_retriever.vdb.operators import RetrieveVdbOperator

        df = pd.DataFrame(
            {
                "text": ["a"],
                "page_scores": [[9.9, 8.8]],
                "text_embeddings_1b_v2": [[0.7, 0.8]],
            }
        )
        op = RetrieveVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp", "table_name": "t"})
        vec = op.preprocess(df)
        assert vec == [[0.7, 0.8]]


class TestRerankLongDataframe:
    def test_groups_by_query_order(self) -> None:
        from nemo_retriever.retriever_graph_utils import rerank_long_dataframe_to_hits

        df = pd.DataFrame(
            [
                {"query": "q1", "text": "b", "_hit": {"text": "b"}, "rerank_score": 0.5},
                {"query": "q1", "text": "a", "_hit": {"text": "a"}, "rerank_score": 0.9},
                {"query": "q2", "text": "c", "_hit": {"text": "c"}, "rerank_score": 0.3},
            ]
        )
        out = rerank_long_dataframe_to_hits(df, query_texts=["q1", "q2"], top_k=1, score_column="rerank_score")
        assert len(out) == 2
        assert out[0][0]["text"] == "a"
        assert out[0][0]["_rerank_score"] == 0.9
        assert out[1][0]["text"] == "c"
