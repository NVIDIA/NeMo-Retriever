# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the VDB-backed Retriever query surface."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


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


def _make_retriever(**overrides: Any):
    from nemo_retriever.retriever import Retriever

    defaults = dict(
        reranker=None,
        top_k=5,
        vdb_op="fake",
        vdb_kwargs={"collection_name": "docs", "model_name": "embedder"},
    )
    defaults.update(overrides)
    return Retriever(**defaults)


class _FakeRetrieveVdbOperator:
    instances: list["_FakeRetrieveVdbOperator"] = []
    next_result: list[list[dict[str, Any]]] = [[{"text": "retrieved", "source": "doc.pdf", "page_number": 1}]]

    def __init__(self, **kwargs: Any) -> None:
        self.constructor_kwargs = kwargs
        self.process_calls: list[tuple[Any, dict[str, Any]]] = []
        self.__class__.instances.append(self)

    def process(self, data: Any, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.process_calls.append((data, kwargs))
        return self.__class__.next_result


@pytest.fixture(autouse=True)
def _reset_fake_operator() -> None:
    _FakeRetrieveVdbOperator.instances = []
    _FakeRetrieveVdbOperator.next_result = [[{"text": "retrieved", "source": "doc.pdf", "page_number": 1}]]


class TestQueriesVdbDelegation:
    def test_empty_queries_returns_empty_without_operator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_retriever.vdb as vdb_pkg

        monkeypatch.setattr(vdb_pkg, "RetrieveVdbOperator", MagicMock())
        assert _make_retriever().queries([]) == []
        vdb_pkg.RetrieveVdbOperator.assert_not_called()

    def test_queries_delegate_raw_strings_to_vdb_operator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_retriever.vdb as vdb_pkg

        monkeypatch.setattr(vdb_pkg, "RetrieveVdbOperator", _FakeRetrieveVdbOperator)
        retriever = _make_retriever(vdb_kwargs={"collection_name": "docs", "milvus_uri": "http://milvus"})

        result = retriever.queries(["q0", 123], top_k=7, vdb_kwargs={"_filter": "content_type == 'text'"})

        assert result == _FakeRetrieveVdbOperator.next_result
        operator = _FakeRetrieveVdbOperator.instances[0]
        expected_kwargs = {
            "collection_name": "docs",
            "milvus_uri": "http://milvus",
            "_filter": "content_type == 'text'",
            "top_k": 7,
        }
        assert operator.constructor_kwargs == {
            "vdb_op": "fake",
            "vdb_kwargs": {"collection_name": "docs", "milvus_uri": "http://milvus"},
        }
        assert operator.process_calls == [(["q0", "123"], expected_kwargs)]

    def test_queries_use_instance_top_k_when_not_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_retriever.vdb as vdb_pkg

        monkeypatch.setattr(vdb_pkg, "RetrieveVdbOperator", _FakeRetrieveVdbOperator)
        _make_retriever(top_k=11).queries(["q"])

        operator = _FakeRetrieveVdbOperator.instances[0]
        assert operator.process_calls[0][1]["top_k"] == 11

    def test_queries_accept_prebuilt_vdb(self) -> None:
        from nemo_retriever.retriever import Retriever

        class FakeVDB:
            def __init__(self) -> None:
                self.calls: list[tuple[Any, dict[str, Any]]] = []

            def retrieval(self, queries: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
                self.calls.append((queries, kwargs))
                return [
                    [
                        {
                            "text": "direct hit",
                            "source": "doc-a.pdf",
                            "content_metadata": {"page_number": 2},
                        }
                    ]
                ]

        vdb = FakeVDB()
        retriever = Retriever(
            vdb=vdb,
            vdb_kwargs={"collection_name": "docs", "model_name": "embedder"},
            top_k=4,
        )

        result = retriever.queries(["q"], vdb_kwargs={"_filter": "content_type == 'text'"})

        assert vdb.calls == [
            (
                ["q"],
                {
                    "collection_name": "docs",
                    "model_name": "embedder",
                    "_filter": "content_type == 'text'",
                    "top_k": 4,
                },
            )
        ]
        assert result[0][0]["text"] == "direct hit"
        assert result[0][0]["pdf_page"] == "doc-a_2"

    def test_queries_embed_then_use_vector_search_for_known_vdb_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_retriever.vdb as vdb_pkg
        from nemo_retriever.retriever import Retriever

        captured: dict[str, Any] = {}

        def fake_search_vdb_with_vectors(**kwargs: Any) -> list[list[dict[str, Any]]]:
            captured.update(kwargs)
            return [
                [
                    {
                        "entity": {
                            "text": "local hit",
                            "source": {"source_id": "doc-local.pdf"},
                            "content_metadata": {"page_number": 3},
                        }
                    }
                ]
            ]

        def fake_supports_vector_search_vdb(vdb_op: str) -> bool:
            captured["vdb_op"] = vdb_op
            return True

        monkeypatch.setattr(vdb_pkg, "supports_vector_search_vdb", fake_supports_vector_search_vdb)
        monkeypatch.setattr(vdb_pkg, "search_vdb_with_vectors", fake_search_vdb_with_vectors)

        retriever = Retriever(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp/lancedb", "model_name": "hf-embedder"}, top_k=6)
        with patch.object(retriever, "_embed_queries_local_hf", return_value=[[0.1, 0.2]]) as mock_embed:
            result = retriever.queries(["q"], vdb_kwargs={"table_name": "nv-ingest"})

        assert captured["vdb_op"] == "lancedb"
        mock_embed.assert_called_once_with(["q"], model_name="hf-embedder")
        assert captured["query_vectors"] == [[0.1, 0.2]]
        assert captured["query_texts"] == ["q"]
        assert captured["top_k"] == 6
        assert captured["vdb_kwargs"] == {
            "uri": "/tmp/lancedb",
            "model_name": "hf-embedder",
            "table_name": "nv-ingest",
            "top_k": 6,
        }
        assert result[0][0]["text"] == "local hit"
        assert result[0][0]["pdf_page"] == "doc-local_3"

    def test_reranker_requests_fanout_and_reranks_to_requested_top_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_retriever.vdb as vdb_pkg

        monkeypatch.setattr(vdb_pkg, "RetrieveVdbOperator", _FakeRetrieveVdbOperator)
        initial = [_make_hits(12)]
        reranked = [_make_hits(3)]
        _FakeRetrieveVdbOperator.next_result = initial
        retriever = _make_retriever(
            top_k=3,
            reranker="nvidia/llama-nemotron-rerank-1b-v2",
            reranker_refine_factor=4,
        )

        with patch.object(retriever, "_rerank_results", return_value=reranked) as mock_rerank:
            result = retriever.queries(["q"])

        operator = _FakeRetrieveVdbOperator.instances[0]
        assert operator.process_calls[0][1]["top_k"] == 12
        mock_rerank.assert_called_once_with(["q"], initial, top_k=3)
        assert result is reranked


class TestQuerySingleConvenience:
    def test_query_delegates_to_queries_and_returns_first_element(self) -> None:
        retriever = _make_retriever()
        expected = _make_hits(5)
        with patch.object(retriever, "queries", return_value=[expected]) as mock_queries:
            result = retriever.query("find something", top_k=4, vdb_kwargs={"collection_name": "docs"})

        mock_queries.assert_called_once_with(["find something"], top_k=4, vdb_kwargs={"collection_name": "docs"})
        assert result is expected


class TestQueriesWithEndpointReranking:
    def test_reranked_results_are_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nemo_retriever.vdb as vdb_pkg

        monkeypatch.setattr(vdb_pkg, "RetrieveVdbOperator", _FakeRetrieveVdbOperator)
        initial = [_make_hits(8)]
        reranked = [_make_hits(2)]
        _FakeRetrieveVdbOperator.next_result = initial
        retriever = _make_retriever(
            reranker="nvidia/llama-nemotron-rerank-1b-v2",
            reranker_endpoint="http://rerank.example.com",
            top_k=2,
        )

        with patch.object(retriever, "_rerank_results", return_value=reranked):
            out = retriever.queries(["q"])

        assert out is reranked

    def test_rerank_results_uses_endpoint_not_local_model(self) -> None:
        retriever = _make_retriever(
            reranker="nvidia/llama-nemotron-rerank-1b-v2",
            reranker_endpoint="http://rerank.example.com",
            top_k=3,
        )
        fake_hits = _make_hits(4)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": [{"index": i, "relevance_score": float(len(fake_hits) - i)} for i in range(len(fake_hits))]
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            out = retriever._rerank_results(["q"], [fake_hits], top_k=retriever.top_k)

        mock_post.assert_called()
        scores = [h["_rerank_score"] for h in out[0]]
        assert scores == sorted(scores, reverse=True)


class TestQueriesWithLocalReranking:
    def test_rerank_results_with_local_model(self) -> None:
        retriever = _make_retriever(reranker="nvidia/llama-nemotron-rerank-1b-v2")
        hits = _make_hits(4)
        fake_model = MagicMock()
        fake_model.score.return_value = [0.1, 0.9, 0.5, 0.3]

        with patch.object(retriever, "_get_reranker_model", return_value=fake_model):
            out = retriever._rerank_results(["q"], [hits], top_k=retriever.top_k)

        scores = [h["_rerank_score"] for h in out[0]]
        assert scores == sorted(scores, reverse=True)
        assert max(scores) == 0.9

    def test_rerank_results_respects_top_k(self) -> None:
        retriever = _make_retriever(reranker="nvidia/llama-nemotron-rerank-1b-v2", top_k=2)
        hits = _make_hits(4)
        fake_model = MagicMock()
        fake_model.score.return_value = [0.1, 0.9, 0.5, 0.3]

        with patch.object(retriever, "_get_reranker_model", return_value=fake_model):
            out = retriever._rerank_results(["q"], [hits], top_k=retriever.top_k)

        assert len(out[0]) == 2

    def test_rerank_results_multiple_queries(self) -> None:
        retriever = _make_retriever(reranker="nvidia/llama-nemotron-rerank-1b-v2", top_k=2)
        hits_a = _make_hits(2)
        hits_b = _make_hits(2)
        fake_model = MagicMock()
        fake_model.score.side_effect = [[0.2, 0.8], [0.6, 0.4]]

        with patch.object(retriever, "_get_reranker_model", return_value=fake_model):
            out = retriever._rerank_results(["q1", "q2"], [hits_a, hits_b], top_k=retriever.top_k)

        assert len(out) == 2
        for per_query in out:
            scores = [h["_rerank_score"] for h in per_query]
            assert scores == sorted(scores, reverse=True)


class TestRetrieverDefaults:
    def test_default_vdb_is_lancedb(self) -> None:
        from nemo_retriever.retriever import Retriever

        retriever = Retriever()
        assert retriever.vdb is None
        assert retriever.vdb_op == "lancedb"
        assert retriever.vdb_kwargs == {}

    def test_default_reranker_is_nemotron_model(self) -> None:
        from nemo_retriever.retriever import Retriever

        retriever = Retriever()
        assert retriever.reranker_model_name == "nvidia/llama-nemotron-rerank-vl-1b-v2"

    def test_reranker_can_be_disabled(self) -> None:
        retriever = _make_retriever(reranker=None)
        assert retriever.reranker is None

    def test_reranker_model_not_initialized_at_construction(self) -> None:
        from nemo_retriever.retriever import Retriever

        retriever = Retriever()
        assert retriever._reranker_model is None

    def test_retriever_alias_is_retriever_class(self) -> None:
        from nemo_retriever.retriever import Retriever, retriever

        assert retriever is Retriever
