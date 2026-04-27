# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Retriever.queries() and Retriever.query().

All external I/O (LanceDB, embedders, requests) is mocked so the tests run
without any GPU, network, or database dependency.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBED_DIM = 4
_DUMMY_VECTOR = [0.1, 0.2, 0.3, 0.4]


def _make_hits(n: int, base_score: float = 0.5) -> list[dict]:
    return [
        {
            "text": f"passage {i}",
            "metadata": "{}",
            "source": "{}",
            "page_number": i,
            "_distance": base_score + i * 0.01,
        }
        for i in range(n)
    ]


def _make_retriever(**overrides):
    """Return a Retriever with reranker disabled by default and sane test values."""
    from nemo_retriever.retriever import Retriever

    defaults = dict(
        reranker=None,
        top_k=5,
        nprobes=16,
    )
    defaults.update(overrides)
    return Retriever(**defaults)


# ---------------------------------------------------------------------------
# Retriever._resolve_embedding_endpoint
# ---------------------------------------------------------------------------


class TestResolveEmbeddingEndpoint:
    def test_returns_none_when_no_endpoints_set(self):
        r = _make_retriever()
        assert r._resolve_embedding_endpoint() is None

    def test_http_endpoint_takes_priority(self):
        r = _make_retriever(
            embedding_http_endpoint="http://embed.example.com",
            embedding_endpoint="http://other.example.com",
        )
        assert r._resolve_embedding_endpoint() == "http://embed.example.com"

    def test_single_endpoint_returned_when_http(self):
        r = _make_retriever(embedding_endpoint="http://embed.example.com")
        assert r._resolve_embedding_endpoint() == "http://embed.example.com"

    def test_grpc_endpoint_raises(self):
        r = _make_retriever(embedding_endpoint="grpc://embed.example.com")
        with pytest.raises(ValueError, match="gRPC"):
            r._resolve_embedding_endpoint()

    def test_whitespace_only_endpoint_treated_as_none(self):
        r = _make_retriever(embedding_http_endpoint="   ")
        assert r._resolve_embedding_endpoint() is None


# ---------------------------------------------------------------------------
# Retriever.queries() — basic (no reranking)
# ---------------------------------------------------------------------------


class TestQueriesNoReranking:
    def _run_queries(self, retriever, query_texts, fake_vectors, fake_hits):
        """Patch embed + search helpers and call queries()."""
        with (
            patch.object(retriever, "_embed_queries_local", return_value=fake_vectors),
            patch.object(retriever, "_search_lancedb", return_value=fake_hits),
        ):
            return retriever.queries(query_texts)

    def test_empty_queries_returns_empty(self):
        r = _make_retriever()
        assert r.queries([]) == []

    def test_single_query_returns_one_result_list(self):
        r = _make_retriever()
        hits = [_make_hits(5)]
        result = self._run_queries(r, ["What is ML?"], [_DUMMY_VECTOR], hits)
        assert len(result) == 1
        assert result[0] is hits[0]

    def test_multiple_queries_return_matching_result_count(self):
        r = _make_retriever()
        n_queries = 3
        fake_hits = [_make_hits(5)] * n_queries
        result = self._run_queries(
            r,
            [f"query {i}" for i in range(n_queries)],
            [_DUMMY_VECTOR] * n_queries,
            fake_hits,
        )
        assert len(result) == n_queries

    def test_embed_local_hf_called_with_query_texts(self):
        r = _make_retriever()
        with (
            patch.object(r, "_embed_queries_local", return_value=[_DUMMY_VECTOR]) as mock_embed,
            patch.object(r, "_search_lancedb", return_value=[_make_hits(5)]),
        ):
            r.queries(["hello world"])

        mock_embed.assert_called_once_with(["hello world"], model_name=r.embedder)

    def test_embed_nim_called_when_endpoint_set(self):
        r = _make_retriever(embedding_http_endpoint="http://nim.example.com")
        with (
            patch.object(r, "_embed_queries_nim", return_value=[_DUMMY_VECTOR]) as mock_nim,
            patch.object(r, "_search_lancedb", return_value=[_make_hits(5)]),
        ):
            r.queries(["hello"])

        mock_nim.assert_called_once()
        call_kwargs = mock_nim.call_args[1]
        assert call_kwargs["endpoint"] == "http://nim.example.com"

    def test_search_lancedb_receives_vectors_and_texts(self):
        r = _make_retriever()
        vecs = [[0.1, 0.2, 0.3, 0.4]]
        with (
            patch.object(r, "_embed_queries_local", return_value=vecs),
            patch.object(r, "_search_lancedb", return_value=[_make_hits(5)]) as mock_search,
        ):
            r.queries(["my query"])

        kwargs = mock_search.call_args[1]
        assert kwargs["query_vectors"] == vecs
        assert kwargs["query_texts"] == ["my query"]

    def test_embedder_override_forwarded(self):
        r = _make_retriever()
        with (
            patch.object(r, "_embed_queries_local", return_value=[_DUMMY_VECTOR]) as mock_embed,
            patch.object(r, "_search_lancedb", return_value=[_make_hits(5)]),
        ):
            r.queries(["q"], embedder="custom/embedder")

        assert mock_embed.call_args[1]["model_name"] == "custom/embedder"

    def test_lancedb_uri_and_table_overrides_forwarded(self):
        r = _make_retriever()
        with (
            patch.object(r, "_embed_queries_local", return_value=[_DUMMY_VECTOR]),
            patch.object(r, "_search_lancedb", return_value=[_make_hits(5)]) as mock_search,
        ):
            r.queries(["q"], lancedb_uri="/tmp/db", lancedb_table="my-table")

        kwargs = mock_search.call_args[1]
        assert kwargs["lancedb_uri"] == "/tmp/db"
        assert kwargs["lancedb_table"] == "my-table"


# ---------------------------------------------------------------------------
# local_query_embed_backend (HF vs vLLM for local query vectors)
# ---------------------------------------------------------------------------


class TestLocalQueryEmbedBackend:
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="local_query_embed_backend"):
            _make_retriever(local_query_embed_backend="nope")

    def test_hf_text_model_uses_create_local_query_embedder(self, monkeypatch):
        recorded: list[tuple[str, str]] = []

        def fake_cq(_model_name: str, *, backend: str = "auto", **_kwargs):
            recorded.append(("cq", backend))
            m = MagicMock()
            import torch

            m.embed_queries.return_value = torch.zeros(1, 2)
            return m

        def fake_ce(*_a, **_k):
            recorded.append(("ce", "vllm"))
            raise AssertionError("create_local_embedder should not run for hf text backend")

        monkeypatch.setattr("nemo_retriever.model.create_local_query_embedder", fake_cq)
        monkeypatch.setattr("nemo_retriever.model.create_local_embedder", fake_ce)

        r = _make_retriever(local_query_embed_backend="hf")
        r._get_local_embedder("nvidia/llama-nemotron-embed-1b-v2")
        r._get_local_embedder("nvidia/llama-nemotron-embed-1b-v2")
        assert recorded == [("cq", "hf")]


# ---------------------------------------------------------------------------
# Retriever.query() — single-query convenience wrapper
# ---------------------------------------------------------------------------


class TestQuerySingleConvenience:
    def test_query_delegates_to_queries_and_returns_first_element(self):
        r = _make_retriever()
        expected = _make_hits(5)
        with patch.object(r, "queries", return_value=[expected]) as mock_queries:
            result = r.query("find something")

        mock_queries.assert_called_once_with(
            ["find something"],
            top_k=None,
            embedder=None,
            lancedb_uri=None,
            lancedb_table=None,
        )
        assert result is expected

    def test_query_passes_through_overrides(self):
        r = _make_retriever()
        with patch.object(r, "queries", return_value=[[]]) as mock_queries:
            r.query("q", embedder="e", lancedb_uri="u", lancedb_table="t")

        mock_queries.assert_called_once_with(["q"], top_k=None, embedder="e", lancedb_uri="u", lancedb_table="t")


# ---------------------------------------------------------------------------
# Retriever.queries() — with reranking via remote endpoint
# ---------------------------------------------------------------------------


class TestQueriesWithEndpointReranking:
    def _retriever_with_endpoint(self, top_k: int = 3, refine: int = 2) -> object:
        return _make_retriever(
            reranker="nvidia/llama-nemotron-rerank-1b-v2",
            reranker_endpoint="http://rerank.example.com",
            top_k=top_k,
            reranker_refine_factor=refine,
        )

    def _fake_search_results(self, retriever) -> list[list[dict]]:
        """Return the number of hits that satisfies the assertion check."""
        n = retriever.top_k * retriever.reranker_refine_factor
        return [_make_hits(n)]

    def test_rerank_results_called_when_reranker_set(self):
        r = self._retriever_with_endpoint()
        fake_results = self._fake_search_results(r)

        with (
            patch.object(r, "_embed_queries_local", return_value=[_DUMMY_VECTOR]),
            patch.object(r, "_search_lancedb", return_value=fake_results),
            patch.object(r, "_rerank_results", return_value=[_make_hits(3)]) as mock_rerank,
        ):
            r.queries(["q"])

        mock_rerank.assert_called_once_with(["q"], fake_results, top_k=r.top_k)

    def test_rerank_not_called_when_reranker_is_none(self):
        r = _make_retriever(reranker=None)
        fake_results = [_make_hits(5)]

        with (
            patch.object(r, "_embed_queries_local", return_value=[_DUMMY_VECTOR]),
            patch.object(r, "_search_lancedb", return_value=fake_results),
            patch.object(r, "_rerank_results") as mock_rerank,
        ):
            r.queries(["q"])

        mock_rerank.assert_not_called()

    def test_reranked_results_are_returned(self):
        r = self._retriever_with_endpoint()
        fake_results = self._fake_search_results(r)
        reranked = [_make_hits(3)]

        with (
            patch.object(r, "_embed_queries_local", return_value=[_DUMMY_VECTOR]),
            patch.object(r, "_search_lancedb", return_value=fake_results),
            patch.object(r, "_rerank_results", return_value=reranked),
        ):
            out = r.queries(["q"])

        assert out is reranked

    def test_rerank_results_uses_endpoint_not_local_model(self):
        r = self._retriever_with_endpoint()
        fake_hits = self._fake_search_results(r)[0]

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        # Return relevance scores in reverse original order
        mock_resp.json.return_value = {
            "results": [{"index": i, "relevance_score": float(len(fake_hits) - i)} for i in range(len(fake_hits))]
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            out = r._rerank_results(["q"], [fake_hits], top_k=r.top_k)

        mock_post.assert_called()
        # Results should be sorted descending
        scores = [h["_rerank_score"] for h in out[0]]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Retriever.queries() — with local reranking model
# ---------------------------------------------------------------------------


class TestQueriesWithLocalReranking:

    def test_rerank_results_with_local_model(self):
        r = _make_retriever(reranker="nvidia/llama-nemotron-rerank-1b-v2")
        hits = _make_hits(4)
        fake_model = MagicMock()
        fake_model.score.return_value = [0.1, 0.9, 0.5, 0.3]

        with patch.object(r, "_get_reranker_model", return_value=fake_model):
            out = r._rerank_results(["q"], [hits], top_k=r.top_k)

        scores = [h["_rerank_score"] for h in out[0]]
        assert scores == sorted(scores, reverse=True)
        assert max(scores) == 0.9

    def test_rerank_results_respects_top_k(self):
        r = _make_retriever(reranker="nvidia/llama-nemotron-rerank-1b-v2", top_k=2)
        hits = _make_hits(4)
        fake_model = MagicMock()
        fake_model.score.return_value = [0.1, 0.9, 0.5, 0.3]

        with patch.object(r, "_get_reranker_model", return_value=fake_model):
            out = r._rerank_results(["q"], [hits], top_k=r.top_k)

        assert len(out[0]) == 2

    def test_rerank_results_multiple_queries(self):
        r = _make_retriever(reranker="nvidia/llama-nemotron-rerank-1b-v2", top_k=2)
        hits_a = _make_hits(2)
        hits_b = _make_hits(2)
        fake_model = MagicMock()
        fake_model.score.side_effect = [[0.2, 0.8], [0.6, 0.4]]

        with patch.object(r, "_get_reranker_model", return_value=fake_model):
            out = r._rerank_results(["q1", "q2"], [hits_a, hits_b], top_k=r.top_k)

        assert len(out) == 2
        # Each per-query list should be sorted descending
        for per_query in out:
            scores = [h["_rerank_score"] for h in per_query]
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Retriever defaults: reranker field behaviour
# ---------------------------------------------------------------------------


class TestRetrieverDefaults:
    def test_default_reranker_is_nemotron_model(self):
        from nemo_retriever.retriever import Retriever

        r = Retriever()
        assert r.reranker_model_name == "nvidia/llama-nemotron-rerank-vl-1b-v2"

    def test_reranker_can_be_disabled(self):
        r = _make_retriever(reranker=None)
        assert r.reranker is None

    def test_reranker_refine_factor_default(self):
        from nemo_retriever.retriever import Retriever

        r = Retriever()
        assert r.reranker_refine_factor == 4

    def test_reranker_max_length_default(self):
        from nemo_retriever.retriever import Retriever

        r = Retriever()
        assert r.reranker_max_length == 10240

    def test_reranker_model_not_initialized_at_construction(self):
        from nemo_retriever.retriever import Retriever

        r = Retriever()
        # Should be None until first use
        assert r._reranker_model is None

    def test_retriever_alias_is_retriever_class(self):
        from nemo_retriever.retriever import retriever, Retriever

        assert retriever is Retriever


# ---------------------------------------------------------------------------
# Retriever._search_lancedb — post-filter strips unwanted columns
# ---------------------------------------------------------------------------


class TestSearchLancedbKeepKeys:
    """Verify that _search_lancedb strips columns not in _KEEP_KEYS."""

    def _fake_table(self, raw_hits: list[dict]):
        """Return a mock LanceDB table whose search chains return *raw_hits*."""
        chain = MagicMock()
        chain.nprobes.return_value = chain
        chain.refine_factor.return_value = chain
        chain.select.return_value = chain
        chain.limit.return_value = chain
        chain.to_list.return_value = raw_hits
        chain.list_indices.return_value = []

        table = MagicMock()
        table.search.return_value = chain
        table.list_indices.return_value = []
        return table

    def test_extra_keys_stripped_from_dense_results(self):
        from nemo_retriever.retriever import _KEEP_KEYS

        raw = [
            {
                "text": "hello",
                "metadata": "{}",
                "source": "{}",
                "page_number": 0,
                "pdf_page": "doc_0",
                "pdf_basename": "doc",
                "source_id": "doc.pdf",
                "path": "/doc.pdf",
                # These should be stripped:
                "vector": [0.1, 0.2, 0.3],
                "_distance": 0.42,
                "_rowid": 7,
            }
        ]
        r = _make_retriever(hybrid=False)
        mock_db = MagicMock()
        mock_db.open_table.return_value = self._fake_table(raw)

        with patch("lancedb.connect", return_value=mock_db):
            results = r._search_lancedb(
                lancedb_uri="fake",
                lancedb_table="t",
                query_vectors=[_DUMMY_VECTOR],
                query_texts=["q"],
                top_k=r.top_k,
            )

        hit = results[0][0]
        assert set(hit.keys()) <= _KEEP_KEYS
        assert "vector" not in hit
        assert "_distance" not in hit
        assert "_rowid" not in hit
        assert hit["text"] == "hello"

    def test_extra_keys_stripped_from_hybrid_results(self):
        from nemo_retriever.retriever import _KEEP_KEYS

        raw = [
            {
                "text": "hello",
                "metadata": "{}",
                "source": "{}",
                "page_number": 0,
                "pdf_page": "doc_0",
                "pdf_basename": "doc",
                "source_id": "doc.pdf",
                "path": "/doc.pdf",
                # These should be stripped:
                "vector": [0.1, 0.2, 0.3],
                "_score": 1.5,
                "_relevance_score": 0.9,
            }
        ]
        r = _make_retriever(hybrid=True)
        mock_db = MagicMock()
        fake_table = self._fake_table(raw)
        # Hybrid chain also needs .vector(), .text(), .rerank()
        chain = fake_table.search.return_value
        chain.vector.return_value = chain
        chain.text.return_value = chain
        chain.rerank.return_value = chain
        mock_db.open_table.return_value = fake_table

        with (
            patch("lancedb.connect", return_value=mock_db),
            patch("lancedb.rerankers.RRFReranker"),
        ):
            results = r._search_lancedb(
                lancedb_uri="fake",
                lancedb_table="t",
                query_vectors=[_DUMMY_VECTOR],
                query_texts=["q"],
                top_k=r.top_k,
            )

        hit = results[0][0]
        assert set(hit.keys()) <= _KEEP_KEYS
        assert "vector" not in hit
        assert "_score" not in hit
        assert "_relevance_score" not in hit
