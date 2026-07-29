# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the agentic retrieval operators.

The ReAct and selection operators delegate all agent logic to the vendored
``nemo_retriever._agentic.nemo_agent`` library, so these tests exercise the operators'
adapter responsibilities — DataFrame translation and the selection gate — by
mocking the ``nemo_agent`` entry points (``Agent.run_sync`` /
``SelectionAgent.select_sync``) rather than the LLM transport.

Run with:
    cd nemo_retriever && uv run pytest tests/test_agentic_operators.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared helpers: canned nemo_agent results
# ---------------------------------------------------------------------------


def _agent_result(*, final_doc_ids=None, retrieval_log=None, error_category=None):
    """Build a canned ``AgentRunResult`` like ``Agent.run``/``SelectionAgent.select``."""
    from nemo_retriever._agentic.nemo_agent.results import AgentError, AgentRunResult

    error = AgentError(category=error_category, message="stub") if error_category else None
    return AgentRunResult(
        final_doc_ids=list(final_doc_ids or []),
        retrieval_log=list(retrieval_log or []),
        error=error,
    )


def _step(docs, query_type="agent"):
    """One retrieval_log entry; ``docs`` is a list of (id, score, text) triples."""
    return {
        "input": {"query": "q", "top_k": len(docs)},
        "tool_name": "retrieve",
        "query_type": query_type,
        "output": [{"id": did, "score": score, "text": text} for did, score, text in docs],
    }


# ---------------------------------------------------------------------------
# RRFAggregatorOperator — pure pandas, no mocking needed
# ---------------------------------------------------------------------------


class TestRRFAggregatorOperator:
    def _make_input(self):
        """Two queries; q1 has doc d1 in both steps, q2 has one step."""
        return pd.DataFrame(
            {
                "query_id": ["q1", "q1", "q1", "q1", "q2", "q2"],
                "query_text": ["inflation"] * 4 + ["vaccines"] * 2,
                "step_idx": [0, 0, 1, 1, 0, 0],
                "doc_id": ["d1", "d2", "d1", "d3", "d4", "d5"],
                "text": ["t1", "t2", "t1", "t3", "t4", "t5"],
                "rank": [1, 2, 1, 2, 1, 2],
            }
        )

    def test_rrf_scores_correct(self):
        from nemo_retriever.operators.graph_ops.rrf_aggregator_operator import RRFAggregatorOperator

        op = RRFAggregatorOperator(k=60)
        result = op.run(self._make_input())

        q1 = result[result["query_id"] == "q1"].set_index("doc_id")
        k = 60
        # d1 appears in step 0 rank 1 and step 1 rank 1
        expected_d1 = 1 / (1 + k) + 1 / (1 + k)
        # d2 appears only in step 0 rank 2
        expected_d2 = 1 / (2 + k)
        assert abs(q1.loc["d1", "rrf_score"] - expected_d1) < 1e-10
        assert abs(q1.loc["d2", "rrf_score"] - expected_d2) < 1e-10

    def test_sorted_descending_per_query(self):
        from nemo_retriever.operators.graph_ops.rrf_aggregator_operator import RRFAggregatorOperator

        op = RRFAggregatorOperator(k=60)
        result = op.run(self._make_input())

        for _, grp in result.groupby("query_id"):
            scores = grp["rrf_score"].tolist()
            assert scores == sorted(scores, reverse=True), "Scores not sorted descending"

    def test_text_carried_through(self):
        from nemo_retriever.operators.graph_ops.rrf_aggregator_operator import RRFAggregatorOperator

        op = RRFAggregatorOperator(k=60)
        result = op.run(self._make_input())
        q1 = result[result["query_id"] == "q1"].set_index("doc_id")
        assert q1.loc["d1", "text"] == "t1"

    def test_output_schema(self):
        from nemo_retriever.operators.graph_ops.rrf_aggregator_operator import RRFAggregatorOperator

        op = RRFAggregatorOperator(k=60)
        result = op.run(self._make_input())
        assert set(result.columns) >= {"query_id", "query_text", "doc_id", "rrf_score", "text"}

    def test_carries_react_final_rank(self):
        from nemo_retriever.operators.graph_ops.rrf_aggregator_operator import RRFAggregatorOperator

        df = self._make_input()
        df["is_final_result"] = [False, False, True, False, False, False]
        op = RRFAggregatorOperator(k=60)
        result = op.run(df)

        q1 = result[result["query_id"] == "q1"].set_index("doc_id")
        assert int(q1.loc["d1", "react_final_rank"]) == 1

    def test_final_result_step_excluded_from_score(self):
        """The synthetic final step must not contribute to the RRF score."""
        from nemo_retriever.operators.graph_ops.rrf_aggregator_operator import RRFAggregatorOperator

        # d1 appears once as a retrieve hit (step 0, rank 1) and once as the
        # final-results selection (step 1, rank 1, is_final_result=True). Only
        # the retrieve hit should score; react_final_rank must still be recorded.
        df = pd.DataFrame(
            {
                "query_id": ["q1", "q1"],
                "query_text": ["q"] * 2,
                "step_idx": [0, 1],
                "doc_id": ["d1", "d1"],
                "text": ["t1", "t1"],
                "rank": [1, 1],
                "is_final_result": [False, True],
            }
        )
        result = op = RRFAggregatorOperator(k=60).run(df)
        row = result[result["doc_id"] == "d1"].iloc[0]
        assert abs(row["rrf_score"] - 1 / (1 + 60)) < 1e-10  # only the retrieve hit
        assert int(row["react_final_rank"]) == 1

    def test_final_only_doc_is_still_emitted(self):
        """A doc that appears only in the final step keeps its react_final_rank."""
        from nemo_retriever.operators.graph_ops.rrf_aggregator_operator import RRFAggregatorOperator

        df = pd.DataFrame(
            {
                "query_id": ["q1", "q1"],
                "query_text": ["q"] * 2,
                "step_idx": [0, 1],
                "doc_id": ["d1", "d2"],  # d2 appears ONLY in the final step
                "text": ["t1", "t2"],
                "rank": [1, 1],
                "is_final_result": [False, True],
            }
        )
        result = RRFAggregatorOperator(k=60).run(df)
        d2 = result[result["doc_id"] == "d2"]
        assert len(d2) == 1
        assert d2.iloc[0]["rrf_score"] == 0.0
        assert int(d2.iloc[0]["react_final_rank"]) == 1

    def test_missing_column_raises(self):
        from nemo_retriever.operators.graph_ops.rrf_aggregator_operator import RRFAggregatorOperator

        op = RRFAggregatorOperator(k=60)
        bad_df = pd.DataFrame({"query_id": ["q1"], "query_text": ["x"]})
        with pytest.raises(ValueError, match="missing required column"):
            op.run(bad_df)


# ---------------------------------------------------------------------------
# ReActAgentOperator — mock nemo_agent.Agent.run_sync
# ---------------------------------------------------------------------------


class TestReActAgentOperator:
    def _op(self, **kwargs):
        from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator

        defaults = dict(llm_model="test-model", retriever_fn=lambda q, k: [], target_top_k=2)
        defaults.update(kwargs)
        return ReActAgentOperator(**defaults)

    def _input(self):
        return pd.DataFrame({"query_id": ["q1"], "query_text": ["What causes inflation?"]})

    def test_retrieve_adapter_renames_and_coerces(self):
        op = self._op(
            retriever_fn=lambda q, k: [
                {"doc_id": "d1", "text": "t", "score": 0.5},
                {"id": "d2", "text": "u", "score": "0.4"},  # already-id + str score
                {"doc_id": "", "text": "skip"},  # empty id dropped
            ]
        )
        out = op._retrieve_adapter("q", 5)
        assert out == [
            {"id": "d1", "score": 0.5, "text": "t"},
            {"id": "d2", "score": 0.4, "text": "u"},
        ]

    def test_translates_retrieval_log_and_final(self):
        from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _agent_result(
            retrieval_log=[
                _step([("d1", 0.9, "A"), ("d2", 0.8, "B")], query_type="main"),
                _step([("d2", 0.7, "B"), ("d3", 0.6, "C")]),
            ],
            final_doc_ids=["d2", "d1"],
        )
        op = self._op()
        with patch.object(ReActAgentOperator, "_ensure_agent", return_value=mock_agent):
            result = op.run(self._input())

        assert set(result.columns) == {
            "query_id", "query_text", "step_idx", "doc_id", "text", "rank",
            "has_valid_final_results", "is_final_result",
        }
        assert result["has_valid_final_results"].all()
        # retrieve steps 0 and 1 present
        assert sorted(result[~result.is_final_result]["step_idx"].unique().tolist()) == [0, 1]
        # synthetic final step carries final_doc_ids in order
        finals = result[result.is_final_result].sort_values("rank")
        assert finals["doc_id"].tolist() == ["d2", "d1"]
        # query_id is bound on the agent run
        assert mock_agent.run_sync.call_args.kwargs["query_id"] == "q1"
        assert mock_agent.run_sync.call_args.kwargs["raw_log_dir"] is None

    def test_empty_final_doc_ids_no_synthetic_step(self):
        from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = _agent_result(
            retrieval_log=[_step([("d1", 0.9, "A")], query_type="main")],
            final_doc_ids=[],
            error_category="max_steps",
        )
        op = self._op()
        with patch.object(ReActAgentOperator, "_ensure_agent", return_value=mock_agent):
            result = op.run(self._input())

        assert not result["has_valid_final_results"].any()
        assert not result["is_final_result"].any()
        assert result["doc_id"].tolist() == ["d1"]  # retrieval log preserved on failure

    def test_empty_input_returns_full_schema(self):
        op = self._op()
        result = op.run(pd.DataFrame({"query_id": [], "query_text": []}))
        assert list(result.columns) == [
            "query_id", "query_text", "step_idx", "doc_id", "text", "rank",
            "has_valid_final_results", "is_final_result",
        ]
        assert result.empty

    def test_multiple_queries_preserve_input_order(self):
        from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator

        mock_agent = MagicMock()

        def run_sync(query, *, query_id=None, raw_log_dir=None):
            return _agent_result(
                retrieval_log=[_step([(f"{query_id}d", 1.0, "x")], query_type="main")],
                final_doc_ids=[],
            )

        mock_agent.run_sync.side_effect = run_sync
        op = self._op(num_concurrent=4)
        df = pd.DataFrame({"query_id": ["qA", "qB", "qC"], "query_text": ["a", "b", "c"]})
        with patch.object(ReActAgentOperator, "_ensure_agent", return_value=mock_agent):
            result = op.run(df)

        # Deterministic input order regardless of thread completion order.
        assert result["query_id"].tolist() == ["qA", "qB", "qC"]
        assert result["doc_id"].tolist() == ["qAd", "qBd", "qCd"]


# ---------------------------------------------------------------------------
# SelectionAgentOperator — mock nemo_agent.SelectionAgent.select_sync
# ---------------------------------------------------------------------------


class TestSelectionAgentOperator:
    def _op(self, **kwargs):
        from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator

        defaults = dict(llm_model="test-model", top_k=2)
        defaults.update(kwargs)
        return SelectionAgentOperator(**defaults)

    def _rrf_frame(self, react_final_rank):
        return pd.DataFrame(
            {
                "query_id": ["q1", "q1", "q1"],
                "query_text": ["What causes inflation?"] * 3,
                "doc_id": ["d1", "d2", "d3"],
                "text": ["doc one", "doc two", "doc three"],
                "rrf_score": [0.9, 0.5, 0.7],
                "react_final_rank": react_final_rank,
            }
        )

    def test_final_results_passthrough(self):
        """Tier 1: a ReAct final list passes through; the selection agent is not run."""
        from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator

        mock_sel = MagicMock()
        op = self._op(top_k=2)
        df = self._rrf_frame(react_final_rank=[2, None, 1])  # d3 rank1, d1 rank2
        with patch.object(SelectionAgentOperator, "_ensure_agent", return_value=mock_sel):
            result = op.run(df)

        assert result["doc_id"].tolist() == ["d3", "d1"]
        assert result["result_source"].tolist() == ["final_results", "final_results"]
        assert result["rank"].tolist() == [1, 2]
        mock_sel.select_sync.assert_not_called()

    def test_selection_runs_when_no_final_results(self):
        """Tier 2: no ReAct final list -> run the selection agent over RRF candidates."""
        from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator

        mock_sel = MagicMock()
        mock_sel.select_sync.return_value = _agent_result(final_doc_ids=["d3", "d2"])
        op = self._op(top_k=2)
        df = self._rrf_frame(react_final_rank=[None, None, None])
        with patch.object(SelectionAgentOperator, "_ensure_agent", return_value=mock_sel):
            result = op.run(df)

        assert result["doc_id"].tolist() == ["d3", "d2"]
        assert result["result_source"].tolist() == ["selection_agent", "selection_agent"]
        mock_sel.select_sync.assert_called_once()
        # scores side-table covers every candidate; candidates are RRF-descending.
        call = mock_sel.select_sync.call_args
        assert call.kwargs["scores"] == {"d1": 0.9, "d2": 0.5, "d3": 0.7}
        assert [d["id"] for d in call.args[1]] == ["d1", "d3", "d2"]

    def test_falls_back_to_rrf_when_selection_returns_empty(self):
        """Tier 3: selection failed/empty -> top RRF-ranked candidates."""
        from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator

        mock_sel = MagicMock()
        mock_sel.select_sync.return_value = _agent_result(final_doc_ids=[], error_category="max_steps")
        op = self._op(top_k=2)
        df = self._rrf_frame(react_final_rank=[None, None, None])
        with patch.object(SelectionAgentOperator, "_ensure_agent", return_value=mock_sel):
            result = op.run(df)

        # RRF-descending top 2: d1 (0.9), d3 (0.7)
        assert result["doc_id"].tolist() == ["d1", "d3"]
        assert result["result_source"].tolist() == ["rrf", "rrf"]
        mock_sel.select_sync.assert_called_once()

    def test_selection_exception_falls_back_to_rrf(self):
        """An unexpected error inside selection degrades to the RRF ranking."""
        from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator

        mock_sel = MagicMock()
        mock_sel.select_sync.side_effect = RuntimeError("boom")
        op = self._op(top_k=2)
        df = self._rrf_frame(react_final_rank=[None, None, None])
        with patch.object(SelectionAgentOperator, "_ensure_agent", return_value=mock_sel):
            result = op.run(df)

        assert result["doc_id"].tolist() == ["d1", "d3"]
        assert result["result_source"].tolist() == ["rrf", "rrf"]

    def test_message_column_empty(self):
        from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator

        mock_sel = MagicMock()
        op = self._op(top_k=2)
        df = self._rrf_frame(react_final_rank=[2, None, 1])
        with patch.object(SelectionAgentOperator, "_ensure_agent", return_value=mock_sel):
            result = op.run(df)
        assert result["message"].tolist() == ["", ""]


# ---------------------------------------------------------------------------
# _parse_json_list — pure Python, no mocking needed
# ---------------------------------------------------------------------------


class TestParseJsonList:
    def _parse(self, raw, fallback="orig"):
        from nemo_retriever.operators.graph_ops.subquery_operator import _parse_json_list

        return _parse_json_list(raw, fallback=fallback)

    def test_plain_json_array(self):
        assert self._parse('["a", "b", "c"]') == ["a", "b", "c"]

    def test_json_fence(self):
        assert self._parse('```json\n["x", "y"]\n```') == ["x", "y"]

    def test_plain_fence(self):
        assert self._parse('```\n["x"]\n```') == ["x"]

    def test_trailing_fence_without_leading_not_stripped(self):
        # A JSON string that happens to end with ``` but has no leading fence.
        # It should still parse because the trailing strip must NOT fire.
        raw = '["valid"]```'
        result = self._parse(raw)
        assert result == ["orig"]  # malformed JSON → fallback

    def test_malformed_json_returns_fallback(self):
        assert self._parse("not json at all", fallback="q") == ["q"]

    def test_empty_list_returns_fallback(self):
        assert self._parse("[]", fallback="q") == ["q"]

    def test_non_string_items_returns_fallback(self):
        assert self._parse("[1, 2, 3]", fallback="q") == ["q"]

    def test_mixed_types_returns_fallback(self):
        assert self._parse('["a", 1]', fallback="q") == ["q"]


# ---------------------------------------------------------------------------
# SubQueryGeneratorOperator.preprocess — no LLM calls needed
# ---------------------------------------------------------------------------


class TestSubQueryPreprocess:
    def _op(self):
        from nemo_retriever.operators.graph_ops.subquery_operator import SubQueryGeneratorOperator

        return SubQueryGeneratorOperator(llm_model="test-model")

    def test_dataframe_accepted(self):
        op = self._op()
        df = pd.DataFrame({"query_id": ["q1"], "query_text": ["hello"]})
        result = op.preprocess(df)
        assert list(result["query_id"]) == ["q1"]

    def test_dataframe_missing_query_id_raises(self):
        op = self._op()
        bad = pd.DataFrame({"query_text": ["hello"]})
        with pytest.raises(ValueError, match="query_id"):
            op.preprocess(bad)

    def test_dataframe_missing_query_text_raises(self):
        op = self._op()
        bad = pd.DataFrame({"query_id": ["q1"]})
        with pytest.raises(ValueError, match="query_text"):
            op.preprocess(bad)

    def test_list_of_strings_auto_ids(self):
        op = self._op()
        result = op.preprocess(["alpha", "beta"])
        assert result["query_id"].tolist() == ["q0", "q1"]
        assert result["query_text"].tolist() == ["alpha", "beta"]

    def test_list_of_tuples(self):
        op = self._op()
        result = op.preprocess([("id1", "alpha"), ("id2", "beta")])
        assert result["query_id"].tolist() == ["id1", "id2"]

    def test_unsupported_type_raises(self):
        op = self._op()
        with pytest.raises(TypeError):
            op.preprocess({"query_id": "q1", "query_text": "hello"})


class TestSubQueryGeneratorOperator:
    """Tests for _build_system_prompt and _generate_one."""

    def _op(self, **kwargs):
        from nemo_retriever.operators.graph_ops.subquery_operator import SubQueryGeneratorOperator

        return SubQueryGeneratorOperator(llm_model="test-model", **kwargs)

    # -- _build_system_prompt -------------------------------------------------

    def test_decompose_prompt_contains_max_subqueries(self):
        op = self._op(strategy="decompose", max_subqueries=6)
        prompt = op._build_system_prompt()
        assert "6" in prompt
        assert "decompos" in prompt.lower()

    def test_hyde_prompt_contains_max_subqueries(self):
        op = self._op(strategy="hyde", max_subqueries=3)
        prompt = op._build_system_prompt()
        assert "3" in prompt
        assert "hypothetical" in prompt.lower()

    def test_multi_perspective_prompt_contains_max_subqueries(self):
        op = self._op(strategy="multi_perspective", max_subqueries=5)
        prompt = op._build_system_prompt()
        assert "5" in prompt
        assert "perspective" in prompt.lower()

    def test_system_prompt_override_used_instead_of_strategy(self):
        op = self._op(system_prompt_override="Custom prompt max={max_subqueries}", max_subqueries=2)
        assert op._build_system_prompt() == "Custom prompt max=2"

    # -- _generate_one --------------------------------------------------------

    @patch("nemo_retriever.operators.graph_ops.subquery_operator.invoke_chat_completions")
    def test_generate_one_happy_path(self, mock_invoke):
        mock_invoke.return_value = ['["sub1", "sub2", "sub3"]']
        op = self._op(max_subqueries=4)
        result = op._generate_one("What causes inflation?", "system prompt")
        assert result == ["sub1", "sub2", "sub3"]
        mock_invoke.assert_called_once()

    @patch("nemo_retriever.operators.graph_ops.subquery_operator.invoke_chat_completions")
    def test_generate_one_fenced_json(self, mock_invoke):
        mock_invoke.return_value = ['```json\n["a", "b"]\n```']
        op = self._op()
        assert op._generate_one("q", "sys") == ["a", "b"]

    @patch("nemo_retriever.operators.graph_ops.subquery_operator.invoke_chat_completions")
    def test_generate_one_malformed_json_falls_back(self, mock_invoke):
        mock_invoke.return_value = ["not valid json"]
        op = self._op()
        assert op._generate_one("original query", "sys") == ["original query"]

    @patch("nemo_retriever.operators.graph_ops.subquery_operator.invoke_chat_completions")
    def test_generate_one_llm_error_falls_back(self, mock_invoke):
        mock_invoke.side_effect = RuntimeError("connection timeout")
        op = self._op()
        assert op._generate_one("original query", "sys") == ["original query"]


# ---------------------------------------------------------------------------
# SelectionAgentOperator.preprocess — no LLM calls needed
# ---------------------------------------------------------------------------


class TestSelectionAgentPreprocess:
    def _op(self):
        from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator

        return SelectionAgentOperator(llm_model="test-model", invoke_url="http://localhost/v1/chat/completions")

    def test_valid_dataframe_accepted(self):
        op = self._op()
        df = pd.DataFrame({"query_id": ["q1"], "query_text": ["q"], "doc_id": ["d1"], "text": ["t"]})
        result = op.preprocess(df)
        assert len(result) == 1

    def test_missing_doc_id_raises(self):
        op = self._op()
        bad = pd.DataFrame({"query_id": ["q1"], "query_text": ["q"], "text": ["t"]})
        with pytest.raises(ValueError, match="doc_id"):
            op.preprocess(bad)

    def test_missing_text_raises(self):
        op = self._op()
        bad = pd.DataFrame({"query_id": ["q1"], "query_text": ["q"], "doc_id": ["d1"]})
        with pytest.raises(ValueError, match="text"):
            op.preprocess(bad)

    def test_non_dataframe_raises(self):
        op = self._op()
        with pytest.raises(TypeError):
            op.preprocess([{"query_id": "q1"}])


class TestBuildLLMForwarding:
    """``_build_llm`` forwards temperature / parallel_tool_calls, preserving falsy values.

    Builds a real backend (imports litellm); skipped when litellm is absent.
    ``temperature=0.0`` / ``parallel_tool_calls=False`` are real settings and must
    not be collapsed to ``None`` (guards against an ``x or None`` regression).
    """

    def test_react_forwards_falsy_sampling_args(self):
        pytest.importorskip("litellm")
        from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator

        op = ReActAgentOperator(
            llm_model="gpt-4o-mini",
            retriever_fn=lambda q, k: [],
            temperature=0.0,
            parallel_tool_calls=False,
        )
        config = op._build_llm().config
        assert config.temperature == 0.0
        assert config.parallel_tool_calls is False

    def test_selection_forwards_falsy_sampling_args(self):
        pytest.importorskip("litellm")
        from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator

        op = SelectionAgentOperator(
            llm_model="gpt-4o-mini",
            temperature=0.0,
            parallel_tool_calls=False,
        )
        config = op._build_llm().config
        assert config.temperature == 0.0
        assert config.parallel_tool_calls is False


# ---------------------------------------------------------------------------
# Callable LLM backend — the in-process (local vLLM) adapter seam
# ---------------------------------------------------------------------------


class TestCallableBackendWiring:
    """The operators forward an injected chat_completion_fn to the callable backend."""

    def test_build_llm_returns_callable_backend_when_completion_fn_set(self):
        from nemo_retriever._agentic.nemo_agent.llm import CallableLLMBackend
        from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator

        def fake_fn(**kwargs):
            return {
                "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]
            }

        op = ReActAgentOperator(
            llm_model="nemotron-8b",
            retriever_fn=lambda q, k: [],
            backend="callable",
            chat_completion_fn=fake_fn,
        )
        # No litellm import required on this path.
        assert isinstance(op._build_llm(), CallableLLMBackend)

    def test_build_llm_uses_configured_backend_without_completion_fn(self):
        pytest.importorskip("litellm")
        from nemo_retriever._agentic.nemo_agent.llm import LiteLLMBackend
        from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator

        op = SelectionAgentOperator(llm_model="gpt-4o")
        assert isinstance(op._build_llm(), LiteLLMBackend)


class TestCallableLLMBackend:
    """The adapter maps an OpenAI chat.completion dict to a CompletionResult."""

    @staticmethod
    def _backend(fn):
        from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config

        return create_llm(create_llm_config("callable", model="nemotron-8b"), completion_fn=fn)

    def test_parses_content_and_stubs_usage(self):
        seen: dict = {}

        def fn(**kwargs):
            seen.update(kwargs)
            return {
                "choices": [{"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 5},
            }

        result = self._backend(fn).completion(messages=[{"role": "user", "content": "q"}])
        assert result.message == {"role": "assistant", "content": "answer"}
        assert result.finish_reason == "stop"
        assert result.usage is None  # usage is intentionally stubbed
        assert seen["tool_choice"] == "none"  # no tools -> suppress tool calls

    def test_passes_tools_and_forwards_tool_calls(self):
        def fn(**kwargs):
            assert kwargs["tool_choice"] == "auto"
            assert kwargs["tools"]
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "retrieve", "arguments": "{}"}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }

        result = self._backend(fn).completion(
            messages=[{"role": "user", "content": "q"}],
            tools=[{"type": "function", "function": {"name": "retrieve"}}],
        )
        assert result.message["tool_calls"][0]["function"]["name"] == "retrieve"
        assert result.finish_reason == "tool_calls"

    def test_requires_completion_fn(self):
        from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config

        with pytest.raises(ValueError, match="completion_fn"):
            create_llm(create_llm_config("callable", model="nemotron-8b"))

    def test_callable_failure_is_translated_to_llm_call_error(self):
        from nemo_retriever._agentic.nemo_agent.llm import LLMCallError

        def fn(**kwargs):
            raise RuntimeError("engine died")

        with pytest.raises(LLMCallError) as excinfo:
            self._backend(fn).completion(messages=[{"role": "user", "content": "q"}])
        # The original exception stays chained so it remains diagnosable.
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    @pytest.mark.parametrize(
        "bad_response",
        [
            "not a dict",
            {},  # no choices
            {"choices": []},  # empty choices
            {"choices": [{"finish_reason": "stop"}]},  # choice missing message
        ],
    )
    def test_malformed_response_is_translated_to_llm_call_error(self, bad_response):
        from nemo_retriever._agentic.nemo_agent.llm import LLMCallError

        with pytest.raises(LLMCallError):
            self._backend(lambda **kwargs: bad_response).completion(
                messages=[{"role": "user", "content": "q"}]
            )

    @staticmethod
    def _seen_kwargs(backend, **call_kwargs):
        seen: dict = {}

        def fn(**kwargs):
            seen.update(kwargs)
            return {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]}

        backend(fn).completion(messages=[{"role": "user", "content": "q"}], **call_kwargs)
        return seen

    def test_known_knobs_routed_into_extra_body(self):
        from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config

        def backend(fn):
            return create_llm(
                create_llm_config(
                    "callable", model="nemotron-8b", parallel_tool_calls=False, reasoning_effort="high"
                ),
                completion_fn=fn,
            )

        seen = self._seen_kwargs(backend)
        assert seen["extra_body"] == {"parallel_tool_calls": False, "reasoning_effort": "high"}

    def test_max_completion_tokens_override_aliases_max_tokens(self):
        seen = self._seen_kwargs(self._backend, max_completion_tokens=64)
        assert seen["max_tokens"] == 64
        assert "max_completion_tokens" not in seen  # aliased, not passed through raw

    def test_unknown_override_is_passed_through_not_dropped(self):
        seen = self._seen_kwargs(self._backend, top_p=0.25)
        assert seen["top_p"] == 0.25  # rides along as a kwarg rather than being silently ignored

    def test_override_wins_over_config_knob_in_extra_body(self):
        from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config

        def backend(fn):
            return create_llm(
                create_llm_config("callable", model="nemotron-8b", reasoning_effort="high"),
                completion_fn=fn,
            )

        seen = self._seen_kwargs(backend, reasoning_effort="low")
        assert seen["extra_body"]["reasoning_effort"] == "low"  # override beats the config value

    def test_temperature_omitted_when_none(self):
        seen = self._seen_kwargs(self._backend)  # default config -> temperature is None
        assert "temperature" not in seen  # unset -> let the callable keep its own default

    def test_temperature_forwarded_when_set_including_zero(self):
        from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config

        def backend(fn):
            return create_llm(
                create_llm_config("callable", model="nemotron-8b", temperature=0.0),
                completion_fn=fn,
            )

        seen = self._seen_kwargs(backend)
        assert seen["temperature"] == 0.0  # explicit falsy value is still forwarded

    def test_capture_raw_io_off_by_default(self):
        result = self._backend(
            lambda **kwargs: {
                "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]
            }
        ).completion(messages=[{"role": "user", "content": "q"}])
        assert result.raw_request is None
        assert result.raw_response is None

    def test_capture_raw_io_populates_request_and_response(self):
        from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config

        response = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 3},
        }
        backend = create_llm(
            create_llm_config("callable", model="nemotron-8b", capture_raw_io=True),
            completion_fn=lambda **kwargs: response,
        )
        result = backend.completion(messages=[{"role": "user", "content": "q"}], max_completion_tokens=64)
        assert result.raw_response == response
        assert result.raw_response is not response  # an independent snapshot, not an alias
        assert result.raw_request["model"] == "nemotron-8b"
        assert result.raw_request["messages"] == [{"role": "user", "content": "q"}]
        assert result.raw_request["max_tokens"] == 64  # not redacted despite containing "token"

    def test_capture_raw_io_redacts_api_key(self):
        from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config
        from nemo_retriever._agentic.nemo_agent.llm.callable_backend import _REDACTED

        backend = create_llm(
            create_llm_config("callable", model="nemotron-8b", capture_raw_io=True),
            completion_fn=lambda **kwargs: {
                "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]
            },
        )
        result = backend.completion(messages=[{"role": "user", "content": "q"}], api_key="super-secret")
        assert result.raw_request["api_key"] == _REDACTED
        assert result.raw_request["messages"] == [{"role": "user", "content": "q"}]  # content preserved
