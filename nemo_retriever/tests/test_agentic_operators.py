# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the agentic retrieval operators.

Run with:
    cd nemo_retriever && uv run pytest tests/test_agentic_operators.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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
        from nemo_retriever.graph.rrf_aggregator_operator import RRFAggregatorOperator

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
        from nemo_retriever.graph.rrf_aggregator_operator import RRFAggregatorOperator

        op = RRFAggregatorOperator(k=60)
        result = op.run(self._make_input())

        for _, grp in result.groupby("query_id"):
            scores = grp["rrf_score"].tolist()
            assert scores == sorted(scores, reverse=True), "Scores not sorted descending"

    def test_text_carried_through(self):
        from nemo_retriever.graph.rrf_aggregator_operator import RRFAggregatorOperator

        op = RRFAggregatorOperator(k=60)
        result = op.run(self._make_input())
        q1 = result[result["query_id"] == "q1"].set_index("doc_id")
        assert q1.loc["d1", "text"] == "t1"

    def test_output_schema(self):
        from nemo_retriever.graph.rrf_aggregator_operator import RRFAggregatorOperator

        op = RRFAggregatorOperator(k=60)
        result = op.run(self._make_input())
        assert set(result.columns) >= {"query_id", "query_text", "doc_id", "rrf_score", "text"}

    def test_missing_column_raises(self):
        from nemo_retriever.graph.rrf_aggregator_operator import RRFAggregatorOperator

        op = RRFAggregatorOperator(k=60)
        bad_df = pd.DataFrame({"query_id": ["q1"], "query_text": ["x"]})
        with pytest.raises(ValueError, match="missing required column"):
            op.run(bad_df)


# ---------------------------------------------------------------------------
# Prompt rendering — pure Python, no mocking needed
# ---------------------------------------------------------------------------


class TestPromptRendering:
    def test_react_prompt_no_extended_relevance(self):
        from nemo_retriever.graph.react_agent_operator import _render_react_agent_prompt

        prompt = _render_react_agent_prompt(10, with_init_docs=True, enforce_top_k=True, extended_relevance=False)
        assert "<Goal>" in prompt
        assert "<WORKFLOW>" in prompt
        assert "<BEST_PRACTICES>" in prompt
        assert "RELEVANCE_DEFINITION" not in prompt
        assert "exactly the 10 most relevant" in prompt
        assert "TIP" in prompt

    def test_react_prompt_with_extended_relevance(self):
        from nemo_retriever.graph.react_agent_operator import _render_react_agent_prompt

        prompt = _render_react_agent_prompt(5, with_init_docs=False, enforce_top_k=False, extended_relevance=True)
        assert "RELEVANCE_DEFINITION" in prompt
        assert "exactly the 5" not in prompt
        assert "TIP" not in prompt

    def test_selection_prompt_no_extended_relevance(self):
        from nemo_retriever.graph.selection_agent_operator import _render_selection_prompt

        prompt = _render_selection_prompt(5, extended_relevance=False)
        assert "<ROLE>" in prompt
        assert "<WORKFLOW>" in prompt
        assert "THINKING TIPS" in prompt
        assert "RELEVANCE_DEFINITION" not in prompt
        assert "5 most relevant" in prompt

    def test_selection_prompt_with_extended_relevance(self):
        from nemo_retriever.graph.selection_agent_operator import _render_selection_prompt

        prompt = _render_selection_prompt(5, extended_relevance=True)
        assert "RELEVANCE_DEFINITION" in prompt
        assert "As explained above" in prompt


# ---------------------------------------------------------------------------
# SelectionAgentOperator — mock invoke_chat_completion_step
# ---------------------------------------------------------------------------


def _make_tool_call_response(fn_name: str, fn_args: dict, tc_id: str = "call_1") -> dict:
    """Build a canned /v1/chat/completions response with one tool call."""
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": json.dumps(fn_args)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


class TestSelectionAgentOperator:
    def _make_input(self):
        return pd.DataFrame(
            {
                "query_id": ["q1", "q1", "q1"],
                "query_text": ["What causes inflation?"] * 3,
                "doc_id": ["d1", "d2", "d3"],
                "text": ["monetary policy doc", "supply chain doc", "unrelated doc"],
            }
        )

    @patch("nemo_retriever.graph.selection_agent_operator.invoke_chat_completion_step")
    def test_happy_path_selects_docs(self, mock_step):
        from nemo_retriever.graph.selection_agent_operator import SelectionAgentOperator

        # LLM immediately calls log_selected_documents
        mock_step.return_value = _make_tool_call_response(
            "log_selected_documents",
            {"doc_ids": ["d1", "d2"], "message": "d1 most relevant"},
        )

        op = SelectionAgentOperator(
            llm_model="test-model",
            invoke_url="http://localhost/v1/chat/completions",
            top_k=2,
        )
        result = op.run(self._make_input())

        assert list(result.columns) == ["query_id", "doc_id", "rank", "message"]
        assert result["query_id"].tolist() == ["q1", "q1"]
        assert result["doc_id"].tolist() == ["d1", "d2"]
        assert result["rank"].tolist() == [1, 2]

    @patch("nemo_retriever.graph.selection_agent_operator.invoke_chat_completion_step")
    def test_think_then_select(self, mock_step):
        from nemo_retriever.graph.selection_agent_operator import SelectionAgentOperator

        # First call: think; second call: log_selected_documents
        mock_step.side_effect = [
            _make_tool_call_response("think", {"thought": "let me reason..."}),
            _make_tool_call_response("log_selected_documents", {"doc_ids": ["d3"], "message": "only d3"}),
        ]

        op = SelectionAgentOperator(
            llm_model="test-model",
            invoke_url="http://localhost/v1/chat/completions",
            top_k=1,
        )
        result = op.run(self._make_input())

        assert result["doc_id"].tolist() == ["d3"]
        assert mock_step.call_count == 2

    @patch("nemo_retriever.graph.selection_agent_operator.invoke_chat_completion_step")
    def test_extended_relevance_in_prompt(self, mock_step):
        from nemo_retriever.graph.selection_agent_operator import SelectionAgentOperator

        captured_prompts = []

        def capture_and_respond(**kwargs):
            captured_prompts.append(kwargs["messages"][0]["content"])
            return _make_tool_call_response("log_selected_documents", {"doc_ids": ["d1"], "message": "ok"})

        mock_step.side_effect = capture_and_respond

        op = SelectionAgentOperator(
            llm_model="test-model",
            invoke_url="http://localhost/v1/chat/completions",
            top_k=1,
            extended_relevance=True,
        )
        op.run(self._make_input())

        assert "RELEVANCE_DEFINITION" in captured_prompts[0]


# ---------------------------------------------------------------------------
# ReActAgentOperator — mock retriever_fn + invoke_chat_completion_step
# ---------------------------------------------------------------------------


class TestReActAgentOperator:
    def _make_input(self):
        return pd.DataFrame(
            {
                "query_id": ["q1"],
                "query_text": ["What causes inflation?"],
            }
        )

    def _make_retriever(self, docs=None):
        """Return a mock retriever_fn that returns canned docs."""
        if docs is None:
            docs = [{"doc_id": "d1", "text": "monetary policy"}, {"doc_id": "d2", "text": "supply chains"}]

        def retriever_fn(query_text: str, top_k: int):
            return docs[:top_k]

        return retriever_fn

    @patch("nemo_retriever.graph.react_agent_operator.invoke_chat_completion_step")
    def test_simple_mode_retrieve_then_final(self, mock_step):
        from nemo_retriever.graph.react_agent_operator import ReActAgentOperator

        # 1) agent calls retrieve("subquery"), 2) agent calls final_results
        mock_step.side_effect = [
            _make_tool_call_response("retrieve", {"query": "inflation monetary policy"}),
            _make_tool_call_response(
                "final_results",
                {"doc_ids": ["d1", "d2"], "message": "found them", "search_successful": "true"},
            ),
        ]

        op = ReActAgentOperator(
            invoke_url="http://localhost/v1/chat/completions",
            llm_model="test-model",
            retriever_fn=self._make_retriever(),
            user_msg_type="simple",
            target_top_k=2,
        )
        result = op.run(self._make_input())

        assert set(result.columns) >= {"query_id", "query_text", "step_idx", "doc_id", "text", "rank"}
        assert result["query_id"].unique().tolist() == ["q1"]
        # step 0 is the retrieve tool call result
        assert 0 in result["step_idx"].values
        assert "d1" in result["doc_id"].values

    @patch("nemo_retriever.graph.react_agent_operator.invoke_chat_completion_step")
    def test_with_results_mode_initial_retrieval(self, mock_step):
        from nemo_retriever.graph.react_agent_operator import ReActAgentOperator

        # with_results=True → retriever_fn called once before LLM, then LLM immediately calls final_results
        mock_step.return_value = _make_tool_call_response(
            "final_results",
            {"doc_ids": ["d1"], "message": "ok", "search_successful": "true"},
        )
        retriever = MagicMock(return_value=[{"doc_id": "d1", "text": "monetary policy"}])

        op = ReActAgentOperator(
            invoke_url="http://localhost/v1/chat/completions",
            llm_model="test-model",
            retriever_fn=retriever,
            user_msg_type="with_results",
            target_top_k=1,
        )
        result = op.run(self._make_input())

        # retriever was called upfront (step_idx=0) before any LLM step
        assert retriever.call_count >= 1
        assert 0 in result["step_idx"].values

    @patch("nemo_retriever.graph.react_agent_operator.invoke_chat_completion_step")
    def test_output_row_structure(self, mock_step):
        from nemo_retriever.graph.react_agent_operator import ReActAgentOperator

        mock_step.side_effect = [
            _make_tool_call_response("retrieve", {"query": "q"}),
            _make_tool_call_response(
                "final_results", {"doc_ids": ["d1"], "message": "ok", "search_successful": "true"}
            ),
        ]

        op = ReActAgentOperator(
            invoke_url="http://localhost/v1/chat/completions",
            llm_model="test-model",
            retriever_fn=self._make_retriever(),
            user_msg_type="simple",
        )
        result = op.run(self._make_input())

        assert (result["rank"] >= 1).all()
        assert result["step_idx"].dtype in (int, "int64")
        assert result["doc_id"].notna().all()

    @patch("nemo_retriever.graph.selection_agent_operator.invoke_chat_completion_step")
    @patch("nemo_retriever.graph.react_agent_operator.invoke_chat_completion_step")
    def test_pipeline_end_to_end_with_mocks(self, mock_react_step, mock_selection_step):
        """Wire ReAct → RRF → Selection with mocks; verify final output shape.

        Each operator imports invoke_chat_completion_step into its own module
        namespace, so both must be patched independently.
        """
        from nemo_retriever.graph.executor import InprocessExecutor
        from nemo_retriever.graph.react_agent_operator import ReActAgentOperator
        from nemo_retriever.graph.rrf_aggregator_operator import RRFAggregatorOperator
        from nemo_retriever.graph.selection_agent_operator import SelectionAgentOperator

        # ReAct: retrieve once, then final_results
        mock_react_step.side_effect = [
            _make_tool_call_response("retrieve", {"query": "inflation"}),
            _make_tool_call_response(
                "final_results", {"doc_ids": ["d1", "d2"], "message": "ok", "search_successful": "true"}
            ),
        ]
        # Selection: immediately log_selected_documents
        mock_selection_step.return_value = _make_tool_call_response(
            "log_selected_documents", {"doc_ids": ["d1"], "message": "d1 best"}
        )

        def retriever_fn(query_text, top_k):
            return [{"doc_id": "d1", "text": "monetary policy"}, {"doc_id": "d2", "text": "supply chains"}]

        pipeline = (
            ReActAgentOperator(
                invoke_url="http://localhost/v1/chat/completions",
                llm_model="test-model",
                retriever_fn=retriever_fn,
                user_msg_type="simple",
                target_top_k=1,
            )
            >> RRFAggregatorOperator(k=60)
            >> SelectionAgentOperator(
                invoke_url="http://localhost/v1/chat/completions",
                llm_model="test-model",
                top_k=1,
            )
        )

        query_df = pd.DataFrame({"query_id": ["q1"], "query_text": ["What causes inflation?"]})
        result = InprocessExecutor(pipeline).ingest(query_df)

        assert list(result.columns) == ["query_id", "doc_id", "rank", "message"]
        assert result["query_id"].tolist() == ["q1"]
        assert result["rank"].tolist() == [1]
