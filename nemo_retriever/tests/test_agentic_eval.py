# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest


def _make_tool_call_response(fn_name: str, fn_args: dict, tc_id: str = "call_1") -> dict:
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


class FakeRetriever:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.graph = kwargs.get("graph")
        self.top_k = int(kwargs.get("top_k", 10))

    def query(self, query: str, *, top_k: int | None = None):
        if self.graph is not None:
            return self.queries([query], top_k=top_k)[0]
        _ = query
        hits = [
            {
                "source": "/tmp/doc.pdf",
                "source_id": "/tmp/doc.pdf",
                "page_number": 1,
                "pdf_page": "doc_1",
                "text": "matching document",
                "_score": 0.9,
            },
            {
                "source": "/tmp/other.pdf",
                "source_id": "/tmp/other.pdf",
                "page_number": 2,
                "pdf_page": "other_2",
                "text": "other document",
                "_score": 0.1,
            },
        ]
        return hits[:top_k]

    def queries(self, queries, *, top_k: int | None = None):
        if self.graph is None:
            return [self.query(query, top_k=top_k) for query in queries]
        limit = int(top_k) if top_k is not None else self.top_k
        df = pd.DataFrame({"query_text": [str(query) for query in queries]})
        graph = self.graph.resolve_for_local_execution()
        raw_hits = graph.execute(df)[0]
        return [list(hits)[:limit] for hits in raw_hits]


@patch("nemo_retriever.operators.graph_ops.selection_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.operators.graph_ops.react_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_runs_graph_with_wrapped_retriever(mock_react_step, mock_selection_step):
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    final_ids = ["doc_1"] + [f"extra_{i}" for i in range(9)]
    mock_react_step.return_value = _make_tool_call_response(
        "final_results",
        {"doc_ids": final_ids, "message": "done", "search_successful": "true"},
    )
    mock_selection_step.return_value = _make_tool_call_response(
        "log_selected_documents",
        {"doc_ids": ["doc_1"], "message": "doc_1 is best"},
    )

    cfg = AgenticRetrievalConfig(llm_model="test-model", invoke_url="http://localhost/v1/chat/completions")
    result = AgenticRetriever(cfg, match_mode="pdf_page").retrieve(["0"], ["find doc"])

    assert list(result.columns) == ["query_id", "doc_id", "rank", "message", "result_source", "hit"]
    assert result["query_id"].tolist() == ["0"] * 10
    assert result["doc_id"].tolist()[0] == "doc_1"
    assert result["rank"].tolist() == list(range(1, 11))
    # doc_1 was actually retrieved, so its row re-hydrates the full hit metadata
    # (dropped at the agent boundary) rather than just a bare doc_id.
    doc1_hit = result.loc[result["doc_id"] == "doc_1", "hit"].iloc[0]
    assert doc1_hit.get("source") == "/tmp/doc.pdf"
    assert doc1_hit.get("page_number") == 1
    assert "text" in doc1_hit


@patch("nemo_retriever.operators.graph_ops.selection_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.operators.graph_ops.react_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_honors_top_k(mock_react_step, mock_selection_step):
    """cfg.top_k drives the pipeline output count, not the hardcoded default of 10."""
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    final_ids = ["doc_1"] + [f"extra_{i}" for i in range(4)]  # exactly 5
    mock_react_step.return_value = _make_tool_call_response(
        "final_results",
        {"doc_ids": final_ids, "message": "done", "search_successful": "true"},
    )
    mock_selection_step.return_value = _make_tool_call_response(
        "log_selected_documents",
        {"doc_ids": ["doc_1"], "message": "doc_1 is best"},
    )

    cfg = AgenticRetrievalConfig(llm_model="test-model", invoke_url="http://localhost/v1/chat/completions", top_k=5)
    result = AgenticRetriever(cfg, match_mode="pdf_page").retrieve(["0"], ["find doc"])

    assert result["rank"].tolist() == list(range(1, 6))  # 5 rows, honoring top_k=5


def test_agentic_config_requires_llm_model():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="llm_model"):
        AgenticRetrievalConfig(llm_model="")
    # None must not slip through as the literal string "None".
    with pytest.raises(ValueError, match="llm_model"):
        AgenticRetrievalConfig(llm_model=None)


def test_agentic_config_rejects_nonpositive_top_k():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="top_k"):
        AgenticRetrievalConfig(llm_model="m", top_k=0)
