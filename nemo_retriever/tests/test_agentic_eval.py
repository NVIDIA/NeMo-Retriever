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


def test_build_qrels_requires_aligned_lengths():
    from nemo_retriever.query.agentic import build_qrels

    with pytest.raises(ValueError, match="same length"):
        build_qrels(["q1"], ["doc_1", "doc_2"])


def test_build_beir_run_from_agentic_result_orders_by_rank():
    from nemo_retriever.query.agentic import build_beir_run_from_agentic_result

    result = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "doc_id": ["d2", "d1", "d3"],
            "rank": [2, 1, 3],
            "message": ["ok", "ok", "ok"],
        }
    )
    run = build_beir_run_from_agentic_result(["q1", "q2"], result)

    assert list(run["q1"]) == ["d1", "d2", "d3"]
    assert run["q1"]["d1"] > run["q1"]["d2"] > run["q1"]["d3"]
    assert run["q2"] == {}


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

    assert list(result.columns) == ["query_id", "doc_id", "rank", "message", "result_source"]
    assert result["query_id"].tolist() == ["0"] * 10
    assert result["doc_id"].tolist()[0] == "doc_1"
    assert result["rank"].tolist() == list(range(1, 11))


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


@patch("nemo_retriever.operators.graph_ops.selection_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.operators.graph_ops.react_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_run_agentic_recall_evaluation_computes_metrics(mock_react_step, mock_selection_step, tmp_path):
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, run_agentic_recall_evaluation

    query_csv = tmp_path / "queries.csv"
    pd.DataFrame({"query": ["find doc"], "pdf_page": ["doc_1"]}).to_csv(query_csv, index=False)

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
    df_query, result, qrels, run, metrics = run_agentic_recall_evaluation(
        query_csv=query_csv,
        cfg=cfg,
        match_mode="pdf_page",
        ks=(1, 5, 10),
    )

    assert df_query["golden_answer"].tolist() == ["doc_1"]
    assert result["doc_id"].tolist()[0] == "doc_1"
    assert qrels == {"0": {"doc_1": 1}}
    assert run["0"]["doc_1"] == 10.0
    assert metrics["recall@1"] == 1.0
    assert metrics["ndcg@1"] == 1.0


@patch("nemo_retriever.operators.graph_ops.selection_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.operators.graph_ops.react_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_run_agentic_beir_evaluation_loads_queries_and_qrels(mock_react_step, mock_selection_step):
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, run_agentic_beir_evaluation
    from nemo_retriever.tools.recall.beir import BeirDataset

    final_ids = ["doc"] + [f"extra_{i}" for i in range(9)]
    mock_react_step.return_value = _make_tool_call_response(
        "final_results",
        {"doc_ids": final_ids, "message": "done", "search_successful": "true"},
    )
    mock_selection_step.return_value = _make_tool_call_response(
        "log_selected_documents",
        {"doc_ids": ["doc"], "message": "doc is best"},
    )

    beir_dataset = BeirDataset(
        dataset_name="vidore_v3_finance_en",
        query_ids=["q1"],
        queries=["find doc"],
        qrels={"q1": {"doc": 1}},
    )
    cfg = AgenticRetrievalConfig(llm_model="test-model", invoke_url="http://localhost/v1/chat/completions")

    with patch("nemo_retriever.query.agentic.load_beir_dataset", return_value=beir_dataset) as mock_loader:
        df_query, result, qrels, run, metrics = run_agentic_beir_evaluation(
            loader="vidore_hf",
            dataset_name="vidore_v3_finance_en",
            cfg=cfg,
            doc_id_field="pdf_basename",
            ks=(1, 5, 10),
        )

    mock_loader.assert_called_once()
    assert df_query["query_id"].tolist() == ["q1"]
    assert result["doc_id"].tolist()[0] == "doc"
    assert qrels == {"q1": {"doc": 1}}
    assert run["q1"]["doc"] == 10.0
    assert metrics["recall@1"] == 1.0


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
