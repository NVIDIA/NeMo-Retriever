# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest
from typer.testing import CliRunner


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
    from nemo_retriever.agentic.retrieval import build_qrels

    with pytest.raises(ValueError, match="same length"):
        build_qrels(["q1"], ["doc_1", "doc_2"])


def test_build_beir_run_from_agentic_result_orders_by_rank():
    from nemo_retriever.agentic.retrieval import build_beir_run_from_agentic_result

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


@patch("nemo_retriever.graph.selection_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.graph.react_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.agentic.retrieval.Retriever", FakeRetriever)
def test_agentic_retriever_runs_graph_with_wrapped_retriever(mock_react_step, mock_selection_step):
    from nemo_retriever.agentic.retrieval import AgenticRetrievalConfig, AgenticRetriever

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


@patch("nemo_retriever.graph.selection_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.graph.react_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.agentic.retrieval.Retriever", FakeRetriever)
def test_run_agentic_recall_evaluation_computes_metrics(mock_react_step, mock_selection_step, tmp_path):
    from nemo_retriever.agentic.retrieval import AgenticRetrievalConfig, run_agentic_recall_evaluation

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


@patch("nemo_retriever.graph.selection_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.graph.react_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.agentic.retrieval.Retriever", FakeRetriever)
def test_run_agentic_beir_evaluation_loads_queries_and_qrels(mock_react_step, mock_selection_step):
    from nemo_retriever.agentic.retrieval import AgenticRetrievalConfig, run_agentic_beir_evaluation
    from nemo_retriever.recall.beir import BeirDataset

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

    with patch("nemo_retriever.agentic.retrieval.load_beir_dataset", return_value=beir_dataset) as mock_loader:
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


def test_pipeline_agentic_beir_wires_config_options():
    from nemo_retriever.pipeline.__main__ import _run_agentic_evaluation

    captured = {}

    def fake_run_agentic_beir_evaluation(**kwargs):
        captured.update(kwargs)
        return (
            pd.DataFrame({"query_id": ["q1"]}),
            pd.DataFrame({"query_id": ["q1"], "doc_id": ["doc"], "rank": [1]}),
            {"q1": {"doc": 1}},
            {"q1": {"doc": 10.0}},
            {"recall@1": 1.0},
        )

    with (
        patch("nemo_retriever.model.resolve_embed_model", return_value="resolved-embed"),
        patch(
            "nemo_retriever.agentic.retrieval.run_agentic_beir_evaluation", side_effect=fake_run_agentic_beir_evaluation
        ),
    ):
        label, _elapsed, metrics, query_count, ran = _run_agentic_evaluation(
            evaluation_mode="beir",
            vdb_op="lancedb",
            vdb_kwargs={"uri": "db", "table_name": "tbl"},
            embed_model_name="embed",
            embed_invoke_url="http://embed/v1",
            embed_remote_api_key="embed-key",
            embed_modality="text",
            query_csv=None,
            recall_match_mode="pdf_page",
            reranker=False,
            reranker_model_name="reranker",
            reranker_invoke_url=None,
            reranker_api_key="",
            local_reranker_backend="vllm",
            local_hf_batch_size=4,
            local_query_embed_backend="hf",
            agentic_llm_model="llm",
            agentic_invoke_url="http://llm/v1/chat/completions",
            agentic_api_key="llm-key",
            agentic_react_max_steps=51,
            agentic_backend_top_k=23,
            agentic_text_truncation=99,
            agentic_reasoning_effort="high",
            agentic_num_concurrent=7,
            beir_loader="vidore_hf",
            beir_dataset_name="vidore_v3_finance_en",
            beir_split="test",
            beir_query_language=None,
            beir_doc_id_field="pdf_basename",
            beir_k=[1, 5],
        )

    cfg = captured["cfg"]
    assert label == "Agentic BEIR"
    assert metrics["recall@1"] == 1.0
    assert query_count == 1
    assert ran is True
    assert captured["loader"] == "vidore_hf"
    assert captured["dataset_name"] == "vidore_v3_finance_en"
    assert captured["doc_id_field"] == "pdf_basename"
    assert captured["ks"] == (1, 5)
    assert cfg.query_embedder == "resolved-embed"
    assert cfg.react_max_steps == 51
    assert cfg.backend_top_k == 23
    assert cfg.text_truncation == 99
    assert cfg.reasoning_effort == "high"
    assert cfg.num_concurrent == 7


def test_agentic_config_requires_llm_model():
    from nemo_retriever.agentic.retrieval import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="llm_model"):
        AgenticRetrievalConfig(llm_model="")


def test_pipeline_rejects_agentic_qa_mode():
    from nemo_retriever.pipeline.__main__ import app

    result = CliRunner().invoke(
        app,
        [
            ".",
            "--evaluation-mode",
            "qa",
            "--retrieval-mode",
            "agentic",
            "--agentic-llm-model",
            "test-model",
        ],
    )

    assert result.exit_code != 0
    assert "--retrieval-mode=agentic is currently supported only with" in result.output
    assert "--evaluation-mode=audio_recall" in result.output
    assert "--evaluation-mode=beir" in result.output


def test_pipeline_invalid_retrieval_mode_falls_back_to_standard():
    from nemo_retriever.pipeline.__main__ import app

    result = CliRunner().invoke(
        app,
        [
            ".",
            "--evaluation-mode",
            "qa",
            "--retrieval-mode",
            "unknown",
        ],
    )

    assert result.exit_code != 0
    assert "falling back to 'standard'" in result.output
    assert "--evaluation-mode=qa requires --eval-config" in result.output


def test_pipeline_requires_agentic_llm_model():
    from nemo_retriever.pipeline.__main__ import app

    result = CliRunner().invoke(
        app,
        [
            ".",
            "--evaluation-mode",
            "audio_recall",
            "--input-type",
            "audio",
            "--recall-match-mode",
            "audio_segment",
            "--retrieval-mode",
            "agentic",
        ],
    )

    assert result.exit_code != 0
    assert "--retrieval-mode=agentic requires --agentic-llm-model" in result.output


def test_pipeline_agentic_recall_wires_query_csv(tmp_path):
    from nemo_retriever.pipeline.__main__ import _run_agentic_evaluation

    captured = {}

    def fake_run_agentic_recall_evaluation(**kwargs):
        captured.update(kwargs)
        return (
            pd.DataFrame({"query_id": ["q1"]}),
            pd.DataFrame({"query_id": ["q1"], "doc_id": ["doc"], "rank": [1]}),
            {"q1": {"doc": 1}},
            {"q1": {"doc": 10.0}},
            {"recall@1": 1.0},
        )

    query_csv = tmp_path / "queries.csv"
    query_csv.write_text("query,golden_answer\nwhat is x,doc\n", encoding="utf-8")

    with (
        patch("nemo_retriever.model.resolve_embed_model", return_value="resolved-embed"),
        patch(
            "nemo_retriever.agentic.retrieval.run_agentic_recall_evaluation",
            side_effect=fake_run_agentic_recall_evaluation,
        ),
    ):
        label, _elapsed, metrics, query_count, ran = _run_agentic_evaluation(
            evaluation_mode="recall",
            vdb_op="lancedb",
            vdb_kwargs={"uri": "db", "table_name": "tbl"},
            embed_model_name="embed",
            embed_invoke_url="http://embed/v1",
            embed_remote_api_key="embed-key",
            embed_modality="text",
            query_csv=query_csv,
            recall_match_mode="pdf_page",
            reranker=False,
            reranker_model_name="reranker",
            reranker_invoke_url=None,
            reranker_api_key="",
            local_reranker_backend="vllm",
            local_hf_batch_size=4,
            local_query_embed_backend="hf",
            agentic_llm_model="llm",
            agentic_invoke_url="http://llm/v1/chat/completions",
            agentic_api_key="llm-key",
            agentic_react_max_steps=50,
            agentic_backend_top_k=20,
            agentic_text_truncation=0,
            agentic_reasoning_effort="high",
            agentic_num_concurrent=1,
            beir_loader=None,
            beir_dataset_name=None,
            beir_split="test",
            beir_query_language=None,
            beir_doc_id_field="pdf_basename",
            beir_k=[1, 5, 10],
        )

    assert label == "Agentic Recall"
    assert ran is True
    assert metrics["recall@1"] == 1.0
    assert query_count == 1
    assert captured["query_csv"] == query_csv
    assert captured["match_mode"] == "pdf_page"


def test_pipeline_recall_agentic_requires_pdf_match_mode():
    from nemo_retriever.pipeline.__main__ import app

    # Default --recall-match-mode is audio_segment, which is invalid for recall.
    result = CliRunner().invoke(
        app,
        [
            ".",
            "--evaluation-mode",
            "recall",
            "--retrieval-mode",
            "agentic",
            "--agentic-llm-model",
            "test-model",
            "--query-csv",
            "queries.csv",
        ],
    )

    assert result.exit_code != 0
    assert "--evaluation-mode=recall requires" in result.output
    assert "pdf_only" in result.output


def test_pipeline_recall_requires_agentic():
    from nemo_retriever.pipeline.__main__ import app

    result = CliRunner().invoke(
        app,
        [
            ".",
            "--evaluation-mode",
            "recall",
            "--retrieval-mode",
            "standard",
            "--recall-match-mode",
            "pdf_page",
        ],
    )

    assert result.exit_code != 0
    assert "--evaluation-mode=recall is currently supported only" in result.output
