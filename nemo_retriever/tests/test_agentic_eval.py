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
        self.query_calls: list[dict] = []

    def query(self, query: str, *, top_k=None, candidate_k=None, page_dedup=False, content_types=None):
        if self.graph is not None:
            return self.queries([query], top_k=top_k)[0]
        is_audio_query = "clip" in str(query).lower()
        self.query_calls.append(
            {"top_k": top_k, "candidate_k": candidate_k, "page_dedup": page_dedup, "content_types": content_types}
        )
        hits = [
            {
                "source": {"source_id": "/tmp/clip.wav"} if is_audio_query else "/tmp/doc.pdf",
                "source_id": "/tmp/clip.wav" if is_audio_query else "/tmp/doc.pdf",
                "page_number": 1,
                "pdf_page": "doc_1",
                "metadata": (
                    {"segment_start_seconds": 1.0, "segment_end_seconds": 3.0}
                    if is_audio_query
                    else {"section": "summary"}
                ),
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


def test_build_beir_run_from_ranked_doc_ids_orders_by_rank():
    from nemo_retriever.tools.recall.beir import build_beir_run_from_ranked_doc_ids

    run = build_beir_run_from_ranked_doc_ids(["q1"], [["d1", "d2", "d3"]])

    assert list(run["q1"]) == ["d1", "d2", "d3"]
    assert run["q1"]["d1"] > run["q1"]["d2"] > run["q1"]["d3"]


def test_build_beir_run_from_ranked_doc_ids_rejects_length_mismatch():
    from nemo_retriever.tools.recall.beir import build_beir_run_from_ranked_doc_ids

    with pytest.raises(ValueError, match="query_ids and ranked_doc_ids must have the same length"):
        build_beir_run_from_ranked_doc_ids(["q1", "q2"], [["d1"]])


def test_agentic_trace_omits_mapping_none_values_but_preserves_list_nulls(tmp_path):
    from nemo_retriever.query.agentic_trace import AgenticTraceWriter

    trace_path = tmp_path / "trace" / "agentic_trace.jsonl"
    writer = AgenticTraceWriter(trace_path)

    writer.event(
        {
            "event": "trace.test",
            "unset": None,
            "nested": {"drop": None, "keep": "value"},
            "items": ["first", None, {"drop": None, "keep": "nested-list-value"}],
        }
    )

    event = json.loads(trace_path.read_text().splitlines()[0])

    assert "unset" not in event
    assert event["nested"] == {"keep": "value"}
    assert event["items"] == ["first", None, {"keep": "nested-list-value"}]


@patch("nemo_retriever.operators.graph_ops.selection_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.operators.graph_ops.react_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_runs_graph_with_wrapped_retriever(mock_react_step, mock_selection_step, tmp_path):
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

    trace_path = tmp_path / "trace" / "agentic_trace.jsonl"
    cfg = AgenticRetrievalConfig(
        llm_model="test-model",
        invoke_url="http://localhost/v1/chat/completions",
        trace_enabled=True,
        trace_path=str(trace_path),
    )
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
    assert doc1_hit.get("metadata") == {"section": "summary"}
    assert "text" in doc1_hit
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    event_names = [event["event"] for event in events]
    assert "agentic.query_start" in event_names
    assert "react.initial_retrieve" in event_names
    assert "react.final_results" in event_names
    assert "selection.query_start" in event_names
    assert "selection.query_done" in event_names


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


def test_build_agentic_config_normalizes_content_type_sequences():
    from nemo_retriever.query.options import QueryAgenticOptions, QueryRequest, QueryRetrievalOptions
    from nemo_retriever.query.workflow import build_agentic_config

    cfg = build_agentic_config(
        QueryRequest(
            query="q",
            retrieval=QueryRetrievalOptions(content_types=["text", "images"]),
            agentic=QueryAgenticOptions(llm_model="m"),
        )
    )

    assert cfg.content_types == "text,image"


@patch("nemo_retriever.operators.graph_ops.selection_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.operators.graph_ops.react_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_forwards_retrieval_knobs(mock_react_step, mock_selection_step):
    """candidate_k/page_dedup/content_types reach the per-hop Retriever.query call."""
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    mock_react_step.return_value = _make_tool_call_response(
        "final_results", {"doc_ids": ["doc_1"], "message": "done", "search_successful": "true"}
    )
    mock_selection_step.return_value = _make_tool_call_response(
        "log_selected_documents", {"doc_ids": ["doc_1"], "message": "best"}
    )

    cfg = AgenticRetrievalConfig(
        llm_model="m",
        invoke_url="http://localhost/v1/chat/completions",
        top_k=1,
        candidate_k=40,
        page_dedup=True,
        content_types="text",
    )
    retriever = AgenticRetriever(cfg, match_mode="pdf_page")
    retriever.retrieve(["0"], ["find doc"])

    calls = retriever._retriever.query_calls
    assert calls, "expected at least one per-hop retriever.query call"
    assert all(c["page_dedup"] is True for c in calls)
    assert all(c["content_types"] == "text" for c in calls)
    assert all(c["candidate_k"] >= c["top_k"] for c in calls)  # floored at the hop's top_k


@patch("nemo_retriever.operators.graph_ops.selection_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.operators.graph_ops.react_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_floors_zero_candidate_k_to_hop_top_k(mock_react_step, mock_selection_step):
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    mock_react_step.return_value = _make_tool_call_response(
        "final_results", {"doc_ids": ["doc_1"], "message": "done", "search_successful": "true"}
    )
    mock_selection_step.return_value = _make_tool_call_response(
        "log_selected_documents", {"doc_ids": ["doc_1"], "message": "best"}
    )

    retriever = AgenticRetriever(
        AgenticRetrievalConfig(
            llm_model="m",
            invoke_url="http://localhost/v1/chat/completions",
            top_k=1,
            candidate_k=0,
        ),
        match_mode="pdf_page",
    )
    retriever.retrieve(["0"], ["find doc"])

    calls = retriever._retriever.query_calls
    assert calls
    assert all(c["candidate_k"] == c["top_k"] for c in calls)


@patch("nemo_retriever.operators.graph_ops.selection_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.operators.graph_ops.react_agent_operator.invoke_chat_completion_step")
@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_run_agentic_audio_recall_evaluation_computes_metrics(mock_react_step, mock_selection_step, tmp_path):
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, run_agentic_audio_recall_evaluation

    query_csv = tmp_path / "queries.csv"
    pd.DataFrame(
        {
            "query": ["find clip"],
            "expected_media_id": ["clip"],
            "expected_start_time": [0.0],
            "expected_end_time": [4.0],
        }
    ).to_csv(query_csv, index=False)

    audio_doc_id = "clip	1.000000	3.000000"
    final_ids = [audio_doc_id] + [f"extra_{i}" for i in range(9)]
    mock_react_step.return_value = _make_tool_call_response(
        "final_results",
        {"doc_ids": final_ids, "message": "done", "search_successful": "true"},
    )
    mock_selection_step.return_value = _make_tool_call_response(
        "log_selected_documents",
        {"doc_ids": [audio_doc_id], "message": "clip is best"},
    )

    cfg = AgenticRetrievalConfig(llm_model="test-model", invoke_url="http://localhost/v1/chat/completions")
    df_query, result, gold, retrieved, metrics = run_agentic_audio_recall_evaluation(
        query_csv=query_csv,
        cfg=cfg,
        ks=(1, 5, 10),
    )

    assert df_query["golden_answer"].tolist() == ["clip	0.000000	4.000000"]
    assert result["doc_id"].tolist()[0] == audio_doc_id
    assert gold == ["clip	0.000000	4.000000"]
    assert retrieved[0][0] == audio_doc_id
    assert metrics["recall@1"] == 1.0


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


def test_agentic_config_rejects_noninteger_top_k():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="top_k must be an integer"):
        AgenticRetrievalConfig(llm_model="m", top_k=1.5)


def test_agentic_config_normalizes_integer_like_values():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    cfg = AgenticRetrievalConfig(
        llm_model="m",
        invoke_url="http://localhost/v1/chat/completions",
        top_k="5.0",
        backend_top_k="6.0",
        temperature="0.25",
    )

    assert cfg.top_k == 5
    assert cfg.backend_top_k == 6
    assert cfg.temperature == 0.25


def test_agentic_config_rejects_backend_top_k_below_target():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="backend_top_k"):
        AgenticRetrievalConfig(llm_model="m", backend_top_k=4, top_k=5)


def test_agentic_config_rejects_nvidia_temperature_above_max():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        AgenticRetrievalConfig(llm_model="m", temperature=1.5)


def test_agentic_config_rejects_nonfinite_temperature():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="temperature must be finite"):
        AgenticRetrievalConfig(llm_model="m", temperature=float("nan"))


def test_agentic_config_normalizes_base_invoke_url_to_chat_completions():
    """A base OPENAI_BASE_URL is extended to /chat/completions; the operators POST
    to invoke_url verbatim, so a bare base URL would otherwise 404."""
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    # Bare base URL -> chat-completions endpoint (with and without trailing slash).
    assert (
        AgenticRetrievalConfig(llm_model="m", invoke_url="https://host/v1").invoke_url
        == "https://host/v1/chat/completions"
    )
    assert (
        AgenticRetrievalConfig(llm_model="m", invoke_url="https://host/v1/").invoke_url
        == "https://host/v1/chat/completions"
    )
    # Already a full chat-completions endpoint -> unchanged.
    assert (
        AgenticRetrievalConfig(llm_model="m", invoke_url="https://host/v1/chat/completions").invoke_url
        == "https://host/v1/chat/completions"
    )
    # Comma-separated multi-endpoint form is normalized per URL.
    assert (
        AgenticRetrievalConfig(llm_model="m", invoke_url="https://a/v1, https://b/v1/chat/completions").invoke_url
        == "https://a/v1/chat/completions,https://b/v1/chat/completions"
    )
    # None/empty pass through so the operator can use its own default endpoint.
    assert AgenticRetrievalConfig(llm_model="m", invoke_url=None).invoke_url is None
    assert AgenticRetrievalConfig(llm_model="m", invoke_url="").invoke_url == ""
