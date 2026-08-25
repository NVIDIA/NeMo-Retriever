# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_tool_call_response(
    fn_name: str,
    fn_args: dict,
    tc_id: str = "call_1",
    usage: dict | None = None,
) -> dict:
    response = {
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
    if usage is not None:
        response["usage"] = usage
    return response


class FakeRetriever:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.graph = kwargs.get("graph")
        self.top_k = int(kwargs.get("top_k", 10))
        self.query_calls = []

    def query(self, query: str, *, top_k: int | None = None, candidate_k: int | None = None):
        self.query_calls.append({"query": query, "top_k": top_k, "candidate_k": candidate_k})
        if self.graph is not None:
            return self.queries([query], top_k=top_k)[0]
        _ = query
        hits = [
            {
                "source": {"source_id": "/tmp/clip.wav"},
                "source_id": "/tmp/doc.pdf",
                "page_number": 1,
                "pdf_page": "doc_1",
                "metadata": {"segment_start_seconds": 1.0, "segment_end_seconds": 3.0},
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


def _dispatch_chat_fn(react_response, selection_response):
    """Fake in-process completion callable shared by both agents.

    The ReAct and selection agents share one injected ``chat_completion_fn``, so
    the fake returns the selection response whenever the selection tool is offered
    and the ReAct response otherwise.
    """

    def fn(**kwargs):
        tool_names = {(tool.get("function") or {}).get("name") for tool in (kwargs.get("tools") or [])}
        if "log_selected_documents" in tool_names:
            return selection_response
        return react_response

    return fn


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_runs_graph_with_wrapped_retriever():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    final_ids = ["doc_1"] + [f"extra_{i}" for i in range(9)]
    chat_fn = _dispatch_chat_fn(
        _make_tool_call_response(
            "final_results", {"doc_ids": final_ids, "message": "done", "search_successful": "true"}
        ),
        _make_tool_call_response("log_selected_documents", {"doc_ids": ["doc_1"], "message": "doc_1 is best"}),
    )

    # In-process path -> callable client backend; inject the fake completion fn.
    cfg = AgenticRetrievalConfig(llm_model="nemotron-8b")
    with patch("nemo_retriever.query.agentic._build_agent_chat_completion_fn", return_value=chat_fn):
        retriever = AgenticRetriever(cfg, match_mode="pdf_page")
        result = retriever.retrieve(["0"], ["find doc"])

    assert "local_ingest_embed_backend" not in retriever._retriever.kwargs["embed_kwargs"]
    assert list(result.columns) == ["query_id", "doc_id", "rank", "message", "result_source", "hit"]
    assert result["query_id"].tolist() == ["0"] * 10
    assert result["doc_id"].tolist()[0] == "doc_1"
    assert result["rank"].tolist() == list(range(1, 11))


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_returns_and_pops_query_usage():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    usage = {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
    response = _make_tool_call_response(
        "final_results",
        {
            "doc_ids": ["doc_1"] + [f"extra_{i}" for i in range(9)],
            "message": "done",
            "search_successful": "true",
        },
        usage=usage,
    )
    cfg = AgenticRetrievalConfig(llm_model="nemotron-8b")
    with patch("nemo_retriever.query.agentic._build_agent_chat_completion_fn", return_value=lambda **_: response):
        result = AgenticRetriever(cfg, match_mode="pdf_page").retrieve_with_usage(["customer-q"], ["find doc"])

    assert result.usage == {"customer-q": {"main_agent": usage}}
    assert result.documents["query_id"].tolist() == ["customer-q"] * 10


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_isolates_usage_for_concurrent_queries():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    response = _make_tool_call_response(
        "final_results",
        {
            "doc_ids": ["doc_1"] + [f"extra_{i}" for i in range(9)],
            "message": "done",
            "search_successful": "true",
        },
        usage=usage,
    )
    cfg = AgenticRetrievalConfig(llm_model="nemotron-8b", num_concurrent=2)
    with patch("nemo_retriever.query.agentic._build_agent_chat_completion_fn", return_value=lambda **_: response):
        result = AgenticRetriever(cfg, match_mode="pdf_page").retrieve_with_usage(
            ["customer-a", "customer-b"],
            ["find a", "find b"],
        )

    assert result.usage == {
        "customer-a": {"main_agent": usage},
        "customer-b": {"main_agent": usage},
    }


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_rehydrates_only_retrieved_documents():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    # doc_1 comes back from the retrieve hop; the extras are ids the agent named
    # without ever retrieving them, so there is nothing to rehydrate for them.
    final_ids = ["doc_1"] + [f"extra_{i}" for i in range(9)]
    chat_fn = _dispatch_chat_fn(
        _make_tool_call_response(
            "final_results", {"doc_ids": final_ids, "message": "done", "search_successful": "true"}
        ),
        _make_tool_call_response("log_selected_documents", {"doc_ids": ["doc_1"], "message": "doc_1 is best"}),
    )

    cfg = AgenticRetrievalConfig(llm_model="nemotron-8b")
    with patch("nemo_retriever.query.agentic._build_agent_chat_completion_fn", return_value=chat_fn):
        result = AgenticRetriever(cfg, match_mode="pdf_page").retrieve(["0"], ["find doc"])

    hits = dict(zip(result["doc_id"], result["hit"]))
    assert hits["doc_1"]["text"] == "matching document"
    assert hits["doc_1"]["source_id"] == "/tmp/doc.pdf"
    assert hits["doc_1"]["page_number"] == 1
    assert hits["doc_1"]["_score"] == 0.9
    assert hits["extra_0"] == {}


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_caches_untruncated_text_for_rehydration():
    """Truncation bounds what the agent sees; rehydration returns the stored text."""
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    cfg = AgenticRetrievalConfig(llm_model="m", invoke_url=_REMOTE_URL, text_truncation=5)
    retriever = AgenticRetriever(cfg, match_mode="pdf_page")

    docs = retriever._retrieve_for_agent("find doc", 10, query_id="q1")

    assert docs[0]["text"] == "match"
    assert retriever._hit_cache[("q1", "doc_1")]["text"] == "matching document"


def test_rehydrated_agentic_hit_layers_annotations_onto_classic_fields():
    from nemo_retriever.query.agentic import rehydrated_agentic_hit

    hit = rehydrated_agentic_hit(
        {"text": "body", "source": "/tmp/doc.pdf", "page_number": 3},
        doc_id="doc_3",
        rank=1,
        result_source="selection_agent",
    )

    assert hit == {
        "text": "body",
        "source": "/tmp/doc.pdf",
        "page_number": 3,
        "doc_id": "doc_3",
        "rank": 1,
        "result_source": "selection_agent",
    }


def test_rehydrated_agentic_hit_without_captured_metadata():
    from nemo_retriever.query.agentic import rehydrated_agentic_hit

    assert rehydrated_agentic_hit({}, doc_id="doc_3", rank=2, result_source="final_results") == {
        "doc_id": "doc_3",
        "rank": 2,
        "result_source": "final_results",
    }


def test_rehydration_miss_severity_depends_on_selecting_stage(caplog):
    """A stage that can only rank retrieved candidates must never miss the cache."""
    from nemo_retriever.query.agentic import _rehydrate_selected_hits

    result = pd.DataFrame(
        {
            "doc_id": ["seen", "unretrieved_candidate", "invented"],
            "rank": [1, 2, 3],
            "result_source": ["selection_agent", "selection_agent", "final_results"],
            "_retrieval_query_id": ["q1", "q1", "q1"],
        }
    )

    with caplog.at_level(logging.WARNING, logger="nemo_retriever.query.agentic"):
        rehydrated = _rehydrate_selected_hits(result, {("q1", "seen"): {"text": "body"}})

    assert rehydrated["hit"].tolist() == [{"text": "body"}, {}, {}]
    assert "_retrieval_query_id" not in rehydrated.columns
    levels = {record.levelno: record.getMessage() for record in caplog.records}
    assert "unretrieved_candidate" in levels[logging.ERROR]
    assert "invented" in levels[logging.WARNING]


def test_rehydration_isolates_same_doc_id_per_query():
    from nemo_retriever.query.agentic import _rehydrate_selected_hits

    result = pd.DataFrame(
        {
            "query_id": ["customer-query-a", "customer-query-b"],
            "doc_id": ["shared_doc", "shared_doc"],
            "rank": [1, 1],
            "result_source": ["selection_agent", "selection_agent"],
            "_retrieval_query_id": ["0", "1"],
        }
    )
    cache = {
        ("0", "shared_doc"): {"text": "query a hit", "_score": 0.9},
        ("1", "shared_doc"): {"text": "query b hit", "_score": 0.4},
    }

    rehydrated = _rehydrate_selected_hits(result, cache)

    assert rehydrated["hit"].tolist() == [
        {"text": "query a hit", "_score": 0.9},
        {"text": "query b hit", "_score": 0.4},
    ]
    assert "_retrieval_query_id" not in rehydrated.columns


def test_agentic_query_documents_returns_classic_hit_fields_with_annotations():
    from nemo_retriever.query.options import QueryAgenticOptions, QueryRequest, QueryRetrievalOptions
    from nemo_retriever.query.workflow import agentic_query_documents

    retriever = MagicMock()
    retriever.retrieve.return_value = pd.DataFrame(
        {
            "query_id": ["0", "0"],
            "doc_id": ["doc_1", "invented"],
            "rank": [1, 2],
            "message": ["", ""],
            "result_source": ["selection_agent", "final_results"],
            "hit": [{"text": "body", "source": "/tmp/doc.pdf", "page_number": 1}, {}],
        }
    )
    request = QueryRequest(
        query="find doc",
        retrieval=QueryRetrievalOptions(top_k=2),
        agentic=QueryAgenticOptions(enabled=True, llm_model="m", invoke_url=_REMOTE_URL),
    )

    with patch("nemo_retriever.query.workflow.build_agentic_retriever", return_value=retriever):
        ranked = agentic_query_documents(request)

    assert ranked == [
        {
            "text": "body",
            "source": "/tmp/doc.pdf",
            "page_number": 1,
            "doc_id": "doc_1",
            "rank": 1,
            "result_source": "selection_agent",
        },
        {"doc_id": "invented", "rank": 2, "result_source": "final_results"},
    ]
    retriever.unload.assert_called_once()


def test_agentic_query_documents_with_metadata_normalizes_usage():
    from nemo_retriever.query.agentic import AgenticRetrieveResult
    from nemo_retriever.query.options import QueryAgenticOptions, QueryRequest, QueryRetrievalOptions
    from nemo_retriever.query.workflow import agentic_query_documents_with_metadata

    retriever = MagicMock()
    retriever.retrieve_with_usage.return_value = AgenticRetrieveResult(
        documents=pd.DataFrame(
            {
                "query_id": ["0"],
                "doc_id": ["doc_1"],
                "rank": [1],
                "result_source": ["final_results"],
                "hit": [{"text": "body"}],
            }
        ),
        usage={
            "0": {
                "main_agent": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                "top1_agent": {"input_tokens": 3, "output_tokens": 2},
            }
        },
    )
    request = QueryRequest(
        query="find doc",
        retrieval=QueryRetrievalOptions(top_k=1),
        agentic=QueryAgenticOptions(enabled=True, llm_model="m", invoke_url=_REMOTE_URL),
    )

    with patch("nemo_retriever.query.workflow.build_agentic_retriever", return_value=retriever):
        result = agentic_query_documents_with_metadata(request)

    assert result.usage["input_tokens"] == 14
    assert result.usage["output_tokens"] == 6
    assert result.usage["total_tokens"] == 20
    assert set(result.usage["stages"]) == {"main_agent", "top1_agent"}
    retriever.unload.assert_called_once()


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_honors_top_k():
    """cfg.top_k drives the pipeline output count, not the hardcoded default of 10."""
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    final_ids = ["doc_1"] + [f"extra_{i}" for i in range(4)]  # exactly 5
    chat_fn = _dispatch_chat_fn(
        _make_tool_call_response(
            "final_results", {"doc_ids": final_ids, "message": "done", "search_successful": "true"}
        ),
        _make_tool_call_response("log_selected_documents", {"doc_ids": ["doc_1"], "message": "doc_1 is best"}),
    )

    cfg = AgenticRetrievalConfig(llm_model="nemotron-8b", top_k=5)
    with patch("nemo_retriever.query.agentic._build_agent_chat_completion_fn", return_value=chat_fn):
        result = AgenticRetriever(cfg, match_mode="pdf_page").retrieve(["0"], ["find doc"])

    assert result["rank"].tolist() == list(range(1, 6))  # 5 rows, honoring top_k=5


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_forwards_candidate_k_per_hop():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    cfg = AgenticRetrievalConfig(llm_model="m", invoke_url=_REMOTE_URL, top_k=10, candidate_k=20)
    retriever = AgenticRetriever(cfg, match_mode="pdf_page")

    retriever._retrieve_for_agent("first", 10, query_id="q1")
    retriever._retrieve_for_agent("later", 25, query_id="q1")

    assert retriever._retriever.query_calls == [
        {"query": "first", "top_k": 10, "candidate_k": 20},
        {"query": "later", "top_k": 25, "candidate_k": 25},
    ]


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_run_agentic_audio_recall_evaluation_computes_metrics(tmp_path):
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
    chat_fn = _dispatch_chat_fn(
        _make_tool_call_response(
            "final_results", {"doc_ids": final_ids, "message": "done", "search_successful": "true"}
        ),
        _make_tool_call_response("log_selected_documents", {"doc_ids": [audio_doc_id], "message": "clip is best"}),
    )

    cfg = AgenticRetrievalConfig(llm_model="nemotron-8b")
    with patch("nemo_retriever.query.agentic._build_agent_chat_completion_fn", return_value=chat_fn):
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


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_forwards_reranker_endpoint_as_rerank_invoke_url():
    """A configured reranker endpoint must reach the remote rerank variant.

    ``NemotronRerankActor`` dispatches on ``rerank_invoke_url``; any other key
    leaves the URL unused and loads the reranker locally instead.
    """
    from nemo_retriever.operators.rerank import NemotronRerankActor
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    cfg = AgenticRetrievalConfig(
        llm_model="test-model",
        invoke_url=_REMOTE_URL,
        reranker="nvidia/llama-nemotron-rerank-vl-1b-v2",
        reranker_endpoint="http://localhost:8015",
    )
    rerank_kwargs = AgenticRetriever(cfg, match_mode="pdf_page")._retriever.kwargs["rerank_kwargs"]

    assert rerank_kwargs["rerank_invoke_url"] == "http://localhost:8015"
    assert "invoke_url" not in rerank_kwargs
    assert NemotronRerankActor.prefers_cpu_variant(rerank_kwargs) is True


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_agentic_retriever_without_reranker_endpoint_uses_local_variant():
    from nemo_retriever.operators.rerank import NemotronRerankActor
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    cfg = AgenticRetrievalConfig(
        llm_model="test-model",
        invoke_url=_REMOTE_URL,
        reranker="nvidia/llama-nemotron-rerank-vl-1b-v2",
        reranker_endpoint="   ",
    )
    rerank_kwargs = AgenticRetriever(cfg, match_mode="pdf_page")._retriever.kwargs["rerank_kwargs"]

    assert rerank_kwargs["rerank_invoke_url"] is None
    assert NemotronRerankActor.prefers_cpu_variant(rerank_kwargs) is False


@patch("nemo_retriever.query.agentic.Retriever", FakeRetriever)
def test_run_agentic_beir_evaluation_loads_queries_and_qrels():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, run_agentic_beir_evaluation
    from nemo_retriever.tools.recall.beir import BeirDataset

    final_ids = ["doc"] + [f"extra_{i}" for i in range(9)]
    chat_fn = _dispatch_chat_fn(
        _make_tool_call_response(
            "final_results", {"doc_ids": final_ids, "message": "done", "search_successful": "true"}
        ),
        _make_tool_call_response("log_selected_documents", {"doc_ids": ["doc"], "message": "doc is best"}),
    )

    beir_dataset = BeirDataset(
        dataset_name="vidore_v3_finance_en",
        query_ids=["q1"],
        queries=["find doc"],
        qrels={"q1": {"doc": 1}},
    )
    cfg = AgenticRetrievalConfig(llm_model="nemotron-8b")

    with patch("nemo_retriever.query.agentic._build_agent_chat_completion_fn", return_value=chat_fn), patch(
        "nemo_retriever.query.agentic.load_beir_dataset", return_value=beir_dataset
    ) as mock_loader:
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


_REMOTE_URL = "http://localhost/v1/chat/completions"


def test_agentic_config_requires_llm_model_on_remote_path():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    # A model is required only on the remote (invoke_url) path; in-process runs
    # default to the local model instead of raising.
    with pytest.raises(ValueError, match="llm_model"):
        AgenticRetrievalConfig(llm_model="", invoke_url=_REMOTE_URL)
    # None must not slip through as the literal string "None".
    with pytest.raises(ValueError, match="llm_model"):
        AgenticRetrievalConfig(llm_model=None, invoke_url=_REMOTE_URL)


def test_agentic_config_defaults_in_process_model_and_client():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    # No invoke_url and no model -> local in-process default with the callable
    # LLM client.
    cfg = AgenticRetrievalConfig(llm_model="")

    assert cfg.llm_backend == "in_process"
    assert cfg.llm_model == "nemotron-8b"
    assert cfg.llm_client == "callable"


def test_agentic_config_rejects_nonpositive_top_k():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="top_k"):
        AgenticRetrievalConfig(llm_model="m", invoke_url=_REMOTE_URL, top_k=0)


def test_agentic_config_rejects_noninteger_top_k():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="top_k must be an integer"):
        AgenticRetrievalConfig(llm_model="m", invoke_url=_REMOTE_URL, top_k=1.5)


def test_agentic_config_rejects_candidate_k_below_top_k():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(
        ValueError,
        match=r"candidate_k \(3\) must be greater than or equal to top_k \(10\)",
    ):
        AgenticRetrievalConfig(llm_model="m", invoke_url=_REMOTE_URL, top_k=10, candidate_k=3)


def test_agentic_config_normalizes_integer_like_values():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    cfg = AgenticRetrievalConfig(
        llm_model="m",
        invoke_url=_REMOTE_URL,
        top_k="5.0",
        temperature="0.25",
    )

    assert cfg.top_k == 5
    assert cfg.temperature == 0.25


def test_agentic_config_allows_none_temperature():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    cfg = AgenticRetrievalConfig(llm_model="m", invoke_url=_REMOTE_URL)

    assert cfg.temperature is None


def test_agentic_config_rejects_nvidia_temperature_above_max():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        AgenticRetrievalConfig(
            llm_model="m",
            invoke_url="https://integrate.api.nvidia.com/v1/chat/completions",
            temperature=1.5,
        )


def test_agentic_config_accepts_in_process_temperature_above_nvidia_limit():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    # In-process uses the OpenAI-compatible bound (2.0), so a value above the
    # hosted-NVIDIA 1.0 cap is accepted.
    cfg = AgenticRetrievalConfig(llm_model="nemotron-8b", temperature=1.5)

    assert cfg.llm_backend == "in_process"
    assert cfg.temperature == pytest.approx(1.5)


def test_agentic_config_rejects_nonfinite_temperature():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="temperature must be finite"):
        AgenticRetrievalConfig(llm_model="m", invoke_url=_REMOTE_URL, temperature=float("nan"))


def test_agentic_config_defaults_client_to_callable():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    # Remote transport (invoke_url set), client unset -> callable default. The
    # same client serves both transports; only the injected completion callable
    # differs.
    cfg = AgenticRetrievalConfig(llm_model="m", invoke_url=_REMOTE_URL)

    assert cfg.llm_backend == "openai_compatible"
    assert cfg.llm_client == "callable"


def test_agentic_config_accepts_and_normalizes_known_client():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    cfg = AgenticRetrievalConfig(llm_model="m", invoke_url=_REMOTE_URL, llm_client=" litellm ")

    assert cfg.llm_client == "litellm"


def test_agentic_config_defaults_callable_client_in_process():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    # In-process transport, client unset or explicitly callable -> callable.
    assert AgenticRetrievalConfig(llm_model="nemotron-8b").llm_client == "callable"
    assert AgenticRetrievalConfig(llm_model="nemotron-8b", llm_client="callable").llm_client == "callable"


def test_agentic_config_rejects_remote_client_without_invoke_url():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    # A non-callable client is a remote client and needs invoke_url; no silent
    # override to callable.
    with pytest.raises(ValueError, match="in-process agentic runs use the 'callable' LLM client"):
        AgenticRetrievalConfig(llm_model="nemotron-8b", llm_client="litellm")


def test_agentic_config_accepts_callable_client_with_invoke_url():
    # `callable` spans both transports: it wraps the in-process engine locally and
    # the shared HTTP client remotely, so pairing it with an invoke_url is valid.
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    cfg = AgenticRetrievalConfig(llm_model="m", invoke_url=_REMOTE_URL, llm_client="callable")

    assert cfg.llm_backend == "openai_compatible"
    assert cfg.llm_client == "callable"


def test_agentic_config_rejects_unknown_client():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="llm_client must be one of"):
        AgenticRetrievalConfig(llm_model="m", invoke_url=_REMOTE_URL, llm_client="bogus")


def test_agentic_config_rejects_invalid_local_llm_backend():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    with pytest.raises(ValueError, match="local_llm_backend"):
        AgenticRetrievalConfig(llm_model="nemotron-8b", local_llm_backend="hf")


def test_agentic_config_validates_local_vllm_knobs():
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    cfg = AgenticRetrievalConfig(
        llm_model="nemotron-8b",
        local_gpu_memory_utilization="0.6",
        local_tensor_parallel_size="2.0",
        local_max_model_len="8192",
        local_max_num_seqs="4.0",
    )

    assert cfg.local_gpu_memory_utilization == pytest.approx(0.6)
    assert cfg.local_tensor_parallel_size == 2
    assert cfg.local_max_model_len == 8192
    assert cfg.local_max_num_seqs == 4


def test_agentic_config_passes_tensor_parallel_size_to_local_llm():
    from nemo_retriever.query.agentic import (
        AgenticRetrievalConfig,
        _build_agent_chat_completion_fn,
    )

    cfg = AgenticRetrievalConfig(
        llm_model="super-49b",
        local_tensor_parallel_size=2,
    )

    with patch(
        "nemo_retriever.models.create_local_agent_llm",
        return_value=object(),
    ) as create_local_llm:
        _build_agent_chat_completion_fn(cfg)

    create_local_llm.assert_called_once_with(
        "super-49b",
        backend="vllm",
        hf_cache_dir=None,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=2,
        max_model_len=None,
        max_num_seqs=None,
    )
