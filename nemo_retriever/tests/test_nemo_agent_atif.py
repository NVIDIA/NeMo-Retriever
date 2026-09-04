# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pandas as pd

from nemo_retriever._agentic.nemo_agent import Agent, AgentConfig, create_retrieve_tool
from nemo_retriever._agentic.nemo_agent.atif import (
    ATIF_SCHEMA_VERSION,
    MAX_OBSERVATION_CHARS,
    build_atif_trajectory,
    persist_atif_trajectory,
)
from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config


def test_agent_run_builds_atif_steps_and_token_metrics():
    def completion(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I found the relevant document.",
                        "tool_calls": [
                            {
                                "id": "call-final",
                                "type": "function",
                                "function": {
                                    "name": "final_results",
                                    "arguments": json.dumps(
                                        {
                                            "doc_ids": ["d1"],
                                            "search_successful": "true",
                                            "message": "Selected d1.",
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "input_tokens_details": {"cached_tokens": 4},
            },
        }

    llm = create_llm(create_llm_config("callable", model="test-model"), completion_fn=completion)
    retrieve = create_retrieve_tool("default", lambda _query, _top_k: [])
    agent = Agent(
        config=AgentConfig(target_top_k=1, user_msg_type="simple", on_error="never_raise"),
        llm=llm,
        retrieve_tool=retrieve,
    )

    result = agent.run_sync("Which document?", query_id="query-1")

    assert result.succeeded
    assert result.atif_trace is not None
    trace = result.atif_trace
    assert trace["schema_version"] == ATIF_SCHEMA_VERSION
    assert trace["agent"]["model_name"] == "test-model"
    agent_step = next(step for step in trace["steps"] if step.get("llm_call_count") == 1)
    assert agent_step["tool_calls"][0]["function_name"] == "final_results"
    assert agent_step["observation"]["results"][0]["source_call_id"] == "call-final"
    assert agent_step["metrics"]["prompt_tokens"] == 11
    assert agent_step["metrics"]["completion_tokens"] == 7
    assert agent_step["metrics"]["cached_tokens"] == 4
    assert trace["final_metrics"]["total_prompt_tokens"] == 11
    assert trace["final_metrics"]["total_completion_tokens"] == 7
    assert trace["final_metrics"]["total_cached_tokens"] == 4


def test_bootstrap_retrieval_is_summarized_and_bounded():
    trace = build_atif_trajectory(
        query="q",
        query_id="query-2",
        stage="main_agent",
        model_name="m",
        message_history=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "query with initial results"},
        ],
        llm_records=[],
        retrieval_log=[
            {
                "query_type": "main",
                "tool_name": "retrieve",
                "input": {"query": "q", "top_k": 1},
                "output": [{"id": "d1", "score": 0.9, "text": "x" * (MAX_OBSERVATION_CHARS * 2)}],
            }
        ],
    )

    bootstrap = next(step for step in trace["steps"] if step.get("extra", {}).get("query_type") == "main")
    content = bootstrap["observation"]["results"][0]["content"]
    assert bootstrap["llm_call_count"] == 0
    assert bootstrap["tool_calls"][0]["function_name"] == "retrieve"
    assert "d1" in content
    assert len(content) < MAX_OBSERVATION_CHARS


def test_trace_persists_under_current_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trace = build_atif_trajectory(
        query="q",
        query_id="query/unsafe",
        stage="main_agent",
        model_name="m",
        message_history=[{"role": "user", "content": "q"}],
        llm_records=[],
        retrieval_log=[],
    )

    path = persist_atif_trajectory(trace)

    assert path is not None
    assert path.parent == tmp_path / "agentic-traces"
    assert path.suffix == ".json"
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == ATIF_SCHEMA_VERSION


def test_react_operator_persists_trace_by_default(tmp_path, monkeypatch):
    from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator

    monkeypatch.chdir(tmp_path)

    def completion(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The bootstrap result is relevant.",
                        "tool_calls": [
                            {
                                "id": "call-final",
                                "type": "function",
                                "function": {
                                    "name": "final_results",
                                    "arguments": json.dumps(
                                        {
                                            "doc_ids": ["d1"],
                                            "search_successful": "true",
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    operator = ReActAgentOperator(
        llm_model="test-model",
        retriever_fn=lambda _query, _top_k: [{"id": "d1", "score": 1.0, "text": "relevant"}],
        target_top_k=1,
        backend="callable",
        chat_completion_fn=completion,
    )

    operator.run(pd.DataFrame({"query_id": ["q1"], "query_text": ["find d1"]}))

    paths = list((tmp_path / "agentic-traces").glob("*.json"))
    assert len(paths) == 1
    trace = json.loads(paths[0].read_text(encoding="utf-8"))
    assert trace["agent"]["extra"]["query_id"] == "q1"
    assert trace["agent"]["extra"]["stage"] == "main_agent"
