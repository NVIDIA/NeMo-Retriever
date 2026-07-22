# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
from typing import Any
from unittest.mock import MagicMock


def test_resolve_agent_llm_profile_aliases() -> None:
    from nemo_retriever.models.local.agent_llm import resolve_agent_llm_model_name

    assert resolve_agent_llm_model_name("nemotron-8b") == "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    assert resolve_agent_llm_model_name("nemotron-super-49b") == "nvidia/Llama-3_3-Nemotron-Super-49B-v1"


def test_local_agent_llm_config_carries_vllm_resource_options() -> None:
    from nemo_retriever.models.local.agent_llm import LocalAgentLLMConfig

    cfg = LocalAgentLLMConfig(
        model_path="nemotron-8b",
        hf_cache_dir="/tmp/hf",
        gpu_memory_utilization=0.7,
        tensor_parallel_size=2,
        max_model_len=8192,
        max_num_seqs=4,
    )

    assert cfg.model_path == "nemotron-8b"
    assert cfg.hf_cache_dir == "/tmp/hf"
    assert cfg.gpu_memory_utilization == 0.7
    assert cfg.tensor_parallel_size == 2
    assert cfg.max_model_len == 8192
    assert cfg.max_num_seqs == 4


def test_vllm_agent_llm_rejects_unsupported_profile_before_vllm_import() -> None:
    import pytest

    from nemo_retriever.models.local.agent_llm import LocalAgentLLMConfig, VLLMAgentChatLLM

    with pytest.raises(ValueError, match="Unsupported local agent LLM model"):
        VLLMAgentChatLLM(LocalAgentLLMConfig(model_path="mistral-7b"))


def test_parse_json_tool_call_output() -> None:
    from nemo_retriever.models.local.agent_llm import parse_tool_calls_from_text

    calls = parse_tool_calls_from_text(json.dumps([{"name": "retrieve", "arguments": {"query": "monetary policy"}}]))

    assert calls == [
        {
            "id": calls[0]["id"],
            "type": "function",
            "function": {"name": "retrieve", "arguments": '{"query": "monetary policy"}'},
        }
    ]


def test_parse_openai_style_tool_call_output() -> None:
    from nemo_retriever.models.local.agent_llm import parse_tool_calls_from_text

    calls = parse_tool_calls_from_text(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "final_results", "arguments": json.dumps({"doc_ids": ["d1"]})},
                    }
                ]
            }
        )
    )

    assert calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "final_results", "arguments": '{"doc_ids": ["d1"]}'},
        }
    ]


def test_parse_tool_call_output_from_code_fence() -> None:
    from nemo_retriever.models.local.agent_llm import parse_tool_calls_from_text

    calls = parse_tool_calls_from_text('```json\n[{"name": "think", "arguments": {"thought": "compare docs"}}]\n```')

    assert calls[0]["function"]["name"] == "think"
    assert json.loads(calls[0]["function"]["arguments"]) == {"thought": "compare docs"}


def test_parse_tool_call_output_ignores_echoed_tool_schema() -> None:
    from nemo_retriever.models.local.agent_llm import parse_tool_calls_from_text

    echoed_schema = json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "retrieve",
                    "description": "Retrieve documents.",
                    "parameters": {"type": "object"},
                },
            }
        ]
    )

    assert parse_tool_calls_from_text(echoed_schema) == []


def test_parse_plain_text_returns_no_tool_calls() -> None:
    from nemo_retriever.models.local.agent_llm import parse_tool_calls_from_text

    assert parse_tool_calls_from_text("I should search again") == []


def test_collapse_parallel_tool_results_for_local_chat_template() -> None:
    from nemo_retriever.models.local.agent_llm import _collapse_consecutive_tool_messages

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "retrieve", "arguments": "{}"},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "final_results", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "Retrieved 3 documents."},
        {"role": "tool", "tool_call_id": "call_b", "content": "Error: doc_ids must be a list."},
        {"role": "assistant", "content": "next"},
    ]

    collapsed = _collapse_consecutive_tool_messages(messages)

    assert [message["role"] for message in collapsed] == ["system", "user", "assistant", "tool", "assistant"]
    assert "Tool result for retrieve (call_a):" in collapsed[3]["content"]
    assert "Retrieved 3 documents." in collapsed[3]["content"]
    assert "Tool result for final_results (call_b):" in collapsed[3]["content"]
    assert "Error: doc_ids must be a list." in collapsed[3]["content"]


def test_normalize_messages_serializes_assistant_tool_calls_for_local_templates() -> None:
    from nemo_retriever.models.local.agent_llm import VLLMAgentChatLLM

    llm = VLLMAgentChatLLM.__new__(VLLMAgentChatLLM)

    normalized = llm._normalize_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "retrieve", "arguments": '{"query": "inflation"}'},
                    }
                ],
            }
        ]
    )

    assert normalized[0]["tool_calls"][0]["function"]["name"] == "retrieve"
    assert normalized[0]["content"].startswith("Assistant tool calls:")
    assert "inflation" in normalized[0]["content"]


def test_malformed_string_tool_arguments_remain_malformed() -> None:
    from nemo_retriever.models.local.agent_llm import parse_tool_calls_from_text

    calls = parse_tool_calls_from_text(json.dumps([{"name": "retrieve", "arguments": "query=inflation"}]))

    assert calls[0]["function"]["arguments"] == "query=inflation"


def test_vllm_agent_llm_unload_releases_engine() -> None:
    import pytest

    from nemo_retriever.models.local.agent_llm import VLLMAgentChatLLM

    engine_core = MagicMock()
    llm_engine = MagicMock()
    llm_engine.engine_core = engine_core
    vllm_llm = MagicMock()
    vllm_llm.llm_engine = llm_engine

    llm = VLLMAgentChatLLM.__new__(VLLMAgentChatLLM)
    llm._llm = vllm_llm
    llm._lock = threading.Lock()

    llm.unload()

    engine_core.shutdown.assert_called_once_with(timeout=30.0)
    assert llm._llm is None
    with pytest.raises(RuntimeError, match="unloaded"):
        llm._require_loaded()


def test_vllm_agent_llm_unload_falls_back_when_shutdown_rejects_timeout() -> None:
    from nemo_retriever.models.local.agent_llm import VLLMAgentChatLLM

    calls: list[dict[str, Any]] = []

    def shutdown(**kwargs: Any) -> None:
        calls.append(kwargs)
        if "timeout" in kwargs:
            raise TypeError("shutdown() got an unexpected keyword argument 'timeout'")

    engine_core = MagicMock()
    engine_core.shutdown.side_effect = shutdown
    llm_engine = MagicMock()
    llm_engine.engine_core = engine_core
    vllm_llm = MagicMock()
    vllm_llm.llm_engine = llm_engine

    llm = VLLMAgentChatLLM.__new__(VLLMAgentChatLLM)
    llm._llm = vllm_llm
    llm._lock = threading.Lock()

    llm.unload()

    assert calls == [{"timeout": 30.0}, {}]
    assert llm._llm is None


def test_create_vllm_agent_chat_llm_returns_fresh_instance() -> None:
    from nemo_retriever.models.local import agent_llm as agent_llm_mod
    from nemo_retriever.models.local.agent_llm import LocalAgentLLMConfig, create_vllm_agent_chat_llm

    created: list[Any] = []

    def fake_ctor(config: LocalAgentLLMConfig) -> Any:
        obj = MagicMock(name="VLLMAgentChatLLM")
        obj.config = config
        created.append(obj)
        return obj

    original = agent_llm_mod.VLLMAgentChatLLM
    agent_llm_mod.VLLMAgentChatLLM = fake_ctor  # type: ignore[misc,assignment]
    try:
        first = create_vllm_agent_chat_llm(LocalAgentLLMConfig(model_path="nemotron-8b"))
        second = create_vllm_agent_chat_llm(LocalAgentLLMConfig(model_path="nemotron-8b"))
    finally:
        agent_llm_mod.VLLMAgentChatLLM = original

    assert first is not second
    assert len(created) == 2


def test_agentic_retriever_unload_releases_owned_llm() -> None:
    from unittest.mock import patch

    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    owned = MagicMock()
    with patch("nemo_retriever.query.agentic.Retriever"):
        retriever = AgenticRetriever(AgenticRetrievalConfig(llm_model="nemotron-8b"))
        retriever._chat_completion_fn = owned
        retriever.unload()

    assert retriever._chat_completion_fn is None
    owned.unload.assert_called_once_with()


def test_agentic_retriever_unload_noop_when_no_local_llm() -> None:
    from unittest.mock import patch

    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    with patch("nemo_retriever.query.agentic.Retriever"):
        retriever = AgenticRetriever(
            AgenticRetrievalConfig(
                llm_model="remote-model",
                invoke_url="http://localhost/v1/chat/completions",
            )
        )
        retriever.unload()

    assert retriever._chat_completion_fn is None
