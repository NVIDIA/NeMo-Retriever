# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Answer mode: the ``log_answer`` end tool and the ``AgentRunResult.answer`` it fills.

Run with:
    cd nemo_retriever && uv run pytest tests/test_nemo_agent_answer_mode.py -v
"""

from __future__ import annotations

import json

import pytest

from nemo_retriever._agentic.nemo_agent import (
    Agent,
    AgentConfig,
    LogAnswer,
    ToolError,
    create_retrieve_tool,
)
from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config

_ANSWER = "The capacity is 42 GB [doc_1]."


def _tool_call_response(name: str, args: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-end",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _run(config: AgentConfig, end_call: str, end_args: dict, seen: list | None = None):
    """Run one agent step whose only LLM reply is the given end-tool call."""

    def completion(**kwargs):
        if seen is not None:
            seen.append(kwargs)
        return _tool_call_response(end_call, end_args)

    agent = Agent(
        config=config,
        llm=create_llm(create_llm_config("callable", model="test-model"), completion_fn=completion),
        retrieve_tool=create_retrieve_tool("default", lambda _query, _top_k: []),
    )
    return agent.run_sync("How much capacity?", query_id="q1")


def _offered_tool_names(call_kwargs: dict) -> set:
    return {spec["function"]["name"] for spec in call_kwargs["tools"]}


class TestAnswerModeRun:
    """End-to-end: an answer-mode agent ends via ``log_answer``."""

    def test_answer_mode_offers_log_answer_instead_of_final_results(self):
        seen: list = []
        _run(
            AgentConfig(mode="answer", user_msg_type="simple", on_error="never_raise"),
            "log_answer",
            {"answer": _ANSWER, "message": "Found it."},
            seen=seen,
        )
        names = _offered_tool_names(seen[0])
        assert "log_answer" in names
        assert "final_results" not in names

    def test_answer_is_exposed_on_the_result(self):
        result = _run(
            AgentConfig(mode="answer", user_msg_type="simple", on_error="never_raise"),
            "log_answer",
            {"answer": _ANSWER, "message": "Found it."},
        )
        assert result.succeeded
        assert result.answer == _ANSWER
        assert result.end_payload == {"answer": _ANSWER, "message": "Found it."}
        assert result.final_doc_ids == []

    def test_select_mode_leaves_answer_unset(self):
        result = _run(
            AgentConfig(mode="select", target_top_k=1, user_msg_type="simple", on_error="never_raise"),
            "final_results",
            {"doc_ids": ["d1"], "search_successful": "true", "message": "Selected d1."},
        )
        assert result.succeeded
        assert result.answer is None
        assert result.final_doc_ids == ["d1"]


class TestLogAnswerSpec:
    """The tool spec the model sees."""

    def test_answer_and_message_are_required_by_default(self):
        spec = LogAnswer().spec["function"]
        assert spec["name"] == "log_answer"
        assert set(spec["parameters"]["required"]) == {"answer", "message"}

    def test_message_is_dropped_when_disabled(self):
        spec = LogAnswer(include_msg=False).spec["function"]
        assert spec["parameters"]["required"] == ["answer"]
        assert "message" not in spec["parameters"]["properties"]


class TestLogAnswerValidation:
    """``_validate_payload`` is the contract; ``try_end`` is how the loop reaches it."""

    def test_valid_call_returns_the_payload(self):
        assert LogAnswer()._validate_payload(answer=_ANSWER, message="ok") == {
            "answer": _ANSWER,
            "message": "ok",
        }

    def test_message_is_omitted_when_not_supplied(self):
        assert LogAnswer(include_msg=False)._validate_payload(answer=_ANSWER) == {"answer": _ANSWER}

    @pytest.mark.parametrize("answer", ["", "   "])
    def test_blank_answer_is_recoverable_error(self, answer):
        with pytest.raises(ToolError):
            LogAnswer(include_msg=False)._validate_payload(answer=answer)

    def test_non_string_answer_is_a_type_error(self):
        with pytest.raises(TypeError):
            LogAnswer(include_msg=False)._validate_payload(answer=["not", "a", "string"])

    def test_missing_message_is_a_type_error(self):
        with pytest.raises(TypeError):
            LogAnswer()._validate_payload(answer=_ANSWER)

    def test_try_end_reports_invalid_calls_without_raising(self):
        payload, text = LogAnswer(include_msg=False).try_end(answer="")
        assert payload is None
        assert "log_answer" in text


class TestLogAnswerSalvage:
    """A failed run still surfaces the model's last answer attempt."""

    def test_answer_is_salvaged_from_an_invalid_call(self):
        assert LogAnswer().salvage_payload({"answer": _ANSWER, "message": "m"}) == {
            "answer": _ANSWER,
            "message": "m",
        }

    def test_message_alone_salvages_nothing(self):
        assert LogAnswer().salvage_payload({"message": "no answer here"}) is None
