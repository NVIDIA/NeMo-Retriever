# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Payload shaping for ``invoke_chat_completion_step``.

This is the completion callable the agent operators inject for remote runs, so
its request body is now an agent-facing contract. The ``temperature`` handling in
particular is load-bearing: the agent forwards ``None`` to mean *unset*, and
sending ``0.0`` instead would silently force greedy decoding on every remote run.
"""

from __future__ import annotations

import pytest

from nemo_retriever.models.nim import chat_completions

_URL = "https://endpoint.invalid/v1/chat/completions"
_MESSAGES = [{"role": "user", "content": "q"}]


@pytest.fixture
def sent(monkeypatch):
    """Capture the kwargs handed to the underlying poster."""
    captured: dict = {}

    def fake_post(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(chat_completions, "_post_with_retries", fake_post)
    return captured


class TestTemperature:
    def test_none_omits_the_field_entirely(self, sent):
        # "Unset" must mean the endpoint/model default applies. Sending 0.0 would
        # force greedy decoding the caller never asked for.
        chat_completions.invoke_chat_completion_step(invoke_url=_URL, messages=_MESSAGES, temperature=None)
        assert "temperature" not in sent["payload"]

    def test_explicit_zero_is_still_sent(self, sent):
        # 0.0 is a real, meaningful value (greedy) and must not be confused with unset.
        chat_completions.invoke_chat_completion_step(invoke_url=_URL, messages=_MESSAGES, temperature=0.0)
        assert sent["payload"]["temperature"] == 0.0

    def test_default_is_unchanged_for_existing_callers(self, sent):
        chat_completions.invoke_chat_completion_step(invoke_url=_URL, messages=_MESSAGES)
        assert sent["payload"]["temperature"] == 0.0

    def test_extra_body_can_still_override(self, sent):
        chat_completions.invoke_chat_completion_step(
            invoke_url=_URL, messages=_MESSAGES, temperature=None, extra_body={"temperature": 0.7}
        )
        assert sent["payload"]["temperature"] == 0.7


class TestPayloadShape:
    def test_tools_and_tool_choice_travel_together(self, sent):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        chat_completions.invoke_chat_completion_step(
            invoke_url=_URL, messages=_MESSAGES, tools=tools, tool_choice="auto"
        )
        assert sent["payload"]["tools"] == tools
        assert sent["payload"]["tool_choice"] == "auto"

    def test_tool_choice_is_dropped_without_tools(self, sent):
        # Suppressing tools is expressed by omitting them, so a stray tool_choice
        # on a tool-less request would be meaningless at best.
        chat_completions.invoke_chat_completion_step(invoke_url=_URL, messages=_MESSAGES, tool_choice="none")
        assert "tool_choice" not in sent["payload"]
        assert "tools" not in sent["payload"]

    def test_max_tokens_omitted_when_none(self, sent):
        chat_completions.invoke_chat_completion_step(invoke_url=_URL, messages=_MESSAGES, max_tokens=None)
        assert "max_tokens" not in sent["payload"]

    def test_extra_body_is_merged_top_level(self, sent):
        chat_completions.invoke_chat_completion_step(
            invoke_url=_URL, messages=_MESSAGES, extra_body={"reasoning_effort": "high"}
        )
        assert sent["payload"]["reasoning_effort"] == "high"

    def test_api_key_becomes_a_bearer_header(self, sent):
        chat_completions.invoke_chat_completion_step(invoke_url=_URL, messages=_MESSAGES, api_key=" sk-live ")
        assert sent["headers"]["Authorization"] == "Bearer sk-live"

    def test_no_authorization_header_without_a_key(self, sent):
        chat_completions.invoke_chat_completion_step(invoke_url=_URL, messages=_MESSAGES, api_key=None)
        assert "Authorization" not in sent["headers"]

    def test_retry_budget_is_forwarded(self, sent):
        chat_completions.invoke_chat_completion_step(
            invoke_url=_URL, messages=_MESSAGES, timeout_s=30.0, max_retries=2, max_429_retries=4
        )
        assert (sent["timeout_s"], sent["max_retries"], sent["max_429_retries"]) == (30.0, 2, 4)

    def test_first_url_of_a_comma_separated_list_is_used(self, sent):
        chat_completions.invoke_chat_completion_step(
            invoke_url=f"{_URL},https://second.invalid/v1/chat/completions", messages=_MESSAGES
        )
        assert sent["invoke_url"] == _URL

    def test_empty_invoke_url_is_rejected(self):
        with pytest.raises(ValueError, match="invoke_url is required"):
            chat_completions.invoke_chat_completion_step(invoke_url="", messages=_MESSAGES)
