# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``openai_http`` LLM backend — direct HTTP, no LLM SDK.

Driven with :class:`httpx.MockTransport` (the house pattern, see
``test_service_client_compat``) so no socket is opened and request bodies can be
asserted on directly. The pure helpers are exercised without any transport at
all.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from copy import deepcopy

import httpx
import pytest

from nemo_retriever._agentic.nemo_agent.llm import (
    ContentPolicyError,
    ContextLimitError,
    LLMCallError,
    RateLimitError,
    create_llm,
    create_llm_config,
)
from nemo_retriever._agentic.nemo_agent.llm.openai_http_backend import (
    _REDACTED,
    _classify_error,
    _normalize_finish_reason,
    _parse_retry_after,
    _redact_headers,
    _redact_url,
)

_URL = "https://endpoint.invalid/v1/chat/completions"
_LOGGER = "nemo_retriever._agentic.nemo_agent.llm.openai_http_backend"

_OK_BODY = {
    "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
}
_OK = (200, {"json": _OK_BODY})


def _transport(*specs):
    """Build a handler returning ``specs`` in order; the last one repeats.

    Each spec is ``(status, httpx.Response kwargs)``. Returns
    ``(handler, calls)`` where ``calls`` accumulates the requests seen.
    """
    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        status, kwargs = specs[min(len(calls) - 1, len(specs) - 1)]
        return httpx.Response(status, **kwargs)

    return handle, calls


def _make_backend(handler, **config_kwargs):
    config_kwargs.setdefault("model", "m")
    config_kwargs.setdefault("base_url", _URL)
    return create_llm(
        create_llm_config("openai_http", **config_kwargs),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _body_of(request: httpx.Request) -> dict:
    return json.loads(request.content)


# ----------------------------------------------------------------------
# Pure helpers — no transport.
# ----------------------------------------------------------------------


class TestClassifyError:
    def _classify(self, status, body, headers=None):
        text = body if isinstance(body, str) else json.dumps(body)
        parsed = None if isinstance(body, str) else body
        return _classify_error(status, text, parsed, headers or {}, _URL)

    def test_rate_limit(self):
        err = self._classify(429, {"error": {"message": "slow down"}}, {"Retry-After": "7"})
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 7.0

    def test_context_limit_structured_code(self):
        err = self._classify(400, {"error": {"code": "context_length_exceeded", "message": "nope"}})
        assert isinstance(err, ContextLimitError)

    def test_context_limit_structured_type(self):
        err = self._classify(400, {"error": {"type": "string_above_max_length"}})
        assert isinstance(err, ContextLimitError)

    @pytest.mark.parametrize(
        "prose",
        [
            "This model's maximum context length is 8192 tokens",
            "The input is longer than the maximum model length",
            "Please reduce the length of the messages",
            "max_tokens must be at least 1, got -37",
        ],
    )
    def test_context_limit_prose(self, prose):
        assert isinstance(self._classify(400, {"error": {"message": prose}}), ContextLimitError)

    def test_content_policy_structured(self):
        err = self._classify(400, {"error": {"code": "content_policy_violation"}})
        assert isinstance(err, ContentPolicyError)

    @pytest.mark.parametrize(
        "prose",
        ["blocked by the content filter", "guardrail intervened", "violates our content policy"],
    )
    def test_content_policy_prose(self, prose):
        assert isinstance(self._classify(400, {"error": {"message": prose}}), ContentPolicyError)

    def test_non_json_body_still_classified(self):
        # body_json is None when the error body is not JSON; prose matching runs
        # against the raw text, which is a superset of it.
        err = _classify_error(400, "maximum context length exceeded", None, {}, _URL)
        assert isinstance(err, ContextLimitError)

    def test_unclassified_is_plain_llm_call_error(self):
        err = self._classify(401, {"error": {"message": "invalid api key"}})
        assert type(err) is LLMCallError

    def test_message_redacts_url(self):
        err = _classify_error(500, "boom", None, {}, "https://user:pw@h/v1/chat/completions?k=secret")
        assert "secret" not in str(err)
        assert "pw" not in str(err)


class TestParseRetryAfter:
    def test_delta_seconds(self):
        assert _parse_retry_after({"Retry-After": "12"}) == 12.0

    def test_case_insensitive(self):
        assert _parse_retry_after({"retry-after": "3"}) == 3.0

    def test_http_date(self):
        value = _parse_retry_after({"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"})
        # Far-future date exceeds the plausibility ceiling and is discarded.
        assert value is None

    def test_zero_is_discarded(self):
        # Would otherwise burn the base class's whole rate-limit budget instantly.
        assert _parse_retry_after({"Retry-After": "0"}) is None

    def test_negative_is_discarded(self):
        assert _parse_retry_after({"Retry-After": "-5"}) is None

    def test_absurdly_large_is_discarded(self):
        assert _parse_retry_after({"Retry-After": "99999"}) is None

    def test_garbage_is_discarded(self):
        assert _parse_retry_after({"Retry-After": "soon"}) is None

    def test_absent(self):
        assert _parse_retry_after({}) is None


class TestNormalizeFinishReason:
    @pytest.mark.parametrize("value", ["stop", "length", "tool_calls", "content_filter"])
    def test_spec_values_pass_through(self, value):
        assert _normalize_finish_reason(value, {}) == value

    def test_deprecated_function_call_alias(self):
        assert _normalize_finish_reason("function_call", {}) == "tool_calls"

    def test_length_survives_alongside_tool_calls(self):
        # A truncated tool-call array must NOT be laundered into "tool_calls";
        # "length" is intentionally terminal.
        message = {"tool_calls": [{"id": "1"}]}
        assert _normalize_finish_reason("length", message) == "length"

    def test_absent_with_tool_calls(self):
        assert _normalize_finish_reason(None, {"tool_calls": [{"id": "1"}]}) == "tool_calls"

    def test_absent_without_tool_calls(self):
        assert _normalize_finish_reason(None, {}) == "stop"

    @pytest.mark.parametrize("raw", ["", "   ", 7, []])
    def test_empty_and_non_string_take_the_infer_branch(self, raw):
        assert _normalize_finish_reason(raw, {}) == "stop"

    def test_unrecognized_passes_through_not_defaulted_to_stop(self):
        # Deliberately unlike litellm's default-to-"stop": a surprise must fail
        # loudly at the agent loop rather than degrade silently.
        assert _normalize_finish_reason("end_turn", {}) == "end_turn"


class TestRedaction:
    def test_url_drops_userinfo_query_and_fragment(self):
        out = _redact_url("https://user:pw@host:8443/v1/chat/completions?token=abc#frag")
        assert out == "https://host:8443/v1/chat/completions"

    def test_headers_redact_authorization_case_insensitively(self):
        out = _redact_headers({"authorization": "Bearer s3cret", "Accept": "application/json"})
        assert out["authorization"] == _REDACTED
        assert out["Accept"] == "application/json"

    def test_headers_not_mutated(self):
        original = {"Authorization": "Bearer s3cret"}
        _redact_headers(original)
        assert original["Authorization"] == "Bearer s3cret"


# ----------------------------------------------------------------------
# Request shaping.
# ----------------------------------------------------------------------


class TestRequestShape:
    def test_private_keys_stripped(self):
        handler, calls = _transport(_OK)
        backend = _make_backend(handler)
        backend.completion(messages=[{"role": "assistant", "content": "a", "__reasoning__": "secret"}])
        assert _body_of(calls[0])["messages"] == [{"role": "assistant", "content": "a"}]

    def test_tool_order_preserved(self):
        handler, calls = _transport(_OK)
        backend = _make_backend(handler)
        tools = [
            {"type": "function", "function": {"name": "z", "parameters": {}}},
            {"type": "function", "function": {"name": "a", "parameters": {}}},
        ]
        backend.completion(messages=[{"role": "user", "content": "q"}], tools=tools)
        names = [t["function"]["name"] for t in _body_of(calls[0])["tools"]]
        assert names == ["z", "a"]

    def test_temperature_omitted_when_none(self):
        handler, calls = _transport(_OK)
        backend = _make_backend(handler)
        backend.completion(messages=[{"role": "user", "content": "q"}])
        assert "temperature" not in _body_of(calls[0])

    def test_temperature_zero_is_sent(self):
        handler, calls = _transport(_OK)
        backend = _make_backend(handler, temperature=0.0)
        backend.completion(messages=[{"role": "user", "content": "q"}])
        assert _body_of(calls[0])["temperature"] == 0.0

    @pytest.mark.parametrize("tools", [None, []])
    def test_tool_only_params_dropped_without_tools(self, tools):
        handler, calls = _transport(_OK)
        backend = _make_backend(handler, parallel_tool_calls=True)
        backend.completion(messages=[{"role": "user", "content": "q"}], tools=tools)
        body = _body_of(calls[0])
        assert "tools" not in body
        assert "tool_choice" not in body
        assert "parallel_tool_calls" not in body

    def test_max_completion_tokens_becomes_max_tokens(self):
        handler, calls = _transport(_OK)
        backend = _make_backend(handler, max_completion_tokens=128)
        backend.completion(messages=[{"role": "user", "content": "q"}])
        body = _body_of(calls[0])
        assert body["max_tokens"] == 128
        assert "max_completion_tokens" not in body

    def test_max_completion_tokens_override_binds(self):
        # The canonical per-call override named by the base class contract. A raw
        # merge would send it alongside max_tokens and silently ignore it.
        handler, calls = _transport(_OK)
        backend = _make_backend(handler, max_completion_tokens=4096)
        backend.completion(messages=[{"role": "user", "content": "q"}], max_completion_tokens=64)
        body = _body_of(calls[0])
        assert body["max_tokens"] == 64
        assert "max_completion_tokens" not in body

    def test_api_key_override_never_reaches_the_wire(self):
        handler, calls = _transport(_OK)
        backend = _make_backend(handler)
        backend.completion(messages=[{"role": "user", "content": "q"}], api_key="super-secret")
        assert "super-secret" not in calls[0].content.decode()

    def test_num_retries_override_is_control_plane_only(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _s: None)
        handler, calls = _transport((503, {"text": "warming up"}))
        backend = _make_backend(handler)
        with pytest.raises(LLMCallError):
            backend.completion(messages=[{"role": "user", "content": "q"}], num_retries=0)
        assert len(calls) == 1
        assert "num_retries" not in _body_of(calls[0])

    def test_tool_choice_override_cannot_reintroduce_tool_params(self):
        handler, calls = _transport(_OK)
        backend = _make_backend(handler)
        backend.completion(messages=[{"role": "user", "content": "q"}], tools=[], tool_choice="required")
        assert "tool_choice" not in _body_of(calls[0])

    def test_unknown_override_passes_through(self):
        handler, calls = _transport(_OK)
        backend = _make_backend(handler)
        backend.completion(messages=[{"role": "user", "content": "q"}], seed=1234)
        assert _body_of(calls[0])["seed"] == 1234

    def test_authorization_present_with_key(self):
        handler, calls = _transport(_OK)
        backend = _make_backend(handler, api_key="k")
        backend.completion(messages=[{"role": "user", "content": "q"}])
        assert calls[0].headers["authorization"] == "Bearer k"

    def test_authorization_absent_without_key(self):
        handler, calls = _transport(_OK)
        backend = _make_backend(handler)
        backend.completion(messages=[{"role": "user", "content": "q"}])
        assert "authorization" not in calls[0].headers

    def test_api_key_resolved_from_environment(self, monkeypatch):
        monkeypatch.setenv("MY_LLM_KEY", "from-env")
        handler, calls = _transport(_OK)
        backend = _make_backend(handler, api_key="os.environ/MY_LLM_KEY")
        backend.completion(messages=[{"role": "user", "content": "q"}])
        assert calls[0].headers["authorization"] == "Bearer from-env"

    def test_missing_environment_variable_names_the_variable(self, monkeypatch):
        monkeypatch.delenv("MY_LLM_KEY", raising=False)
        handler, _ = _transport(_OK)
        with pytest.raises(ValueError, match="MY_LLM_KEY"):
            _make_backend(handler, api_key="os.environ/MY_LLM_KEY")

    def test_injected_client_is_reconfigured(self):
        # A bare httpx.Client defaults to a 5s timeout and follow_redirects=False,
        # which would time out every real call and read a 3xx as success.
        handler, _ = _transport(_OK)
        client = httpx.Client(transport=httpx.MockTransport(handler))
        create_llm(
            create_llm_config("openai_http", model="m", base_url=_URL, timeout_s=99.0),
            http_client=client,
        )
        assert client.timeout.read == 99.0
        assert client.timeout.connect == 10.0
        assert client.follow_redirects is True

    def test_caller_inputs_are_not_mutated(self):
        handler, _ = _transport(_OK)
        backend = _make_backend(handler, parallel_tool_calls=True)
        messages = [{"role": "user", "content": "q", "__reasoning__": "r"}]
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        before_messages, before_tools = deepcopy(messages), deepcopy(tools)
        backend.completion(messages=messages, tools=tools)
        assert messages == before_messages
        assert tools == before_tools


# ----------------------------------------------------------------------
# Result assembly.
# ----------------------------------------------------------------------


class TestResult:
    def test_basic_envelope(self):
        handler, _ = _transport(_OK)
        backend = _make_backend(handler)
        result = backend.completion(messages=[{"role": "user", "content": "q"}])
        assert result.message == {"role": "assistant", "content": "hi"}
        assert result.finish_reason == "stop"
        assert result.usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "a\nb"),
            # Empty block list on a tool-call turn: coerced to None, not rejected.
            ([], None),
        ],
    )
    def test_block_list_content_is_coerced_not_rejected(self, content, expected):
        # Some OpenAI-compatible endpoints return content as a block list. Shared
        # with the litellm backend via helpers.extract_text_content so the default
        # backend degrades the same way instead of hard-failing every step.
        body = {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}
        handler, _ = _transport((200, {"json": body}))
        backend = _make_backend(handler)
        result = backend.completion(messages=[{"role": "user", "content": "q"}])
        assert result.message["content"] == expected

    def test_tool_call_arguments_stay_a_json_string(self):
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "retrieve", "arguments": '{"query": "x"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        handler, _ = _transport((200, {"json": body}))
        backend = _make_backend(handler)
        result = backend.completion(messages=[{"role": "user", "content": "q"}])
        assert result.message["tool_calls"][0]["function"]["arguments"] == '{"query": "x"}'
        assert result.finish_reason == "tool_calls"

    def test_legacy_function_call_is_normalized_to_a_tool_call(self):
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {"name": "retrieve", "arguments": '{"query": "x"}'},
                    },
                    "finish_reason": "function_call",
                }
            ]
        }
        handler, _ = _transport((200, {"json": body}))
        backend = _make_backend(handler)

        result = backend.completion(messages=[{"role": "user", "content": "q"}])

        call = result.message["tool_calls"][0]
        assert call["id"].startswith("call_legacy_")
        assert call["type"] == "function"
        assert call["function"] == {"name": "retrieve", "arguments": '{"query": "x"}'}
        assert result.finish_reason == "tool_calls"

    def test_multiple_choices_are_rejected(self):
        body = {"choices": [_OK_BODY["choices"][0], _OK_BODY["choices"][0]]}
        handler, _ = _transport((200, {"json": body}))
        backend = _make_backend(handler)

        with pytest.raises(LLMCallError, match="exactly 1 choice"):
            backend.completion(messages=[{"role": "user", "content": "q"}])

    @pytest.mark.parametrize(
        "message",
        [
            {"role": "assistant", "content": 7},
            {"role": "assistant", "content": None, "tool_calls": {}},
            {"role": "assistant", "content": None, "tool_calls": ["not-an-object"]},
            {"role": "assistant", "content": None, "tool_calls": [{"function": "not-an-object"}]},
            {"role": "assistant", "content": None, "function_call": "not-an-object"},
        ],
    )
    def test_malformed_message_fields_are_rejected(self, message):
        body = {"choices": [{"message": message, "finish_reason": "stop"}]}
        handler, _ = _transport((200, {"json": body}))
        backend = _make_backend(handler)

        with pytest.raises(LLMCallError):
            backend.completion(messages=[{"role": "user", "content": "q"}])

    def test_tool_call_finish_without_calls_is_rejected(self):
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "tool_calls",
                }
            ]
        }
        handler, _ = _transport((200, {"json": body}))
        backend = _make_backend(handler)

        with pytest.raises(LLMCallError, match="supplied no tool-call objects"):
            backend.completion(messages=[{"role": "user", "content": "q"}])

    def test_reasoning_content_extracted(self):
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "a", "reasoning_content": "because"},
                    "finish_reason": "stop",
                }
            ]
        }
        handler, _ = _transport((200, {"json": body}))
        backend = _make_backend(handler)
        assert backend.completion(messages=[{"role": "user", "content": "q"}]).reasoning == "because"

    def test_missing_finish_reason_is_inferred_and_warned(self, caplog):
        body = {"choices": [{"message": {"role": "assistant", "content": "a"}}]}
        handler, _ = _transport((200, {"json": body}))
        backend = _make_backend(handler)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = backend.completion(messages=[{"role": "user", "content": "q"}])
        assert result.finish_reason == "stop"
        assert any("finish_reason" in r.getMessage() for r in caplog.records)

    def test_ratelimit_headers_surface_json_serializably(self):
        headers = {
            "X-RateLimit-Remaining-Tokens": "900",
            "x-ratelimit-remaining-requests": "5",
        }
        handler, _ = _transport((200, {"json": _OK_BODY, "headers": headers}))
        backend = _make_backend(handler)
        info = backend.completion(messages=[{"role": "user", "content": "q"}]).extra_response_info
        assert info["ratelimit"] == {"TPM": "900", "RQ": "5"}
        assert info["http"] == {"status": 200, "attempts": 1}
        json.dumps(info)  # must not raise: the agent dumps this to disk

    @pytest.mark.parametrize(
        "body",
        [
            {"choices": []},
            {"choices": "nope"},
            {},
            {"choices": [{"finish_reason": "stop"}]},
            {"choices": ["not-an-object"]},
        ],
    )
    def test_malformed_envelope_raises_llm_call_error(self, body):
        handler, _ = _transport((200, {"json": body}))
        backend = _make_backend(handler)
        with pytest.raises(LLMCallError):
            backend.completion(messages=[{"role": "user", "content": "q"}])

    def test_assistant_message_round_trips(self):
        # result.py contracts the message as "valid to send back on the next call".
        # vLLM emits a provider-only `index` key on tool calls; it must survive
        # and stay serializable.
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "index": 0,
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        handler, calls = _transport((200, {"json": body}), _OK)
        backend = _make_backend(handler)
        first = backend.completion(messages=[{"role": "user", "content": "q"}])

        history = [
            {"role": "user", "content": "q"},
            first.message,
            {"role": "tool", "tool_call_id": "call_1", "name": "f", "content": "done"},
        ]
        backend.completion(messages=history)
        assert _body_of(calls[1])["messages"][1] == first.message


# ----------------------------------------------------------------------
# Retry, error translation, and the base-class boundary.
# ----------------------------------------------------------------------


class TestRetryAndErrors:
    @pytest.fixture
    def sleeps(self, monkeypatch):
        # One binding only: this module and base_backend both `import time`, so
        # `openai_http_backend.time is base_backend.time is time`. In-backend and
        # base-class sleeps are separated by CALL LEVEL, not by patch target.
        recorded: list[float] = []
        monkeypatch.setattr(time, "sleep", recorded.append)
        return recorded

    def test_transient_5xx_retried_then_succeeds(self, sleeps):
        handler, calls = _transport((503, {"text": "warming up"}), (503, {"text": "warming up"}), _OK)
        backend = _make_backend(handler)
        result = backend.completion(messages=[{"role": "user", "content": "q"}])
        assert result.finish_reason == "stop"
        assert len(calls) == 3
        assert result.extra_response_info["http"]["attempts"] == 3

    def test_transient_5xx_exhausted(self, sleeps):
        handler, calls = _transport((503, {"text": "still down"}))
        backend = _make_backend(handler, max_transient_retries=2)
        with pytest.raises(LLMCallError):
            backend.completion(messages=[{"role": "user", "content": "q"}])
        assert len(calls) == 3

    def test_zero_retries_means_one_attempt(self, sleeps):
        handler, calls = _transport((500, {"text": "boom"}))
        backend = _make_backend(handler, max_transient_retries=0)
        with pytest.raises(LLMCallError):
            backend.completion(messages=[{"role": "user", "content": "q"}])
        assert len(calls) == 1

    def test_4xx_never_retried(self, sleeps):
        handler, calls = _transport((400, {"json": {"error": {"message": "bad"}}}))
        backend = _make_backend(handler)
        with pytest.raises(LLMCallError):
            backend.completion(messages=[{"role": "user", "content": "q"}])
        assert len(calls) == 1

    def test_non_json_success_body_not_retried(self, sleeps):
        handler, calls = _transport((200, {"text": "<html>gateway</html>"}))
        backend = _make_backend(handler)
        with pytest.raises(LLMCallError, match="non-JSON"):
            backend.completion(messages=[{"role": "user", "content": "q"}])
        assert len(calls) == 1

    def test_non_json_error_body_is_translated_not_raw_valueerror(self, sleeps):
        handler, calls = _transport((400, {"text": "<html>bad request</html>"}))
        backend = _make_backend(handler)
        with pytest.raises(LLMCallError):
            backend.completion(messages=[{"role": "user", "content": "q"}])
        assert len(calls) == 1

    def test_transport_error_retried_then_wrapped(self, sleeps):
        calls: list[httpx.Request] = []

        def handle(request):
            calls.append(request)
            raise httpx.ConnectError("refused")

        backend = _make_backend(handle, max_transient_retries=1)
        with pytest.raises(LLMCallError):
            backend.completion(messages=[{"role": "user", "content": "q"}])
        assert len(calls) == 2

    def test_context_limit_reaches_the_caller(self, sleeps):
        body = {"error": {"code": "context_length_exceeded", "message": "too long"}}
        handler, _ = _transport((400, {"json": body}))
        backend = _make_backend(handler)
        with pytest.raises(ContextLimitError):
            backend.completion(messages=[{"role": "user", "content": "q"}])

    def test_429_raises_immediately_with_no_in_backend_sleep(self, sleeps):
        handler, calls = _transport((429, {"text": "slow down", "headers": {"Retry-After": "9"}}))
        backend = _make_backend(handler)
        with pytest.raises(RateLimitError) as exc:
            backend._completion_impl([{"role": "user", "content": "q"}])
        assert exc.value.retry_after == 9.0
        assert len(calls) == 1
        assert sleeps == []

    def test_429_retried_by_the_base_template_only(self, sleeps):
        # The base owns the rate-limit loop; the backend must not nest one inside
        # it. Default rate_limit_max_retries=3 => 4 attempts total.
        handler, calls = _transport((429, {"text": "slow down", "headers": {"Retry-After": "1"}}))
        backend = _make_backend(handler)
        with pytest.raises(RateLimitError):
            backend.completion(messages=[{"role": "user", "content": "q"}])
        assert len(calls) == 4
        assert sleeps == [1.0, 1.0, 1.0]

    def test_unclassified_4xx_warns_once(self, sleeps, caplog):
        handler, _ = _transport((401, {"json": {"error": {"message": "invalid key"}}}))
        backend = _make_backend(handler)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            with pytest.raises(LLMCallError):
                backend.completion(messages=[{"role": "user", "content": "q"}])
        warnings = [r for r in caplog.records if "unclassified" in r.getMessage()]
        assert len(warnings) == 1

    def test_retried_5xx_does_not_warn(self, sleeps, caplog):
        handler, _ = _transport((503, {"text": "warming up"}), _OK)
        backend = _make_backend(handler)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            backend.completion(messages=[{"role": "user", "content": "q"}])
        assert [r for r in caplog.records if "unclassified" in r.getMessage()] == []


# ----------------------------------------------------------------------
# Async path.
# ----------------------------------------------------------------------


class TestAsyncPath:
    def test_one_backend_survives_successive_event_loops(self):
        # The agent runs `asyncio.run` per query on a pool thread, so a backend
        # that cached an event-loop-bound async client would fail the second call.
        # This is the test that would catch a future `_acompletion_impl` override.
        handler, calls = _transport(_OK)
        backend = _make_backend(handler)

        first = asyncio.run(backend.acompletion(messages=[{"role": "user", "content": "q"}]))
        second = asyncio.run(backend.acompletion(messages=[{"role": "user", "content": "q"}]))

        assert first.finish_reason == "stop"
        assert second.finish_reason == "stop"
        assert len(calls) == 2


# ----------------------------------------------------------------------
# Raw-IO capture.
# ----------------------------------------------------------------------


class TestCaptureRawIO:
    def test_disabled_by_default(self):
        handler, _ = _transport(_OK)
        backend = _make_backend(handler, api_key="k")
        result = backend.completion(messages=[{"role": "user", "content": "q"}])
        assert result.raw_request is None
        assert result.raw_response is None

    def test_enabled_redacts_credentials_only(self):
        handler, _ = _transport(_OK)
        backend = _make_backend(
            handler,
            api_key="k",
            capture_raw_io=True,
            max_completion_tokens=64,
            base_url="https://user:pw@h/v1/chat/completions?token=abc",
        )
        result = backend.completion(messages=[{"role": "user", "content": "q"}])

        assert result.raw_request["headers"]["Authorization"] == _REDACTED
        assert result.raw_request["url"] == "https://h/v1/chat/completions"
        # Exact-key redaction: max_tokens is not a credential despite the substring.
        assert result.raw_request["body"]["max_tokens"] == 64
        assert result.raw_response == _OK_BODY

    def test_redaction_does_not_mutate_live_headers(self):
        # Redacting in place would strip Authorization from every later request.
        handler, calls = _transport(_OK)
        backend = _make_backend(handler, api_key="k", capture_raw_io=True)
        backend.completion(messages=[{"role": "user", "content": "q"}])
        backend.completion(messages=[{"role": "user", "content": "q"}])
        assert calls[1].headers["authorization"] == "Bearer k"
