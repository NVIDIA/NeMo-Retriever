# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""The contract between CallableLLMBackend and the completion callables it drives.

Two callables ship in this repo — the hosted ``invoke_chat_completion_step`` and
the in-process ``VLLMAgentChatLLM`` — and the backend must speak to both through
one keyword set. These tests pin that seam plus the request shaping that used to
live in the retired HTTP backend.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from nemo_retriever._agentic.nemo_agent.llm import (
    CallableLLMConfig,
    ContextLimitError,
    LLMCallError,
    create_llm,
    create_llm_config,
)

_OK = {
    "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
}


def _backend(fn, **config_kwargs):
    config_kwargs.setdefault("model", "m")
    return create_llm(create_llm_config("callable", **config_kwargs), completion_fn=fn)


def _seen(**config_kwargs):
    """Run one completion against a recording callable and return its kwargs."""
    seen: dict = {}

    def fn(**kwargs):
        seen.update(kwargs)
        return _OK

    _backend(fn, **config_kwargs).completion(messages=[{"role": "user", "content": "q"}])
    return seen


class TestKeywordContract:
    #: Every keyword the backend promises to send. Both shipped callables accept
    #: all of them; a callable that does not is a contract violation, and a loud
    #: TypeError is the intended failure.
    CONTRACT = frozenset(
        {
            "invoke_url",
            "messages",
            "model",
            "api_key",
            "tools",
            "tool_choice",
            "timeout_s",
            "temperature",
            "max_tokens",
            "extra_body",
            "max_retries",
            "max_429_retries",
        }
    )

    def test_every_contract_keyword_is_sent(self):
        assert set(_seen()) == self.CONTRACT

    @pytest.mark.parametrize(
        "target",
        [
            "nemo_retriever.models.nim.chat_completions.invoke_chat_completion_step",
            "nemo_retriever.models.local.agent_llm.VLLMAgentChatLLM.__call__",
        ],
    )
    def test_both_shipped_callables_accept_the_contract(self, target):
        module_path, _, attr = target.rpartition(".")
        if attr == "__call__":
            module_path, _, cls_name = module_path.rpartition(".")
            module = pytest.importorskip(module_path)
            fn = getattr(module, cls_name).__call__
        else:
            module = pytest.importorskip(module_path)
            fn = getattr(module, attr)

        accepted = set(inspect.signature(fn).parameters)
        missing = self.CONTRACT - accepted
        assert not missing, f"{target} does not accept {sorted(missing)}"


class TestRetryKnobs:
    def test_defaults_are_bounded_for_agent_use(self):
        config = CallableLLMConfig(model="m")
        assert config.max_retries == 3
        assert config.max_429_retries == 6
        assert config.timeout_s == 120.0
        # The callable owns retrying, so the base template must make one attempt.
        assert config.rate_limit_max_retries == 0

    def test_zero_max_retries_is_rejected(self):
        # A callable that loops `while attempt < max_retries` issues ZERO requests
        # at 0 and then reports retries-exhausted, which reads as an endpoint
        # failure. The bound makes that unreachable.
        with pytest.raises(ValidationError):
            CallableLLMConfig(model="m", max_retries=0)

    @pytest.mark.parametrize("field", ["max_429_retries"])
    def test_other_retry_bounds(self, field):
        with pytest.raises(ValidationError):
            CallableLLMConfig(model="m", **{field: 0})

    def test_non_positive_timeout_is_rejected(self):
        with pytest.raises(ValidationError):
            CallableLLMConfig(model="m", timeout_s=0)

    def test_knobs_are_forwarded_to_the_callable(self):
        seen = _seen(timeout_s=30.0, max_retries=2, max_429_retries=4)
        assert seen["timeout_s"] == 30.0
        assert seen["max_retries"] == 2
        assert seen["max_429_retries"] == 4


class TestApiKeyResolution:
    def test_env_indirection_resolved_once_at_construction(self, monkeypatch):
        monkeypatch.setenv("AGENT_TEST_KEY", "  sk-live  ")
        assert _seen(api_key="os.environ/AGENT_TEST_KEY")["api_key"] == "sk-live"

    def test_literal_key_passes_through(self):
        assert _seen(api_key="sk-literal")["api_key"] == "sk-literal"

    def test_absent_key_is_none_not_empty_string(self):
        # An empty Bearer header is worse than none at all.
        assert _seen()["api_key"] is None

    def test_missing_env_var_fails_at_build_time(self, monkeypatch):
        # Must fail while the pipeline is being constructed, not several hundred
        # agent steps into a run.
        monkeypatch.delenv("AGENT_TEST_MISSING", raising=False)
        with pytest.raises(ValueError, match="AGENT_TEST_MISSING"):
            _backend(lambda **kw: _OK, api_key="os.environ/AGENT_TEST_MISSING")


class TestRequestShaping:
    def test_block_list_content_is_normalized_to_a_string(self):
        # The agent builds every message as a content-block list, including the
        # system message. Endpoints that only accept string content for some roles
        # would otherwise see a shape this repo has never exercised.
        seen: dict = {}

        def fn(**kwargs):
            seen.update(kwargs)
            return _OK

        _backend(fn).completion(
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "S"}]},
                {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            ]
        )
        assert seen["messages"][0]["content"] == "S"
        assert seen["messages"][1]["content"] == "a\nb"

    def test_private_keys_are_stripped_before_normalization(self):
        seen: dict = {}

        def fn(**kwargs):
            seen.update(kwargs)
            return _OK

        _backend(fn).completion(messages=[{"role": "user", "content": "q", "__reasoning__": "secret"}])
        assert "__reasoning__" not in seen["messages"][0]

    def test_caller_messages_are_not_mutated(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "q"}], "__reasoning__": "r"}]
        _backend(lambda **kw: _OK).completion(messages=messages)
        assert messages[0]["content"] == [{"type": "text", "text": "q"}]
        assert messages[0]["__reasoning__"] == "r"

    def test_base_url_is_forwarded_verbatim_as_invoke_url(self):
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        assert _seen(base_url=url)["invoke_url"] == url

    def test_tool_choice_suppressed_without_tools(self):
        assert _seen()["tool_choice"] == "none"


class TestResponseHandling:
    def test_usage_is_recorded_not_discarded(self):
        result = _backend(lambda **kw: _OK).completion(messages=[{"role": "user", "content": "q"}])
        assert result.usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}

    def test_extra_response_info_is_always_empty(self):
        # Structural: the callable returns a decoded body, so headers/status never
        # cross the boundary. Pinned so nobody "fixes" it by plumbing them through.
        result = _backend(lambda **kw: _OK).completion(messages=[{"role": "user", "content": "q"}])
        assert result.extra_response_info == {}

    def test_padded_finish_reason_is_stripped(self):
        # The agent loop treats any finish reason outside ("stop", "tool_calls") as
        # terminal, so a padded value would end an otherwise healthy run.
        body = {"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "  tool_calls  "}]}
        result = _backend(lambda **kw: body).completion(messages=[{"role": "user", "content": "q"}])
        assert result.finish_reason == "tool_calls"

    def test_missing_finish_reason_defaults_to_stop(self):
        body = {"choices": [{"message": {"role": "assistant", "content": "x"}}]}
        result = _backend(lambda **kw: body).completion(messages=[{"role": "user", "content": "q"}])
        assert result.finish_reason == "stop"

    def test_block_list_response_content_is_coerced(self):
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": blocks},
                    "finish_reason": "stop",
                }
            ]
        }
        result = _backend(lambda **kw: body).completion(messages=[{"role": "user", "content": "q"}])
        assert result.message["content"] == "a\nb"

    @pytest.mark.parametrize(
        "body",
        [None, {}, {"choices": []}, {"choices": [None]}, {"choices": [{}]}, "not-a-dict"],
    )
    def test_malformed_responses_raise_llm_call_error_not_key_error(self, body):
        with pytest.raises(LLMCallError):
            _backend(lambda **kw: body).completion(messages=[{"role": "user", "content": "q"}])


class TestErrorTranslation:
    def test_callable_failures_are_classified(self):
        def boom(**kwargs):
            raise ValueError("prompt is longer than the maximum model length of 8192")

        with pytest.raises(ContextLimitError):
            _backend(boom).completion(messages=[{"role": "user", "content": "q"}])

    def test_library_errors_keep_their_subclass(self):
        # Re-wrapping would erase the class the selection agent's shrink loop
        # branches on.
        def boom(**kwargs):
            raise ContextLimitError("already ours")

        with pytest.raises(ContextLimitError, match="already ours"):
            _backend(boom).completion(messages=[{"role": "user", "content": "q"}])

    def test_original_exception_stays_chained(self):
        original = RuntimeError("wire down")

        def boom(**kwargs):
            raise original

        with pytest.raises(LLMCallError) as excinfo:
            _backend(boom).completion(messages=[{"role": "user", "content": "q"}])
        assert excinfo.value.__cause__ is original

    def test_cancellation_is_not_swallowed(self):
        # BaseException must propagate untouched or task cancellation breaks.
        def boom(**kwargs):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            _backend(boom).completion(messages=[{"role": "user", "content": "q"}])


class TestCaptureRawIO:
    def test_credentials_and_url_are_redacted(self):
        backend = _backend(
            lambda **kw: _OK,
            api_key="sk-secret",
            base_url="https://user:pw@endpoint.invalid/v1/chat/completions?token=leaked",
            capture_raw_io=True,
        )
        result = backend.completion(messages=[{"role": "user", "content": "q"}])
        dumped = str(result.raw_request)
        assert "sk-secret" not in dumped
        assert "pw" not in dumped
        assert "token=leaked" not in dumped
        assert "endpoint.invalid" in dumped

    def test_disabled_by_default(self):
        result = _backend(lambda **kw: _OK).completion(messages=[{"role": "user", "content": "q"}])
        assert result.raw_request is None
        assert result.raw_response is None
