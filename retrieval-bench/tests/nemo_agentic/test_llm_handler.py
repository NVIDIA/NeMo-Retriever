# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LLM handler module.

These tests mock the underlying completion API so they run without network access
or API keys. They verify the contract that callers of `LLM.acompletion` depend on:

- Successful completions return a response with `.model_dump()`.
- Rate-limit errors are retried up to 3 times then re-raised.
- Content-policy / context-window errors are caught and returned as error strings
  (unless strict_error_handling is enabled).
- `is_error()` correctly identifies error-string responses.
- `normalize_messages_for_api()` collapses text-only content blocks.
- Request kwargs are assembled correctly from LLMConfig + per-call overrides.
- Logging writes request/response JSON when instant_log is enabled.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from retrieval_bench.nemo_agentic.configs import LLMConfig
from retrieval_bench.nemo_agentic.llm_handler import (
    LLM,
    LLM_ERROR_PREFIX,
    is_error,
    make_acall_with_ratelimit_pause,
    normalize_messages_for_api,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_response(**overrides: Any) -> MagicMock:
    """Create a mock that quacks like a ChatCompletion."""
    resp = MagicMock()
    dump = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    dump.update(overrides)
    resp.model_dump.return_value = dump
    resp._hidden_params = {
        "additional_headers": {
            "llm_provider-x-ratelimit-remaining-tokens": "5000",
            "llm_provider-x-ratelimit-remaining-requests": "100",
        }
    }
    return resp


def _make_llm(**config_overrides: Any) -> LLM:
    """Create an LLM instance with sensible test defaults."""
    defaults = dict(model="test-model", api_key="test-key")
    defaults.update(config_overrides)
    return LLM(LLMConfig(**defaults))


def _make_bad_request_error(message: str) -> openai.BadRequestError:
    """Create an openai.BadRequestError with the given message."""
    response = httpx.Response(400, request=httpx.Request("POST", "https://test"))
    return openai.BadRequestError(message=message, response=response, body=None)


# ---------------------------------------------------------------------------
# is_error
# ---------------------------------------------------------------------------


class TestIsError:
    def test_error_string_detected(self):
        assert is_error(f"{LLM_ERROR_PREFIX} something went wrong") is True

    def test_plain_string_not_error(self):
        assert is_error("just a string") is False

    def test_non_string_not_error(self):
        assert is_error(42) is False
        assert is_error(None) is False
        assert is_error({"key": "val"}) is False

    def test_model_response_not_error(self):
        assert is_error(_make_model_response()) is False


# ---------------------------------------------------------------------------
# normalize_messages_for_api
# ---------------------------------------------------------------------------


class TestNormalizeMessages:
    def test_plain_string_content_unchanged(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert normalize_messages_for_api(msgs) == msgs

    def test_single_text_block_collapsed(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        result = normalize_messages_for_api(msgs)
        assert result[0]["content"] == "hi"

    def test_multiple_text_blocks_joined(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "line1"},
                    {"type": "text", "text": "line2"},
                ],
            }
        ]
        result = normalize_messages_for_api(msgs)
        assert result[0]["content"] == "line1\nline2"

    def test_non_text_block_preserved(self):
        content = [{"type": "image_url", "image_url": {"url": "data:..."}}]
        msgs = [{"role": "user", "content": content}]
        result = normalize_messages_for_api(msgs)
        assert result[0]["content"] == content

    def test_empty_content_list_becomes_none(self):
        msgs = [{"role": "user", "content": []}]
        result = normalize_messages_for_api(msgs)
        assert result[0]["content"] is None

    def test_original_messages_not_mutated(self):
        original_content = [{"type": "text", "text": "hi"}]
        msgs = [{"role": "user", "content": original_content}]
        normalize_messages_for_api(msgs)
        # Original list should still be a list
        assert isinstance(msgs[0]["content"], list)


# ---------------------------------------------------------------------------
# make_acall_with_ratelimit_pause
# ---------------------------------------------------------------------------


def _make_mock_client(side_effect=None, return_value=None):
    """Create a mock AsyncOpenAI client."""
    client = MagicMock()
    create_mock = AsyncMock(side_effect=side_effect, return_value=return_value)
    client.chat.completions.create = create_mock
    return client, create_mock


class TestRateLimitRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        expected = _make_model_response()
        client, _ = _make_mock_client(return_value=expected)
        result = await make_acall_with_ratelimit_pause(client, model="m", messages=[])
        assert result is expected

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(self):
        expected = _make_model_response()
        response_429 = httpx.Response(429, request=httpx.Request("POST", "https://test"))
        rate_limit_exc = openai.RateLimitError(message="rate limited", response=response_429, body=None)
        client, _ = _make_mock_client(side_effect=[rate_limit_exc, expected])

        with patch("retrieval_bench.nemo_agentic.llm_handler.asyncio.sleep", new_callable=AsyncMock):
            result = await make_acall_with_ratelimit_pause(client, model="m", messages=[])

        assert result is expected

    @pytest.mark.asyncio
    async def test_gives_up_after_3_retries(self):
        response_429 = httpx.Response(429, request=httpx.Request("POST", "https://test"))
        exc = openai.RateLimitError(message="rate limited", response=response_429, body=None)
        client, create_mock = _make_mock_client(side_effect=[exc, exc, exc, exc])

        with patch("retrieval_bench.nemo_agentic.llm_handler.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(openai.RateLimitError):
                await make_acall_with_ratelimit_pause(client, model="m", messages=[])

        # 1 initial + 3 retries = 4 calls total
        assert create_mock.await_count == 4


# ---------------------------------------------------------------------------
# LLM.__init__ — completion_kwargs assembly
# ---------------------------------------------------------------------------


class TestLLMInit:
    def test_basic_kwargs(self):
        llm = _make_llm(model="gpt-4", base_url="http://localhost:8000")
        assert llm.completion_kwargs["model"] == "gpt-4"
        assert str(llm._client.base_url).rstrip("/") == "http://localhost:8000"

    def test_optional_fields_omitted_when_none(self):
        llm = _make_llm()
        assert "reasoning_effort" not in llm.completion_kwargs

    def test_reasoning_effort_included(self):
        llm = _make_llm(reasoning_effort="medium")
        assert llm.completion_kwargs["reasoning_effort"] == "medium"

    def test_api_key_env_resolution(self):
        import os

        os.environ["_TEST_LLM_KEY"] = "secret123"
        try:
            llm = _make_llm(api_key="os.environ/_TEST_LLM_KEY")
            assert llm._api_key == "secret123"
        finally:
            del os.environ["_TEST_LLM_KEY"]

    def test_num_retries_passed_to_client(self):
        llm = _make_llm(num_retries=7)
        assert llm._client.max_retries == 7


# ---------------------------------------------------------------------------
# LLM.pre_completion
# ---------------------------------------------------------------------------


class TestPreCompletion:
    def test_tools_included_in_kwargs(self):
        llm = _make_llm()
        tools = [{"type": "function", "function": {"name": "foo"}}]
        request_kwargs, _, _ = llm.pre_completion(messages=[{"role": "user", "content": "hi"}], tools=tools)
        assert request_kwargs["tools"] is tools

    def test_tool_choice_removed_when_no_tools(self):
        llm = _make_llm(tool_choice="auto")
        request_kwargs, _, _ = llm.pre_completion(messages=[{"role": "user", "content": "hi"}], tools=None)
        assert "tool_choice" not in request_kwargs

    def test_per_call_kwargs_override_defaults(self):
        llm = _make_llm(max_completion_tokens=100)
        request_kwargs, _, _ = llm.pre_completion(
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=200,
        )
        assert request_kwargs["max_completion_tokens"] == 200


# ---------------------------------------------------------------------------
# LLM.post_completion — metadata extraction
# ---------------------------------------------------------------------------


class TestPostCompletion:
    def test_extracts_rate_limit_metadata(self):
        llm = _make_llm()
        resp = _make_model_response()
        logging_ctx = {"json_log_dir": None, "curr_step": "1"}

        _, metadata_kv, _ = llm.post_completion(request_kwargs={}, response=resp, logging_ctx=logging_ctx)

        assert metadata_kv["TPM"] == "5,000"
        assert metadata_kv["RQ"] == "100"

    def test_missing_headers_default_to_negative_one(self):
        llm = _make_llm()
        resp = _make_model_response()
        resp._hidden_params = {}
        logging_ctx = {"json_log_dir": None, "curr_step": "1"}

        _, metadata_kv, _ = llm.post_completion(request_kwargs={}, response=resp, logging_ctx=logging_ctx)

        assert metadata_kv["TPM"] == "-1"
        assert metadata_kv["RQ"] == "-1"

    def test_response_dump_in_io_log(self):
        llm = _make_llm()
        resp = _make_model_response()
        logging_ctx = {"json_log_dir": "/tmp/logs", "curr_step": "42"}

        _, _, io_log_kwargs = llm.post_completion(request_kwargs={}, response=resp, logging_ctx=logging_ctx)

        assert io_log_kwargs["output_json"]["filename"] == "42_response.json"


# ---------------------------------------------------------------------------
# LLM.acompletion — end-to-end async
# ---------------------------------------------------------------------------


class TestACompletion:
    @pytest.mark.asyncio
    async def test_successful_completion_returns_response(self):
        llm = _make_llm()
        expected = _make_model_response()

        with patch(
            "retrieval_bench.nemo_agentic.llm_handler.make_acall_with_ratelimit_pause",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            result = await llm.acompletion(messages=[{"role": "user", "content": "hi"}])

        assert result is expected

    @pytest.mark.asyncio
    async def test_content_policy_error_returns_error_string(self):
        llm = _make_llm()

        with patch(
            "retrieval_bench.nemo_agentic.llm_handler.make_acall_with_ratelimit_pause", new_callable=AsyncMock
        ) as mock_call:
            mock_call.side_effect = _make_bad_request_error("content policy violation")
            result = await llm.acompletion(messages=[{"role": "user", "content": "bad"}])

        assert is_error(result)
        assert "content policy violation" in result.lower()

    @pytest.mark.asyncio
    async def test_context_window_error_returns_error_string(self):
        llm = _make_llm()

        with patch(
            "retrieval_bench.nemo_agentic.llm_handler.make_acall_with_ratelimit_pause", new_callable=AsyncMock
        ) as mock_call:
            mock_call.side_effect = _make_bad_request_error("context window exceeded")
            result = await llm.acompletion(messages=[{"role": "user", "content": "long"}])

        assert is_error(result)

    @pytest.mark.asyncio
    async def test_strict_error_handling_reraises(self):
        llm = _make_llm(strict_error_handling=True)

        with patch(
            "retrieval_bench.nemo_agentic.llm_handler.make_acall_with_ratelimit_pause", new_callable=AsyncMock
        ) as mock_call:
            mock_call.side_effect = _make_bad_request_error("content policy violation")
            with pytest.raises(openai.BadRequestError):
                await llm.acompletion(messages=[{"role": "user", "content": "bad"}])

    @pytest.mark.asyncio
    async def test_bad_request_without_known_keyword_reraises(self):
        """BadRequestError that isn't context-window or content-policy should propagate."""
        llm = _make_llm()

        with patch(
            "retrieval_bench.nemo_agentic.llm_handler.make_acall_with_ratelimit_pause", new_callable=AsyncMock
        ) as mock_call:
            mock_call.side_effect = _make_bad_request_error("something completely different")
            with pytest.raises(openai.BadRequestError):
                await llm.acompletion(messages=[{"role": "user", "content": "bad"}])

    @pytest.mark.asyncio
    async def test_bad_request_with_context_window_keyword_handled(self):
        """BadRequestError mentioning context window should be caught."""
        llm = _make_llm()

        with patch(
            "retrieval_bench.nemo_agentic.llm_handler.make_acall_with_ratelimit_pause", new_callable=AsyncMock
        ) as mock_call:
            mock_call.side_effect = _make_bad_request_error("This model's maximum context window is exceeded")
            result = await llm.acompletion(messages=[{"role": "user", "content": "long"}])

        assert is_error(result)

    @pytest.mark.asyncio
    async def test_return_metadata_mode(self):
        llm = _make_llm()
        expected = _make_model_response()

        with patch(
            "retrieval_bench.nemo_agentic.llm_handler.make_acall_with_ratelimit_pause",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            result = await llm.acompletion(
                messages=[{"role": "user", "content": "hi"}],
                return_metadata=True,
            )

        assert isinstance(result, dict)
        assert "response" in result
        assert "metadata_kv" in result
        assert "io_log_kwargs" in result
        assert result["response"] is expected

    @pytest.mark.asyncio
    async def test_instant_log_writes_files(self, tmp_path):
        llm = _make_llm(raw_log_pardir=str(tmp_path), instant_log=True)
        expected = _make_model_response()

        with patch(
            "retrieval_bench.nemo_agentic.llm_handler.make_acall_with_ratelimit_pause",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            await llm.acompletion(
                messages=[{"role": "user", "content": "hi"}],
                logging_kwargs={"subdir": "run1", "step": "5"},
            )

        log_dir = tmp_path / "run1"
        prompt_file = log_dir / "5_prompt.json"
        response_file = log_dir / "5_response.json"
        assert prompt_file.exists()
        assert response_file.exists()

        # Verify JSON is valid
        with open(prompt_file) as f:
            data = json.load(f)
        assert "messages" in data

    @pytest.mark.asyncio
    async def test_api_key_env_override_used_in_client(self):
        """When api_key uses os.environ/ syntax, the resolved key is set on the client."""
        import os

        os.environ["_TEST_ACOMP_KEY"] = "resolved-key"
        try:
            llm = _make_llm(api_key="os.environ/_TEST_ACOMP_KEY")
            assert llm._api_key == "resolved-key"
            assert llm._client.api_key == "resolved-key"
        finally:
            del os.environ["_TEST_ACOMP_KEY"]
