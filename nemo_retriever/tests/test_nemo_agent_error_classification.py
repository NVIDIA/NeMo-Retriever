# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Exception -> error-hierarchy mapping for the callable LLM backend.

The agent branches on these classes: ``loop.py`` records ContextLimitError and
ContentPolicyError as expected outcomes, and the selection agent retries a
shrunken candidate list only on a context limit. A misclassification therefore
silently changes agent behavior, so the table below is the contract.
"""

from __future__ import annotations

import json

import pytest
import requests

from nemo_retriever._agentic.nemo_agent.llm import (
    ContentPolicyError,
    ContextLimitError,
    LLMCallError,
    RateLimitError,
)
from nemo_retriever._agentic.nemo_agent.llm.error_classification import (
    _parse_retry_after,
    classify_call_exception,
    classify_http_error,
)

_URL = "https://user:secret@endpoint.invalid/v1/chat/completions?api_key=leaked"


def _response(status: int, text: str = "", headers: dict | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = text.encode()
    response.url = _URL
    response.headers.update(headers or {})
    return response


def _http_error(status: int, text: str = "", headers: dict | None = None) -> requests.HTTPError:
    return requests.HTTPError(f"HTTP {status}", response=_response(status, text, headers))


class TestResponseCarryingErrors:
    def test_429_becomes_a_rate_limit_error_with_retry_after(self):
        error = classify_call_exception(_http_error(429, "slow down", {"Retry-After": "12"}))
        assert isinstance(error, RateLimitError)
        assert error.retry_after == 12.0

    def test_error_response_objects_are_falsy_and_still_classified(self):
        # requests.Response.__bool__ is `status_code < 400`, so EVERY response this
        # module exists to classify is falsy. A truthiness check instead of
        # `is None` would skip classification for all of them.
        response = _response(429, "slow down")
        assert not response
        assert isinstance(classify_call_exception(requests.HTTPError("x", response=response)), RateLimitError)

    @pytest.mark.parametrize("code", ["context_length_exceeded", "string_above_max_length"])
    def test_structured_context_limit_codes(self, code):
        body = '{"error": {"code": "%s", "message": "too long"}}' % code
        assert isinstance(classify_call_exception(_http_error(400, body)), ContextLimitError)

    @pytest.mark.parametrize(
        "body",
        [
            "This model's maximum context length is 8192 tokens",
            "Please reduce the length of the messages",
            "max_tokens must be at least 1, got -42",
        ],
    )
    def test_prose_context_limit_markers(self, body):
        assert isinstance(classify_call_exception(_http_error(400, body)), ContextLimitError)

    def test_content_policy(self):
        assert isinstance(classify_call_exception(_http_error(400, "blocked by content policy")), ContentPolicyError)

    def test_unclassified_4xx_stays_a_plain_llm_call_error(self):
        error = classify_call_exception(_http_error(401, "invalid api key"))
        assert type(error) is LLMCallError

    def test_message_redacts_url_credentials_and_query(self):
        message = str(classify_call_exception(_http_error(500, "boom")))
        assert "secret" not in message
        assert "api_key=leaked" not in message
        assert "endpoint.invalid" in message

    def test_message_excerpts_a_huge_body(self):
        message = str(classify_call_exception(_http_error(500, "x" * 50_000)))
        assert len(message) < 3_000
        assert message.endswith("...")


class TestErrorsWithoutAResponse:
    """No response object: timeouts, transport errors, and in-process backends."""

    @pytest.mark.parametrize(
        "exc",
        [
            TimeoutError("Request timed out after 3 attempts."),
            RuntimeError("Failed to get a successful response after 3 retries."),
            ValueError("invoke_url is required"),
        ],
    )
    def test_degrade_to_plain_llm_call_error(self, exc):
        assert type(classify_call_exception(exc)) is LLMCallError

    def test_prose_still_recovers_a_context_limit(self):
        # An in-process engine reports overflow as a bare ValueError with no
        # response to inspect. Recovering ContextLimitError from the text is what
        # keeps the selection agent's shrink-and-retry loop alive locally.
        exc = ValueError("The decoder prompt (length 9000) is longer than the maximum model length of 8192.")
        assert isinstance(classify_call_exception(exc), ContextLimitError)


class TestTotality:
    """The classifier must never raise — a bug here would mask the real failure."""

    def test_a_response_whose_text_raises_is_survived(self):
        class Hostile:
            status_code = 500

            @property
            def text(self):
                raise RuntimeError("content already consumed")

        exc = requests.HTTPError("x")
        exc.response = Hostile()
        assert isinstance(classify_call_exception(exc), LLMCallError)

    def test_a_non_integer_status_code_falls_back_to_prose(self):
        class Weird:
            status_code = "not-a-number"

        exc = RuntimeError("maximum context length exceeded")
        exc.response = Weird()
        assert isinstance(classify_call_exception(exc), ContextLimitError)

    def test_an_exception_whose_str_raises_is_survived(self):
        class Explosive(Exception):
            def __str__(self):
                raise RuntimeError("nope")

        assert isinstance(classify_call_exception(Explosive()), LLMCallError)

    def test_library_errors_pass_through_with_their_subclass_intact(self):
        # Re-wrapping would erase the class the agent branches on.
        original = ContextLimitError("prompt too long")
        assert classify_call_exception(original) is original


# ----------------------------------------------------------------------
# Pure helpers, called directly. These were previously exercised through the
# retired HTTP backend; the logic moved here, so the coverage moves with it.
# ----------------------------------------------------------------------


class TestClassifyHttpError:
    """``classify_http_error`` on its own, without an exception wrapper."""

    def _classify(self, status, body, headers=None):
        text = body if isinstance(body, str) else json.dumps(body)
        parsed = None if isinstance(body, str) else body
        return classify_http_error(status, text, parsed, headers or {}, _URL)

    def test_rate_limit(self):
        error = self._classify(429, {"error": {"message": "slow down"}}, {"Retry-After": "7"})
        assert isinstance(error, RateLimitError)
        assert error.retry_after == 7.0

    def test_context_limit_structured_code(self):
        assert isinstance(self._classify(400, {"error": {"code": "context_length_exceeded"}}), ContextLimitError)

    def test_context_limit_structured_type(self):
        # `type` is checked as well as `code`: providers populate one or the other.
        assert isinstance(self._classify(400, {"error": {"type": "string_above_max_length"}}), ContextLimitError)

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
        assert isinstance(self._classify(400, {"error": {"code": "content_policy_violation"}}), ContentPolicyError)

    @pytest.mark.parametrize(
        "prose",
        ["blocked by the content filter", "guardrail intervened", "violates our content policy"],
    )
    def test_content_policy_prose(self, prose):
        assert isinstance(self._classify(400, {"error": {"message": prose}}), ContentPolicyError)

    def test_non_json_body_still_classified(self):
        # body_json is None when the error body is not JSON; prose matching runs
        # against the raw text, which is a superset of it.
        error = classify_http_error(400, "maximum context length exceeded", None, {}, _URL)
        assert isinstance(error, ContextLimitError)

    def test_unclassified_is_plain_llm_call_error(self):
        assert type(self._classify(401, {"error": {"message": "invalid api key"}})) is LLMCallError

    def test_structured_code_wins_over_absent_prose(self):
        # The structured check runs first precisely because provider wording drifts.
        assert isinstance(self._classify(400, {"error": {"code": "context_length_exceeded"}}), ContextLimitError)


class TestParseRetryAfter:
    """Header parsing for the 429 path.

    The base class accepts any finite ``>= 0`` and caps at its own ceiling, so a
    bogus value must be rejected *here* or it would either burn the whole
    rate-limit budget instantly or stall every retry.
    """

    def test_delta_seconds(self):
        assert _parse_retry_after({"Retry-After": "12"}) == 12.0

    def test_case_insensitive(self):
        assert _parse_retry_after({"retry-after": "3"}) == 3.0

    def test_http_date_far_future_is_discarded(self):
        assert _parse_retry_after({"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"}) is None

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

    def test_no_headers_at_all(self):
        assert _parse_retry_after(None) is None


class TestCredentialScrubbing:
    """Messages this module builds must not carry a URL's credentials.

    These land in ``AgentError.message`` and from there in the run trajectory, so
    a query-string API key would be persisted alongside ordinary results. Bounds
    disclosure in what we surface; the original is still chained on ``__cause__``.
    """

    def test_transport_error_query_string_is_scrubbed(self):
        # The regression that motivated this: a connection error carries no
        # response, so it takes the prose path — and urllib3's message keeps the
        # path and query even though it drops the scheme and host.
        exc = requests.ConnectionError(
            "HTTPSConnectionPool(host='h', port=443): Max retries exceeded with "
            "url: /v1/chat/completions?api_key=SECRET123 (Caused by NewConnectionError(...))"
        )
        message = str(classify_call_exception(exc))
        assert "SECRET123" not in message
        # Still diagnosable: the failure mode survives scrubbing.
        assert "Max retries exceeded" in message

    def test_full_url_in_a_prose_message_is_scrubbed(self):
        exc = RuntimeError("failed calling https://user:pw@h.invalid/v1/chat?api_key=SECRET123")
        message = str(classify_call_exception(exc))
        assert "SECRET123" not in message
        assert "user:pw" not in message
        assert "h.invalid" in message

    def test_scrubbing_does_not_cost_a_classification(self):
        # The message is scrubbed but the marker matching runs against the raw
        # text, so a URL sitting next to a marker cannot suppress the verdict.
        exc = ValueError("prompt is longer than the maximum model length at https://h/v1?api_key=SECRET123")
        error = classify_call_exception(exc)
        assert isinstance(error, ContextLimitError)
        assert "SECRET123" not in str(error)

    def test_response_body_echoing_a_url_is_scrubbed(self):
        # Defense in depth: some providers echo the request back in the error body.
        error = classify_call_exception(_http_error(400, "bad request to https://h/v1?api_key=SECRET123"))
        assert "SECRET123" not in str(error)

    def test_original_exception_is_still_chained_unscrubbed(self):
        # Scrubbing is about what we surface, not about destroying evidence. The
        # backend attaches __cause__, so a debugger still sees the real message.
        original = requests.ConnectionError("url: /v1?api_key=SECRET123")
        assert "SECRET123" in str(original)
