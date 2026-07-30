# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct-HTTP implementation of :class:`BaseLLMBackend` for OpenAI-compatible endpoints.

Talks to any endpoint implementing the ``/v1/chat/completions`` contract
(build.nvidia.com, self-hosted NIMs, vLLM, ...) with ``httpx`` and no LLM SDK.
``base_url`` is the POST target and is used **verbatim** — normalizing the
caller's endpoint is not this backend's job.

Scope note: this backend targets OpenAI-compatible wire formats only. It does
not carry the multi-provider response translation an SDK like litellm does;
see :func:`_normalize_finish_reason`.
"""

from __future__ import annotations

import logging
import os
import random
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import Field

from .base_backend import BaseLLMBackend, BaseLLMConfig
from .errors import ContentPolicyError, ContextLimitError, LLMCallError, RateLimitError
from .helpers import (
    extract_reasoning_from_message,
    extract_text_content,
    normalize_messages_for_api,
    strip_private_message_keys,
)
from .result import CompletionResult
from .usage import coerce_usage_to_dict

logger = logging.getLogger(__name__)

_REDACTED = "***REDACTED***"
#: Exact-match (never substring) so e.g. ``max_tokens`` is not caught by a naive
#: "token" check.
_SENSITIVE_HEADER_KEYS = frozenset({"authorization"})
_SENSITIVE_BODY_KEYS = frozenset({"api_key"})

_ENV_PREFIX = "os.environ/"

#: Statuses retried inside the backend. 429 is deliberately ABSENT: the base
#: class owns rate-limit pauses (see ``OpenAIHTTPBackend`` docstring).
_RETRYABLE_STATUS = frozenset({408, 500, 502, 503, 504})
_RETRY_BACKOFF_BASE_S = 0.5
_RETRY_BACKOFF_MAX_S = 8.0

#: ``Retry-After`` values above this are treated as unusable; the base class's
#: configured sleep is a better answer than a 15-minute stall.
_RETRY_AFTER_MAX_S = 300.0

#: Cap on provider text echoed into exception messages and logs. The legacy
#: implementation embedded whole response bodies.
_BODY_EXCERPT_CHARS = 2000

#: The only alias worth carrying for OpenAI-compatible endpoints: ``function_call``
#: is the spec's deprecated tool-call reason, and litellm maps it for free, so
#: omitting it would regress this backend against ``litellm`` on the same endpoint.
#: Anthropic/Gemini/Cohere families are deliberately NOT here — they are
#: unreachable through an OpenAI-compatible route, and unmapped values pass
#: through so a surprise fails loudly instead of degrading silently.
_FINISH_REASON_ALIASES = {"function_call": "tool_calls"}

_CONTEXT_LIMIT_CODES = frozenset({"context_length_exceeded", "string_above_max_length"})
_CONTENT_POLICY_CODES = frozenset({"content_filter", "content_policy_violation"})

_CONTENT_POLICY_MARKERS = (
    "content filter",
    "content_filter",
    "content policy",
    "content_policy",
    "content management policy",
    "guardrail",
    "responsible ai",
)


# ----------------------------------------------------------------------
# Pure helpers. None of these log — their callers do.
# ----------------------------------------------------------------------


def _header(headers: Optional[Mapping[str, Any]], name: str) -> Optional[Any]:
    """Case-insensitive header lookup that works on any mapping."""
    if not headers:
        return None
    for key, value in headers.items():
        if str(key).lower() == name:
            return value
    return None


def _redact_url(url: str) -> str:
    """Drop userinfo, query, and fragment so a URL is safe to log or persist."""
    try:
        parts = urlsplit(str(url))
        netloc = parts.hostname or ""
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except Exception:
        return "<unparseable-url>"


def _redact_headers(headers: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy of ``headers`` with credential values replaced. Never mutates the input."""
    return {
        key: (_REDACTED if str(key).lower() in _SENSITIVE_HEADER_KEYS else deepcopy(value))
        for key, value in headers.items()
    }


def _redact_body(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-copied, credential-redacted snapshot of the request body."""
    return {key: (_REDACTED if str(key) in _SENSITIVE_BODY_KEYS else deepcopy(value)) for key, value in payload.items()}


def _excerpt(text: Any) -> str:
    body = str(text or "")
    return body if len(body) <= _BODY_EXCERPT_CHARS else body[:_BODY_EXCERPT_CHARS] + "..."


def _pop_retries(overrides: Dict[str, Any], default: int) -> int:
    """Consume the transient-retry override so it can never reach the wire.

    Both keys are popped unconditionally. ``num_retries`` is litellm's spelling
    and is named as an example override in the base class contract;
    ``max_transient_retries`` is this backend's own name and wins when both are
    given.
    """
    value = overrides.pop("num_retries", None)
    explicit = overrides.pop("max_transient_retries", None)
    if explicit is not None:
        value = explicit
    if value is None:
        return int(default)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return int(default)


def _build_payload(
    config: "OpenAIHTTPConfig",
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the chat-completions request body. Never mutates its inputs.

    Order is load-bearing: private-key stripping runs BEFORE normalization,
    which collapses text-only content-block lists and discards block metadata.

    ``overrides`` are consumed in three steps here (the transient-retry keys are
    already gone, see :func:`_pop_retries`): credentials are discarded, keys that
    name a config field are remapped/gated exactly like the config value they
    shadow, and whatever remains is merged verbatim.
    """
    overrides = dict(overrides)
    # A credential never belongs in the body; the resolved key lives in the
    # Authorization header. Callers do pass this override (it is exercised
    # against the callable backend in the operator test suite).
    overrides.pop("api_key", None)

    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": normalize_messages_for_api(strip_private_message_keys(messages)),
    }

    # The wire key is `max_tokens`; `max_completion_tokens` is the library-side
    # name and is the canonical per-call override, so it must be remapped rather
    # than merged, or it would be sent alongside and silently ignored.
    max_tokens = overrides.pop("max_completion_tokens", config.max_completion_tokens)
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    # None means "provider default" — send nothing rather than forcing 0.0.
    temperature = overrides.pop("temperature", config.temperature)
    if temperature is not None:
        payload["temperature"] = temperature

    reasoning_effort = overrides.pop("reasoning_effort", config.reasoning_effort)
    if isinstance(reasoning_effort, str) and reasoning_effort.strip():
        payload["reasoning_effort"] = reasoning_effort.strip()

    # Popped whether or not tools are present, so an override cannot re-introduce
    # tool-only params on a tool-less request.
    tool_choice = overrides.pop("tool_choice", config.tool_choice)
    parallel_tool_calls = overrides.pop("parallel_tool_calls", config.parallel_tool_calls)
    if tools:
        # Shallow copy: tool order is load-bearing, so never sort or dedupe.
        payload["tools"] = list(tools)
        payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls

    payload.update(overrides)
    return payload


def _parse_retry_after(headers: Mapping[str, Any]) -> Optional[float]:
    """Seconds from a ``Retry-After`` header, or None when unusable.

    Accepts delta-seconds and RFC-9110 HTTP-date. Returns None for ``<= 0`` and
    for implausibly large values: the base class accepts any finite ``>= 0`` and
    caps at its own ceiling, so ``Retry-After: 0`` would burn the whole
    rate-limit budget instantly and a skewed date would stall every retry.
    """
    raw = _header(headers, "retry-after")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()

    if seconds <= 0 or seconds > _RETRY_AFTER_MAX_S:
        return None
    return seconds


def _normalize_finish_reason(raw: Any, message: Dict[str, Any]) -> str:
    """Coerce a provider finish reason to the non-empty string the envelope requires.

    A wire value is passed through (after alias mapping) and the message shape is
    NOT consulted, so ``length`` and ``content_filter`` stay terminal even when a
    truncated ``tool_calls`` array is present. Only an absent/empty/non-string
    value is inferred from the message — that branch exists because the result
    envelope rejects an empty finish reason outright.
    """
    if isinstance(raw, str) and raw.strip():
        value = raw.strip()
        return _FINISH_REASON_ALIASES.get(value, value)
    return "tool_calls" if message.get("tool_calls") else "stop"


def _error_fields(body_json: Any) -> Tuple[str, str]:
    """``(code, type)`` from an OpenAI-style error envelope, lowercased; ``("", "")`` if absent."""
    if not isinstance(body_json, dict):
        return "", ""
    error = body_json.get("error")
    if not isinstance(error, dict):
        return "", ""
    code = str(error.get("code") or "").strip().lower()
    type_ = str(error.get("type") or "").strip().lower()
    return code, type_


def _classify_error(
    status_code: int,
    body_text: str,
    body_json: Any,
    headers: Mapping[str, Any],
    url: str,
) -> LLMCallError:
    """Map an HTTP error response to the exception class the agent branches on.

    Structured ``error.code`` / ``error.type`` are checked first because they are
    stable; prose markers are a fallback because provider wording drifts.
    ``body_json`` may be None (an error body is not always JSON) — every prose
    check runs against ``body_text``, which is a superset of it.

    Pure: an unclassified error is returned as a plain ``LLMCallError`` and the
    caller decides whether to log it.
    """
    message = f"HTTP {status_code} from {_redact_url(url)}: {_excerpt(body_text)}"

    if status_code == 429:
        return RateLimitError(message, retry_after=_parse_retry_after(headers))

    code, type_ = _error_fields(body_json)
    if code in _CONTEXT_LIMIT_CODES or type_ in _CONTEXT_LIMIT_CODES:
        return ContextLimitError(message)
    if code in _CONTENT_POLICY_CODES or type_ in _CONTENT_POLICY_CODES:
        return ContentPolicyError(message)

    lowered = str(body_text or "").lower()
    if (
        "contextwindowexceedederror" in lowered
        or ("context" in lowered and "window" in lowered)
        or ("context" in lowered and "reduce" in lowered)
        or "maximum context length" in lowered
        or "is longer than the maximum model length" in lowered
        or "please reduce the length" in lowered
        # Prompt so long the completion budget went negative — a context-overflow
        # symptom rather than a literal context-window message.
        or ("max_tokens must be at least 1" in lowered and "got -" in lowered)
    ):
        return ContextLimitError(message)
    if any(marker in lowered for marker in _CONTENT_POLICY_MARKERS):
        return ContentPolicyError(message)

    return LLMCallError(message)


def _full_jitter(attempt: int) -> float:
    ceiling = min(_RETRY_BACKOFF_BASE_S * (2**attempt), _RETRY_BACKOFF_MAX_S)
    return random.uniform(0.0, ceiling)


def _try_json(response: httpx.Response) -> Any:
    """Best-effort decode for the error path; None when the body is not JSON."""
    try:
        return response.json()
    except ValueError:
        return None


def _post_with_retries(
    client: httpx.Client,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    max_transient_retries: int,
) -> Tuple[httpx.Response, int]:
    """POST with bounded retries for transient failures.

    Returns ``(response, attempts_used)``. Every path returns or raises.

    Only transport errors and ``_RETRYABLE_STATUS`` are retried. 429 is excluded
    on purpose: it becomes a ``RateLimitError`` and the base class's own loop owns
    the pause, so retrying here would multiply the two budgets.
    """
    last = int(max_transient_retries)
    for attempt in range(last + 1):
        try:
            # The client call: the ONLY statement wrapped for error translation.
            # `HTTPError`, not `TransportError` — DecodingError and TooManyRedirects
            # are the former but not the latter. Never BaseException, so task
            # cancellation still propagates.
            response = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as e:
            if attempt == last:
                raise LLMCallError(f"request to {_redact_url(url)} failed: {e}") from e
        else:
            if response.status_code < 400:
                return response, attempt + 1
            if response.status_code not in _RETRYABLE_STATUS or attempt == last:
                error = _classify_error(response.status_code, response.text, _try_json(response), response.headers, url)
                if 400 <= response.status_code < 500 and type(error) is LLMCallError:
                    # The only signal that a provider rephrased an error we used to
                    # classify. Emitted here, on the final non-retried failure, so
                    # transient 5xx retries never drown it out.
                    logger.warning(
                        "openai_http: unclassified HTTP %s from %s; body: %s",
                        response.status_code,
                        _redact_url(url),
                        _excerpt(response.text),
                    )
                raise error
        time.sleep(_full_jitter(attempt))
    # Unreachable while max_transient_retries >= 0 (enforced by the config), but
    # it keeps the helper total.
    raise LLMCallError(f"retry loop exited without a response after {last + 1} attempts")


def _resolve_api_key(api_key: Optional[str]) -> str:
    """Resolve the configured key, following an ``os.environ/VAR`` indirection."""
    raw = (api_key or "").strip()
    if not raw.startswith(_ENV_PREFIX):
        return raw
    var = raw[len(_ENV_PREFIX) :].strip()
    try:
        return os.environ[var].strip()
    except KeyError:
        raise ValueError(f"Environment variable '{var}' is not set. Set it with: export {var}=<your-api-key>") from None


# ----------------------------------------------------------------------
# Config + backend.
# ----------------------------------------------------------------------


class OpenAIHTTPConfig(BaseLLMConfig):
    """Configuration for :class:`OpenAIHTTPBackend`.

    ``backend`` is pinned by type: an ``OpenAIHTTPConfig`` cannot be constructed
    with (and therefore never routed to) a different backend.

    ``base_url`` is redeclared as **required** and is the POST target, used
    verbatim — deliberately unlike ``LiteLLMConfig``, which strips a trailing
    ``/chat/completions`` because litellm re-appends it.

    Attributes
    ----------
    timeout_s / connect_timeout_s:
        Read and connect timeouts, applied to the underlying client. A wire
        timeout is functionally required, not cosmetic: the async path runs this
        backend in a worker thread, and a blocking socket read there cannot be
        cancelled.
    max_transient_retries:
        In-backend retries for transport errors and retryable 5xx/408. Attempts
        are ``max_transient_retries + 1``. Overridable per call. Rate limits are
        NOT retried here; see :class:`OpenAIHTTPBackend`.
    """

    backend: Literal["openai_http"] = "openai_http"
    base_url: str
    timeout_s: float = 120.0
    connect_timeout_s: float = 10.0
    max_transient_retries: int = Field(default=2, ge=0)


class OpenAIHTTPBackend(BaseLLMBackend):
    """LLM backend that calls OpenAI-compatible endpoints over direct HTTP.

    No LLM SDK: request shaping, retry, error classification, and response
    parsing all live here.

    Retry split
    -----------
    Transport errors and ``_RETRYABLE_STATUS`` are retried in-backend. 429 raises
    ``RateLimitError`` immediately with a parsed ``Retry-After`` and is retried by
    the base class instead. That split is deliberate: the base re-invokes the impl
    from scratch on each rate-limit retry, so an in-backend 429 loop would
    multiply the two budgets — worst case
    ``(max_transient_retries + 1) x (rate_limit_max_retries + 1)`` requests, i.e.
    12 with the defaults.

    Concurrency
    -----------
    One instance is shared across queries. ``_acompletion_impl`` is deliberately
    NOT overridden: the agent runs one event loop per query on a pool thread, so a
    cached async client would bind to a loop that is closed after the first query.
    The inherited thread bridge propagates contextvars, so usage attribution
    survives it. ``httpx.Client`` is thread-safe and pools connections.
    """

    config_cls = OpenAIHTTPConfig

    def __init__(self, config: OpenAIHTTPConfig, *, http_client: Optional[httpx.Client] = None) -> None:
        super().__init__(config)
        self.config: OpenAIHTTPConfig

        self._url = config.base_url
        api_key = _resolve_api_key(config.api_key)
        self._headers: Dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        # Headers are sent per request rather than bound to the client, so an
        # injected client still carries them.
        self._owns_client = http_client is None
        self._client = http_client if http_client is not None else httpx.Client()
        # Applied regardless of ownership: a bare httpx.Client() defaults to a 5s
        # timeout and follow_redirects=False, which would time out every real call
        # and read a 3xx as success. Ownership governs close() only.
        self._client.timeout = httpx.Timeout(config.timeout_s, connect=config.connect_timeout_s)
        self._client.follow_redirects = True

    def close(self) -> None:
        """Release the connection pool if this backend created it."""
        if self._owns_client:
            self._client.close()

    # ------------------------------------------------------------------
    # Impl.
    # ------------------------------------------------------------------

    def _completion_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> CompletionResult:
        call_overrides = dict(overrides)
        retries = _pop_retries(call_overrides, self.config.max_transient_retries)
        payload = _build_payload(self.config, messages, tools, call_overrides)

        response, attempts = _post_with_retries(self._client, self._url, payload, self._headers, retries)

        # Parsing stays outside every translating try: a failure here is our bug,
        # not the wire's, except for the explicit malformed-body cases below.
        body = self._decode(response)
        return self._build_result(response, body, payload, attempts)

    def _decode(self, response: httpx.Response) -> Any:
        """Decode a successful response body.

        Not retried: a non-JSON 2xx is an HTML error page from a proxy or a
        misrouted endpoint, and replaying it would burn the whole retry budget.
        """
        try:
            return response.json()
        except ValueError as e:
            raise LLMCallError(
                f"non-JSON response body from {_redact_url(self._url)}: {_excerpt(response.text)!r}"
            ) from e

    # ------------------------------------------------------------------
    # Result assembly.
    # ------------------------------------------------------------------

    def _build_result(
        self,
        response: httpx.Response,
        body: Any,
        payload: Dict[str, Any],
        attempts: int,
    ) -> CompletionResult:
        raw_message = self._extract_message(body)

        # Spec says `content` is a string, but some OpenAI-compatible endpoints
        # return a block list; shared with the litellm backend so both degrade
        # the same way instead of one hard-failing the run. A shape that is
        # neither raises LLMCallError from there.
        content = extract_text_content(raw_message.get("content"))

        message: Dict[str, Any] = {"role": "assistant", "content": content}
        tool_calls = self._extract_tool_calls(raw_message)
        if tool_calls:
            # `arguments` stays the raw JSON string; the agent owns parsing it and
            # recovering from malformed arguments. Copied so nothing is aliased.
            message["tool_calls"] = tool_calls

        raw_finish = self._choice(body).get("finish_reason")
        if not (isinstance(raw_finish, str) and raw_finish.strip()):
            logger.warning(
                "openai_http: response omitted finish_reason; inferring %r from message shape.",
                "tool_calls" if message.get("tool_calls") else "stop",
            )

        raw_request: Optional[Dict[str, Any]] = None
        raw_response: Optional[Dict[str, Any]] = None
        if self.config.capture_raw_io:
            # Best-effort: capturing artifacts must never fail an otherwise-good call.
            try:
                raw_request = {
                    "url": _redact_url(self._url),
                    "headers": _redact_headers(self._headers),
                    "body": _redact_body(payload),
                }
                raw_response = deepcopy(body)
            except Exception:
                logger.warning("Failed to capture raw LLM IO; continuing without it.", exc_info=True)

        finish_reason = _normalize_finish_reason(raw_finish, message)
        if finish_reason == "tool_calls" and not message.get("tool_calls"):
            raise LLMCallError("Chat-completions response finished with tool calls but supplied no tool-call objects.")

        return CompletionResult(
            message=message,
            finish_reason=finish_reason,
            reasoning=extract_reasoning_from_message(raw_message),
            usage=coerce_usage_to_dict(body.get("usage")),
            extra_response_info=self._extra_response_info(response, attempts),
            raw_request=raw_request,
            raw_response=raw_response,
        )

    def _choice(self, body: Any) -> Dict[str, Any]:
        """The single choice, or an explicit ``LLMCallError`` — never a bare KeyError."""
        if not isinstance(body, dict):
            raise LLMCallError(f"Expected a JSON object from the chat-completions endpoint, got {type(body).__name__}.")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            count = len(choices) if isinstance(choices, list) else 0
            raise LLMCallError(f"Expected exactly 1 choice in the chat-completions response, got {count}.")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMCallError(f"Chat-completions choice must be an object, got {type(choice).__name__}.")
        return choice

    def _extract_message(self, body: Any) -> Dict[str, Any]:
        raw_message = self._choice(body).get("message")
        if not isinstance(raw_message, dict):
            raise LLMCallError("Chat-completions choice is missing a 'message' object.")
        return raw_message

    def _extract_tool_calls(self, raw_message: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Validate modern tool calls and normalize the deprecated function-call shape."""
        tool_calls = raw_message.get("tool_calls")
        if tool_calls is None:
            function_call = raw_message.get("function_call")
            if function_call is None:
                return None
            if not isinstance(function_call, dict):
                raise LLMCallError(
                    f"Chat-completions message 'function_call' must be an object, "
                    f"got {type(function_call).__name__}."
                )
            tool_calls = [
                {
                    "id": f"call_legacy_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": function_call,
                }
            ]

        if not isinstance(tool_calls, list):
            raise LLMCallError(
                f"Chat-completions message 'tool_calls' must be a list, got {type(tool_calls).__name__}."
            )

        validated = deepcopy(tool_calls)
        for index, call in enumerate(validated):
            if not isinstance(call, dict):
                raise LLMCallError(
                    f"Chat-completions tool call at index {index} must be an object, got {type(call).__name__}."
                )
            function = call.get("function")
            if not isinstance(function, dict):
                raise LLMCallError(
                    f"Chat-completions tool call at index {index} is missing a 'function' object."
                )
        return validated

    def _extra_response_info(self, response: httpx.Response, attempts: int) -> Dict[str, Any]:
        """Schema-unstable, log-only. Degrades to ``{}`` rather than failing a good call.

        The ``ratelimit`` keys match the litellm backend's so the agent's progress
        line renders identically across backends. Everything here must stay
        JSON-serializable — the agent dumps it to disk.
        """
        try:
            tpm = _header(response.headers, "x-ratelimit-remaining-tokens")
            rq = _header(response.headers, "x-ratelimit-remaining-requests")
            return {
                "ratelimit": {
                    "TPM": None if tpm is None else str(tpm),
                    "RQ": None if rq is None else str(rq),
                },
                "http": {"status": int(response.status_code), "attempts": int(attempts)},
            }
        except Exception:
            return {}
