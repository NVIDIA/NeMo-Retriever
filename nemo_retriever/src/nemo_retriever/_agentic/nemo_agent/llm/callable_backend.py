# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM backend that adapts an external chat-completion *callable* to the backend contract.

This is the library's default backend. It carries no HTTP client or LLM serving logic of its own:
everything related to LLM serving or HTTP client logic lives inside the injected callable.

The callable contract
---------------------
The callable is keyword-only and returns an OpenAI-compatible ``chat.completion``
dict. It must accept the full keyword set this backend sends::

    invoke_url, messages, model, api_key, tools, tool_choice, timeout_s,
    temperature, max_tokens, extra_body, max_retries, max_429_retries

Every keyword is sent on every call.

``temperature`` may be ``None``, meaning *unset*. Callables interpret that
themselves — a hosted endpoint should omit the field so the provider default
applies, while an in-process engine that has no provider to defer to should pick
a concrete value. That asymmetry is intentional; see the callables' docstrings.

Unlike the registry-buildable backends, :class:`CallableLLMBackend` needs a live
``completion_fn`` and therefore cannot be constructed from config alone. Callers
inject it: ``create_llm(config, completion_fn=fn)``.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import Field

from .base_backend import BaseLLMBackend, BaseLLMConfig
from .error_classification import classify_call_exception
from .errors import LLMCallError
from .helpers import (
    extract_reasoning_from_message,
    extract_text_content,
    normalize_messages_for_api,
    redact_url,
    resolve_api_key,
    strip_private_message_keys,
)
from .result import CompletionResult
from .usage import coerce_usage_to_dict

logger = logging.getLogger(__name__)

#: The completion callable contract. See the module docstring for the required
#: keyword set — ``Callable[..., Dict]`` cannot express it.
CompletionFn = Callable[..., Dict[str, Any]]

_REDACTED = "***REDACTED***"
#: Request keys whose values are credentials and must be scrubbed from a captured
#: ``raw_request``. Exact-match (not substring) so e.g. ``max_tokens`` is never
#: caught by a naive "token" check.
_SENSITIVE_REQUEST_KEYS = frozenset({"api_key"})
#: Request keys holding a URL, which may carry credentials in userinfo or query.
_URL_REQUEST_KEYS = frozenset({"invoke_url"})


def _redacted_request(call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copied, credential-redacted snapshot of the request."""
    out: Dict[str, Any] = {}
    for key, value in call_kwargs.items():
        if key in _SENSITIVE_REQUEST_KEYS:
            out[key] = _REDACTED
        elif key in _URL_REQUEST_KEYS and value is not None:
            out[key] = redact_url(str(value))
        else:
            out[key] = deepcopy(value)
    return out


class CallableLLMConfig(BaseLLMConfig):
    """Configuration for :class:`CallableLLMBackend`.

    ``backend`` is pinned by type, so a ``CallableLLMConfig`` is only ever routed
    to :class:`CallableLLMBackend`. Request knobs are inherited from
    :class:`BaseLLMConfig` (``model``, ``temperature``, ``tool_choice``,
    ``max_completion_tokens``, ...).

    ``base_url`` (inherited, optional) is the callable's ``invoke_url``, forwarded
    **verbatim**. It stays optional and unvalidated on purpose: an in-process
    callable has no endpoint, and neither this config nor the backend can tell
    a remote callable from an in-process one. A remote callable handed ``None``
    fails on its first call with its own error.

    Attributes
    ----------
    timeout_s:
        Per-request wire timeout handed to the callable.
    max_retries / max_429_retries:
        Retry budget handed to the callable, which owns retrying. Both are
        forwarded rather than acted on here.
    rate_limit_max_retries:
        Redeclared as ``0``. The callable owns retries, so the base template must
        make exactly one attempt; retrying on top would multiply the two budgets.
    """

    backend: Literal["callable"] = "callable"

    timeout_s: float = Field(default=120.0, gt=0)
    #: A TOTAL ATTEMPT count in the callable's own spelling, not a retry count:
    #: 3 means at most 3 requests. ``ge=1`` is load-bearing — a callable that
    #: loops ``while attempt < max_retries`` issues ZERO requests at 0 and then
    #: reports retries-exhausted, which reads as an endpoint failure.
    max_retries: int = Field(default=3, ge=1)
    #: Responses tolerated before giving up on a rate-limited endpoint. Sized so a
    #: sustained 429 fails in roughly two minutes instead of stalling the agent.
    max_429_retries: int = Field(default=6, ge=1)
    #: The callable retries internally, so the base template must not retry too.
    #: Currently reached only when the callable surfaces a rate limit as a typed
    #: ``RateLimitError``; kept explicit so a future callable that reports rate
    #: limits WITHOUT retrying can opt back in by overriding it.
    rate_limit_max_retries: int = 0


class CallableLLMBackend(BaseLLMBackend):
    """Adapter from an OpenAI-compatible completion callable to :class:`BaseLLMBackend`.

    The base class owns usage recording and the rate-limit retry template; this
    subclass implements :meth:`_completion_impl`, shapes the request, parses the
    response, and translates failures.

    Error translation
    -----------------
    An exception raised by the callable is classified by
    :func:`~nemo_agent.llm.error_classification.classify_call_exception` and
    re-raised as the matching :class:`~nemo_agent.llm.errors.LLMCallError`
    subclass, with the original chained on ``__cause__``. Classification is
    best-effort and degrades to a plain ``LLMCallError``: a failure carrying a
    response object is classified from its status and body, anything else from
    its message text. A malformed response dict is surfaced as an explicit
    ``LLMCallError`` rather than a raw ``KeyError``.

    Not populated
    -------------
    :attr:`CompletionResult.extra_response_info` is always ``{}``.  The callable
    returns a decoded response *body*, so response headers, HTTP status, and
    attempt counts do not survive the call boundary and cannot be recovered here.
    """

    config_cls = CallableLLMConfig

    def __init__(self, config: CallableLLMConfig, completion_fn: Optional[CompletionFn] = None) -> None:
        super().__init__(config)
        self.config: CallableLLMConfig
        if completion_fn is None:
            raise ValueError(
                "CallableLLMBackend requires a completion_fn; it cannot be built from config "
                "alone. Pass it via create_llm(config, completion_fn=...)."
            )
        self._completion_fn = completion_fn
        self._api_key = resolve_api_key(config.api_key) or None

    def _completion_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> CompletionResult:
        # Order is load-bearing: private-key stripping runs BEFORE normalization,
        # which collapses text-only content-block lists and discards block metadata.
        prepared = normalize_messages_for_api(strip_private_message_keys(messages))
        call_kwargs = self._build_call_kwargs(prepared, tools, overrides)

        # Client call: the ONLY statement wrapped for error translation. We catch
        # Exception (never BaseException), so cancellation / KeyboardInterrupt
        # still propagate.
        try:
            response = self._completion_fn(**call_kwargs)
        except LLMCallError:
            # Already one of ours — a backend-shaped callable, or a classifier that
            # ran closer to the wire. Re-wrapping would erase the subclass the
            # agent branches on.
            raise
        except Exception as e:
            raise classify_call_exception(e) from e

        # Response parsing stays OUTSIDE the try (a bug here is ours, not the
        # wire's); a malformed response is surfaced as an explicit LLMCallError.
        return self._build_result(response, call_kwargs)

    def _build_call_kwargs(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble the keyword arguments for the completion callable."""
        overrides = dict(overrides)

        extra_body: Dict[str, Any] = {}
        parallel_tool_calls = overrides.pop("parallel_tool_calls", self.config.parallel_tool_calls)
        if parallel_tool_calls is not None:
            extra_body["parallel_tool_calls"] = parallel_tool_calls
        reasoning_effort = overrides.pop("reasoning_effort", self.config.reasoning_effort)
        if reasoning_effort:
            extra_body["reasoning_effort"] = reasoning_effort
        override_extra_body = overrides.pop("extra_body", None)
        if isinstance(override_extra_body, dict):
            extra_body.update(override_extra_body)

        # The FULL contract, always, in the callable's own spelling. Nothing is
        # conditionally omitted (see the module docstring).
        call_kwargs: Dict[str, Any] = {
            "invoke_url": self.config.base_url,
            "messages": messages,
            "model": self.config.model,
            "api_key": self._api_key,
            "tools": tools,
            "tool_choice": "none" if not tools else self.config.tool_choice,
            "timeout_s": self.config.timeout_s,
            # None means "unset" and is forwarded as such; the callable decides
            # whether that means "omit the field" or a concrete default.
            "temperature": overrides.pop("temperature", self.config.temperature),
            "max_tokens": overrides.pop("max_completion_tokens", self.config.max_completion_tokens),
            "extra_body": extra_body,
            "max_retries": self.config.max_retries,
            "max_429_retries": self.config.max_429_retries,
        }
        # Remaining overrides are passed through as keyword arguments so nothing is
        # silently ignored.
        call_kwargs.update(overrides)
        return call_kwargs

    def _build_result(self, response: Any, call_kwargs: Dict[str, Any]) -> CompletionResult:
        if not isinstance(response, dict):
            raise LLMCallError(f"Callable returned {type(response).__name__}, expected an OpenAI chat.completion dict.")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMCallError("Callable response is missing a non-empty 'choices' list.")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMCallError(f"Callable response choice must be a dict, got {type(choice).__name__}.")
        raw_message = choice.get("message")
        if not isinstance(raw_message, dict):
            raise LLMCallError("Callable response choice is missing a 'message' object.")

        # Spec says `content` is a string, but some OpenAI-compatible endpoints
        # return a block list; coerce so one exotic response cannot fail a run.
        message: Dict[str, Any] = {"role": "assistant", "content": extract_text_content(raw_message.get("content"))}
        tool_calls = raw_message.get("tool_calls")
        if tool_calls:
            # Already OpenAI-shaped with ``arguments`` as a JSON string — pass through verbatim.
            message["tool_calls"] = tool_calls

        raw_request: Optional[Dict[str, Any]] = None
        raw_response: Optional[Dict[str, Any]] = None
        if self.config.capture_raw_io:
            # Best-effort: capturing artifacts must never fail an otherwise-good call.
            try:
                raw_request = _redacted_request(call_kwargs)
                raw_response = deepcopy(response)
            except Exception:
                logger.warning("Failed to capture raw LLM IO; continuing without it.", exc_info=True)

        # `.strip()` is not cosmetic: the agent loop treats any finish reason
        # outside ("stop", "tool_calls") as terminal, so a padded " tool_calls "
        # would end an otherwise healthy run.
        raw_finish = choice.get("finish_reason")
        finish_reason = raw_finish.strip() if isinstance(raw_finish, str) and raw_finish.strip() else "stop"

        return CompletionResult(
            message=message,
            finish_reason=finish_reason,
            reasoning=extract_reasoning_from_message(raw_message),
            usage=coerce_usage_to_dict(response.get("usage")),
            raw_request=raw_request,
            raw_response=raw_response,
        )
