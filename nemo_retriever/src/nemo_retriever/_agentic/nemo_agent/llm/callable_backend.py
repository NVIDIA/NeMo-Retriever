# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM backend that adapts an external chat-completion *callable* to the backend contract.

The callable must accept OpenAI-style ``messages`` / ``tools`` (keyword-only) and
return an OpenAI-compatible ``chat.completion`` dict — the shape produced by
``nemo_retriever.models.local.agent_llm.VLLMAgentChatLLM.__call__`` and by the
hosted ``invoke_chat_completion_step``. This backend does the request shaping and
response parsing itself; everything provider-specific lives inside the callable.

Unlike the registry-buildable backends, :class:`CallableLLMBackend` needs a live
``completion_fn`` and therefore cannot be constructed from config alone. Callers
inject it: ``create_llm(config, completion_fn=fn)``.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional

from .base_backend import BaseLLMBackend, BaseLLMConfig
from .errors import LLMCallError
from .helpers import extract_reasoning_from_message, strip_private_message_keys
from .result import CompletionResult

logger = logging.getLogger(__name__)

#: The completion callable contract: keyword-only, returns an OpenAI ``chat.completion`` dict.
CompletionFn = Callable[..., Dict[str, Any]]

_REDACTED = "***REDACTED***"
#: Request keys whose values are credentials and must be scrubbed from a captured
#: ``raw_request``. Exact-match (not substring) so e.g. ``max_tokens`` is never
#: caught by a naive "token" check.
_SENSITIVE_REQUEST_KEYS = frozenset({"api_key"})


def _redacted_request(call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copied, credential-redacted snapshot of the request."""
    out: Dict[str, Any] = {}
    for key, value in call_kwargs.items():
        out[key] = _REDACTED if key in _SENSITIVE_REQUEST_KEYS else deepcopy(value)
    return out


class CallableLLMConfig(BaseLLMConfig):
    """Configuration for :class:`CallableLLMBackend`.

    ``backend`` is pinned by type, so a ``CallableLLMConfig`` is only ever routed
    to :class:`CallableLLMBackend`. All request knobs are inherited from
    :class:`BaseLLMConfig` (``model``, ``temperature``, ``tool_choice``,
    ``max_completion_tokens``, ...).
    """

    backend: Literal["callable"] = "callable"


class CallableLLMBackend(BaseLLMBackend):
    """Adapter from an OpenAI-compatible completion callable to :class:`BaseLLMBackend`.

    The base class owns rate-limit retry and usage recording; this subclass
    implements :meth:`_completion_impl` and translates failures: an exception
    raised by the callable is wrapped as
    :class:`~nemo_agent.llm.errors.LLMCallError` (the original chained on
    ``__cause__``), and a malformed response dict is surfaced as an
    ``LLMCallError`` rather than a raw ``KeyError``. Provider-specific
    classification (e.g. context-limit) is intentionally not attempted, since the
    callable is provider-agnostic. Usage tracking is intentionally stubbed:
    :attr:`CompletionResult.usage` is ``None`` (the base's accumulator
    short-circuits on falsy usage), so no per-(query, stage) tokens are recorded
    for callable-backed runs.
    """

    config_cls = CallableLLMConfig

    def __init__(self, config: CallableLLMConfig, completion_fn: Optional[CompletionFn] = None) -> None:
        super().__init__(config)
        if completion_fn is None:
            raise ValueError(
                "CallableLLMBackend requires a completion_fn; it cannot be built from config "
                "alone. Pass it via create_llm(config, completion_fn=...)."
            )
        self._completion_fn = completion_fn

    def _completion_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> CompletionResult:
        prepared = strip_private_message_keys(messages)
        call_kwargs = self._build_call_kwargs(prepared, tools, overrides)

        # Client call: the ONLY statement wrapped for error translation. The
        # callable is the "wire" — any failure it raises becomes an LLMCallError
        # with the original chained on ``__cause__``. We catch Exception (never
        # BaseException), so cancellation / KeyboardInterrupt still propagate.
        try:
            response = self._completion_fn(**call_kwargs)
        except Exception as e:
            raise LLMCallError(f"completion callable failed: {e}") from e

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

        call_kwargs: Dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "tool_choice": "none" if not tools else self.config.tool_choice,
            "max_tokens": overrides.pop("max_completion_tokens", self.config.max_completion_tokens),
            "model": self.config.model,
            "extra_body": extra_body,
        }
        # Temperature is forwarded only when set. When None we omit it entirely so
        # the callable keeps its own default — its parameter is a plain float that
        # cannot accept None, and sending 0.0 would force a value the caller never
        # asked for.
        if self.config.temperature is not None:
            call_kwargs["temperature"] = self.config.temperature
        # Remaining overrides: a key naming a real callable parameter binds to it
        # (a ``temperature`` override lands here too); anything else is passed
        # through as a keyword argument so nothing is silently ignored.
        call_kwargs.update(overrides)
        return call_kwargs

    def _build_result(self, response: Any, call_kwargs: Dict[str, Any]) -> CompletionResult:
        if not isinstance(response, dict):
            raise LLMCallError(
                f"Callable returned {type(response).__name__}, expected an OpenAI chat.completion dict."
            )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMCallError("Callable response is missing a non-empty 'choices' list.")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMCallError(f"Callable response choice must be a dict, got {type(choice).__name__}.")
        raw_message = choice.get("message")
        if not isinstance(raw_message, dict):
            raise LLMCallError("Callable response choice is missing a 'message' object.")

        message: Dict[str, Any] = {"role": "assistant", "content": raw_message.get("content")}
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

        return CompletionResult(
            message=message,
            finish_reason=choice.get("finish_reason") or "stop",
            reasoning=extract_reasoning_from_message(raw_message),
            usage=None,
            raw_request=raw_request,
            raw_response=raw_response,
        )
