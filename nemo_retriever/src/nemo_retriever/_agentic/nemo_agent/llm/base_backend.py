# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class every LLM backend implements, and its base configuration."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from .errors import RateLimitError
from .result import CompletionResult
from .usage import (
    UNSET_QUERY,
    UNSET_STAGE,
    deep_merge_usage,
    deep_merge_usage_breakdown,
    get_query_id,
    get_stage,
    sum_usage_breakdown,
)

logger = logging.getLogger(__name__)

# Ceiling for provider-communicated `RateLimitError.retry_after` values, so a
# bogus Retry-After header can never stall the retry loop for e.g. an hour.
_RETRY_AFTER_CAP_S = 300.0


class BaseLLMConfig(BaseModel):
    """Backend-agnostic LLM configuration.

    Each backend subclasses this with its own fields, colocated with the
    backend class (see ``LiteLLMConfig`` in ``litellm_backend.py``). Note what
    is deliberately absent: raw-IO log paths and error-handling policy belong
    to the caller (agent/pipeline), not the LLM — do not add them back here.

    Attributes
    ----------
    backend:
        Discriminator used by :func:`nemo_agent.llm.create_llm` to pick the
        backend class (kept as a string so yaml/hydra-driven config works).
        Defaults to ``"callable"``, the library's default backend. Note each
        backend subclass pins this by ``Literal``, so the default here only
        applies to a bare ``BaseLLMConfig`` — which cannot build a backend, since
        every backend requires its own config subclass.
    model:
        Model identifier, in whatever form the backend expects.
    api_key:
        API key, or the indirection ``"os.environ/VAR_NAME"`` to resolve the
        key from the environment at backend construction.
    base_url:
        Endpoint base URL, if the backend needs one.
    tool_choice:
        OpenAI-style tool-choice policy; sent only when tools are provided.
    max_completion_tokens:
        Per-request completion budget, if set. Overridable per call.
    reasoning_effort:
        Provider reasoning-effort knob (e.g. ``"low"`` / ``"medium"`` /
        ``"high"``, provider-dependent). Backend-agnostic *declaration*; each
        backend applies or ignores it.
    temperature:
        Sampling temperature. Backend-agnostic *declaration*; each backend
        forwards it when set and leaves it unset (provider default) when ``None``.
    parallel_tool_calls:
        Whether the model may emit multiple tool calls in one turn. Backend-agnostic
        *declaration*; forwarded when set, left unset (provider default) when ``None``.
    capture_raw_io:
        When true, backends populate ``CompletionResult.raw_request`` /
        ``raw_response`` (JSON-serializable, credential-redacted). Off by
        default — the dumps cost memory and nobody should pay for artifacts
        they don't persist.
    rate_limit_max_retries / rate_limit_retry_sleep_s:
        Policy for the base class's retry-on-:class:`RateLimitError` loop
        wrapped around every call. An exception-provided ``retry_after`` wins
        over the configured sleep but never exceeds a hardcoded cap.
    """

    model_config = ConfigDict(extra="forbid")

    backend: str = "callable"
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    tool_choice: str = "auto"
    max_completion_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = None
    parallel_tool_calls: Optional[bool] = None
    capture_raw_io: bool = False
    rate_limit_max_retries: int = 3
    rate_limit_retry_sleep_s: float = 60.0


class BaseLLMBackend(ABC):
    """Chat-completion client with built-in usage tracking and rate-limit retry.

    Subclassing
    -----------
    Implement :meth:`_completion_impl` (sync). Backends with a native async
    client should also override :meth:`_acompletion_impl`; otherwise the
    default bridges via ``asyncio.to_thread`` (contextvars propagate, so usage
    attribution survives the bridge). NEVER override :meth:`completion` /
    :meth:`acompletion` — they are the templates that guarantee usage
    recording and retry policy run on every call.

    Contract for implementations
    ----------------------------
    - Return a fully-assembled :class:`CompletionResult` (see its docstring for
      per-field contracts, including ``capture_raw_io`` gating and
      ``raw_request`` credential redaction).
    - Raise only :class:`~nemo_agent.llm.errors.LLMCallError` subclasses for
      call failures (``raise ... from e``, wrapping only the client call);
      anything else escaping is treated as a bug. See ``errors`` module.
    - Never mutate ``messages`` / ``tools`` in place — callers pass live
      history.
    - ``**overrides`` are per-call, backend-interpreted request overrides
      (e.g. ``max_completion_tokens=64, num_retries=0`` for a preflight
      probe); apply what you understand, tolerate what you don't.

    Usage tracking
    --------------
    After every successful call the template deep-merges ``result.usage`` into
    ``usage[query_id][stage]``, keyed by the ambient
    :func:`~nemo_agent.llm.usage.bind_query_id` /
    :func:`~nemo_agent.llm.usage.bind_stage` bindings (falling back to
    ``UNSET_QUERY`` / ``UNSET_STAGE``). One backend instance is typically
    shared across all queries and stages; accumulation is guarded by a lock
    and is safe under both threads and asyncio. A failure to record usage is
    logged loudly but never fails a completed call.
    """

    #: Each concrete backend MUST set this to its paired config subclass. Read by
    #: ``create_llm_config`` WITHOUT instantiating the backend, and used by
    #: ``__init__`` below for the generic config-type check.
    config_cls: ClassVar[type[BaseLLMConfig]]

    def __init__(self, config: BaseLLMConfig) -> None:
        expected = type(self).config_cls
        if not isinstance(config, expected):
            raise TypeError(f"{type(self).__name__} requires a {expected.__name__}, " f"got {type(config).__name__}")
        self.config = config
        # Name-mangled on purpose: subclasses own the plain `_` namespace, and
        # accidentally clobbering the accumulator must be impossible. Access
        # goes through the public usage accessors below.
        self.__usage: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.__usage_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public templates — do not override.
    # ------------------------------------------------------------------

    def completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> CompletionResult:
        """Make a chat-completion call (sync). See class docstring for the contract."""
        attempt = 0
        while True:
            try:
                result = self._completion_impl(messages, tools=tools, **overrides)
                break
            except RateLimitError as e:
                delay = self._rate_limit_retry_delay(e, attempt)
                if delay is None:
                    raise
                time.sleep(delay)
                attempt += 1
        result = self._require_completion_result(result, "_completion_impl")
        self._record_usage(result.usage)
        return result

    async def acompletion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> CompletionResult:
        """Make a chat-completion call (async). See class docstring for the contract."""
        attempt = 0
        while True:
            try:
                result = await self._acompletion_impl(messages, tools=tools, **overrides)
                break
            except RateLimitError as e:
                delay = self._rate_limit_retry_delay(e, attempt)
                if delay is None:
                    raise
                await asyncio.sleep(delay)
                attempt += 1
        result = self._require_completion_result(result, "_acompletion_impl")
        self._record_usage(result.usage)
        return result

    # ------------------------------------------------------------------
    # Implementation hooks.
    # ------------------------------------------------------------------

    @abstractmethod
    def _completion_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> CompletionResult:
        """Backend-specific synchronous completion. See class docstring."""

    async def _acompletion_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> CompletionResult:
        """Backend-specific async completion.

        Default bridges to :meth:`_completion_impl` in a worker thread.
        Override with a native async client when available.
        """
        return await asyncio.to_thread(self._completion_impl, messages, tools=tools, **overrides)

    def _require_completion_result(self, result: Any, impl_name: str) -> CompletionResult:
        """Make a contract violation self-diagnosing instead of a downstream AttributeError."""
        if not isinstance(result, CompletionResult):
            raise TypeError(
                f"{type(self).__name__}.{impl_name} must return a CompletionResult, " f"got {type(result).__name__}"
            )
        return result

    # ------------------------------------------------------------------
    # Rate-limit retry policy (shared by both templates so they can't drift).
    # ------------------------------------------------------------------

    def _rate_limit_retry_delay(self, e: RateLimitError, attempt: int) -> Optional[float]:
        """Seconds to sleep before retrying ``attempt`` (0-based), or None to re-raise.

        A provider-communicated ``retry_after`` wins over the configured sleep
        but is capped by ``_RETRY_AFTER_CAP_S``; garbage values (negative, NaN,
        non-numeric) fall back to the configured sleep.
        """
        if attempt >= self.config.rate_limit_max_retries:
            return None
        retry_after: Optional[float] = None
        if e.retry_after is not None:
            try:
                retry_after = float(e.retry_after)
            except (TypeError, ValueError):
                retry_after = None
        if retry_after is not None and 0 <= retry_after < float("inf"):
            delay = min(retry_after, _RETRY_AFTER_CAP_S)
        else:
            delay = self.config.rate_limit_retry_sleep_s
        logger.info(
            "Rate limited; sleeping %.1fs before retry %d/%d.",
            delay,
            attempt + 1,
            self.config.rate_limit_max_retries,
        )
        return float(delay)

    # ------------------------------------------------------------------
    # Usage tracking.
    # ------------------------------------------------------------------

    def _record_usage(self, usage: Optional[Dict[str, Any]]) -> None:
        if not usage:
            return
        query_id = get_query_id() or UNSET_QUERY
        stage = get_stage() or UNSET_STAGE
        try:
            with self.__usage_lock:
                bucket = self.__usage.setdefault(query_id, {}).setdefault(stage, {})
                deep_merge_usage(bucket, usage)
        except Exception:
            logger.warning(
                "Failed to record LLM token usage for query_id=%r stage=%r; " "usage totals will undercount.",
                query_id,
                stage,
                exc_info=True,
            )

    def get_usage(self, query_id: Optional[str] = None) -> Dict[str, Any]:
        """Deep copy of accumulated usage.

        With ``query_id``: that query's ``{stage: usage}`` breakdown (``{}`` if
        unknown). Without: the full ``{query_id: {stage: usage}}`` mapping.
        """
        with self.__usage_lock:
            if query_id is None:
                return deepcopy(self.__usage)
            return deepcopy(self.__usage.get(query_id, {}))

    def pop_query_usage(self, query_id: str) -> Dict[str, Any]:
        """Remove and return one query's ``{stage: usage}`` breakdown (``{}`` if unknown)."""
        with self.__usage_lock:
            return self.__usage.pop(query_id, {})

    def get_usage_by_stage(self) -> Dict[str, Any]:
        """Derived run totals: ``{stage: usage}`` merged across all queries."""
        with self.__usage_lock:
            totals: Dict[str, Any] = {}
            for stage_breakdown in self.__usage.values():
                deep_merge_usage_breakdown(totals, stage_breakdown)
            return totals

    def get_total_usage(self) -> Dict[str, Any]:
        """Derived grand total across all queries and stages."""
        return sum_usage_breakdown(self.get_usage_by_stage())

    def reset_usage(self) -> None:
        """Drop all accumulated usage."""
        with self.__usage_lock:
            self.__usage = {}
