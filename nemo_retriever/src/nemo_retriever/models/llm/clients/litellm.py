# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified LLM answer generation client.

LiteLLMClient wraps the litellm library which provides a single interface
for routing to NVIDIA NIM, OpenAI, HuggingFace Inference Endpoints, and
local vLLM / Ollama servers via a model name prefix convention.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Optional

from nemo_retriever.common.params.models import (
    LLMInferenceParams,
    LLMRemoteClientParams,
    resolve_environment_reference,
    validate_llm_extra_params,
)
from nemo_retriever.models.llm.tasks.rag_answer import (
    RagAnswerTask,
    _build_rag_prompt as _task_build_rag_prompt,
    _deep_merge_dicts,
)
from nemo_retriever.models.llm.text_utils import ThinkTagStreamFilter
from nemo_retriever.models.llm.types import (
    GenerationResult,
    UnsupportedTextResponseError,
)

logger = logging.getLogger(__name__)
# Backwards-compatible helper export retained for existing callers.
_build_rag_prompt = _task_build_rag_prompt
_NO_AUTH_API_KEY = "no-api-key"


def _field(value: object, name: str, default: object = None) -> object:
    """Read a provider response field from either a mapping or an object."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class LiteLLMClient:
    """Unified LLM client backed by litellm.

    A single model string change routes to any supported provider:
    - NVIDIA NIM:  nvidia_nim/<org>/<model>
    - OpenAI:      openai/<model>
    - Any OpenAI-compatible server (vLLM, Ollama): openai/<model> + api_base
    - HuggingFace: huggingface/<org>/<model>

    Provider API keys are read from environment variables automatically
    (NVIDIA_API_KEY, OPENAI_API_KEY, HUGGINGFACE_API_KEY, etc.).

    Configuration is split into two orthogonal Pydantic objects:

    * ``transport``: :class:`~nemo_retriever.common.params.LLMRemoteClientParams`
      owns provider endpoint, authentication, retry, and timeout.
    * ``sampling``: :class:`~nemo_retriever.common.params.LLMInferenceParams`
      owns ``temperature``, ``top_p``, and ``max_tokens``.

    Use :meth:`from_kwargs` for a flat, backwards-compatible constructor.
    """

    _DEFAULT_MODEL: str = "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5"
    supports_concurrent_calls: bool = True

    def __init__(
        self,
        transport: LLMRemoteClientParams,
        sampling: Optional[LLMInferenceParams] = None,
    ):
        self.transport = transport
        # Default to ``temperature=0.0, max_tokens=4096`` so the structured
        # constructor matches ``from_kwargs`` and keeps RAG-eval runs
        # deterministic.  ``LLMInferenceParams`` itself defaults to
        # ``max_tokens=1024`` for captioning/summarization workloads; RAG
        # answers routinely exceed that, so the client overrides it.
        self.sampling = sampling if sampling is not None else LLMInferenceParams(temperature=0.0, max_tokens=4096)

    @property
    def model(self) -> str:
        """Return the model identifier from the transport params."""
        return self.transport.model

    @classmethod
    def from_kwargs(
        cls,
        *,
        model: str = _DEFAULT_MODEL,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        max_tokens: int = 4096,
        extra_params: Optional[dict[str, Any]] = None,
        num_retries: int = 3,
        timeout: float = 120.0,
        rag_system_prompt: Optional[str] = None,
        rag_system_prompt_prefix: Optional[str] = None,
        reasoning_enabled: bool = True,
    ) -> "LiteLLMClient":
        """Flat-kwarg constructor for zero-churn migration from the old signature.

        Splits the flat kwargs into the two structured params objects. All
        validation (temperature range, ``num_retries >= 0``, ``timeout > 0``)
        is delegated to the Pydantic models.
        """
        transport = LLMRemoteClientParams(
            model=model,
            api_base=api_base,
            api_key=api_key,
            num_retries=num_retries,
            timeout=timeout,
            extra_params=extra_params or {},
            rag_system_prompt=rag_system_prompt,
            rag_system_prompt_prefix=rag_system_prompt_prefix,
            reasoning_enabled=reasoning_enabled,
        )
        sampling = LLMInferenceParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return cls(transport=transport, sampling=sampling)

    def _build_call_kwargs(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        extra_params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build provider-neutral kwargs shared by sync and streaming completion."""
        validate_llm_extra_params(self.transport.extra_params, source="LLMRemoteClientParams.extra_params")
        validate_llm_extra_params(extra_params, source="GenerationRequest.extra_params")

        sampling_kwargs = self.sampling.to_sampling_kwargs()
        if max_tokens is not None:
            sampling_kwargs["max_tokens"] = max_tokens

        call_kwargs: dict[str, Any] = {
            "model": self.transport.model,
            "messages": messages,
            "num_retries": self.transport.num_retries,
            "timeout": self.transport.timeout,
            **sampling_kwargs,
        }
        if self.transport.api_base:
            call_kwargs["api_base"] = self.transport.api_base
        if self.transport._uses_no_api_key("api_key"):
            call_kwargs["api_key"] = _NO_AUTH_API_KEY
        elif self.transport.api_key is not None:
            resolved_api_key = resolve_environment_reference(self.transport.api_key)
            if resolved_api_key:
                call_kwargs["api_key"] = resolved_api_key
        call_kwargs.update(_deep_merge_dicts(self.transport.extra_params, extra_params or {}))
        return call_kwargs

    @staticmethod
    def _delta_from_stream_chunk(chunk: object) -> str | None:
        """Extract a text delta from one LiteLLM/OpenAI-compatible stream chunk."""
        choices = _field(chunk, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            return None
        choice = choices[0]
        delta = _field(choice, "delta")
        if delta is None:
            message = _field(choice, "message")
            if message is not None:
                delta = message
        if delta is None:
            return None
        content = _field(delta, "content")
        if content is None:
            return None
        if not isinstance(content, str) or not content:
            return None
        return content

    def complete(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        extra_params: Optional[dict[str, Any]] = None,
    ) -> tuple[str, float]:
        """Raw litellm completion call. Returns (content_text, latency_s)."""
        import litellm

        call_kwargs = self._build_call_kwargs(messages, max_tokens, extra_params)

        t0 = time.monotonic()
        try:
            response = litellm.completion(**call_kwargs)
        except Exception as exc:
            err = str(exc)
            if "temperature" in err and "top_p" in err:
                logger.error(
                    "Model %s rejected the request because both `temperature` "
                    "and `top_p` were specified. Some providers (e.g. Bedrock) "
                    "only accept one. Either remove `top_p` from the model "
                    "config or set `temperature` to null. Sent: "
                    "temperature=%s, top_p=%s",
                    self.transport.model,
                    call_kwargs.get("temperature"),
                    call_kwargs.get("top_p"),
                )
            raise
        latency = time.monotonic() - t0

        choices = _field(response, "choices")
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise UnsupportedTextResponseError("provider response must contain exactly one choice")
        choice = choices[0]
        message = _field(choice, "message")
        if message is None:
            raise UnsupportedTextResponseError("provider response choice has no message")
        if _field(choice, "finish_reason") in {"tool_calls", "function_call"}:
            raise UnsupportedTextResponseError("tool-call responses are unsupported by the text completion contract")
        if _field(message, "tool_calls") or _field(message, "function_call"):
            raise UnsupportedTextResponseError("tool-call responses are unsupported by the text completion contract")
        if _field(message, "refusal"):
            raise UnsupportedTextResponseError("refusal responses are unsupported by the text completion contract")
        if any(_field(message, field_name) is not None for field_name in ("audio", "images", "videos")):
            raise UnsupportedTextResponseError("non-text response modalities are unsupported")
        content = _field(message, "content")
        if not isinstance(content, str):
            raise UnsupportedTextResponseError("provider response content must be plain text")
        content = content.strip()
        return content, latency

    async def stream_complete(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        extra_params: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Yield raw text deltas from ``litellm.acompletion(..., stream=True)``."""
        import litellm

        call_kwargs = self._build_call_kwargs(messages, max_tokens, extra_params)
        call_kwargs["stream"] = True

        try:
            response = await litellm.acompletion(**call_kwargs)
        except Exception as exc:
            err = str(exc)
            if "temperature" in err and "top_p" in err:
                logger.error(
                    "Model %s rejected the request because both `temperature` "
                    "and `top_p` were specified. Some providers (e.g. Bedrock) "
                    "only accept one. Either remove `top_p` from the model "
                    "config or set `temperature` to null. Sent: "
                    "temperature=%s, top_p=%s",
                    self.transport.model,
                    call_kwargs.get("temperature"),
                    call_kwargs.get("top_p"),
                )
            raise

        async for chunk in response:
            delta = self._delta_from_stream_chunk(chunk)
            if delta is not None:
                yield delta

    async def stream_generate(
        self,
        query: str,
        chunks: list[str],
        *,
        reasoning_enabled: Optional[bool] = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Stream visible RAG answer tokens and final generation metadata.

        Yields ``("token", {"delta": str, "index": int})`` events followed by
        ``("complete", {"answer": str, "latency_s": float, "model": str, "error": str | None})``.
        """
        effective_reasoning_enabled = (
            self.transport.reasoning_enabled if reasoning_enabled is None else reasoning_enabled
        )
        task = RagAnswerTask(
            system_prompt=self.transport.rag_system_prompt,
            system_prompt_prefix=self.transport.rag_system_prompt_prefix,
            reasoning_enabled=effective_reasoning_enabled,
        )
        request = task.build_request(
            query=query,
            chunks=chunks,
            reasoning_enabled=reasoning_enabled,
        )
        think_filter = ThinkTagStreamFilter() if effective_reasoning_enabled is not False else None

        t0 = time.monotonic()
        ttft_s: float | None = None
        raw_parts: list[str] = []
        token_index = 0

        try:
            async for delta in self.stream_complete(
                request.messages,
                max_tokens=request.max_tokens,
                extra_params=request.extra_params,
            ):
                raw_parts.append(delta)
                visible_deltas = [delta] if think_filter is None else think_filter.feed(delta)
                for visible in visible_deltas:
                    if ttft_s is None:
                        ttft_s = time.monotonic() - t0
                        yield "metrics", {"ttft_s": ttft_s}
                    yield "token", {"delta": visible, "index": token_index}
                    token_index += 1
        except Exception as exc:
            yield "complete", {
                "answer": "",
                "latency_s": time.monotonic() - t0,
                "model": self.model,
                "error": str(exc),
                "ttft_s": ttft_s,
            }
            return

        raw_text = "".join(raw_parts)
        parsed = task.parse(raw_text)
        latency_s = time.monotonic() - t0
        error = None if parsed else task.empty_output_error
        if ttft_s is not None:
            yield "metrics", {"ttft_s": ttft_s, "generation_latency_s": latency_s}
        yield "complete", {
            "answer": parsed,
            "latency_s": latency_s,
            "model": self.model,
            "error": error,
            "ttft_s": ttft_s,
        }

    def generate(
        self,
        query: str,
        chunks: list[str],
        *,
        reasoning_enabled: Optional[bool] = None,
    ) -> GenerationResult:
        """Generate an answer for the given query using retrieved chunks as context."""
        effective_reasoning_enabled = (
            self.transport.reasoning_enabled if reasoning_enabled is None else reasoning_enabled
        )
        task = RagAnswerTask(
            system_prompt=self.transport.rag_system_prompt,
            system_prompt_prefix=self.transport.rag_system_prompt_prefix,
            reasoning_enabled=effective_reasoning_enabled,
        )
        result = task.execute(self, query=query, chunks=chunks)
        return GenerationResult(
            answer=result.text,
            latency_s=result.latency_s,
            model=result.model,
            error=result.error,
        )
