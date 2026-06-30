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
from typing import Any, Optional

from nemo_retriever.models.llm.tasks.rag_answer import (
    RagAnswerTask,
    _build_rag_prompt as _task_build_rag_prompt,
    _deep_merge_dicts,
    _format_rag_system_prompt,
)
from nemo_retriever.models.llm.types import GenerationResult
from nemo_retriever.common.params.models import LLMInferenceParams, LLMRemoteClientParams

logger = logging.getLogger(__name__)
# Backwards-compatible helper export retained for existing callers.
_build_rag_prompt = _task_build_rag_prompt


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
        self._formatted_rag_system_prompt = _format_rag_system_prompt(
            rag_system_prompt=transport.rag_system_prompt,
            rag_system_prompt_prefix=transport.rag_system_prompt_prefix,
        )

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

    def complete(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        extra_params: Optional[dict[str, Any]] = None,
    ) -> tuple[str, float]:
        """Raw litellm completion call. Returns (content_text, latency_s)."""
        import litellm

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
        if self.transport.api_key:
            call_kwargs["api_key"] = self.transport.api_key
        call_kwargs.update(_deep_merge_dicts(self.transport.extra_params, extra_params or {}))

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
        content = (response.choices[0].message.content or "").strip()
        return content, latency

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
            system_prompt=self._formatted_rag_system_prompt,
            reasoning_enabled=effective_reasoning_enabled,
        )
        result = task.execute(self, query=query, chunks=chunks)
        return GenerationResult(
            answer=result.text,
            latency_s=result.latency_s,
            model=result.model,
            error=result.error,
        )
