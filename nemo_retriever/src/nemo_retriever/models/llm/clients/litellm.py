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

from copy import deepcopy
import logging
import time
from typing import Any, Optional

from nemo_retriever.models.llm.text_utils import strip_think_tags
from nemo_retriever.models.llm.types import GenerationResult
from nemo_retriever.common.params.models import LLMInferenceParams, LLMRemoteClientParams

logger = logging.getLogger(__name__)

_RAG_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the information provided in the context below. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Be concise and factual."
)

_RAG_USER_TEMPLATE = """\
Context:
{context}

Question: {query}

Answer:"""

_NO_REASONING_SYSTEM_DIRECTIVE = "/no_think"
_NO_REASONING_EXTRA_PARAMS = {"chat_template_kwargs": {"enable_thinking": False}}


def _format_rag_system_prompt(
    *,
    rag_system_prompt: Optional[str] = None,
    rag_system_prompt_prefix: Optional[str] = None,
) -> str:
    """Resolve the system prompt used for RAG answer generation."""
    prompt = (rag_system_prompt if rag_system_prompt is not None else _RAG_SYSTEM_PROMPT).strip()
    prefix = (rag_system_prompt_prefix or "").strip()
    if not prefix:
        return prompt
    if not prompt:
        return prefix
    return f"{prefix}\n{prompt}"


def _build_rag_prompt(
    query: str,
    chunks: list[str],
    *,
    formatted_rag_system_prompt: str,
) -> list[dict]:
    """Build the OpenAI-style messages list for a RAG prompt."""
    context = "\n\n---\n\n".join(chunks) if chunks else "(no context retrieved)"
    user_content = _RAG_USER_TEMPLATE.format(context=context, query=query)
    return [
        {"role": "system", "content": formatted_rag_system_prompt},
        {"role": "user", "content": user_content},
    ]


def _deep_merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge where ``right`` wins without mutating inputs."""
    merged = deepcopy(left)
    for key, value in right.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _with_no_reasoning_controls(messages: list[dict]) -> list[dict]:
    """Add no-reasoning prompt metadata understood by current Nemotron LLM NIMs."""
    updated = [dict(message) for message in messages]
    if updated and updated[0].get("role") == "system":
        content = str(updated[0].get("content") or "").strip()
        if _NO_REASONING_SYSTEM_DIRECTIVE not in content:
            content = f"{_NO_REASONING_SYSTEM_DIRECTIVE}\n{content}" if content else _NO_REASONING_SYSTEM_DIRECTIVE
        updated[0]["content"] = content
        return updated
    updated.insert(0, {"role": "system", "content": _NO_REASONING_SYSTEM_DIRECTIVE})
    return updated


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

    * ``transport``: :class:`~nemo_retriever.params.LLMRemoteClientParams`
      owns provider endpoint, authentication, retry, and timeout.
    * ``sampling``: :class:`~nemo_retriever.params.LLMInferenceParams`
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
        temperature: float = 0.0,
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
        messages = _build_rag_prompt(
            query,
            chunks,
            formatted_rag_system_prompt=self._formatted_rag_system_prompt,
        )
        request_extra_params: dict[str, Any] | None = None
        effective_reasoning_enabled = (
            self.transport.reasoning_enabled if reasoning_enabled is None else reasoning_enabled
        )
        if not effective_reasoning_enabled:
            messages = _with_no_reasoning_controls(messages)
            request_extra_params = _NO_REASONING_EXTRA_PARAMS
        try:
            raw_answer, latency = self.complete(messages, extra_params=request_extra_params)
            answer = strip_think_tags(raw_answer)
            if not answer:
                return GenerationResult(
                    answer="",
                    latency_s=latency,
                    model=self.transport.model,
                    error="thinking_truncated",
                )
            return GenerationResult(answer=answer, latency_s=latency, model=self.transport.model)
        except Exception as exc:
            logger.debug("Generation failed for model=%s: %s", self.transport.model, exc)
            return GenerationResult(
                answer="",
                latency_s=0.0,
                model=self.transport.model,
                error=str(exc),
            )
