# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLM answer generation client for the QA evaluation pipeline.

LiteLLMClient wraps the litellm library which provides a single interface
for routing to NVIDIA NIM, OpenAI, HuggingFace Inference Endpoints, and
local vLLM / Ollama servers via a model name prefix convention.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from nemo_retriever.evaluation.text_utils import strip_think_tags
from nemo_retriever.evaluation.types import GenerationResult

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


def _build_rag_prompt(query: str, chunks: list[str]) -> list[dict]:
    """Build the OpenAI-style messages list for a RAG prompt."""
    context = "\n\n---\n\n".join(chunks) if chunks else "(no context retrieved)"
    user_content = _RAG_USER_TEMPLATE.format(context=context, query=query)
    return [
        {"role": "system", "content": _RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class LiteLLMClient:
    """Unified LLM client backed by litellm.

    A single model string change routes to any supported provider:
    - NVIDIA NIM:  nvidia_nim/<org>/<model>
    - OpenAI:      openai/<model>
    - Any OpenAI-compatible server (vLLM, Ollama): openai/<model> + api_base
    - HuggingFace: huggingface/<org>/<model>

    Provider API keys are read from environment variables automatically
    (NVIDIA_API_KEY, OPENAI_API_KEY, HUGGINGFACE_API_KEY, etc.).
    """

    def __init__(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        extra_params: Optional[dict[str, Any]] = None,
        num_retries: int = 3,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = extra_params or {}
        self.num_retries = num_retries

    def complete(self, messages: list[dict], max_tokens: Optional[int] = None) -> tuple[str, float]:
        """Raw litellm completion call. Returns (content_text, latency_s)."""
        import litellm

        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "num_retries": self.num_retries,
        }
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        call_kwargs.update(self.extra_params)

        t0 = time.monotonic()
        response = litellm.completion(**call_kwargs)
        latency = time.monotonic() - t0
        content = (response.choices[0].message.content or "").strip()
        return content, latency

    def generate(self, query: str, chunks: list[str]) -> GenerationResult:
        """Generate an answer for the given query using retrieved chunks as context."""
        messages = _build_rag_prompt(query, chunks)
        try:
            raw_answer, latency = self.complete(messages)
            answer = strip_think_tags(raw_answer)
            if not answer:
                return GenerationResult(
                    answer="",
                    latency_s=latency,
                    model=self.model,
                    error="thinking_truncated",
                )
            return GenerationResult(answer=answer, latency_s=latency, model=self.model)
        except Exception as exc:
            return GenerationResult(
                answer="",
                latency_s=0.0,
                model=self.model,
                error=str(exc),
            )
