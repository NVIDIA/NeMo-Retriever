"""
LLM answer generation client for the QA evaluation pipeline.

LiteLLMClient wraps the litellm library which provides a single interface
for routing to NVIDIA NIM, OpenAI, HuggingFace Inference Endpoints, and
local vLLM / Ollama servers via a model name prefix convention:

  nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5  -> NVIDIA NIM
  openai/gpt-4o                                       -> OpenAI
  openai/my-model                                     -> any OpenAI-spec server (+ api_base)
  huggingface/meta-llama/Llama-3-70b-instruct         -> HF Inference Endpoints

Provider-specific API keys are read automatically from environment variables
(NVIDIA_API_KEY, OPENAI_API_KEY, etc.). Do not embed keys in config files.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

from nv_ingest_harness.utils.qa.types import GenerationResult


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output.

    Handles both closed tags (<think>...</think>) and unclosed tags where the
    model hit the token limit mid-reasoning and never emitted </think>.
    Returns empty string if nothing remains after stripping so callers can
    detect thinking_truncated.
    """
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*", "", stripped, flags=re.DOTALL)
    return stripped.strip()


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
    """
    Unified LLM client backed by litellm.

    A single model string change routes to any supported provider:
    - NVIDIA NIM:  nvidia_nim/<org>/<model>
    - OpenAI:      openai/<model>
    - Any OpenAI-compatible server (vLLM, Ollama): openai/<model> + api_base
    - HuggingFace: huggingface/<org>/<model>

    Provider API keys are read from environment variables automatically
    (NVIDIA_API_KEY, OPENAI_API_KEY, HUGGINGFACE_API_KEY, etc.).

    Args:
        model: litellm model string with provider prefix.
        api_base: Override endpoint URL for private / local deployments.
        api_key: Explicit API key (prefer env vars; only use for non-standard setups).
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in the generated response.
        extra_params: Additional kwargs forwarded verbatim to litellm.completion.
                      Use this for provider-specific options such as reasoning mode:
                      {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        num_retries: Number of retry attempts on transient errors.
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
        """
        Raw litellm completion call. Returns (content_text, latency_s).

        This is the single place where the litellm API is called. Both
        generate() and external callers (e.g. LLMJudge) use this method so
        retry logic, auth, and extra_params stay in one place.

        Args:
            messages: OpenAI-style messages list.
            max_tokens: Override max_tokens for this call (uses self.max_tokens if None).

        Returns:
            Tuple of (response text, wall-clock latency in seconds).

        Raises:
            Exception: Re-raises litellm errors after exhausting retries.
        """
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
        """
        Generate an answer for the given query using retrieved chunks as context.

        Args:
            query: The question to answer.
            chunks: Retrieved text chunks providing context.

        Returns:
            GenerationResult with answer text and wall-clock latency.
        """
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
