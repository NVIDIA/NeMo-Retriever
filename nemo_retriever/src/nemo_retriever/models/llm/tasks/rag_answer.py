# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieval-augmented answer-generation task and prompt helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from string import Formatter
from typing import Any, ClassVar, Optional

from nemo_retriever.models.llm.tasks.base import TextGenerationTask
from nemo_retriever.models.llm.text_utils import strip_think_tags
from nemo_retriever.models.llm.types import GenerationRequest

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


def _validate_rag_prompt(prompt: str) -> None:
    """Allow only simple ``context`` and ``query`` placeholders."""
    try:
        parsed_fields = list(Formatter().parse(prompt))
    except ValueError as exc:
        raise ValueError(f"invalid RAG prompt: {exc}") from exc
    for _, field_name, format_spec, conversion in parsed_fields:
        if field_name is None:
            continue
        if field_name not in {"context", "query"} or format_spec or conversion:
            raise ValueError("RAG prompt may only use simple {context} and {query} placeholders")


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
) -> list[dict[str, Any]]:
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


def _with_no_reasoning_controls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add no-reasoning prompt metadata understood by current Nemotron NIMs."""
    updated = [dict(message) for message in messages]
    if updated and updated[0].get("role") == "system":
        content = str(updated[0].get("content") or "").strip()
        if _NO_REASONING_SYSTEM_DIRECTIVE not in content:
            content = f"{_NO_REASONING_SYSTEM_DIRECTIVE}\n{content}" if content else _NO_REASONING_SYSTEM_DIRECTIVE
        updated[0]["content"] = content
        return updated
    updated.insert(0, {"role": "system", "content": _NO_REASONING_SYSTEM_DIRECTIVE})
    return updated


def _apply_reasoning_control(
    messages: list[dict[str, Any]],
    reasoning_enabled: Optional[bool],
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    """Apply task-level reasoning controls to messages and request extras."""
    if reasoning_enabled is not False:
        return messages, None
    return _with_no_reasoning_controls(messages), deepcopy(_NO_REASONING_EXTRA_PARAMS)


@dataclass(frozen=True)
class RagAnswerTask(TextGenerationTask):
    """Generate a grounded answer from a query and retrieved text chunks."""

    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    system_prompt_prefix: Optional[str] = None
    reasoning_enabled: Optional[bool] = None

    required_inputs: ClassVar[tuple[str, ...]] = ("query", "chunks")
    _default_sampling: ClassVar[dict[str, Any]] = {
        "temperature": 0.0,
        "top_p": None,
        "max_tokens": 4096,
    }
    empty_output_error: ClassVar[str] = "thinking_truncated"

    def __post_init__(self) -> None:
        if self.prompt is not None:
            _validate_rag_prompt(self.prompt)

    def build_request(self, **inputs: object) -> GenerationRequest:
        """Build a grounded answer request, including optional reasoning controls."""
        query = inputs["query"]
        chunks = inputs["chunks"]
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
            raise TypeError("chunks must be a list of strings")

        formatted_system_prompt = _format_rag_system_prompt(
            rag_system_prompt=self.system_prompt,
            rag_system_prompt_prefix=self.system_prompt_prefix,
        )
        messages = _build_rag_prompt(
            query,
            chunks,
            formatted_rag_system_prompt=formatted_system_prompt,
        )
        if self.prompt is not None:
            context = "\n\n---\n\n".join(chunks) if chunks else "(no context retrieved)"
            messages[-1]["content"] = self.prompt.format(context=context, query=query)

        per_request_reasoning = inputs.get("reasoning_enabled")
        effective_reasoning = self.reasoning_enabled if per_request_reasoning is None else bool(per_request_reasoning)
        messages, extra_params = _apply_reasoning_control(messages, effective_reasoning)
        return GenerationRequest(messages=messages, extra_params=extra_params)

    def parse(self, raw_text: str) -> str:
        """Remove visible model reasoning from the answer."""
        return strip_think_tags(raw_text)


__all__ = ["RagAnswerTask"]
