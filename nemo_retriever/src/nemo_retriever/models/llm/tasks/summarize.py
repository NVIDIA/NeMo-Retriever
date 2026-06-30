# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-request text summarization task."""

from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import Any, ClassVar, Optional

from nemo_retriever.models.llm.tasks.base import GenerationTask
from nemo_retriever.models.llm.tasks.rag_answer import _apply_reasoning_control
from nemo_retriever.models.llm.text_utils import strip_think_tags
from nemo_retriever.models.llm.types import GenerationRequest

_SUMMARIZE_SYSTEM_PROMPT = (
    "You are a precise summarization assistant. Produce a faithful, concise summary "
    "without adding information that is not present in the source text."
)
_SUMMARIZE_USER_TEMPLATE = "Summarize the following text:\n\n{text}"


def _summary_prompt_fields(prompt: str) -> set[str]:
    """Validate and return fields used by a custom summarization prompt."""
    fields: set[str] = set()
    try:
        parsed = Formatter().parse(prompt)
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if field_name != "text" or format_spec or conversion:
                raise ValueError("summarization prompt may only use the simple {text} placeholder")
            fields.add(field_name)
    except ValueError as exc:
        if str(exc) == "summarization prompt may only use the simple {text} placeholder":
            raise
        raise ValueError(f"invalid summarization prompt: {exc}") from exc
    return fields


@dataclass(frozen=True)
class SummarizeTask(GenerationTask):
    """Summarize one text value without truncation or hidden map-reduce."""

    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    reasoning_enabled: Optional[bool] = None

    required_inputs: ClassVar[tuple[str, ...]] = ("text",)
    _default_sampling: ClassVar[dict[str, Any]] = {
        "temperature": 0.0,
        "top_p": None,
        "max_tokens": 1024,
    }

    def __post_init__(self) -> None:
        if self.prompt is not None:
            _summary_prompt_fields(self.prompt)

    def _preflight_error(self, **inputs: object) -> Optional[str]:
        text = inputs.get("text")
        if isinstance(text, str) and not text.strip():
            return "empty_input"
        return None

    def build_request(self, **inputs: object) -> GenerationRequest:
        """Build one faithful-summary request for the supplied text."""
        text = inputs["text"]
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        prompt = self.prompt if self.prompt is not None else _SUMMARIZE_USER_TEMPLATE
        fields = _summary_prompt_fields(prompt)
        user_content = prompt.format(text=text) if fields else f"{prompt}\n\n{text}"
        system_content = self.system_prompt if self.system_prompt is not None else _SUMMARIZE_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        messages, extra_params = _apply_reasoning_control(messages, self.reasoning_enabled)
        return GenerationRequest(messages=messages, extra_params=extra_params)

    def parse(self, raw_text: str) -> str:
        """Remove visible model reasoning from the summary."""
        return strip_think_tags(raw_text)


__all__ = ["SummarizeTask"]
