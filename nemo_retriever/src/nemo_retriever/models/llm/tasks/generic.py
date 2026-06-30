# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated prompt-template task for general text generation."""

from __future__ import annotations

from string import Formatter
from typing import Any, ClassVar, Optional, Sequence

from nemo_retriever.models.llm.tasks.base import GenerationTask
from nemo_retriever.models.llm.tasks.rag_answer import _apply_reasoning_control
from nemo_retriever.models.llm.text_utils import strip_think_tags
from nemo_retriever.models.llm.types import GenerationRequest


def _validate_prompt_template(prompt: str, input_names: tuple[str, ...]) -> None:
    """Require a one-to-one match between simple fields and declared inputs."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not input_names:
        raise ValueError("input_names must declare at least one input")
    if len(set(input_names)) != len(input_names):
        raise ValueError("input_names must not contain duplicates")
    if any(not name.isidentifier() for name in input_names):
        raise ValueError("input_names must contain valid Python identifiers")

    fields: list[str] = []
    try:
        for _, field_name, format_spec, conversion in Formatter().parse(prompt):
            if field_name is None:
                continue
            if (
                not field_name
                or not field_name.isidentifier()
                or "." in field_name
                or "[" in field_name
                or format_spec
                or conversion
            ):
                raise ValueError("prompt placeholders must be simple names")
            fields.append(field_name)
    except ValueError as exc:
        if str(exc) == "prompt placeholders must be simple names":
            raise
        raise ValueError(f"invalid prompt template: {exc}") from exc

    if not fields:
        raise ValueError("prompt must contain at least one declared placeholder")
    declared = set(input_names)
    referenced = set(fields)
    missing = declared - referenced
    undeclared = referenced - declared
    if missing:
        raise ValueError(f"prompt is missing declared placeholders: {sorted(missing)}")
    if undeclared:
        raise ValueError(f"prompt contains undeclared placeholders: {sorted(undeclared)}")


class GenericPromptTask(GenerationTask):
    """Render declared row inputs into a validated prompt template."""

    _default_sampling: ClassVar[dict[str, Any]] = {
        "temperature": 1.0,
        "top_p": None,
        "max_tokens": 1024,
    }

    def __init__(
        self,
        *,
        prompt: str,
        input_names: Sequence[str],
        system_prompt: Optional[str] = None,
        reasoning_enabled: Optional[bool] = None,
    ) -> None:
        if isinstance(input_names, str):
            raise TypeError("input_names must be a sequence of names, not a string")
        names = tuple(input_names)
        _validate_prompt_template(prompt, names)
        self.prompt = prompt
        self.required_inputs = names
        self.system_prompt = system_prompt
        self.reasoning_enabled = reasoning_enabled

    def build_request(self, **inputs: object) -> GenerationRequest:
        """Render declared inputs and build one completion request."""
        missing = [name for name in self.required_inputs if name not in inputs]
        if missing:
            raise KeyError(f"missing required inputs: {missing}")
        values = {name: inputs[name] for name in self.required_inputs}
        user_content = self.prompt.format(**values)
        messages: list[dict[str, Any]] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        messages, extra_params = _apply_reasoning_control(messages, self.reasoning_enabled)
        return GenerationRequest(messages=messages, extra_params=extra_params)

    def parse(self, raw_text: str) -> str:
        """Remove visible model reasoning from generated text."""
        return strip_think_tags(raw_text)


__all__ = ["GenericPromptTask"]
