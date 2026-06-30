# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable request/response lifecycle for text-generation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from nemo_retriever.common.params.models import LLMInferenceParams
from nemo_retriever.models.llm.types import (
    CompletionClient,
    GeneratedTextResult,
    GenerationRequest,
)


class GenerationTask(ABC):
    """Stateless strategy that turns logical inputs into one completion call."""

    required_inputs: tuple[str, ...] = ()
    _default_sampling: ClassVar[dict[str, Any]] = {
        "temperature": 1.0,
        "top_p": None,
        "max_tokens": 1024,
    }
    empty_output_error: ClassVar[str] = "empty_output"

    @property
    def default_sampling(self) -> LLMInferenceParams:
        """Return a fresh copy of this task's sampling defaults."""
        return LLMInferenceParams(**self._default_sampling)

    @abstractmethod
    def build_request(self, **inputs: object) -> GenerationRequest:
        """Build one provider-neutral request from logical task inputs."""

    def parse(self, raw_text: str) -> str:
        """Parse completion text into the task's text result."""
        return raw_text.strip()

    def _preflight_error(self, **inputs: object) -> Optional[str]:
        """Return an error code when no provider request should be made."""
        return None

    def execute(self, client: CompletionClient, **inputs: object) -> GeneratedTextResult:
        """Build, execute, and parse one request without leaking row failures."""
        latency_s = 0.0
        try:
            preflight_error = self._preflight_error(**inputs)
            if preflight_error is not None:
                return GeneratedTextResult(
                    text="",
                    latency_s=0.0,
                    model=client.model,
                    error=preflight_error,
                )

            request = self.build_request(**inputs)
            raw_text, latency_s = client.complete(
                request.messages,
                max_tokens=request.max_tokens,
                extra_params=request.extra_params,
            )
            text = self.parse(raw_text)
            return GeneratedTextResult(
                text=text,
                latency_s=latency_s,
                model=client.model,
                error=None if text else self.empty_output_error,
            )
        except Exception as exc:
            return GeneratedTextResult(
                text="",
                latency_s=latency_s,
                model=client.model,
                error=str(exc),
            )
