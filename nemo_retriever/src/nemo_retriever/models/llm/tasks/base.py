# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable request/response lifecycle for text-generation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, ClassVar, Literal, Optional

from nemo_retriever.common.params.models import LLMInferenceParams
from nemo_retriever.models.llm.types import (
    GeneratedTextResult,
    GenerationRequest,
    TextCompletionClient,
    UnsupportedTextResponseError,
)


class GenerationTaskError(RuntimeError):
    """Sanitized failure raised by the strict generation-task lifecycle."""

    def __init__(
        self,
        *,
        code: str,
        phase: Literal["request", "transport", "response", "parse"],
        retryable: bool,
        public_message: str,
        latency_s: float,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.phase = phase
        self.retryable = retryable
        self.public_message = public_message
        self.latency_s = latency_s


class TextGenerationTask(ABC):
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

    @staticmethod
    def _client_model(client: object) -> str:
        """Read a client model identifier without making error handling fail."""
        try:
            model = getattr(client, "model", "")
        except Exception:
            return ""
        return model if isinstance(model, str) else ""

    @staticmethod
    def _elapsed(started_at: float) -> float:
        return time.monotonic() - started_at

    def invoke(self, client: TextCompletionClient, **inputs: object) -> GeneratedTextResult:
        """Strictly build, execute, and parse one text request.

        Failures are raised as :class:`GenerationTaskError` with stable codes
        and sanitized messages. Batch operators collect those errors at the row
        boundary; callers that need the historical collecting behavior can use
        :meth:`execute`.
        """
        started_at = time.monotonic()

        preflight_failure: Optional[GenerationTaskError] = None
        try:
            preflight_error = self._preflight_error(**inputs)
        except Exception:
            preflight_failure = GenerationTaskError(
                code="request_error",
                phase="request",
                retryable=False,
                public_message="generation request validation failed",
                latency_s=self._elapsed(started_at),
            )
        if preflight_failure is not None:
            raise preflight_failure
        if preflight_error is not None:
            raise GenerationTaskError(
                code=preflight_error,
                phase="request",
                retryable=False,
                public_message="generation request was skipped",
                latency_s=self._elapsed(started_at),
            )

        request_failure: Optional[GenerationTaskError] = None
        try:
            request = self.build_request(**inputs)
            if not isinstance(request, GenerationRequest):
                raise TypeError("build_request must return GenerationRequest")
            request = GenerationRequest(
                messages=request.messages,
                max_tokens=request.max_tokens,
                extra_params=request.extra_params,
            )
        except Exception:
            request_failure = GenerationTaskError(
                code="request_error",
                phase="request",
                retryable=False,
                public_message="generation request construction failed",
                latency_s=self._elapsed(started_at),
            )
        if request_failure is not None:
            raise request_failure

        transport_failure: Optional[GenerationTaskError] = None
        try:
            raw_text, latency_s = client.complete(
                request.messages,
                max_tokens=request.max_tokens,
                extra_params=request.extra_params,
            )
        except UnsupportedTextResponseError:
            transport_failure = GenerationTaskError(
                code="unsupported_response",
                phase="response",
                retryable=False,
                public_message="provider response is not representable as a text completion",
                latency_s=self._elapsed(started_at),
            )
        except Exception as exc:
            try:
                retryable = bool(getattr(exc, "retryable", False))
            except Exception:
                retryable = False
            transport_failure = GenerationTaskError(
                code="transport_error",
                phase="transport",
                retryable=retryable,
                public_message="text completion request failed",
                latency_s=self._elapsed(started_at),
            )
        if transport_failure is not None:
            raise transport_failure

        parse_failure: Optional[GenerationTaskError] = None
        try:
            text = self.parse(raw_text)
            if not isinstance(text, str):
                raise TypeError("parse must return text")
        except Exception:
            parse_failure = GenerationTaskError(
                code="parse_error",
                phase="parse",
                retryable=False,
                public_message="generation response parsing failed",
                latency_s=self._elapsed(started_at),
            )
        if parse_failure is not None:
            raise parse_failure
        if not text:
            raise GenerationTaskError(
                code=self.empty_output_error,
                phase="response",
                retryable=False,
                public_message="generation produced no usable text",
                latency_s=latency_s,
            )
        return GeneratedTextResult(
            text=text,
            latency_s=latency_s,
            model=self._client_model(client),
            error=None,
        )

    def execute(self, client: TextCompletionClient, **inputs: object) -> GeneratedTextResult:
        """Compatibility facade that collects strict failures into a result."""
        try:
            return self.invoke(client, **inputs)
        except GenerationTaskError as exc:
            return GeneratedTextResult(
                text="",
                latency_s=exc.latency_s,
                model=self._client_model(client),
                error=exc.code,
            )
