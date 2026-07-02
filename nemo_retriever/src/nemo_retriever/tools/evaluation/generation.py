# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QAGenerationOperator -- DataFrame-in/out answer generation."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, ClassVar, Optional

from nemo_retriever.common.params import TextGenerationParams
from nemo_retriever.models.llm.tasks import GenerationTaskError, RagAnswerTask
from nemo_retriever.models.llm.types import (
    GeneratedTextResult,
    LLMClient,
    TextCompletionClient,
)
from nemo_retriever.operators.generation import TextGenerationOperator


class QAGenerationOperator(TextGenerationOperator):
    """Generate answers for each row using a single LLM.

    Input DataFrame must have ``query`` and ``context`` columns.
    ``context`` is a list[str] of retrieved chunks per row.

    Adds columns: ``answer``, ``latency_s``, ``model``, ``gen_error``.
    """

    required_columns: ClassVar[tuple[str, ...]] = ("query", "context")
    output_columns: ClassVar[tuple[str, ...]] = (
        "answer",
        "latency_s",
        "model",
        "gen_error",
    )

    def __init__(
        self,
        model: str,
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        max_tokens: int = 4096,
        extra_params: Optional[dict[str, Any]] = None,
        num_retries: int = 3,
        timeout: float = 120.0,
        max_workers: int = 8,
        rag_system_prompt: Optional[str] = None,
        rag_system_prompt_prefix: Optional[str] = None,
        reasoning_enabled: bool = True,
    ) -> None:
        params = TextGenerationParams.from_kwargs(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_params=extra_params,
            num_retries=num_retries,
            timeout=timeout,
            max_workers=max_workers,
            rag_system_prompt=rag_system_prompt,
            rag_system_prompt_prefix=rag_system_prompt_prefix,
            reasoning_enabled=reasoning_enabled,
        )
        task_reasoning_enabled = (
            params.reasoning_enabled if params.reasoning_enabled is not None else params.transport.reasoning_enabled
        )
        task = RagAnswerTask(
            prompt=params.prompt,
            system_prompt=params.transport.rag_system_prompt,
            system_prompt_prefix=params.transport.rag_system_prompt_prefix,
            reasoning_enabled=task_reasoning_enabled,
        )
        super().__init__(
            params,
            task=task,
            input_columns={"query": "query", "chunks": "context"},
            output_column="answer",
            latency_column="latency_s",
            model_column="model",
            error_column="gen_error",
            overwrite=True,
        )
        self._qa_constructor_kwargs = deepcopy(
            {
                "model": model,
                "api_base": api_base,
                "api_key": api_key,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "extra_params": extra_params,
                "num_retries": num_retries,
                "timeout": timeout,
                "max_workers": max_workers,
                "rag_system_prompt": rag_system_prompt,
                "rag_system_prompt_prefix": rag_system_prompt_prefix,
                "reasoning_enabled": reasoning_enabled,
            }
        )

    def _get_generation_constructor_kwargs(self) -> dict[str, Any]:
        """Preserve the legacy flat QA constructor contract for graph workers."""
        return deepcopy(self._qa_constructor_kwargs)

    def _execute_task(self, inputs: dict[str, Any]) -> GeneratedTextResult:
        """Prefer completion tasks while adapting legacy generate-only clients."""
        client = self._client
        if isinstance(client, TextCompletionClient):
            return super()._execute_task(inputs)
        if isinstance(client, LLMClient):
            started_at = time.monotonic()
            failure: GenerationTaskError | None = None
            try:
                result = client.generate(inputs["query"], inputs["chunks"])
            except Exception:
                failure = GenerationTaskError(
                    code="transport_error",
                    phase="transport",
                    retryable=False,
                    public_message="Legacy text generation request failed.",
                    latency_s=time.monotonic() - started_at,
                )
            if failure is not None:
                raise failure
            return GeneratedTextResult(
                text=result.answer,
                latency_s=result.latency_s,
                model=result.model,
                error=result.error,
            )
        raise GenerationTaskError(
            code="request_error",
            phase="request",
            retryable=False,
            public_message=("QAGenerationOperator client must implement " "TextCompletionClient or LLMClient."),
            latency_s=0.0,
        )
