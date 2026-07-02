# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarization operator built on the reusable generation layer."""

from __future__ import annotations

from nemo_retriever.common.params import TextGenerationParams
from nemo_retriever.models.llm.tasks import SummarizeTask
from nemo_retriever.models.llm.types import TextCompletionClient
from nemo_retriever.operators.generation.base import TextGenerationOperator


class SummarizationOperator(TextGenerationOperator):
    """Summarize the text in one DataFrame column per row."""

    def __init__(
        self,
        params: TextGenerationParams,
        input_column: str = "text",
        output_column: str = "summary",
        *,
        latency_column: str | None = None,
        model_column: str | None = None,
        error_column: str | None = None,
        overwrite: bool = False,
        client: TextCompletionClient | None = None,
    ) -> None:
        reasoning_enabled = (
            params.reasoning_enabled if params.reasoning_enabled is not None else params.transport.reasoning_enabled
        )
        task = SummarizeTask(
            prompt=params.prompt,
            system_prompt=params.system_prompt,
            reasoning_enabled=reasoning_enabled,
        )
        super().__init__(
            params,
            task=task,
            input_columns={"text": input_column},
            output_column=output_column,
            latency_column=latency_column,
            model_column=model_column,
            error_column=error_column,
            overwrite=overwrite,
            client=client,
        )

    def _get_generation_constructor_kwargs(self) -> dict[str, object]:
        return {
            "params": self._params.model_copy(deep=True),
            "input_column": self._input_columns["text"],
            "output_column": self._output_column,
            "latency_column": self._latency_column_arg,
            "model_column": self._model_column_arg,
            "error_column": self._error_column_arg,
            "overwrite": self._overwrite,
        }
