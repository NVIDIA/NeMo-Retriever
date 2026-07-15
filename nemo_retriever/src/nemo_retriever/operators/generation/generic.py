# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic prompt-template operator built on the generation layer."""

from __future__ import annotations

from collections.abc import Mapping

from nemo_retriever.common.params import TextGenerationParams
from nemo_retriever.models.llm.tasks import GenericPromptTask
from nemo_retriever.models.llm.types import TextCompletionClient
from nemo_retriever.operators.generation.base import TextGenerationOperator


class GenericGenerationOperator(TextGenerationOperator):
    """Generate text from a validated prompt template and mapped row inputs."""

    def __init__(
        self,
        params: TextGenerationParams,
        input_columns: Mapping[str, str],
        output_column: str = "generated_text",
        *,
        latency_column: str | None = None,
        model_column: str | None = None,
        error_column: str | None = None,
        overwrite: bool = False,
        client: TextCompletionClient | None = None,
    ) -> None:
        normalized_input_columns = dict(input_columns)
        if params.prompt is None:
            raise ValueError("GenericGenerationOperator requires params.prompt")
        reasoning_enabled = (
            params.reasoning_enabled if params.reasoning_enabled is not None else params.transport.reasoning_enabled
        )
        task = GenericPromptTask(
            prompt=params.prompt,
            input_names=tuple(normalized_input_columns),
            system_prompt=params.system_prompt,
            reasoning_enabled=reasoning_enabled,
        )
        super().__init__(
            params,
            task=task,
            input_columns=normalized_input_columns,
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
            "input_columns": self._input_columns.copy(),
            "output_column": self._output_column,
            "latency_column": self._latency_column_arg,
            "model_column": self._model_column_arg,
            "error_column": self._error_column_arg,
            "overwrite": self._overwrite,
        }
