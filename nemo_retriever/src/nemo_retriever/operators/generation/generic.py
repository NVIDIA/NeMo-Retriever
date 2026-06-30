# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic prompt-template operator built on the generation layer."""

from __future__ import annotations

from collections.abc import Mapping

from nemo_retriever.common.params import TextGenerationParams
from nemo_retriever.models.llm.tasks import GenerationTask, GenericPromptTask
from nemo_retriever.models.llm.types import CompletionClient
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
        client: CompletionClient | None = None,
    ) -> None:
        normalized_input_columns = dict(input_columns)
        super().__init__(
            params,
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
            "params": self._params,
            "input_columns": dict(self._input_columns),
            "output_column": self._output_column,
            "latency_column": self._latency_column_arg,
            "model_column": self._model_column_arg,
            "error_column": self._error_column_arg,
            "overwrite": self._overwrite,
        }

    def _create_task(
        self,
        params: TextGenerationParams,
        logical_inputs: tuple[str, ...],
    ) -> GenerationTask:
        if params.prompt is None:
            raise ValueError("GenericGenerationOperator requires params.prompt")
        reasoning_enabled = (
            params.reasoning_enabled if params.reasoning_enabled is not None else params.transport.reasoning_enabled
        )
        return GenericPromptTask(
            prompt=params.prompt,
            input_names=logical_inputs,
            system_prompt=params.system_prompt,
            reasoning_enabled=reasoning_enabled,
        )
