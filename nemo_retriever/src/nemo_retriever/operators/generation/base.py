# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable DataFrame operator for text-generation tasks."""

from __future__ import annotations

import inspect
import logging
import time
from abc import abstractmethod
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, ClassVar

import pandas as pd

from pydantic import BaseModel
from nemo_retriever.common.params import LLMInferenceParams, TextGenerationParams
from nemo_retriever.models.llm.clients import LiteLLMClient
from nemo_retriever.models.llm.tasks import GenerationTask, GenerationTaskError
from nemo_retriever.models.llm.types import GeneratedTextResult, TextCompletionClient
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator

logger = logging.getLogger(__name__)


class TextGenerationOperator(AbstractOperator, CPUOperator):
    """Base operator for one text-generation request per DataFrame row.

    Concrete operators construct an immutable :class:`GenerationTask` before
    calling this base. The task and client are runtime-only state; graph
    reconstruction uses only defensive constructor state. The base owns
    validation, safe bounded execution, positional ordering, and stable
    output metadata.

    ``input_columns`` maps each task-level input name to a physical DataFrame
    column. Results are tracked by row position rather than index label so
    duplicate DataFrame indices remain safe.
    """

    required_columns: ClassVar[tuple[str, ...]] = ()
    output_columns: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        params: TextGenerationParams,
        *,
        task: GenerationTask,
        input_columns: Mapping[str, str],
        output_column: str,
        latency_column: str | None = None,
        model_column: str | None = None,
        error_column: str | None = None,
        overwrite: bool = False,
        client: TextCompletionClient | None = None,
    ) -> None:
        copied_params = params.model_copy(deep=True)
        logical_columns = dict(input_columns)
        self._validate_input_mapping(logical_columns)

        resolved_latency_column = latency_column if latency_column is not None else f"{output_column}_latency_s"
        resolved_model_column = model_column if model_column is not None else f"{output_column}_model"
        resolved_error_column = error_column if error_column is not None else f"{output_column}_error"
        output_columns = (
            output_column,
            resolved_latency_column,
            resolved_model_column,
            resolved_error_column,
        )
        self._validate_output_columns(output_columns)

        super().__init__()

        self._params = copied_params
        self._input_columns = logical_columns
        self._output_column = output_column
        self._latency_column = resolved_latency_column
        self._model_column = resolved_model_column
        self._error_column = resolved_error_column
        self._latency_column_arg = latency_column
        self._model_column_arg = model_column
        self._error_column_arg = error_column
        self._overwrite = overwrite
        self._max_workers = copied_params.max_workers
        self._configured_model = copied_params.transport.model

        self.required_columns = tuple(dict.fromkeys(logical_columns.values()))
        self.output_columns = output_columns

        self._task = task
        missing_inputs = [name for name in self._task.required_inputs if name not in logical_columns]
        if missing_inputs:
            raise ValueError(f"{type(self).__name__} is missing task input mappings: {missing_inputs}")

        if client is None:
            sampling = copied_params.resolve_sampling(self._task.default_sampling)
            self._client: TextCompletionClient = self._create_client(copied_params, sampling)
        else:
            self._client = client

    def _create_client(
        self,
        params: TextGenerationParams,
        sampling: LLMInferenceParams,
    ) -> TextCompletionClient:
        """Create the default client without introducing a global registry."""
        return LiteLLMClient(transport=params.transport, sampling=sampling)

    @abstractmethod
    def _get_generation_constructor_kwargs(self) -> dict[str, Any]:
        """Return reconstructible public constructor state for this operator."""
        ...

    @classmethod
    def _contains_runtime_object(
        cls,
        value: Any,
        targets: tuple[object, ...],
        seen: set[int],
    ) -> bool:
        if any(value is target for target in targets):
            return True
        if isinstance(value, (str, bytes, bytearray, memoryview)):
            return False
        value_id = id(value)
        if value_id in seen:
            return False
        seen.add(value_id)
        if isinstance(value, BaseModel):
            return any(
                cls._contains_runtime_object(getattr(value, name), targets, seen) for name in type(value).model_fields
            )
        if isinstance(value, Mapping):
            return any(cls._contains_runtime_object(item, targets, seen) for pair in value.items() for item in pair)
        if isinstance(value, (list, tuple, set, frozenset)):
            return any(cls._contains_runtime_object(item, targets, seen) for item in value)
        return False

    def get_constructor_kwargs(self) -> dict[str, Any]:
        """Return validated graph state without capturing a live task or client."""
        kwargs = dict(self._get_generation_constructor_kwargs())
        forbidden_keys = {"client", "task", "_client", "_task"}.intersection(kwargs)
        if forbidden_keys:
            raise TypeError(
                f"{type(self).__name__} graph constructor hook returned runtime-only keys: " f"{sorted(forbidden_keys)}"
            )
        if self._contains_runtime_object(
            kwargs,
            (self._client, self._task),
            set(),
        ):
            raise TypeError(f"{type(self).__name__} graph constructor hook captured a live client or task")

        signature = inspect.signature(type(self).__init__)
        try:
            signature.bind(None, **kwargs)
        except TypeError as exc:
            raise TypeError(f"{type(self).__name__} returned invalid graph constructor kwargs: {exc}") from exc
        try:
            return deepcopy(kwargs)
        except Exception as exc:
            raise TypeError(f"{type(self).__name__} graph constructor kwargs could not be copied safely") from exc

    @staticmethod
    def _validate_input_mapping(input_columns: Mapping[str, str]) -> None:
        if not input_columns:
            raise ValueError("input_columns must contain at least one task input mapping")
        for logical_name, column_name in input_columns.items():
            if not isinstance(logical_name, str) or not logical_name:
                raise ValueError("input_columns task input names must be non-empty strings")
            if not isinstance(column_name, str) or not column_name:
                raise ValueError("input_columns DataFrame column names must be non-empty strings")

    @staticmethod
    def _validate_output_columns(output_columns: tuple[str, ...]) -> None:
        if any(not isinstance(column, str) or not column for column in output_columns):
            raise ValueError("output column names must be non-empty strings")
        if len(set(output_columns)) != len(output_columns):
            raise ValueError(f"output column names must be distinct: {list(output_columns)}")

    @staticmethod
    def _label_positions(data: pd.DataFrame, label: str) -> list[int]:
        return [int(position) for position in data.columns.get_indexer_for([label]) if position >= 0]

    def _validate_and_resolve_dataframe(
        self,
        data: Any,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} requires a pandas DataFrame")

        input_positions: dict[str, int] = {}
        missing: list[str] = []
        ambiguous_inputs: list[str] = []
        for logical_name, column_name in self._input_columns.items():
            positions = self._label_positions(data, column_name)
            if not positions:
                missing.append(column_name)
            elif len(positions) > 1:
                ambiguous_inputs.append(column_name)
            else:
                input_positions[logical_name] = positions[0]
        if missing:
            missing = list(dict.fromkeys(missing))
            raise ValueError(f"{type(self).__name__} requires missing columns: {missing}")
        if ambiguous_inputs:
            ambiguous_inputs = list(dict.fromkeys(ambiguous_inputs))
            raise ValueError(
                f"{type(self).__name__} mapped input columns are ambiguous because their labels "
                f"are duplicated: {ambiguous_inputs}"
            )

        if not self._overwrite:
            collisions = [column for column in self.output_columns if self._label_positions(data, column)]
            if collisions:
                raise ValueError(
                    f"{type(self).__name__} output columns already exist: {collisions}; "
                    "set overwrite=True to replace them"
                )
        else:
            ambiguous_outputs = [
                column for column in self.output_columns if len(self._label_positions(data, column)) > 1
            ]
            if ambiguous_outputs:
                raise ValueError(
                    f"{type(self).__name__} cannot overwrite ambiguous duplicate output " f"labels: {ambiguous_outputs}"
                )
        return data, input_positions

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        df, _ = self._validate_and_resolve_dataframe(data)
        return df

    def _execute_task(self, inputs: dict[str, Any]) -> GeneratedTextResult:
        """Execute the configured task; subclasses may adapt legacy clients."""
        return self._task.invoke(self._client, **inputs)

    def _execute_row(self, position: int, inputs: dict[str, Any]) -> tuple[int, GeneratedTextResult]:
        started_at = time.monotonic()
        try:
            result = self._execute_task(inputs)
        except GenerationTaskError as exc:
            # Keep the strict task's measured lifecycle while covering custom
            # adapters that report a shorter or zero failure duration.
            elapsed = max(exc.latency_s, time.monotonic() - started_at)
            result = self._failure_result(exc.code, elapsed)
            logger.warning("Row %d generation failed (%s)", position, exc.code)
        except Exception:
            # Unexpected adapter/client failures remain isolated by row. Raw
            # provider exception text is never persisted or logged.
            result = self._failure_result("request_error", time.monotonic() - started_at)
            logger.warning("Row %d generation failed (request_error)", position)
        return position, result

    def _failure_model(self) -> str:
        try:
            model = self._client.model
        except Exception:
            return self._configured_model
        return model if isinstance(model, str) and model else self._configured_model

    def _failure_result(self, error: str, latency_s: float) -> GeneratedTextResult:
        return GeneratedTextResult(
            text="",
            latency_s=latency_s,
            model=self._failure_model(),
            error=error,
        )

    def _effective_max_workers(self, row_count: int) -> int:
        """Return safe concurrency for the current runtime client."""
        try:
            supports_concurrency = getattr(self._client, "supports_concurrent_calls", False) is True
        except Exception:
            supports_concurrency = False
        configured_workers = self._max_workers if supports_concurrency else 1
        return min(configured_workers, row_count)

    def process(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        df, input_positions = self._validate_and_resolve_dataframe(data)
        results: list[GeneratedTextResult | None] = [None] * len(df)

        if len(df):
            futures: dict[Future[tuple[int, GeneratedTextResult]], int] = {}
            with ThreadPoolExecutor(max_workers=self._effective_max_workers(len(df))) as pool:
                for position in range(len(df)):
                    inputs = {
                        name: df.iat[position, column_position] for name, column_position in input_positions.items()
                    }
                    future = pool.submit(self._execute_row, position, inputs)
                    futures[future] = position

                for future in as_completed(futures):
                    position = futures[future]
                    try:
                        result_position, result = future.result()
                        if result_position != position:
                            raise RuntimeError(
                                f"generation result position {result_position} does not match "
                                f"submitted position {position}"
                            )
                        results[position] = result
                    except Exception:
                        logger.warning("Row %d generation failed (request_error)", position)
                        results[position] = self._failure_result("request_error", 0.0)

        # Every non-empty row is assigned either a task result or a failure
        # result above. The cast-free local assertion catches future changes to
        # that invariant before partially writing output columns.
        if any(result is None for result in results):
            raise RuntimeError("generation completed without a result for every row")
        completed_results = [result for result in results if result is not None]

        out = df.copy()
        # Explicit dtypes keep empty and non-empty Ray/Arrow batches
        # schema-compatible while retaining positional duplicate-index writes.
        out[self._output_column] = pd.array([result.text for result in completed_results], dtype="object")
        out[self._latency_column] = pd.array([result.latency_s for result in completed_results], dtype="float64")
        out[self._model_column] = pd.array([result.model for result in completed_results], dtype="object")
        out[self._error_column] = pd.array([result.error for result in completed_results], dtype="object")
        return out

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
