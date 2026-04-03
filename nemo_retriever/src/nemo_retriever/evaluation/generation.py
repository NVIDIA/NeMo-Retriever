# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QAGenerationOperator -- DataFrame-in/out answer generation."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, ClassVar, Optional

from nemo_retriever.evaluation.eval_operator import EvalOperator
from nemo_retriever.evaluation.generators import LiteLLMClient
from nemo_retriever.evaluation.types import GenerationResult

logger = logging.getLogger(__name__)


class QAGenerationOperator(EvalOperator):
    """Generate answers for each row using a single LLM.

    Input DataFrame must have ``query`` and ``context`` columns.
    ``context`` is a list[str] of retrieved chunks per row.

    Adds columns: ``answer``, ``latency_s``, ``model``, ``gen_error``.
    """

    required_columns: ClassVar[tuple[str, ...]] = ("query", "context")
    output_columns: ClassVar[tuple[str, ...]] = ("answer", "latency_s", "model", "gen_error")

    def __init__(
        self,
        model: str,
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        extra_params: Optional[dict[str, Any]] = None,
        num_retries: int = 3,
        max_workers: int = 8,
    ) -> None:
        super().__init__(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=extra_params,
            num_retries=num_retries,
            max_workers=max_workers,
        )
        self._client = LiteLLMClient(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_params=extra_params,
            num_retries=num_retries,
        )
        self._max_workers = max_workers

    def process(self, data: Any, **kwargs: Any) -> Any:
        df = data
        results: list = [None] * len(df)

        def _generate(idx: int, query: str, context: list[str]):
            gen = self._client.generate(query, context)
            return idx, gen

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(_generate, i, row["query"], row["context"]): i for i, (_, row) in enumerate(df.iterrows())
            }
            for future in as_completed(futures):
                try:
                    idx, gen = future.result()
                    results[idx] = gen
                except Exception as exc:
                    idx = futures[future]
                    logger.warning("Row %d generation failed: %s", idx, exc)
                    results[idx] = GenerationResult(answer="", latency_s=0.0, model=self._client.model, error=str(exc))

        out = df.copy()
        out["answer"] = [r.answer for r in results]
        out["latency_s"] = [r.latency_s for r in results]
        out["model"] = [r.model for r in results]
        out["gen_error"] = [r.error for r in results]
        return out
