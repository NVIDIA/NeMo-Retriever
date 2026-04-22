# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ScoringOperator -- graph-chainable programmatic scoring."""

from __future__ import annotations

from typing import Any, ClassVar

from nemo_retriever.evaluation.eval_operator import EvalOperator
from nemo_retriever.evaluation.scoring import score_dataframe


class ScoringOperator(EvalOperator):
    """Apply all programmatic scoring metrics to a DataFrame.

    Wraps :func:`score_dataframe` as a graph-chainable operator so it can
    participate in ``>>`` chains alongside generation and judging operators.

    Input must have ``reference_answer``, ``answer``, ``context`` columns.
    Adds: ``answer_in_context``, ``token_f1``, ``exact_match``, ``failure_mode``.

    No LLM dependency -- pure computation.
    """

    required_columns: ClassVar[tuple[str, ...]] = ("reference_answer", "answer", "context")
    output_columns: ClassVar[tuple[str, ...]] = ("answer_in_context", "token_f1", "exact_match", "failure_mode")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def process(self, data: Any, **kwargs: Any) -> Any:
        return score_dataframe(data)
