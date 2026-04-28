# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-as-judge scoring (Tier 3) as a DataFrame transform.

Accepts a pre-built :class:`~nemo_retriever.llm.clients.LLMJudge` so the
judge's transport (``api_base``, ``api_key``, ``num_retries``, etc.) is
configured by the caller exactly once and reused across rows.  Delegates
to :class:`~nemo_retriever.evaluation.judging.JudgingOperator` for the
actual ThreadPool / retry / error-bucketing logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from nemo_retriever.llm.clients import LLMJudge


def judge(df: pd.DataFrame, *, judge: "LLMJudge") -> pd.DataFrame:
    """Run LLM-as-judge scoring over a DataFrame of answered queries.

    The input DataFrame is expected to carry ``query``,
    ``reference_answer``, and ``answer`` columns -- the shape produced by
    :func:`~nemo_retriever.generation.answer` combined with a caller-supplied
    ``reference_answer`` column.  Each row is scored by the supplied
    :class:`LLMJudge`, and the following columns are appended:

    * ``judge_score``     -- ``Optional[int]`` (``None`` on error).
    * ``judge_reasoning`` -- ``str`` explanation from the judge.
    * ``judge_error``     -- ``Optional[str]`` error message if the call
      failed; ``None`` on success.

    Args:
        df: DataFrame with ``query``, ``reference_answer``, and
            ``answer`` columns.  Other columns are preserved untouched.
        judge: A pre-built :class:`~nemo_retriever.llm.clients.LLMJudge`.
            The caller owns its transport configuration; this function
            will not re-wrap or reconfigure it.

    Returns:
        A new :class:`pandas.DataFrame` with the three judging columns
        appended.  The input is not mutated.

    Raises:
        ValueError: If any required column is missing.
    """
    from nemo_retriever.evaluation.judging import JudgingOperator

    transport = judge.transport
    operator = JudgingOperator(
        model=transport.model,
        api_base=transport.api_base,
        api_key=transport.api_key,
        extra_params=dict(transport.extra_params) if transport.extra_params else None,
        num_retries=transport.num_retries,
        timeout=transport.timeout,
    )
    return operator.run(df)
