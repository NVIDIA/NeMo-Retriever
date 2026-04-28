# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programmatic scoring (Tier 1 + Tier 2) as a DataFrame transform.

Delegates to :class:`~nemo_retriever.evaluation.scoring_operator.ScoringOperator`
so the underlying metric implementations (``answer_in_context``,
``token_f1``, ``classify_failure``) are shared with the evaluation
framework.  No LLM dependency -- this is pure computation.
"""

from __future__ import annotations

import pandas as pd


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Tier-1 and Tier-2 scoring to a DataFrame of answered queries.

    The input DataFrame is expected to carry ``query``, ``reference_answer``,
    ``answer`` and ``context`` columns -- the shape produced by
    :func:`~nemo_retriever.generation.answer` after a ``reference_answer``
    column has been attached, or by a user that has loaded results from
    parquet / JSONL.  Returns a new DataFrame with the following columns
    appended:

    * ``answer_in_context``  -- Tier-1 retrieval-quality boolean.
    * ``token_f1``           -- Tier-2 SQuAD-style F1 float.
    * ``exact_match``        -- Tier-2 normalised exact-match boolean.
    * ``failure_mode``       -- Classification string (``correct``,
      ``refused_with_context``, ``judge_error``, ...).

    Args:
        df: DataFrame with ``reference_answer``, ``answer``, and
            ``context`` columns.  Additional columns are preserved
            untouched.

    Returns:
        A new :class:`pandas.DataFrame` with the four scoring columns
        appended.  The input is not mutated.

    Raises:
        ValueError: If any required column is missing.
    """
    from nemo_retriever.evaluation.scoring_operator import ScoringOperator

    return ScoringOperator().run(df)
