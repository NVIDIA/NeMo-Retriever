# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full retrieve -> answer -> score -> judge chain as a DataFrame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

if TYPE_CHECKING:
    import pandas as pd

    from nemo_retriever.llm.clients import LLMJudge, LiteLLMClient
    from nemo_retriever.retriever import Retriever


def eval(  # noqa: A001 -- matches the CLI subcommand name `retriever eval`
    retriever: "Retriever",
    queries: Union[str, Sequence[str]],
    *,
    llm: "LiteLLMClient",
    reference: Union[str, Sequence[str]],
    judge: Optional["LLMJudge"] = None,
    top_k: Optional[int] = None,
    embedder: Optional[str] = None,
    lancedb_uri: Optional[str] = None,
    lancedb_table: Optional[str] = None,
) -> "pd.DataFrame":
    """Run the full live-RAG evaluation chain for a batch of queries.

    This is the batch-mode counterpart to ``retriever eval`` on the CLI:
    every row goes through retrieval, answer generation, programmatic
    scoring (token-F1 / exact-match / answer-in-context /
    failure-mode), and optional LLM-as-judge scoring.  A ground-truth
    ``reference`` is required because the scoring tier depends on it.

    Args:
        retriever: :class:`Retriever` used for retrieval.
        queries: Query string(s) to evaluate.
        llm: Pre-built :class:`LiteLLMClient` used for answer generation.
        reference: Ground-truth answer(s).  A single string is broadcast
            to every row; a sequence must have the same length as
            ``queries``.
        judge: Optional :class:`LLMJudge` for Tier-3 scoring.  When
            omitted the ``judge_score``, ``judge_reasoning``,
            ``judge_error`` columns are not added.
        top_k: Per-call ``top_k`` override.
        embedder: Per-call embedder override.
        lancedb_uri: Per-call LanceDB URI override.
        lancedb_table: Per-call LanceDB table override.

    Returns:
        A :class:`pandas.DataFrame` with one row per query carrying the
        union of columns produced by :func:`answer`, :func:`score`, and
        (optionally) :func:`judge`.
    """
    from nemo_retriever.generation.answer import answer
    from nemo_retriever.generation.judge import judge as judge_fn
    from nemo_retriever.generation.score import score

    df = answer(
        retriever,
        queries,
        llm=llm,
        reference=reference,
        top_k=top_k,
        embedder=embedder,
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
    )

    if judge is not None:
        df = judge_fn(df, judge=judge)

    df = score(df)
    return df
