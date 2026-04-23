# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Live RAG generation surface.

This package is the public, composable API for every LLM-backed operation
that used to live on :class:`~nemo_retriever.retriever.Retriever` before
the structural refactor that split "retrieve" (kept on ``Retriever``)
from "generate / score / judge / evaluate" (moved here).  The design
goals are:

* **Single return type.**  Every function in this package returns a
  :class:`pandas.DataFrame`.  Columns are additive: ``retrieve`` produces
  ``[query, chunks, metadata]``; ``answer`` extends that with
  ``[answer, model, latency_s, gen_error]``; ``score`` adds
  ``[answer_in_context, token_f1, exact_match, failure_mode]``;
  ``judge`` adds ``[judge_score, judge_reasoning, judge_error]``.
* **Composability.**  The output of any function is a valid input to the
  next, so an ad-hoc pipeline is just ``judge(score(answer(retrieve(...),
  llm=...), reference=...), judge=...)``.
* **Delegation, not reimplementation.**  Each function is a thin shim
  over the corresponding graph operator in
  :mod:`nemo_retriever.evaluation` (``LiveRetrievalOperator``,
  ``QAGenerationOperator``, ``ScoringOperator``, ``JudgingOperator``)
  so every optimization / retry / logging behaviour that lives in the
  operator is available here for free.

See each submodule for the per-function contract.
"""

from nemo_retriever.generation.answer import answer
from nemo_retriever.generation.eval import eval
from nemo_retriever.generation.judge import judge
from nemo_retriever.generation.retrieve import retrieve
from nemo_retriever.generation.score import score

__all__ = [
    "answer",
    "eval",
    "judge",
    "retrieve",
    "score",
]
