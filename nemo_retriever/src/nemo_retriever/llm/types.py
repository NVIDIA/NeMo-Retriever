# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Protocol definitions and result dataclasses for LLM-based pipelines.

These abstractions allow retrieval strategies, LLM clients, and judges
to be swapped independently.  They are consumed by both the evaluation
framework (``nemo_retriever.evaluation``) and the live RAG surface
(:mod:`nemo_retriever.generation`).

Note on legacy shapes:
    ``RetrievalResult`` and ``AnswerResult`` predate the current
    DataFrame-centric contract.  They are kept around because they
    mirror (respectively) the per-row ``[chunks, metadata]`` shape and
    the ``retriever answer`` JSON schema, so user code that imports them
    continues to work.  Production paths now move data as
    ``pandas.DataFrame`` values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class RetrieverStrategy(Protocol):
    """Pluggable retrieval strategy interface.

    Implementations return a :class:`pandas.DataFrame` with exactly the
    columns ``[query, chunks, metadata]`` and one row per invocation.
    ``chunks`` is a ``list[str]`` in rank order; ``metadata`` is the
    aligned ``list[dict]`` holding everything else a hit record carries
    (source, page_number, distance scores, ...).

    The DataFrame return type is the single contract shared with
    :mod:`nemo_retriever.generation`, so an implementation that satisfies
    this Protocol is immediately composable with ``generation.answer``,
    ``generation.score``, ``generation.judge``, and ``generation.eval``.
    """

    def retrieve(self, query: str, top_k: int) -> "pd.DataFrame": ...


@runtime_checkable
class LLMClient(Protocol):
    """Pluggable LLM answer generation interface."""

    def generate(self, query: str, chunks: list[str]) -> "GenerationResult": ...


@runtime_checkable
class AnswerJudge(Protocol):
    """Pluggable answer scoring interface."""

    def judge(self, query: str, reference: str, candidate: str) -> "JudgeResult": ...


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    chunks: list[str]
    metadata: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Result from a single LLM generation call."""

    answer: str
    latency_s: float
    model: str
    error: Optional[str] = None


@dataclass
class JudgeResult:
    """Result from a single judge evaluation.

    ``score`` is ``None`` when the judge could not produce a score
    (API error, parse failure, empty candidate).  Valid scores are 1-5.
    """

    score: Optional[int] = None
    reasoning: str = ""
    error: Optional[str] = None


@dataclass
class AnswerResult:
    """Legacy shape mirroring one row of the ``generation.eval`` DataFrame.

    Holds the generated answer alongside the retrieved context that was used
    to produce it and -- when a ``reference`` answer and/or ``judge`` are
    supplied -- the Tier-1 / Tier-2 / Tier-3 scoring artefacts produced by
    :mod:`nemo_retriever.evaluation.scoring` and
    :class:`~nemo_retriever.llm.clients.judge.LLMJudge`.

    Production code no longer returns ``AnswerResult`` -- call
    :func:`nemo_retriever.generation.answer` or
    :func:`nemo_retriever.generation.eval` instead, which return a
    :class:`pandas.DataFrame` whose column union is this dataclass's
    fields.  ``AnswerResult`` is retained because the ``retriever
    answer`` CLI and the MCP server advertise this exact JSON schema
    (see :func:`nemo_retriever.answer_cli.row_to_answer_dict`).

    Attributes:
        query: The question that was answered.
        answer: The generated answer text.
        chunks: Retrieved chunk texts used as context, in rank order.
        metadata: Per-chunk metadata (source, page_number, etc.), aligned
            with ``chunks``.
        model: Model identifier that produced ``answer``.
        latency_s: Wall-clock latency of the generation call in seconds.
        error: Non-None when generation failed.  Scoring and judge are
            skipped when ``error`` is set.
        judge_score: LLM-judge Tier-3 score (1-5) when a judge was run.
        judge_reasoning: One-sentence rationale emitted by the judge.
        judge_error: Non-None when the judge call failed.
        token_f1: Tier-2 token-level F1 between ``answer`` and the
            reference answer (0.0-1.0).
        exact_match: Tier-2 normalised exact-match flag.
        answer_in_context: Tier-1 flag -- True if at least half of the
            reference answer's content words appear in the retrieved chunks.
        failure_mode: Classification produced by
            :func:`~nemo_retriever.evaluation.scoring.classify_failure`.
    """

    query: str
    answer: str
    chunks: list[str]
    metadata: list[dict[str, Any]]
    model: str
    latency_s: float
    error: Optional[str] = None
    judge_score: Optional[int] = None
    judge_reasoning: Optional[str] = None
    judge_error: Optional[str] = None
    token_f1: Optional[float] = None
    exact_match: Optional[bool] = None
    answer_in_context: Optional[bool] = None
    failure_mode: Optional[str] = None
