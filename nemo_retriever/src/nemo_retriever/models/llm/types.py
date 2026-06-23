# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Protocol definitions and result models for LLM-based pipelines.

These abstractions allow retrieval strategies, LLM clients, and judges
to be swapped independently. They are consumed by both the evaluation
framework (``nemo_retriever.evaluation``) and the live RAG surface on
``nemo_retriever.retriever.Retriever``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


@runtime_checkable
class RetrieverStrategy(Protocol):
    """Pluggable retrieval strategy interface."""

    def retrieve(self, query: str, top_k: int) -> "RetrievalResult": ...


@runtime_checkable
class LLMClient(Protocol):
    """Pluggable LLM answer generation interface."""

    def generate(
        self,
        query: str,
        chunks: list[str],
        *,
        reasoning_enabled: Optional[bool] = None,
    ) -> "GenerationResult": ...


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
    (API error, empty candidate, or no valid rating). Valid scores are
    ragas ``AnswerAccuracy`` values on a ``0.0-1.0`` scale (higher is
    better). ``reasoning`` is empty -- ``AnswerAccuracy`` emits only a
    numeric rating.
    """

    score: Optional[float] = None
    reasoning: str = ""
    error: Optional[str] = None


class AnswerRequest(BaseModel):
    """Shared internal request model for answer generation.

    Public callers may continue using ergonomic keyword arguments on
    ``Retriever.answer``. Service and local code normalize into this model so
    query, retrieval, and per-call generation controls stay aligned.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = Field(default=5, ge=1)
    reasoning_enabled: Optional[bool] = None
    reference: Optional[str] = None
    judge_enabled: bool = False


class AnswerResult(BaseModel):
    """Result from a single live-RAG call to ``Retriever.answer``.

    Holds the generated answer alongside the retrieved context that was used
    to produce it and -- when a ``reference`` answer and/or ``judge`` are
    supplied -- the Tier-1 / Tier-2 / Tier-3 scoring artefacts produced by
    :mod:`nemo_retriever.evaluation.scoring` and
    :class:`~nemo_retriever.llm.clients.judge.LLMJudge`.

    Attributes:
        query: The question that was answered.
        answer: The generated answer text.
        chunks: Retrieved chunk texts used as context, in rank order.
        metadata: Per-chunk metadata (source, page_number, etc.), aligned
            with ``chunks``.
        model: Model identifier that produced ``answer``.
        latency_s: Wall-clock latency of the generation call in seconds.
        chunk_count: Number of retrieved chunks used for generation.
        error: Non-None when generation failed. Scoring and judge are
            skipped when ``error`` is set.
        judge_score: ragas AnswerAccuracy Tier-3 score (0.0-1.0) when a
            judge was run.
        judge_reasoning: Empty -- AnswerAccuracy emits only a numeric rating.
        judge_error: Non-None when the judge call failed.
        token_f1: Tier-2 token-level F1 between ``answer`` and the
            reference answer (0.0-1.0).
        exact_match: Tier-2 normalised exact-match flag.
        answer_in_context: Tier-1 flag -- True if at least half of the
            reference answer's content words appear in the retrieved chunks.
        failure_mode: Classification produced by
            :func:`~nemo_retriever.evaluation.scoring.classify_failure`.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    answer: str
    model: str
    latency_s: float
    chunk_count: int
    chunks: Optional[list[str]] = None
    metadata: Optional[list[dict[str, Any]]] = None
    error: Optional[str] = None
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None
    judge_error: Optional[str] = None
    token_f1: Optional[float] = None
    exact_match: Optional[bool] = None
    answer_in_context: Optional[bool] = None
    failure_mode: Optional[str] = None


def build_answer_result(
    *,
    query: str,
    retrieval: RetrievalResult,
    generation: GenerationResult,
    reference: Optional[str] = None,
    judge: Optional[AnswerJudge] = None,
) -> AnswerResult:
    """Build the shared answer result and optional scoring artefacts."""

    result = AnswerResult(
        query=query,
        answer=generation.answer,
        chunks=retrieval.chunks,
        metadata=retrieval.metadata,
        model=generation.model,
        latency_s=generation.latency_s,
        chunk_count=len(retrieval.chunks),
        error=generation.error,
    )

    if generation.error is not None or (reference is None and judge is None):
        return result

    populate_answer_scores(
        result,
        query=query,
        reference=reference,
        judge=judge,
        gen_error=generation.error,
    )
    return result


def populate_answer_scores(
    result: AnswerResult,
    *,
    query: str,
    reference: Optional[str],
    judge: Optional[AnswerJudge],
    gen_error: Optional[str],
) -> None:
    """Populate local/service answer scoring fields in-place."""

    from concurrent.futures import ThreadPoolExecutor

    from nemo_retriever.tools.evaluation.scoring import (
        answer_in_context,
        classify_failure,
        token_f1,
    )

    chunks = result.chunks or []

    def _scoring() -> tuple[Optional[bool], Optional[float], Optional[bool]]:
        if reference is None:
            return None, None, None
        aic = answer_in_context(reference, chunks)
        f1 = token_f1(reference, result.answer)
        return aic, float(f1.get("f1", 0.0)), bool(f1.get("exact_match", False))

    def _judging() -> tuple[Optional[float], Optional[str], Optional[str]]:
        if judge is None or reference is None:
            return None, None, None
        jr = judge.judge(query, reference, result.answer)
        return jr.score, jr.reasoning, jr.error

    with ThreadPoolExecutor(max_workers=2) as pool:
        scoring_future = pool.submit(_scoring)
        judge_future = pool.submit(_judging)
        aic, f1, em = scoring_future.result()
        judge_score, judge_reasoning, judge_error = judge_future.result()

    result.answer_in_context = aic
    result.token_f1 = f1
    result.exact_match = em
    result.judge_score = judge_score
    result.judge_reasoning = judge_reasoning
    result.judge_error = judge_error
    if reference is not None and aic is not None:
        result.failure_mode = classify_failure(
            ref_in_chunks=aic,
            judge_score=judge_score,
            gen_error=gen_error,
            candidate=result.answer,
        )
