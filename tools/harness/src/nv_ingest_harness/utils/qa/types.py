"""
Protocol definitions and dataclasses for the QA evaluation pipeline.

These abstractions allow retrieval strategies, LLM clients, and judges
to be swapped independently without modifying the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class RetrieverStrategy(Protocol):
    """Pluggable retrieval strategy interface."""

    def retrieve(self, query: str, top_k: int) -> "RetrievalResult":
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Maximum number of chunks to return.

        Returns:
            RetrievalResult with chunks and optional metadata.
        """
        ...


@runtime_checkable
class LLMClient(Protocol):
    """Pluggable LLM answer generation interface."""

    def generate(self, query: str, chunks: list[str]) -> "GenerationResult":
        """
        Generate an answer given a query and retrieved context chunks.

        Args:
            query: The user question.
            chunks: Retrieved text chunks to use as context.

        Returns:
            GenerationResult with the generated answer and latency.
        """
        ...


@runtime_checkable
class AnswerJudge(Protocol):
    """Pluggable answer scoring interface."""

    def judge(self, query: str, reference: str, candidate: str) -> "JudgeResult":
        """
        Score a candidate answer against a reference answer.

        Args:
            query: The original question.
            reference: Ground-truth reference answer.
            candidate: LLM-generated candidate answer to evaluate.

        Returns:
            JudgeResult with a 1-5 score and reasoning.
        """
        ...


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
    """Result from a single judge evaluation."""

    score: int
    reasoning: str
    error: Optional[str] = None


@dataclass
class QAQueryResult:
    """
    Aggregated per-query results across all configured LLMs.

    Each key in `generations` and `judgements` maps to a named LLM
    from qa_llm_configs (e.g., "nemotron_super_49b").

    Multi-tier fields (populated by orchestrator after generation/judging):
      answer_in_context: Tier-1 retrieval quality flag.
      token_f1: Tier-2 programmatic answer quality per model.
      failure_mode: Per-model failure classification.
    """

    query: str
    reference_answer: str
    retrieval: RetrievalResult
    generations: dict[str, GenerationResult] = field(default_factory=dict)
    judgements: dict[str, JudgeResult] = field(default_factory=dict)
    answer_in_context: Optional[bool] = None
    token_f1: dict[str, dict] = field(default_factory=dict)
    failure_mode: dict[str, str] = field(default_factory=dict)
