# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

    def retrieve(self, query: str, top_k: int) -> "RetrievalResult": ...


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
