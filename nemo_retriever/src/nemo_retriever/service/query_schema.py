# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

QueryFormat = Literal["hits", "evidence"]


class QueryRequest(BaseModel):
    query: str | list[str]
    top_k: int = Field(default=10, ge=1, le=1000)
    collection_name: str | None = Field(default=None, min_length=1, max_length=128)
    # Kept only for legacy fixed-table callers. Collection-aware requests may
    # never select physical storage.
    table_name: str | None = None
    format: QueryFormat = Field(
        default="hits",
        description=(
            "Output shape: 'hits' (default) returns raw retrieval hits; 'evidence' "
            "returns the fidelity-tagged, citation-ready {evidence, coverage} shape."
        ),
    )

    def model_post_init(self, __context: Any) -> None:
        if self.collection_name and self.table_name:
            raise ValueError("collection_name and table_name cannot be supplied together")


class QueryResult(BaseModel):
    hits: list[dict[str, Any]]


class QueryResponse(BaseModel):
    results: list[QueryResult]

    def hits_by_query(self, *, expected_results: int | None = None) -> list[list[dict[str, Any]]]:
        if expected_results is not None and len(self.results) != expected_results:
            raise ValueError(f"expected {expected_results} result set(s), got {len(self.results)}")
        return [result.hits for result in self.results]


class Locator(BaseModel):
    """Where an evidence item lives in its source (page / segment / timestamp / bbox)."""

    kind: str
    value: Any = None


class EvidenceItem(BaseModel):
    """One fidelity-tagged, citation-ready evidence span."""

    text: str
    source: str
    locator: Locator
    modality: str
    fidelity: str
    score: float
    citation: str


class Coverage(BaseModel):
    """Summary of what was searched, plus flagged thin spots."""

    strategies_used: list[str]
    n_docs_seen: int
    thin_spots: list[str]


class EvidenceResult(BaseModel):
    """One query's answer-ready evidence, mirroring ``retriever query --format evidence``."""

    evidence: list[EvidenceItem]
    coverage: Coverage


class EvidenceQueryResponse(BaseModel):
    results: list[EvidenceResult]
