# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

QueryFormat = Literal["hits", "evidence"]
QueryMode = Literal["classic", "agentic"]

# Agentic queries are replayed into every step of a multi-step LLM loop, so an
# oversized query multiplies prompt cost and latency. Roughly 1k tokens of
# natural-language question is far above any realistic retrieval query.
MAX_AGENTIC_QUERY_CHARS = 4096


class QueryRequest(BaseModel):
    query: str | list[str]
    top_k: int = Field(default=10, ge=1, le=1000)
    collection_name: str | None = Field(default=None, min_length=1, max_length=128)
    format: QueryFormat = Field(
        default="hits",
        description=(
            "Output shape: 'hits' (default) returns raw retrieval hits; 'evidence' "
            "returns the fidelity-tagged, citation-ready {evidence, coverage} shape. "
            "Agentic queries require format='hits'."
        ),
    )
    agentic: bool = Field(
        default=False,
        description=(
            "When true, run the server-configured agentic (ReAct) retrieval workflow. "
            "Requires agentic.enabled in service configuration. Response uses the same "
            "hits envelope as dense/hybrid query; document-level agentic results map "
            "doc_id onto source, keep result_source/rank in metadata, and leave "
            "chunk-level fields unset (null)."
        ),
    )

    @model_validator(mode="after")
    def _validate_agentic_request(self) -> "QueryRequest":
        if not self.agentic:
            return self
        if not isinstance(self.query, str):
            raise ValueError("agentic queries require a single query string, not a list")
        if not self.query.strip():
            raise ValueError("agentic query must be a non-empty string")
        if len(self.query) > MAX_AGENTIC_QUERY_CHARS:
            raise ValueError(f"agentic query exceeds max length of {MAX_AGENTIC_QUERY_CHARS} characters")
        if self.format != "hits":
            raise ValueError("agentic queries require format='hits'")
        return self

    @model_validator(mode="before")
    @classmethod
    def _reject_raw_storage_keys(cls, value: Any) -> Any:
        if isinstance(value, dict):
            raw_keys = {
                "table_name",
                "table",
                "physical_table",
                "lancedb_uri",
                "lance_uri",
                "uri",
                "table_path",
                "database_uri",
                "vdb_uri",
            }
            supplied = sorted(raw_keys.intersection(value))
            if supplied:
                raise ValueError(f"client-selected storage is not supported: {', '.join(supplied)}")
        return value


class QueryResult(BaseModel):
    hits: list[dict[str, Any]]


class QueryResponse(BaseModel):
    results: list[QueryResult]
    query_mode: QueryMode = Field(
        default="classic",
        description=(
            "Which /v1/query workflow produced this response: 'classic' (dense/hybrid) "
            "or 'agentic' (ReAct document ranking)."
        ),
    )

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
    query_mode: QueryMode = Field(
        default="classic",
        description="Evidence format is classic retrieval only; always 'classic'.",
    )
