"""Shared data contract between Phase 1 (Retrieval Agent) and Phase 2 (SQL Agent).

``RetrievalContext`` is the single structured object that Phase 1 produces and
Phase 2 consumes.  It is injected into the Phase 2 system prompt so that the
SQL generation agent starts with a clean context window — no tool-call history
from the retrieval phase is carried over.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class EntityCoverage(TypedDict):
    """Coverage record for a single entity extracted from the user question."""

    entity: str
    entity_type: Literal["metric", "dimension", "time_filter", "value"]
    resolved_as: Literal["column", "custom_analysis", "expression", "unresolved", "value", "time_filter"]
    candidates: list[dict]
    sql_expression: str | None  # only populated when resolved_as == "expression"
    matched_table: str | None  # populated when covered_by_existing_table is True
    matched_column: str | None  # populated when covered_by_existing_table is True


class RetrievalContext(TypedDict):
    """Output produced by Phase 1 and consumed by Phase 2.

    Attributes:
        entity_coverage: Per-entity grounding result list. One entry per
            metric/dimension entity extracted from the user question.
        relevant_tables: FK-expanded, deduplicated table dicts (same schema as
            ``get_relevant_fks_from_candidates_tables`` output).
        relevant_fks: FK relationship dicts.
        complex_candidates_str: Formatted strings for custom-analysis / certified
            SQL snippets (highest-priority reference for Phase 2).
        relevant_queries: Example queries from the knowledge base.
        coverage_complete: ``True`` when every metric/dimension entity resolved
            to at least one candidate or expression.  Phase 2 can use this flag
            to decide whether to proceed confidently or warn about missing data.
    """

    entity_coverage: list[EntityCoverage]
    relevant_tables: list[dict]
    relevant_fks: list[dict]
    complex_candidates_str: list[str]
    relevant_queries: list
    coverage_complete: bool


__all__ = ["EntityCoverage", "RetrievalContext"]
