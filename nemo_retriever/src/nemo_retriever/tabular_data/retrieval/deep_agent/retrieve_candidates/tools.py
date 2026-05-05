"""LangChain tools for the Phase 1 Retrieval Deep Agent.

Each tool writes its results directly into a shared ``RetrievalStore`` — the
same pattern used by ``ExecutionStore`` in Phase 2.  The agent never needs to
pass tables or entities as JSON arguments between calls; the store accumulates
state internally.

Tools:
- ``decompose_question``   — splits the question into typed, priority-ordered entities
- ``retrieve_for_entity``  — per-entity candidate + table/FK retrieval with intra-table
                             coverage check (no state arguments required)
- ``synthesize_expression`` — derives a SQL expression for zero-coverage entities

Use ``build_retrieval_tools(payload, llm)`` to get ``(tools, store)``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from nemo_retriever.tabular_data.retrieval.deep_agent.retrieve_candidates.candidates_preparation import (
    CandidatePreparationAgent,
)
from nemo_retriever.tabular_data.retrieval.llm_invoke import invoke_with_structured_output
from nemo_retriever.tabular_data.retrieval.deep_agent.context import EntityCoverage, RetrievalContext
from nemo_retriever.tabular_data.retrieval.deep_agent.state import AgentPayload
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from nemo_retriever.tabular_data.retrieval.utils import (
    _apply_foreign_key_hints,
    clean_results,
    dedupe_merge_relevant_tables,
    extract_candidates,
    get_relevant_fks_from_candidates_tables,
    get_relevant_tables_with_fks,
)

logger = logging.getLogger(__name__)

_prep_agent = CandidatePreparationAgent()


# ---------------------------------------------------------------------------
# RetrievalStore
# ---------------------------------------------------------------------------


class RetrievalStore:
    """Mutable per-request store written by the retrieval tools, read by the runtime.

    Mirrors the ``ExecutionStore`` pattern from Phase 2:
    - ``decompose_question`` writes to ``entities`` and ``question``.
    - ``retrieve_for_entity`` appends to ``entity_results``, ``accumulated_tables``,
      ``accumulated_fks``, and ``custom_candidates``.
    - ``synthesize_expression`` patches the matching entry in ``entity_results``.
    - ``filter_relevant_tables`` prunes ``accumulated_tables`` and ``accumulated_fks``
      in-place based on LLM relevance judgement.

    The runtime calls ``as_context()`` after the agent finishes to build the
    ``RetrievalContext`` directly from store state — no JSON parsing of agent
    messages required.
    """

    def __init__(self, retriever=None) -> None:
        self.retriever = retriever  # shared DeepAgentRetriever — init once in main.py
        self.question: str = ""
        self.entities: list[dict] = []
        self.entity_results: list[dict] = []
        self.accumulated_tables: list[dict] = []
        self.accumulated_fks: list[dict] = []
        self.custom_candidates: list[dict] = []

    # ------------------------------------------------------------------
    # Helpers used by tools
    # ------------------------------------------------------------------

    def _resolved_as(self, result: dict) -> str:
        entity_type = result.get("entity_type", "")
        if entity_type == "value":
            return "value"
        if entity_type == "time_filter":
            return "time_filter"
        candidates = result.get("candidates", [])
        if any(c.get("label") == Labels.CUSTOM_ANALYSIS for c in candidates):
            return "custom_analysis"
        if candidates or result.get("relevant_tables"):
            return "column"
        if result.get("sql_expression"):
            return "expression"
        return "unresolved"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def as_context(self) -> RetrievalContext | None:
        """Build a ``RetrievalContext`` from accumulated store state.

        Returns ``None`` when no tools have written any data yet.
        """
        if not self.entity_results and not self.entities:
            return None

        entity_coverage: list[EntityCoverage] = []
        for r in self.entity_results:
            entity_coverage.append(
                {
                    "entity": r.get("entity", ""),
                    "entity_type": r.get("entity_type", "dimension"),
                    "resolved_as": self._resolved_as(r),
                    "candidates": r.get("candidates", []),
                    "sql_expression": r.get("sql_expression"),
                    "matched_table": None,
                    "matched_column": None,
                }
            )

        coverage_complete = all(
            ec["resolved_as"] != "unresolved" for ec in entity_coverage if ec["entity_type"] in ("metric", "dimension")
        )

        complex_candidates_str = _prep_agent._build_complex_candidates_str(self.custom_candidates)

        return {
            "entity_coverage": entity_coverage,
            "relevant_tables": self.accumulated_tables,
            "relevant_fks": self.accumulated_fks,
            "complex_candidates_str": complex_candidates_str,
            "relevant_queries": [],
            "coverage_complete": coverage_complete,
        }


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM outputs
# ---------------------------------------------------------------------------


class _EntityItem(BaseModel):
    term: str = Field(
        ...,
        description=(
            "A semantic concept or phrase from the question — may be multiple words. "
            "Plain natural-language text only — spell out comparators in words. "
            "Never split a multi-word concept into individual words."
        ),
    )
    entity_type: Literal["metric", "dimension", "time_filter", "value"] = Field(
        ...,
        description=(
            "metric — a measurable value to compute (revenue, profit, count, average …); "
            "dimension — a schema concept that maps to a table/column (product, student, customer …); "
            "time_filter — a time period or date (last month, Q1 2024, yesterday …); "
            "value — a specific named literal that will become a WHERE filter "
            "(e.g. 'Seattle' in 'students from Seattle', 'Enterprise' in 'Enterprise customers'). "
        ),
    )
    priority: int = Field(
        ...,
        description=(
            "Importance order for retrieval. Lower = retrieved first. "
            "1 = primary subject (the main thing being queried, e.g. 'students', 'orders'); "
            "2 = grouping/join dimension (related objects, e.g. 'product', 'department'); "
            "3 = filter attribute concept (a field used to narrow results, e.g. 'city', 'status'); "
            "4 = filter value or time period (specific literal, e.g. 'Seattle', 'last month')."
        ),
    )


class _DecomposeResult(BaseModel):
    entities: list[_EntityItem] = Field(
        ...,
        description=("All entities extracted from the question, ordered by priority ascending " "(priority=1 first)."),
    )


class _ExpressionResult(BaseModel):
    expression: str = Field(
        ...,
        description=(
            "A SQL expression (NOT a full query) that computes the entity from the given columns. "
            "Example: 'income - outcome' or 'SUM(s.sales_amount) / COUNT(DISTINCT s.customer_id)'. "
            "Use ONLY column names from the provided available_columns list."
        ),
    )
    columns_used: list[str] = Field(
        ...,
        description="Fully-qualified column names used in the expression.",
    )


# ---------------------------------------------------------------------------
# Tool 1 — decompose_question
# ---------------------------------------------------------------------------


def _make_decompose_question_tool(llm: Any, store: RetrievalStore):
    """Return a ``decompose_question`` tool that writes entities into *store*."""

    @tool
    def decompose_question(question: str) -> str:
        """Decompose the user question into typed, priority-ordered entities.

        Call this as the FIRST step.  The entities are stored internally —
        you do not need to pass them to subsequent tool calls.

        Returns a confirmation listing the entities found and their priority order.

        Args:
            question: The raw user question.
        """
        prompt = f"""You are extracting database-retrieval entities from a user question.

User Question:
{question}

GOAL — atomic decomposition.
Every distinct DB concept in the question must become its own entity. Each
entity must map to ONE artifact: one table, column, measurable, time period,
or literal value. Never combine multiple concepts into a single entity.

How to split:
- Each output column the user asks for → its own entity.
- Each subject / object noun → its own entity.
- Each predicate / filter / threshold → its own entity, paired with the field
  it modifies in the SAME entity.
- Each time period or proper-noun literal → its own entity.

HARD CONSTRAINT — every entity term MUST contain a concrete noun (the
subject/field/concept being referred to). Any term made up only of comparators,
numbers, qualifiers, quantifiers, or prepositions — with no noun — is INVALID
and MUST be merged into the entity for the noun it modifies, by including that
noun in the term itself. Before emitting an entity, check: does this term name
a thing that can be looked up in a database? If not, fix it.

Normally an entity should be up to 3 words.

Each entity term must be PLAIN NATURAL-LANGUAGE TEXT — spell out comparators
in words rather than using mathematical symbols.

Entity types:
- metric: a measurable value or aggregate the user wants to compute
- dimension: a schema object — table or column concept
- time_filter: a time period or date expression
- value: a SPECIFIC NAMED LITERAL or threshold for a WHERE clause

Priority (lower = retrieved first):
1 = primary subject being queried
2 = related object / join target
3 = filter attribute concept
4 = filter value or time literal

Order entities by priority ascending in your output."""

        messages = [SystemMessage(content=prompt)]
        result = invoke_with_structured_output(llm, messages, _DecomposeResult)

        if result is None:
            store.entities = []
            return "decompose_question failed — no entities extracted."

        entities = sorted(result.model_dump()["entities"], key=lambda e: e.get("priority", 99))
        store.question = question
        store.entities = entities

        lines = [f"Extracted {len(entities)} entities (call retrieve_for_entity for each):"]
        for i, e in enumerate(entities, 1):
            lines.append(f"  {i}. [{e['entity_type']} p={e['priority']}] {e['term']}")
        return "\n".join(lines)

    return decompose_question


# ---------------------------------------------------------------------------
# Tool 2 — retrieve_for_entity
# ---------------------------------------------------------------------------


def _make_retrieve_for_entity_tool(store: RetrievalStore):
    """Return a ``retrieve_for_entity`` tool that reads/writes *store*."""

    @tool
    def retrieve_for_entity(entity_term: str, entity_type: str = "dimension") -> str:
        """Retrieve semantically relevant candidates for a single entity term.

        Call this ONCE for EACH entity returned by decompose_question, in priority order.

        Args:
            entity_term: The entity term to search for (e.g. "revenue", "city", "students").
            entity_type: The entity type from decompose_question
                         ("metric", "dimension", "time_filter", or "value").

        Returns:
            A short summary: covered/not-covered, tables found.
        """
        # ── Vector/graph search ────────────────────────────────────────────
        try:
            custom_raw, column_raw = extract_candidates(
                retriever=store.retriever,
                entities=[entity_term],
                query_no_values=entity_term,
                query_with_values=entity_term,
            )
            custom_candidates = clean_results(list(custom_raw or []))
            column_candidates = clean_results(list(column_raw or []))

            # Column candidates are used only to derive their parent tables.
            # Custom analysis candidates go into the candidates list.
            all_for_tables = custom_candidates + column_candidates
            relevant_tables, relevant_fks = get_relevant_fks_from_candidates_tables(all_for_tables)
            add_tables, add_fks = get_relevant_tables_with_fks(store.retriever, entity_term, k=3)
            relevant_tables.extend(add_tables)
            relevant_fks.extend(add_fks)
            relevant_tables = dedupe_merge_relevant_tables(relevant_tables)
            _apply_foreign_key_hints(relevant_tables, relevant_fks)

            # Accumulate into store
            store.custom_candidates.extend(custom_candidates)
            store.accumulated_tables = dedupe_merge_relevant_tables(store.accumulated_tables + relevant_tables)
            store.accumulated_fks.extend(relevant_fks)

            covered = len(custom_candidates) > 0 or len(relevant_tables) > 0

            store.entity_results.append(
                {
                    "entity": entity_term,
                    "entity_type": entity_type,
                    "candidates": custom_candidates,
                    "relevant_tables": relevant_tables,
                    "relevant_fks": relevant_fks,
                    "sql_expression": None,
                }
            )

            status = "COVERED" if covered else "NOT COVERED"
            return (
                f"'{entity_term}' — {status}. "
                f"Found {len(custom_candidates)} custom analyses, "
                f"{len(relevant_tables)} tables."
            )

        except Exception as exc:
            logger.warning("retrieve_for_entity failed for %r: %s", entity_term, exc)
            store.entity_results.append(
                {
                    "entity": entity_term,
                    "entity_type": entity_type,
                    "candidates": [],
                    "relevant_tables": [],
                    "relevant_fks": [],
                    "sql_expression": None,
                }
            )
            return f"'{entity_term}' — retrieval failed: {exc}"

    return retrieve_for_entity


# ---------------------------------------------------------------------------
# Tool 3 — synthesize_expression
# ---------------------------------------------------------------------------


def _make_synthesize_expression_tool(llm: Any, store: RetrievalStore):
    """Return a ``synthesize_expression`` tool that reads columns from *store*
    and patches the matching entity result.
    """

    @tool
    def synthesize_expression(entity_term: str) -> str:
        """Derive a SQL expression for an entity that has no direct candidate match.

        Call this ONLY when retrieve_for_entity returned NOT COVERED for an entity.
        Uses all columns accumulated in the store — no column list argument needed.

        Args:
            entity_term: The entity that has no direct candidate (e.g. "revenue").

        Returns:
            A summary: expression derived or failure reason.
        """
        # Collect all column names from accumulated tables
        col_names: list[str] = []
        for table in store.accumulated_tables:
            for col in table.get("columns") or []:
                if isinstance(col, dict):
                    name = col.get("name") or col.get("id") or ""
                elif isinstance(col, str):
                    name = col
                else:
                    continue
                if name:
                    col_names.append(name)

        if not col_names:
            _patch_expression(store, entity_term, "", False)
            return f"'{entity_term}' — no columns available for synthesis."

        prompt = f"""You are a SQL expression composer.

The user question refers to "{entity_term}", but there is no database column with that exact name.
Compose a SQL expression for "{entity_term}" using ONLY the columns listed below.

Available columns (ONLY use these — never invent new column names):
{json.dumps(col_names, indent=2)}

Rules:
1. Use ONLY column names from the list above.
2. Output a SQL expression fragment (NOT a full SELECT statement).
3. Common patterns: subtraction (income - cost = profit), ratio, SUM, COUNT, AVG.
4. If you cannot express "{entity_term}" from these columns, leave expression empty.

Return a JSON object with keys: expression, columns_used."""

        messages = [SystemMessage(content=prompt)]
        result = invoke_with_structured_output(llm, messages, _ExpressionResult)

        if result is None:
            _patch_expression(store, entity_term, "", False)
            return f"'{entity_term}' — synthesis LLM call failed."

        valid_col_set = {n.lower() for n in col_names}
        validated_cols = [c for c in result.columns_used if str(c).split(".")[-1].lower() in valid_col_set]
        success = bool(result.expression)
        _patch_expression(store, entity_term, result.expression if success else "", success)

        if success:
            return f"'{entity_term}' — expression synthesized: {result.expression} (columns={validated_cols})"
        return f"'{entity_term}' — synthesis produced no expression."

    return synthesize_expression


def _patch_expression(store: RetrievalStore, entity_term: str, expression: str, success: bool) -> None:
    """Write the synthesized expression back into the matching entity_result entry."""
    for r in store.entity_results:
        if r.get("entity") == entity_term:
            r["sql_expression"] = expression if success else None
            return
    # No existing entry — add one marked as unresolved
    store.entity_results.append(
        {
            "entity": entity_term,
            "entity_type": "metric",
            "candidates": [],
            "relevant_tables": [],
            "relevant_fks": [],
            "sql_expression": expression if success else None,
        }
    )


# ---------------------------------------------------------------------------
# Tool 4 — filter_relevant_tables
# ---------------------------------------------------------------------------


class _TableFilterResult(BaseModel):
    relevant_table_ids: list[str] = Field(
        ...,
        description=(
            "IDs of tables (exact match from the provided schema) that are genuinely needed "
            "to answer the user question. Omit any table whose subject domain does not match "
            "the question's intent, even if one of its columns happened to match a search term. "
            "Use the table id (not the name) since the same table name can appear in different schemas."
        ),
    )


def _make_filter_relevant_tables_tool(store: RetrievalStore, llm: Any):
    """Return a ``filter_relevant_tables`` tool that prunes *store.accumulated_tables* in-place."""

    @tool
    def filter_relevant_tables() -> str:
        """Remove tables that are not relevant to the user question.

        Call this ONCE after all retrieve_for_entity (and synthesize_expression) calls
        are complete. The tool compares every accumulated table against the question's
        intent and drops tables whose domain does not match — even if the vector search
        retrieved them because they share a column name with a search term.

        Tables are identified by their unique ``id`` (not name) since the same table
        name can appear in different schemas. The returned summary lists kept and
        removed tables as ``id (name)`` pairs — always refer to tables by ``id``
        downstream.

        No arguments needed; the tool reads the question and tables from the store.

        Returns:
            A summary of which tables were kept and which were removed, formatted as
            ``id (name)`` so callers can reference tables unambiguously.
        """
        tables = store.accumulated_tables
        if not tables:
            return "No tables in store — nothing to filter."

        schema_lines: list[str] = []
        for t in tables:
            t_id = str(t.get("id") or "").strip()
            if not t_id:
                continue
            t_name = t.get("name") or ""
            t_desc = t.get("description") or ""
            col_names = []
            for col in t.get("columns") or []:
                if isinstance(col, dict):
                    col_names.append(col.get("name") or "")
                elif isinstance(col, str):
                    col_names.append(col)
            header = f"id={t_id} | name={t_name}" + (f" — {t_desc}" if t_desc else "")
            schema_lines.append(f"  {header}: [{', '.join(c for c in col_names if c)}]")

        schema_str = "\n".join(schema_lines)

        prompt = f"""You are a database schema relevance filter.

User question: "{store.question}"

The following tables were retrieved by a vector search. Your job is to keep ONLY the
tables that are genuinely needed to answer this question. Remove any table whose
subject domain does not match the question's intent, even if one of its columns
coincidentally matched a search term.

Each table is listed with a unique id and a name. The same table name can appear
in different schemas, so always identify tables by their id.

Retrieved tables:
{schema_str}

Return only the ids of relevant tables. Use the exact id values from the list above."""

        messages = [SystemMessage(content=prompt)]
        result = invoke_with_structured_output(llm, messages, _TableFilterResult)

        if result is None:
            return "filter_relevant_tables: LLM call failed — tables unchanged."

        keep_ids = {str(i) for i in result.relevant_table_ids}
        before = [(str(t.get("id") or ""), t.get("name")) for t in tables]
        store.accumulated_tables = [t for t in tables if str(t.get("id") or "") in keep_ids]

        # Remove FKs whose both sides are no longer present
        remaining_names = {t.get("name") for t in store.accumulated_tables}
        store.accumulated_fks = [
            fk
            for fk in store.accumulated_fks
            if fk.get("from_table") in remaining_names or fk.get("to_table") in remaining_names
        ]

        def _fmt(tid: str, name: Any) -> str:
            return f"{tid} ({name})" if name else tid

        kept = [_fmt(tid, name) for tid, name in before if tid in keep_ids]
        removed = [_fmt(tid, name) for tid, name in before if tid not in keep_ids]
        parts = [f"Kept {len(kept)} (id (name)): {kept}"]
        if removed:
            parts.append(f"Removed {len(removed)} (id (name)): {removed}")
        return " | ".join(parts)

    return filter_relevant_tables


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_retrieval_tools(payload: AgentPayload, llm: Any, retriever=None) -> tuple[list, RetrievalStore]:
    """Build and return the list of Phase 1 Retrieval Agent tools and a shared ``RetrievalStore``.

    Tools close over the store so all state is accumulated automatically as the
    agent calls them — no JSON passing between tool calls required.

    Args:
        payload: The ``AgentPayload`` from the caller.
        llm: The LLM client used by ``decompose_question`` and
            ``synthesize_expression``.
        retriever: Optional pre-built ``DeepAgentRetriever`` singleton from
            ``main.py``.  When provided, the same instance is reused across all
            ``retrieve_for_entity`` calls in this session instead of creating a
            new one each time.

    Returns:
        ``(tools, store)`` — tools list and the ``RetrievalStore`` that will be
        populated as the agent runs.
    """
    store = RetrievalStore(retriever=retriever)
    return [
        _make_decompose_question_tool(llm, store),
        _make_retrieve_for_entity_tool(store),
        _make_synthesize_expression_tool(llm, store),
        _make_filter_relevant_tables_tool(store, llm),
    ], store


__all__ = ["build_retrieval_tools", "RetrievalStore"]
