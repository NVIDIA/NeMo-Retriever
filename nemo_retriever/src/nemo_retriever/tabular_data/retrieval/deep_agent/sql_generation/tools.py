"""LangChain tools for the Phase 2 SQL Deep Agent.

State is accumulated in a ``SqlGenerationStore`` — the same pattern as
``RetrievalStore`` in Phase 1.  Tools read/write the store; the agent never
needs to pass SQL strings or plans between calls.

Tools
-----
- ``plan_query``   — Step 1: produce a structured query plan from the RetrievalContext
- ``generate_sql`` — Step 2: write SQL from the plan
- ``validate_sql`` — Step 3: validate SQL (reads ``store.draft_sql``)
- ``fix_sql``      — Step 4 (conditional): targeted fix on validation error

Use ``build_sql_tools(payload, llm)`` → ``(tools, store)``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from nemo_retriever.tabular_data.retrieval.deep_agent.sql_generation.query_validation import (
    query_validation,
)
from nemo_retriever.tabular_data.retrieval.llm_invoke import invoke_with_structured_output
from nemo_retriever.tabular_data.retrieval.deep_agent.context import RetrievalContext
from nemo_retriever.tabular_data.retrieval.deep_agent.state import AgentPayload, get_dialect

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SqlGenerationStore
# ---------------------------------------------------------------------------


class SqlGenerationStore:
    """Mutable per-request store written by the SQL generation tools, read by Phase 3.

    Mirrors the ``RetrievalStore`` pattern from Phase 1:
    - ``plan_query``   writes to ``plan``.
    - ``generate_sql`` writes to ``draft_sql``.
    - ``validate_sql`` writes to ``sql`` on success.
    - ``fix_sql``      overwrites ``draft_sql`` with the corrected version.

    ``main.py`` reads ``store.sql`` (the last validated SQL) for Phase 3 execution.
    """

    def __init__(self, question: str, retrieval_ctx: RetrievalContext) -> None:
        self.question: str = question
        self.retrieval_ctx: RetrievalContext = retrieval_ctx
        self.plan: dict | None = None
        self.draft_sql: str | None = None
        self.sql: str | None = None

    def as_answer(self) -> dict | None:
        """Return a partial answer dict if validated SQL is present, else ``None``."""
        if self.sql is None:
            return None
        return {
            "sql_code": self.sql,
            "answer": "",
            "result": None,
            "semantic_elements": [],
        }


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class _QueryPlan(BaseModel):
    tables_to_use: list[str] = Field(
        ...,
        description=(
            "Fully-qualified table references in the form 'SCHEMA.TABLE AS alias' "
            "(e.g. 'school_scheduling.Students AS s'). "
            "NEVER omit the schema prefix.  NEVER write just the table name or alias alone. "
            "Include only tables that contribute a SELECT column, a WHERE condition, "
            "or a required JOIN leg."
        ),
    )
    join_conditions: list[str] = Field(
        default_factory=list,
        description=("FK-based JOIN conditions in the form 'alias1.col = alias2.col'. "),
    )
    select_expressions: list[str] = Field(
        ...,
        description=(
            "Exactly what to put in the SELECT clause: column references "
            "(alias.column), aggregation expressions (SUM(alias.col)), "
            "or sql_expression fragments from entity_coverage."
        ),
    )
    where_conditions: list[str] = Field(
        default_factory=list,
        description="WHERE predicates, e.g. \"s.city = 'Seattle'\".",
    )
    group_by: list[str] = Field(
        default_factory=list,
        description="Columns/expressions for GROUP BY.",
    )
    having_conditions: list[str] = Field(
        default_factory=list,
        description=(
            "HAVING predicates that compare against aggregate expressions, "
            'e.g. "COUNT(s.seat_id) > 2". '
            "Use this — together with group_by — instead of a correlated "
            "subquery in where_conditions whenever the user filters on a "
            "count/sum/avg of a related table."
        ),
    )
    order_by: list[str] = Field(
        default_factory=list,
        description="Columns/expressions for ORDER BY (include ASC/DESC).",
    )
    use_cte: bool = Field(
        default=False,
        description="True if CTEs improve clarity or are needed for multi-step logic.",
    )
    cte_descriptions: list[str] = Field(
        default_factory=list,
        description="One plain-English description per CTE (only when use_cte=True).",
    )
    notes: str = Field(
        default="",
        description="Any special considerations: dialect quirks, missing coverage, etc.",
    )


class _GeneratedSQL(BaseModel):
    sql: str = Field(
        ...,
        description=(
            "Complete, fully-qualified SELECT statement following the query plan. "
            "Every table must be SCHEMA.TABLE AS alias; every column must be alias.column. "
            "No markdown fences, no comments inside the SQL."
        ),
    )


class _FixedSQL(BaseModel):
    sql: str = Field(
        ...,
        description="The corrected SQL statement.  Must differ from the failing SQL.",
    )
    changes_summary: str = Field(
        ...,
        description="One sentence describing what was changed and why.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_sql_fences(sql: str) -> str:
    """Strip markdown code fences from a SQL string (kept for backward compat)."""
    stripped = sql.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        return "\n".join(lines[start:end]).strip()
    return sql


def _compact_schema(retrieval_ctx: RetrievalContext) -> str:
    """Return a compact schema summary for LLM prompts."""
    lines: list[str] = []
    for t in retrieval_ctx.get("relevant_tables") or []:
        name = t.get("table_name") or t.get("name") or ""
        schema = t.get("schema_name") or ""
        fqn = f"{schema}.{name}" if schema else name
        cols = []
        for col in t.get("columns") or []:
            if isinstance(col, dict):
                name = col.get("name") or ""
                dtype = col.get("data_type") or ""
                cols.append(f"{name} ({dtype})" if dtype else name)
            elif isinstance(col, str):
                cols.append(col)
        lines.append(f"  {fqn}: [{', '.join(c for c in cols if c)}]")
    return "\n".join(lines) if lines else "  (no tables)"


def _compact_schema_for_sql(retrieval_ctx: RetrievalContext, sql: str) -> str:
    """Compact schema filtered to tables actually referenced in *sql*.

    Falls back to the full ``_compact_schema`` when no table name matches —
    typically because the draft SQL is empty or completely off-schema.
    """
    sql_lc = (sql or "").lower()
    tables = retrieval_ctx.get("relevant_tables") or []
    matched = []
    for t in tables:
        name = (t.get("table_name") or t.get("name") or "").lower()
        if name and name in sql_lc:
            matched.append(t)
    if not matched:
        return _compact_schema(retrieval_ctx)
    return _compact_schema({**retrieval_ctx, "relevant_tables": matched})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tool 1 — plan_query
# ---------------------------------------------------------------------------


def _make_plan_query_tool(store: SqlGenerationStore, llm: Any):
    """Return a ``plan_query`` tool that reads the store and writes ``store.plan``."""

    @tool
    def plan_query() -> str:
        """Plan the SQL query structure from the RetrievalContext.

        Call this as the FIRST step.  The tool reads the question and the full
        RetrievalContext from the store automatically.

        Produces a structured query plan (tables, joins, select expressions,
        WHERE conditions, GROUP BY, ORDER BY, CTE flag) and stores it
        internally for generate_sql to consume.

        Returns:
            A summary of the query plan.
        """
        ctx = store.retrieval_ctx
        logger.info(
            "plan_query invoked | question_len=%d | entity_coverage=%d | relevant_tables=%d | relevant_fks=%d | "
            "coverage_complete=%s",
            len(store.question or ""),
            len(ctx.get("entity_coverage") or []),
            len(ctx.get("relevant_tables") or []),
            len(ctx.get("relevant_fks") or []),
            ctx.get("coverage_complete"),
        )
        entity_lines = []
        for ec in ctx.get("entity_coverage") or []:
            expr = f" → sql_expression: {ec['sql_expression']}" if ec.get("sql_expression") else ""
            entity_lines.append(f"  - {ec['entity']} ({ec['entity_type']}, resolved_as={ec['resolved_as']}){expr}")

        fk_lines = []
        for fk in ctx.get("relevant_fks") or []:
            fk_lines.append(
                f"  - {fk.get('from_table')}.{fk.get('from_column')} → " f"{fk.get('to_table')}.{fk.get('to_column')}"
            )

        # Hard SQL syntax rules + planning discipline live in the
        # ``sql-rules`` and ``query-planning`` skills loaded by the
        # ``query-planner`` subagent.  This prompt only carries the inputs
        # specific to this question.
        prompt = f"""You are producing a structured SQL query plan.

User question: "{store.question}"

Entities identified:
{chr(10).join(entity_lines) or "  (none)"}

Available tables (schema.table: [columns]):
{_compact_schema(ctx)}

Foreign-key relationships:
{chr(10).join(fk_lines) or "  (none)"}

coverage_complete: {ctx.get('coverage_complete', False)}

Apply the planning rules from your skills. Produce a structured plan
whose strings are already fully qualified ("SCHEMA.TABLE AS alias",
"alias.column") so the SQL author can translate it verbatim."""

        messages = [SystemMessage(content=prompt)]
        result = invoke_with_structured_output(llm, messages, _QueryPlan)

        if result is None:
            return "plan_query: LLM call failed — no plan produced."

        store.plan = result.model_dump()
        summary_lines = [
            "Query plan ready:",
            f"  tables: {result.tables_to_use}",
            f"  joins: {result.join_conditions}",
            f"  select: {result.select_expressions}",
            f"  where: {result.where_conditions}",
            f"  group_by: {result.group_by}",
            f"  having: {result.having_conditions}",
            f"  order_by: {result.order_by}",
            f"  use_cte: {result.use_cte}",
        ]
        if result.notes:
            summary_lines.append(f"  notes: {result.notes}")
        return "\n".join(summary_lines)

    return plan_query


# ---------------------------------------------------------------------------
# Tool 2 — generate_sql
# ---------------------------------------------------------------------------


def _make_generate_sql_tool(store: SqlGenerationStore, llm: Any):
    """Return a ``generate_sql`` tool that reads ``store.plan`` and writes ``store.draft_sql``."""

    @tool
    def generate_sql() -> str:
        """Generate SQL from the query plan.

        Call this AFTER plan_query.  The tool reads the plan and the full
        table schema from the store — no arguments needed.

        Writes the generated SQL to the store for validate_sql to consume.

        Returns:
            The generated SQL string (also stored internally).
        """
        if store.plan is None:
            return "generate_sql: no plan in store — call plan_query first."

        plan = store.plan

        # Hard SQL syntax rules and constructability heuristics live in the
        # ``sql-rules`` and ``sql-generation`` skills loaded by the
        # ``sql-author`` subagent.  The plan already lists fully-qualified
        # identifiers and the only valid FKs / tables, so this prompt does
        # not re-ship the schema or FK list.
        prompt = f"""You are translating a structured plan into a single SELECT statement.

User question: "{store.question}"

Query plan (use ONLY tables_to_use, join_conditions, and select / where /
group / having / order / having_conditions present in this plan):
{json.dumps(plan, indent=2)}

Apply your skills:
- Every identifier in the plan is already fully qualified — preserve that.
- Place every join predicate from join_conditions in an explicit JOIN ... ON,
  never in WHERE.
- Translate having_conditions verbatim into HAVING.
- Embed any sql_expression fragments verbatim in SELECT / HAVING.
- SELECT-only. No SQL comments. No markdown fences."""

        messages = [SystemMessage(content=prompt)]
        result = invoke_with_structured_output(llm, messages, _GeneratedSQL)

        if result is None or not result.sql.strip():
            return "generate_sql: LLM call failed — no SQL produced."

        store.draft_sql = result.sql.strip()
        return store.draft_sql

    return generate_sql


# ---------------------------------------------------------------------------
# Tool 3 — validate_sql
# ---------------------------------------------------------------------------


def _make_validate_sql_tool(dialects: list[str] | None, store: SqlGenerationStore):
    """Return a ``validate_sql`` tool bound to *dialects* and *store*."""

    _dialects = dialects or []

    @tool
    def validate_sql() -> str:
        """Validate the current draft SQL for schema correctness and SELECT-only enforcement.

        Call this AFTER generate_sql (or fix_sql).  Reads the SQL from the store
        automatically — no argument needed.

        If valid, the SQL is saved internally as the final validated result.
        If invalid, call fix_sql with the returned error, then call validate_sql again.
        Retry up to 4 times total.

        Returns:
            JSON object with:
              - ``valid``: bool
              - ``error``: error message string (empty when valid)
              - ``sql_columns``: list of column IDs referenced by the query
        """
        sql = store.draft_sql
        if not sql:
            return json.dumps(
                {"valid": False, "error": "No SQL in store — call generate_sql first.", "sql_columns": []}
            )

        sql = _strip_sql_fences(sql)
        try:
            result = query_validation(None, sql, _dialects, user_participants=[])
            if result.get("error"):
                return json.dumps(
                    {
                        "valid": False,
                        "error": result["error"],
                        "sql_columns": [],
                    }
                )
            store.sql = sql
            return json.dumps(
                {
                    "valid": True,
                    "error": "",
                    "sql_columns": result.get("sql_columns") or [],
                }
            )
        except Exception as exc:
            logger.warning("validate_sql failed: %s", exc)
            return json.dumps({"valid": False, "error": str(exc), "sql_columns": []})

    return validate_sql


# ---------------------------------------------------------------------------
# Tool 4 — fix_sql
# ---------------------------------------------------------------------------


def _make_fix_sql_tool(store: SqlGenerationStore, llm: Any):
    """Return a ``fix_sql`` tool that reads ``store.draft_sql`` and overwrites it with a fix."""

    @tool
    def fix_sql(error: str) -> str:
        """Fix the current draft SQL based on a validation error.

        Call this when validate_sql returns valid=false.
        Pass the exact error string from the validate_sql result.
        The tool reads the current draft SQL from the store, applies a targeted
        fix, and writes the corrected SQL back to the store.
        Then call validate_sql again.

        Args:
            error: The exact error message returned by validate_sql.

        Returns:
            The corrected SQL string (also stored internally for the next validate_sql call).
        """
        if not store.draft_sql:
            return "fix_sql: no draft SQL in store — call generate_sql first."

        hint = ""
        if "unknown column name" in error.lower() and "in table" in error.lower():
            hint = (
                "\nHINT: 'Unknown column name in table' usually means you used the right table "
                "name but from the wrong schema.  Check if the same table exists under a "
                "different schema in the available schema list, or remove the column."
            )

        # Hard SQL syntax rules live in the ``sql-rules`` skill loaded by
        # the ``sql-author`` subagent.  Schema is filtered to the tables
        # actually referenced in the failing SQL — anything else is just
        # noise for a targeted fix.
        prompt = f"""You are fixing a SQL query that failed validation.

User question: "{store.question}"

Failing SQL:
{store.draft_sql}

Validation error:
{error}{hint}

Schema for tables referenced in the failing SQL (schema.table: [columns]):
{_compact_schema_for_sql(store.retrieval_ctx, store.draft_sql)}

Apply your skills:
- Fix ONLY what the error describes; do not rewrite unrelated parts.
- Return SQL that is different from the failing SQL.
- No markdown fences, no comments."""

        messages = [SystemMessage(content=prompt)]
        result = invoke_with_structured_output(llm, messages, _FixedSQL)

        if result is None or not result.sql.strip():
            return "fix_sql: LLM call failed — draft SQL unchanged."

        store.draft_sql = result.sql.strip()
        return f"Fixed SQL ({result.changes_summary}):\n{store.draft_sql}"

    return fix_sql


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_sql_tools(
    payload: AgentPayload, llm: Any, retrieval_ctx: RetrievalContext | None = None
) -> tuple[list, SqlGenerationStore]:
    """Build and return Phase 2 SQL Agent tools and a shared ``SqlGenerationStore``.

    Tools close over the store so all state is accumulated automatically as
    the agent calls them — no SQL strings or plans pass between tool calls.

    Args:
        payload: The ``AgentPayload`` from the caller.
        llm: The LLM client used by ``plan_query``, ``generate_sql``, and ``fix_sql``.
        retrieval_ctx: The ``RetrievalContext`` from Phase 1.  When ``None``,
            an empty context is used (safe fallback).

    Returns:
        ``(tools, store)`` — tools list and the ``SqlGenerationStore`` that will
        be populated as the agent runs.
    """
    dialect = get_dialect(payload)
    dialects = [dialect] if dialect else []
    question = payload.get("question") or ""
    ctx: RetrievalContext = retrieval_ctx or {
        "entity_coverage": [],
        "relevant_tables": [],
        "relevant_fks": [],
        "relevant_queries": [],
        "coverage_complete": False,
    }

    store = SqlGenerationStore(question=question, retrieval_ctx=ctx)
    planner_tools = build_planner_tools(store, llm)
    author_tools = build_author_tools(store, llm, dialects)
    return planner_tools + author_tools, store


def build_planner_tools(store: SqlGenerationStore, llm: Any) -> list:
    """Tools owned by the ``query-planner`` subagent."""
    return [_make_plan_query_tool(store, llm)]


def build_author_tools(store: SqlGenerationStore, llm: Any, dialects: list[str] | None) -> list:
    """Tools owned by the ``sql-author`` subagent (generate / validate / fix loop)."""
    return [
        _make_generate_sql_tool(store, llm),
        _make_validate_sql_tool(dialects, store),
        _make_fix_sql_tool(store, llm),
    ]


def build_sql_store(payload: AgentPayload, retrieval_ctx: RetrievalContext | None = None) -> SqlGenerationStore:
    """Build just the per-request store, without instantiating any tools."""
    question = payload.get("question") or ""
    ctx: RetrievalContext = retrieval_ctx or {
        "entity_coverage": [],
        "relevant_tables": [],
        "relevant_fks": [],
        "relevant_queries": [],
        "coverage_complete": False,
    }
    return SqlGenerationStore(question=question, retrieval_ctx=ctx)


# Keep ExecutionStore as a thin alias so any existing imports don't break
ExecutionStore = SqlGenerationStore

__all__ = [
    "build_sql_tools",
    "build_planner_tools",
    "build_author_tools",
    "build_sql_store",
    "SqlGenerationStore",
    "ExecutionStore",
    "_strip_sql_fences",
]
