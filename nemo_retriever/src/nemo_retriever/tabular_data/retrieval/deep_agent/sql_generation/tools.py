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
from nemo_retriever.tabular_data.retrieval.deep_agent.state import AgentPayload

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
        entity_lines = []
        for ec in ctx.get("entity_coverage") or []:
            expr = f" → sql_expression: {ec['sql_expression']}" if ec.get("sql_expression") else ""
            entity_lines.append(f"  - {ec['entity']} ({ec['entity_type']}, resolved_as={ec['resolved_as']}){expr}")

        fk_lines = []
        for fk in ctx.get("relevant_fks") or []:
            fk_lines.append(
                f"  - {fk.get('from_table')}.{fk.get('from_column')} → " f"{fk.get('to_table')}.{fk.get('to_column')}"
            )

        snippets = "\n".join(ctx.get("complex_candidates_str") or []) or "  (none)"

        prompt = f"""You are planning a SQL query from a pre-analyzed database context.

User question: "{store.question}"

Entities identified:
{chr(10).join(entity_lines) or "  (none)"}

Available tables (schema.table: [columns]):
{_compact_schema(ctx)}

Foreign-key relationships:
{chr(10).join(fk_lines) or "  (none)"}

Certified SQL snippets / custom analyses (highest-priority reference):
{snippets}

coverage_complete: {ctx.get('coverage_complete', False)}

Rules:
- Use ONLY tables listed above.  Never reference unlisted tables.
- For entities with resolved_as="expression", embed their sql_expression directly.
- If coverage_complete=false, note which entity is unresolved in the notes field.
- Prefer certified SQL snippets as structural references when available.
- JOIN vs WHERE: whenever more than one table is involved, link them with an
  explicit JOIN ... ON ... clause and put the FK predicate in `join_conditions`.
  NEVER place a join predicate (e.g. t1.id = t2.id) in `where_conditions`.
  `where_conditions` is exclusively for row-level filters.
- No correlated subqueries for joinable conditions. If a related table is used
  to compute a COUNT / SUM / EXISTS / comparison against the main table's row,
  add it to `tables_to_use`, declare the FK predicate in `join_conditions`,
  group by the main-table key in `group_by`, and put the comparison in
  `having_conditions` (NOT `where_conditions`). Do not emit
  `(SELECT ... WHERE outer.col = inner.col) <op> ...` patterns.

MANDATORY — Fully-Qualified Identifiers (apply in every expression you write in the plan):
- Every table reference MUST be SCHEMA.TABLE AS alias  (e.g. school_scheduling.Students AS s).
- Every column reference MUST be alias.column           (e.g. s.StudentCity).
- join_conditions must use alias.col = alias.col        (e.g. s.DeptID = d.DeptID).
- select_expressions, where_conditions, group_by, order_by must all use alias.column.
- Never write a bare table name, bare column name, or column without an alias prefix.

Produce a structured query plan."""

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
        ctx = store.retrieval_ctx

        prompt = f"""You are writing a SQL query from a structured plan.

User question: "{store.question}"

Query plan:
{json.dumps(plan, indent=2)}

Full table schema (schema.table: [columns]):
{_compact_schema(ctx)}

Foreign keys:
{json.dumps(ctx.get('relevant_fks') or [], indent=2)}

MANDATORY RULES — every violation causes a validation failure:
1. Every table MUST be written as SCHEMA.TABLE AS alias.
2. Every column MUST be prefixed with its alias: alias.column.
3. Every alias in SELECT/WHERE/GROUP BY/HAVING/ORDER BY MUST be defined in FROM/JOIN.
4. Use ONLY tables from the plan's tables_to_use list.
5. Use ONLY FK conditions listed in the plan for JOIN predicates. Place every
   join predicate in an explicit JOIN ... ON ... clause — NEVER in WHERE.
6. Translate the plan's having_conditions verbatim into a HAVING clause. Do
   NOT rewrite aggregate filters as correlated subqueries in WHERE.
7. For entities with a sql_expression in the plan, embed that expression directly.
8. String literals MUST use single quotes.  Never use double quotes for values.
9. No SQL comments.  No markdown fences.
10. SELECT-only — no INSERT/UPDATE/DELETE/DROP/ALTER/CREATE.

Write the complete SQL query."""

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

        prompt = f"""You are fixing a SQL query that failed validation.

User question: "{store.question}"

Failing SQL:
{store.draft_sql}

Validation error:
{error}{hint}

Available schema (schema.table: [columns]):
{_compact_schema(store.retrieval_ctx)}

Rules:
- Fix ONLY what the error describes.  Do not rewrite unrelated parts.
- Fully-qualified identifiers (SCHEMA.TABLE AS alias, alias.column) are mandatory.
- String literals MUST use single quotes.
- Return a corrected SQL that is different from the failing SQL.
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
    dialect = payload.get("dialect")
    dialects = [dialect] if dialect else []
    question = payload.get("question") or ""
    ctx: RetrievalContext = retrieval_ctx or {
        "entity_coverage": [],
        "relevant_tables": [],
        "relevant_fks": [],
        "complex_candidates_str": [],
        "relevant_queries": [],
        "coverage_complete": False,
    }

    store = SqlGenerationStore(question=question, retrieval_ctx=ctx)

    return [
        _make_plan_query_tool(store, llm),
        _make_generate_sql_tool(store, llm),
        _make_validate_sql_tool(dialects, store),
        _make_fix_sql_tool(store, llm),
    ], store


# Keep ExecutionStore as a thin alias so any existing imports don't break
ExecutionStore = SqlGenerationStore

__all__ = ["build_sql_tools", "SqlGenerationStore", "ExecutionStore", "_strip_sql_fences"]
