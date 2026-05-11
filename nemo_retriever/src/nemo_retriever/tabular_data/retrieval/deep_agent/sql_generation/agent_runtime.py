"""Phase 2 — SQL Deep Agent runtime.

Creates a Deep Agent whose only job is to generate and validate SQL from the
``RetrievalContext`` produced by Phase 1.  The context is injected into the
agent's system prompt so the SQL generation starts with a clean context
window that contains no tool-call history from the retrieval phase.

Usage
-----
::

    from nemo_retriever.tabular_data.retrieval.deep_agent.sql_generation.agent_runtime import (
        create_sql_agent,
        extract_structured_answer,
        format_sql_user_prompt,
    )

    agent, store = create_sql_agent(payload, retrieval_ctx, llm=llm_client)
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    answer = extract_structured_answer(result)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from nemo_retriever.tabular_data.retrieval.deep_agent.context import RetrievalContext
from nemo_retriever.tabular_data.retrieval.deep_agent.state import AgentPayload
from nemo_retriever.tabular_data.retrieval.deep_agent.sql_generation.tools import (
    SqlGenerationStore,
    build_author_tools,
    build_planner_tools,
    build_sql_store,
)
from nemo_retriever.tabular_data.retrieval.utils import get_llm_client

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent

# Required keys in the final JSON answer
_REQUIRED_ANSWER_KEYS = frozenset({"sql_code", "answer", "result"})


def _resolve_skill_dirs(names: list[str]) -> list[str]:
    """Return skill directory paths under ``_BASE_DIR/skills`` that exist on disk."""
    dirs: list[str] = []
    for name in names:
        path = _BASE_DIR / "skills" / name
        if path.is_dir():
            dirs.append(str(path))
        else:
            logger.debug("Skill directory not found, skipping: %s", path)
    return dirs


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return ""


# Cached at module import — these never change at runtime, so re-reading them
# per question wastes I/O and slows agent creation.
_AGENTS_MD_PATH = _BASE_DIR / "AGENTS.md"
_AGENTS_MD_PATH_STR = str(_AGENTS_MD_PATH)
_AGENTS_MD_EXISTS = _AGENTS_MD_PATH.exists()


def _resolve_skill_dir(name: str) -> str | None:
    """Return a single skill dir path or None when missing."""
    path = _BASE_DIR / "skills" / name
    return str(path) if path.is_dir() else None


# Resolve every skill dir we may want exactly once.  Subagents pick the
# subset they need from this map — orchestrator gets nothing (it delegates).
_SKILL_PATHS: dict[str, str] = {
    name: path
    for name, path in (
        (n, _resolve_skill_dir(n)) for n in ("sql-generation", "answer-formatting", "sql-rules", "query-planning")
    )
    if path is not None
}


def _skills_for(*names: str) -> list[str] | None:
    """Pick the resolved skill paths for the requested names (preserves order)."""
    paths = [_SKILL_PATHS[n] for n in names if n in _SKILL_PATHS]
    return paths or None


# Subagent system prompts — read once at import.
_QUERY_PLANNER_PROMPT = _read_text_if_exists(_BASE_DIR / "subagents" / "query_planner.md") or (
    "You are the query-planner subagent. Call plan_query() exactly once and reply 'Plan ready.'."
)
_SQL_AUTHOR_PROMPT = _read_text_if_exists(_BASE_DIR / "subagents" / "sql_author.md") or (
    "You are the sql-author subagent. Loop generate_sql -> validate_sql -> fix_sql up to 4 times "
    "and reply with 'SQL: <validated SQL>'."
)


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


def create_sql_agent(
    payload: AgentPayload,
    retrieval_ctx: RetrievalContext,
    llm: Any | None = None,
) -> tuple[Any, SqlGenerationStore]:
    """Create a Phase 2 SQL Deep Agent.

    The agent is equipped with four tools (``plan_query``, ``generate_sql``,
    ``validate_sql``, ``fix_sql``) and the ``AGENTS.md`` persistent memory.
    The ``RetrievalContext`` from Phase 1 is stored in the ``SqlGenerationStore``
    so tools can access the full schema context without the agent needing to
    pass it as arguments.

    Args:
        payload: The ``AgentPayload`` from the caller.
        retrieval_ctx: The ``RetrievalContext`` produced by Phase 1.
        llm: Optional pre-built LLM client.  When ``None``, ``get_llm_client()`` is
            called automatically.

    Returns:
        Tuple of (agent, store):
        - agent: Deep Agent instance ready for ``agent.invoke()``.
        - store: ``SqlGenerationStore`` populated in-place as the agent runs.
          ``store.sql`` holds the last validated SQL.
    """
    if llm is None:
        llm = get_llm_client()

    store = build_sql_store(payload, retrieval_ctx=retrieval_ctx)
    dialect = payload.get("dialect")
    dialects = [dialect] if dialect else []

    planner_tools = build_planner_tools(store, llm)
    author_tools = build_author_tools(store, llm, dialects)

    subagents = [
        {
            "name": "query-planner",
            "description": (
                "Produces the structured query plan from the RetrievalContext "
                "(entities, relevant tables, FKs, certified snippets). Call once "
                "before SQL generation. No SQL writing."
            ),
            "system_prompt": _QUERY_PLANNER_PROMPT,
            "tools": planner_tools,
            "skills": _skills_for("query-planning", "sql-rules"),
        },
        {
            "name": "sql-author",
            "description": (
                "Generates, validates, and self-corrects the SQL using the plan "
                "in the store. Owns the generate -> validate -> fix loop up to "
                "4 iterations and returns the final validated SQL."
            ),
            "system_prompt": _SQL_AUTHOR_PROMPT,
            "tools": author_tools,
            "skills": _skills_for("sql-generation", "sql-rules", "answer-formatting"),
        },
    ]

    memory = [_AGENTS_MD_PATH_STR] if _AGENTS_MD_EXISTS else []
    system_prompt = _build_system_prompt(payload, retrieval_ctx)

    logger.info(
        "Creating SQL Deep Agent (Phase 2) | subagents=%s | memory=%s",
        [s["name"] for s in subagents],
        memory,
    )

    agent = create_deep_agent(
        model=llm,
        system_prompt=system_prompt,
        memory=memory,
        skills=None,
        tools=[],
        subagents=subagents,
        backend=FilesystemBackend(root_dir=str(_BASE_DIR)),
    )
    return agent, store


def _build_system_prompt(payload: AgentPayload, retrieval_ctx: RetrievalContext) -> str:
    """Build the system prompt for the Phase 2 orchestrator.

    The orchestrator delegates to subagents via the framework's ``task``
    tool — it does not call SQL tools directly.  The full
    ``RetrievalContext`` lives in the shared ``SqlGenerationStore`` and is
    accessed by subagent tools, not the orchestrator.
    """
    now = datetime.now()
    dialect = payload.get("dialect")

    lines: list[str] = [
        f"Today's date: {now.year}-{now.month:02d}-{now.day:02d} " f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}.",
    ]

    if dialect:
        lines.append(f"Allowed SQL dialect: {dialect}.")

    lines.append("")
    lines.append(
        "You are the Phase 2 SQL orchestrator.  You do NOT call SQL tools yourself; "
        "you delegate via the framework's `task` tool to two subagents:\n"
        "  1. `query-planner` — produces the structured plan.\n"
        "  2. `sql-author` — generates, validates, and self-corrects the SQL.\n"
        "Phase 1's RetrievalContext is already in the shared store; the subagents' "
        "tools read it directly, so you do NOT need to repeat schema, entities, or FKs "
        "in the task descriptions."
    )

    entities = retrieval_ctx.get("entity_coverage") or []
    if entities:
        lines.append("")
        lines.append("Entities resolved by Phase 1 (informational only):")
        for ec in entities:
            expr = f"  sql_expression={ec['sql_expression']}" if ec.get("sql_expression") else ""
            lines.append(f"  - {ec['entity']} ({ec['entity_type']}, resolved_as={ec['resolved_as']}){expr}")

    if not retrieval_ctx.get("coverage_complete"):
        lines.append(
            "\nNote: coverage_complete=false — one or more entities were unresolved by Phase 1. "
            "Tell `sql-author` to construct the best SQL possible and note limitations in the final answer."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# User-prompt formatting
# ---------------------------------------------------------------------------


def format_sql_user_prompt(
    question: str,
    dialect: str | None = None,
) -> str:
    """Format the user-turn message sent to the Phase 2 SQL Deep Agent.

    Args:
        question: The current user question.
        dialect: Optional allowed SQL dialect name (informational;
            also injected via system prompt).

    Returns:
        A formatted string ready to pass as the user message content.
    """
    parts: list[str] = []

    if dialect:
        parts.append(f"Allowed SQL dialect: {dialect}.")

    parts.append(f"User question: {question.strip()}")
    parts.append("")
    parts.append(
        "Delegate via the `task` tool — never call SQL tools directly. "
        "The shared store already holds Phase 1's RetrievalContext; the "
        "subagents' tools read it themselves, so a one-line task description "
        "is enough.\n\n"
        'Step 1 — call `task` with `subagent_type="query-planner"` and a '
        "description like: 'Produce the structured plan for the user question "
        "using the RetrievalContext in the store.' The subagent will reply "
        "`Plan ready.`.\n\n"
        'Step 2 — call `task` with `subagent_type="sql-author"` and a '
        "description like: 'Generate, validate, and (if needed) fix the SQL "
        "for the plan now in the store. Return the validated SQL.' The "
        "subagent will reply with `SQL: <validated SQL>`.\n\n"
        "Step 3 — emit your final answer as a single JSON object (nothing "
        "before { or after }):\n"
        '  {"sql_code": "<validated SQL>", "answer": "<1-3 sentence explanation>", '
        '"result": null, "semantic_elements": []}'
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_structured_answer(result: dict) -> dict | None:
    """Scan Deep Agent messages (newest-first) for the structured JSON answer.

    Tries in order:
    1. Full JSON object with all required keys.
    2. Markdown-style answer with a ```sql block.
    3. Plain-prose SQL — a SQL statement starting at a line boundary and
       delimited by blank lines or end-of-text.

    Args:
        result: The dict returned by ``agent.invoke()``.

    Returns:
        ``{"sql_code", "answer", "result"}`` dict or ``None`` if not found.
    """
    messages = result.get("messages") or []
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            obj = _extract_json_answer_object(content)
            if obj is not None:
                return obj
            md = _parse_markdown_answer(content)
            if md is not None:
                return md
            prose = _extract_sql_from_prose(content)
            if prose is not None:
                return prose
        elif isinstance(content, dict):
            if _REQUIRED_ANSWER_KEYS.issubset(content.keys()):
                return content
    return None


def _extract_json_answer_object(content: str) -> dict | None:
    if not content.strip():
        return None
    text = content.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and _REQUIRED_ANSWER_KEYS.issubset(obj.keys()):
            return obj
    except Exception:
        pass
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, dict) and _REQUIRED_ANSWER_KEYS.issubset(obj.keys()):
                    return obj
            except json.JSONDecodeError:
                pass
        i += 1
    return None


def _parse_markdown_answer(text: str) -> dict | None:
    sql_code = answer = None
    result_value: Any = None

    start = text.find("```sql")
    if start != -1:
        start = text.find("\n", start)
        if start != -1:
            end = text.find("```", start)
            if end != -1:
                sql_code = text[start:end].strip()

    answer_marker = "**Answer:**"
    idx = text.find(answer_marker)
    if idx != -1:
        answer = text[idx + len(answer_marker) :].strip()

    result_marker = "**Result:**"
    r_idx = text.find(result_marker)
    if r_idx != -1:
        r_start = r_idx + len(result_marker)
        r_end = text.find("**Answer:**", r_start)
        r_end = r_end if r_end != -1 else len(text)
        section = text[r_start:r_end]
        m = re.search(r"-?\d+(\.\d+)?", section)
        if m:
            try:
                result_value = float(m.group(0))
            except ValueError:
                result_value = m.group(0)
        elif section.strip():
            result_value = section.strip()

    if sql_code and answer:
        return {"sql_code": sql_code, "answer": answer, "result": result_value}
    return None


_SQL_PROSE_RE = re.compile(
    r"(?m)^" r"((?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|EXPLAIN)\b" r".*?)" r"(?=\n[ \t]*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_sql_from_prose(text: str) -> dict | None:
    """Extract SQL embedded in plain prose (no fences, no JSON)."""
    m = _SQL_PROSE_RE.search(text)
    if not m:
        return None
    sql_code = m.group(1).strip()
    if not sql_code:
        return None
    answer = text[: m.start()].strip() + " " + text[m.end() :].strip()
    answer = re.sub(r"\s+", " ", answer).strip()
    logger.debug("_extract_sql_from_prose: extracted SQL from prose (%d chars)", len(sql_code))
    return {"sql_code": sql_code, "answer": answer or text.strip(), "result": None, "semantic_elements": []}


__all__ = [
    "create_sql_agent",
    "extract_structured_answer",
    "format_sql_user_prompt",
]
