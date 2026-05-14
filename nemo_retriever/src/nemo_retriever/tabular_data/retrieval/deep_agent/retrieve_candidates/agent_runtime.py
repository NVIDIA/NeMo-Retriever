"""Phase 1 — Retrieval Deep Agent runtime.

Creates a Deep Agent whose only job is semantic grounding: decompose the user
question into typed entities, retrieve per-entity candidates, and synthesize
SQL expressions for entities that have no direct match.

State is accumulated in a ``RetrievalStore`` as the agent calls tools — the
same pattern as ``ExecutionStore`` in Phase 2.  The runtime reads
``store.as_context()`` directly after the agent finishes, which is more
reliable than parsing the agent's final JSON message.

Usage
-----
::

    from nemo_retriever.tabular_data.retrieval.deep_agent.retrieve_candidates.agent_runtime import (
        run_retrieval_agent,
    )

    retrieval_ctx = run_retrieval_agent(payload, llm=llm_client)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from nemo_retriever.tabular_data.retrieval.deep_agent.context import RetrievalContext
from nemo_retriever.tabular_data.retrieval.deep_agent.retrieve_candidates.tools import (
    RetrievalStore,
    build_decomposer_tools,
    build_grounder_tools,
    build_relevance_filter_tools,
    build_retrieval_store,
)
from nemo_retriever.tabular_data.retrieval.deep_agent.state import AgentPayload
from nemo_retriever.tabular_data.retrieval.llm_invoke import get_llm_client

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent

# Keys required in the agent's final JSON message (fallback path only)
_REQUIRED_KEYS = frozenset({"entity_coverage", "relevant_tables", "relevant_fks", "coverage_complete"})

# Empty RetrievalContext returned on total failure
_EMPTY_CONTEXT: RetrievalContext = {
    "entity_coverage": [],
    "relevant_tables": [],
    "relevant_fks": [],
    "complex_candidates_str": [],
    "relevant_queries": [],
    "coverage_complete": False,
}


def _resolve_skill_dir(name: str) -> str | None:
    """Return a single skill dir path or None when missing."""
    path = _BASE_DIR / "skills" / name
    return str(path) if path.is_dir() else None


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return ""


def _read_agents_md() -> str:
    """Read AGENTS.md once at module import; fall back to a tiny built-in prompt."""
    text = _read_text_if_exists(_BASE_DIR / "AGENTS.md")
    if text:
        return text
    return (
        "You are the Retrieval Deep Agent (Phase 1 of a 3-phase Text-to-SQL pipeline).\n"
        "Delegate decomposition, grounding, and filtering to subagents.\n"
        "You MUST NOT generate SQL queries."
    )


# Cached at module import — AGENTS.md and the skill paths never change at
# runtime, so reading them per question wastes I/O.
_ORCH_SYSTEM_PROMPT: str = _read_agents_md()

_SKILL_PATHS: dict[str, str] = {
    name: path
    for name, path in (
        (n, _resolve_skill_dir(n)) for n in ("entity-decomposition", "entity-grounding", "table-relevance")
    )
    if path is not None
}


def _skills_for(*names: str) -> list[str] | None:
    paths = [_SKILL_PATHS[n] for n in names if n in _SKILL_PATHS]
    return paths or None


# Subagent system prompts — read once at import.
_DECOMPOSER_PROMPT = _read_text_if_exists(_BASE_DIR / "subagents" / "decomposer.md") or (
    "You are the decomposer subagent. Call decompose_question(question) once and reply 'Entities decomposed.'."
)
_GROUNDER_PROMPT = _read_text_if_exists(_BASE_DIR / "subagents" / "entity_grounder.md") or (
    "You are the entity-grounder subagent. For each entity in the store call retrieve_for_entity, "
    "and call synthesize_expression for entities that are NOT COVERED."
)
_RELEVANCE_PROMPT = _read_text_if_exists(_BASE_DIR / "subagents" / "relevance_filter.md") or (
    "You are the relevance-filter subagent. Call filter_relevant_tables() once."
)


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


def _create_retrieval_agent(payload: AgentPayload, llm: Any, retriever=None) -> tuple[Any, RetrievalStore]:
    """Instantiate the Phase 1 Retrieval Deep Agent.

    Returns:
        ``(agent, store)`` — the agent ready for ``invoke()`` and the
        ``RetrievalStore`` that will be populated as the subagents run.
    """
    store = build_retrieval_store(retriever=retriever)

    decomposer_tools = build_decomposer_tools(store, llm)
    grounder_tools = build_grounder_tools(store, llm)
    relevance_tools = build_relevance_filter_tools(store, llm)

    subagents = [
        {
            "name": "decomposer",
            "description": (
                "Splits the user question into typed entities. "
                "Call once at the very start. Writes entities to the shared store; "
                "does not call retrieval / synthesis / filtering tools."
            ),
            "system_prompt": _DECOMPOSER_PROMPT,
            "tools": decomposer_tools,
            "skills": _skills_for("entity-decomposition"),
        },
        {
            "name": "entity-grounder",
            "description": (
                "Grounds every entity in the store: calls retrieve_for_entity once "
                "per entity, and synthesize_expression for entities reported NOT "
                "COVERED. Writes accumulated tables / FKs / sql_expressions back "
                "into the store."
            ),
            "system_prompt": _GROUNDER_PROMPT,
            "tools": grounder_tools,
            "skills": _skills_for("entity-grounding"),
        },
        {
            "name": "relevance-filter",
            "description": (
                "Prunes accumulated tables down to those genuinely needed to "
                "answer the question. Call once after grounding is complete."
            ),
            "system_prompt": _RELEVANCE_PROMPT,
            "tools": relevance_tools,
            "skills": _skills_for("table-relevance"),
        },
    ]

    logger.info(
        "Creating Retrieval Deep Agent | subagents=%s | system_prompt_len=%d",
        [s["name"] for s in subagents],
        len(_ORCH_SYSTEM_PROMPT),
    )

    agent = create_deep_agent(
        model=llm,
        system_prompt=_ORCH_SYSTEM_PROMPT,
        memory=[],
        skills=None,
        tools=[],
        subagents=subagents,
        backend=FilesystemBackend(root_dir=str(_BASE_DIR)),
    )
    return agent, store


# ---------------------------------------------------------------------------
# Fallback: extract RetrievalContext from agent messages
# ---------------------------------------------------------------------------


def _extract_retrieval_context_from_messages(result: dict) -> RetrievalContext | None:
    """Scan the agent's message list (newest-first) for a RetrievalContext JSON.

    Used only when the store is empty (e.g. all tool calls failed).
    """
    messages = result.get("messages") or []
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if not isinstance(content, str):
            if isinstance(content, dict) and _REQUIRED_KEYS.issubset(content.keys()):
                return content  # type: ignore[return-value]
            continue

        text = content.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and _REQUIRED_KEYS.issubset(obj.keys()):
                return obj  # type: ignore[return-value]
        except Exception:
            pass

        decoder = json.JSONDecoder()
        i = 0
        while i < len(text):
            if text[i] == "{":
                try:
                    obj, _ = decoder.raw_decode(text, i)
                    if isinstance(obj, dict) and _REQUIRED_KEYS.issubset(obj.keys()):
                        return obj  # type: ignore[return-value]
                except json.JSONDecodeError:
                    pass
            i += 1

    return None


# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------


def _format_user_prompt(question: str) -> str:
    """Build the user-turn message for the Retrieval Agent orchestrator."""
    return (
        f"User question: {question.strip()}\n\n"
        "Delegate to the subagents in order. Never call tools yourself — "
        "only the framework's `task` tool:\n\n"
        "Step 1 — delegate to `decomposer` with the user question. It "
        "returns a bullet list of typed entities of the form:\n"
        "    Extracted <N> entities (call retrieve_for_entity for each):\n"
        "      1. [<entity_type>] <term>\n"
        "      ...\n\n"
        "Step 2 — delegate to `entity-grounder`. The task description "
        "you give it MUST include the exact bullet list returned by "
        "`decomposer`, verbatim, including the header and every numbered "
        "bullet. The grounder iterates over that list to call "
        "retrieve_for_entity / synthesize_expression — without the list "
        "it has nothing to do.\n\n"
        "Step 3 — delegate to `relevance-filter`. It runs "
        "filter_relevant_tables() once.\n\n"
        "Step 4 — reply with: 'Retrieval complete.'"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_retrieval_agent(
    payload: AgentPayload,
    llm: Any | None = None,
    retriever: Any | None = None,
    max_retries: int = 2,
) -> RetrievalContext:
    """Run the Phase 1 Retrieval Deep Agent and return a ``RetrievalContext``.

    Reads ``store.as_context()`` first (primary path — built from tool call
    results written directly into the store).  Falls back to parsing the agent's
    final JSON message if the store is empty.

    Args:
        payload: Caller-supplied payload.  Required: ``question``.
        llm: Optional pre-built LLM client.
        retriever: Optional pre-built :class:`~nemo_retriever.retriever.Retriever`
            singleton.  When provided, the same instance is reused for all
            LanceDB searches in this session — avoids re-initializing the
            embedder model each call.
        max_retries: Number of agent invocation attempts before giving up.

    Returns:
        ``RetrievalContext`` dict.  Falls back to ``_EMPTY_CONTEXT`` on total failure.
    """
    if llm is None:
        llm = get_llm_client()

    question = payload["question"]
    agent, store = _create_retrieval_agent(payload, llm, retriever=retriever)
    prompt = _format_user_prompt(question)

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

            # Primary: build context directly from store state (most reliable)
            ctx = store.as_context()
            if ctx is not None:
                logger.info(
                    "Retrieval Agent store context (attempt %d) | entities=%d | coverage_complete=%s",
                    attempt,
                    len(ctx.get("entity_coverage", [])),
                    ctx.get("coverage_complete"),
                )
                return ctx

            # Fallback: parse the agent's final JSON message
            ctx = _extract_retrieval_context_from_messages(result)
            if ctx is not None:
                logger.info(
                    "Retrieval Agent message fallback (attempt %d) | entities=%d",
                    attempt,
                    len(ctx.get("entity_coverage", [])),
                )
                return ctx

            logger.warning(
                "Retrieval Agent attempt %d/%d: store empty and no RetrievalContext in messages",
                attempt,
                max_retries,
            )
        except Exception as exc:
            logger.error(
                "Retrieval Agent failed (attempt %d/%d): %s",
                attempt,
                max_retries,
                exc,
            )
            last_error = exc

    logger.error(
        "Retrieval Agent gave up after %d attempts. Last error: %s.",
        max_retries,
        last_error,
    )
    return dict(_EMPTY_CONTEXT)  # type: ignore[return-value]


__all__ = ["run_retrieval_agent"]
