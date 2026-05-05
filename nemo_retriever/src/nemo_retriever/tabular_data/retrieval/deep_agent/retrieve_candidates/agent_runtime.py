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
    build_retrieval_tools,
)
from nemo_retriever.tabular_data.retrieval.deep_agent.state import AgentPayload
from nemo_retriever.tabular_data.retrieval.utils import get_llm_client

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent
_AGENTS_MD = str(_BASE_DIR / "AGENTS.md")

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


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


def _build_retrieval_system_prompt() -> str:
    """Read AGENTS.md and return it as the system prompt.

    The deep agent framework requires a non-null system_prompt to engage the
    tool-calling loop.  Without it the LLM generates a single plain-text reply
    and terminates immediately without calling any tools.
    """
    agents_md_path = Path(_AGENTS_MD)
    if agents_md_path.exists():
        try:
            return agents_md_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not read AGENTS.md: %s", exc)
    return (
        "You are the Retrieval Deep Agent (Phase 1 of a 3-phase Text-to-SQL pipeline).\n"
        "You MUST call tools in order: decompose_question → retrieve_for_entity (once per entity) "
        "→ synthesize_expression (for uncovered entities only).\n"
        "You MUST NOT generate SQL queries."
    )


def _create_retrieval_agent(payload: AgentPayload, llm: Any, retriever=None) -> tuple[Any, RetrievalStore]:
    """Instantiate the Phase 1 Retrieval Deep Agent.

    Returns:
        ``(agent, store)`` — the agent ready for ``invoke()`` and the
        ``RetrievalStore`` that will be populated as tools are called.
    """
    tools, store = build_retrieval_tools(payload, llm, retriever=retriever)
    system_prompt = _build_retrieval_system_prompt()

    logger.info(
        "Creating Retrieval Deep Agent | tools=%s | system_prompt_len=%d",
        [getattr(t, "name", "?") for t in tools],
        len(system_prompt),
    )

    agent = create_deep_agent(
        model=llm,
        system_prompt=system_prompt,
        memory=[],
        skills=None,
        tools=tools,
        subagents=[],
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
    """Build the user-turn message for the Retrieval Agent."""
    return (
        f"User question: {question.strip()}\n\n"
        "Follow these steps exactly:\n\n"
        "Step 1 — call decompose_question with the question.\n\n"
        "Step 2 — call retrieve_for_entity ONE TIME FOR EACH entity listed in the decompose_question result.\n"
        "  - Process in priority order (lowest number first).\n"
        "  - You MUST call retrieve_for_entity for EVERY entity. Do NOT stop after the first one.\n"
        "  - The tool manages accumulated state automatically — just pass entity_term and entity_type.\n\n"
        "Step 3 — for any entity returned as NOT COVERED, call synthesize_expression.\n\n"
        "Step 4 — call filter_relevant_tables() once to remove off-topic tables.\n\n"
        "Step 5 — reply with: 'Retrieval complete.'"
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
        retriever: Optional pre-built ``DeepAgentRetriever`` singleton.  When
            provided, the same instance is reused for all LanceDB searches in
            this session — avoids re-initializing the embedder model each call.
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
