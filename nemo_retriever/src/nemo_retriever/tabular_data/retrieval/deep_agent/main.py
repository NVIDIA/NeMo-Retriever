"""3-Phase Text-to-SQL orchestrator.

Phases
------
1. **Retrieval Agent** (``retrieve_candidates.agent_runtime.run_retrieval_agent``)
   Decomposes the question into typed entities, retrieves per-entity
   candidates, and synthesizes SQL expressions for uncovered entities.
   Returns a ``RetrievalContext``.

2. **SQL Agent** (``sql_generation.agent_runtime.create_sql_agent``)
   Receives the ``RetrievalContext`` in its system prompt (no retrieval
   tool-call history) and generates + validates SQL.  Returns a validated
   SQL string via ``ExecutionStore``.

3. **Execute** (``_execute_sql`` plain function)
   Runs the validated SQL against the database connector and returns rows.
"""

import logging
import os

from nemo_retriever.retriever import Retriever
from nemo_retriever.tabular_data.retrieval.deep_agent.context import RetrievalContext
from nemo_retriever.tabular_data.retrieval.deep_agent.sql_generation.agent_runtime import (
    create_sql_agent,
    extract_structured_answer,
    format_sql_user_prompt,
)
from nemo_retriever.tabular_data.retrieval.deep_agent.retrieve_candidates.agent_runtime import (
    run_retrieval_agent,
)
from nemo_retriever.tabular_data.retrieval.deep_agent.state import AgentPayload
from nemo_retriever.tabular_data.retrieval.deep_agent.sql_generation.tools import ExecutionStore
from nemo_retriever.tabular_data.retrieval.utils import (
    _semantic_retriever_init_kwargs,
    get_llm_client,
)

logger = logging.getLogger(__name__)

try:
    llm_client = get_llm_client()
except ValueError as e:
    logger.error("Failed to initialize LLM client: %s", e)
    llm_client = None

try:
    _uri = os.environ.get("OMNI_SEMANTIC_LANCEDB_URI", "lancedb")
    _table = os.environ.get("OMNI_SEMANTIC_LANCEDB_TABLE", "nv-ingest-tabular")
    # top_k=30 covers the highest k used anywhere; per-call limits are applied
    # downstream by _hits_to_semantic_rows so no results are wasted.
    # Translate the legacy ``lancedb_uri``/``lancedb_table`` keys produced by
    # ``_semantic_retriever_init_kwargs`` (kept as-is for the text_to_sql pipeline)
    # into the ``vdb`` / ``vdb_kwargs`` shape that ``Retriever.__init__`` expects.
    _kwargs = _semantic_retriever_init_kwargs(_uri, _table, top_k=30)
    _lancedb_uri = _kwargs.pop("lancedb_uri")
    _lancedb_table = _kwargs.pop("lancedb_table")
    retriever_client = Retriever(
        vdb="lancedb",
        vdb_kwargs={"uri": _lancedb_uri, "table_name": _lancedb_table},
        **_kwargs,
    )
    logger.info("Retriever initialised once at startup (uri=%s table=%s)", _lancedb_uri, _lancedb_table)
except Exception as e:
    logger.error("Failed to initialize Retriever: %s", e)
    retriever_client = None


# ---------------------------------------------------------------------------
# Phase 3 — plain execute function
# ---------------------------------------------------------------------------


def _execute_sql(sql: str, db_connector, store: ExecutionStore) -> list | None:
    """Execute validated SQL and return rows.

    Prefers ``store.sql`` (the SQL captured by ``validate_sql``) over the
    agent-supplied argument, which may be truncated due to context overflow.

    Args:
        sql: SQL string from the Phase 2 agent's final message.
        db_connector: Database connector (e.g. DuckDB) with an ``execute``
            method that returns a DataFrame.
        store: ``ExecutionStore`` from Phase 2 (``store.sql`` holds the last
            validated SQL).

    Returns:
        List of row dicts, or ``None`` on failure / missing connector.
    """
    effective_sql = store.sql or sql
    if not effective_sql:
        logger.warning("_execute_sql: no SQL to execute")
        return None
    if db_connector is None:
        logger.warning("_execute_sql: no db_connector provided — skipping execution")
        return None
    try:
        df = db_connector.execute(effective_sql)
        if df is not None and not df.empty:
            return df.to_dict(orient="records")
        return []
    except Exception as exc:
        logger.error("_execute_sql failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Phase 2 — SQL Agent
# ---------------------------------------------------------------------------


def _run_sql_agent(
    payload: AgentPayload,
    retrieval_ctx: RetrievalContext,
    llm,
    max_retries: int = 3,
) -> tuple[str, ExecutionStore]:
    """Run the Phase 2 SQL Deep Agent and return (sql_string, store).

    Args:
        payload: Original caller payload.
        retrieval_ctx: Output from Phase 1.
        llm: Shared LLM client.
        max_retries: Number of agent invocation attempts.

    Returns:
        ``(sql, store)`` — ``sql`` is the validated SQL string (may be empty
        on failure), ``store`` carries ``store.sql`` for Phase 3.
    """
    agent, store = create_sql_agent(payload, retrieval_ctx, llm=llm)

    prompt = format_sql_user_prompt(
        question=payload["question"],
        history=payload.get("history"),
        dialects=payload.get("dialects"),
    )

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

            # Use the store's validated SQL if available (most reliable source)
            if store.sql:
                logger.info("SQL Agent store SQL (attempt %d): %s …", attempt, store.sql[:80])
                return store.sql, store

            # Fall back to parsing the agent's final message
            parsed = extract_structured_answer(result)
            if parsed and parsed.get("sql_code"):
                sql = parsed["sql_code"].strip()
                logger.info("SQL Agent parsed SQL (attempt %d): %s …", attempt, sql[:80])
                store.sql = sql
                return sql, store

            logger.warning("SQL Agent attempt %d/%d: no SQL found in output", attempt, max_retries)

        except Exception as exc:
            logger.error("SQL Agent failed (attempt %d/%d): %s", attempt, max_retries, exc)
            last_error = exc

    logger.error("SQL Agent gave up after %d attempts. Last error: %s", max_retries, last_error)
    return "", store


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def get_agent_response(payload: AgentPayload) -> dict:
    """Convert a natural-language question to SQL and execute it.

    Orchestrates the 3-phase pipeline:
    1. Retrieval Agent → RetrievalContext
    2. SQL Agent       → validated SQL
    3. Execute         → result rows

    Args:
        payload: Caller-supplied payload.  Required: ``question``.
            Optional: ``history``, ``dialects``, ``db_connector``, ``path_state``.

    Returns:
        dict with keys ``sql_code``, ``answer``, ``result``,
        ``semantic_elements``.
    """
    question = payload["question"]
    db_connector = payload.get("db_connector")

    # ── Phase 1: Retrieval ─────────────────────────────────────────────────
    logger.info("Phase 1 — Retrieval Agent starting for question: %s …", question[:80])
    retrieval_ctx = run_retrieval_agent(payload, llm=llm_client, retriever=retriever_client)
    logger.info(
        "Phase 1 — complete | entities=%d | coverage_complete=%s",
        len(retrieval_ctx.get("entity_coverage", [])),
        retrieval_ctx.get("coverage_complete"),
    )

    # ── Phase 2: SQL Generation + Validation ───────────────────────────────
    logger.info("Phase 2 — SQL Agent starting")
    sql, store = _run_sql_agent(payload, retrieval_ctx, llm=llm_client)
    logger.info("Phase 2 — complete | sql_len=%d", len(sql))

    # ── Phase 3: Execute ───────────────────────────────────────────────────
    logger.info("Phase 3 — executing SQL")
    result = _execute_sql(sql, db_connector, store)
    logger.info(
        "Phase 3 — complete | rows=%s",
        len(result) if isinstance(result, list) else result,
    )

    return {
        "sql_code": sql,
        "answer": "",
        "result": result,
        "semantic_elements": [],
    }


__all__ = ["get_agent_response", "llm_client", "retriever_client"]
