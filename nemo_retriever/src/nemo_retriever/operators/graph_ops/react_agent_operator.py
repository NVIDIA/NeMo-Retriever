# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator that runs a ReAct agentic retrieval loop per query.

The agent logic itself lives in the private :mod:`nemo_retriever._agentic.nemo_agent`
library. This operator is a thin adapter: it builds an
:class:`~nemo_retriever._agentic.nemo_agent.Agent` (select mode) from a subset of the
library's configuration, runs it once per query, and flattens the resulting
retrieval log and final doc-id list into the exploded DataFrame that
:class:`RRFAggregatorOperator` and :class:`SelectionAgentOperator` consume.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from nemo_retriever._agentic.nemo_agent import (
    ERROR_LLM_CALL_FAILED,
    ERROR_TOOL_FAILED,
    ERROR_UNEXPECTED,
    Agent,
    AgentConfig,
    create_retrieve_tool,
)
from nemo_retriever._agentic.nemo_agent.atif import persist_atif_trajectory
from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator

logger = logging.getLogger(__name__)

_LOG_PREVIEW_CHARS = 300
_LOG_DOC_ID_LIMIT = 20
_FATAL_AGENT_ERROR_CATEGORIES = frozenset({ERROR_LLM_CALL_FAILED, ERROR_TOOL_FAILED, ERROR_UNEXPECTED})
_ACTIVE_QUERY_ID: ContextVar[Optional[str]] = ContextVar("react_agent_query_id", default=None)


class _FatalAgentError(RuntimeError):
    """Fatal recorded agent error that must abort single-query and batch execution."""


#: Output DataFrame columns emitted by :func:`_build_output_rows` / this operator.
_OUTPUT_COLUMNS = [
    "query_id",
    "query_text",
    "step_idx",
    "doc_id",
    "text",
    "rank",
    "has_valid_final_results",
    "is_final_result",
]


def _preview_text(value: Any, *, limit: int = _LOG_PREVIEW_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


class ReActAgentOperator(AbstractOperator, CPUOperator):
    """Run an iterative ReAct retrieval loop per query and emit the full retrieval log.

    Each query row is processed independently by an
    :class:`~nemo_retriever._agentic.nemo_agent.Agent` running in ``select`` mode. The
    agent owns the ReAct loop, prompt rendering, tool schemas, and the
    over-fetch/dedup retrieval bookkeeping; this operator only supplies the
    retrieval callback and translates results into the library's exploded
    DataFrame convention. The operator emits one output row per retrieved
    document per retrieval step (plus a synthetic final step for the agent's
    ``final_results`` selection), enabling downstream
    :class:`RRFAggregatorOperator` to fuse the ranked lists with Reciprocal
    Rank Fusion.

    Input DataFrame schema
    ----------------------
    query_id   : str  — unique query identifier
    query_text : str  — the search query text
    (additional columns are ignored)

    Output DataFrame schema
    -----------------------
    query_id                : str  — same ``query_id`` as the input
    query_text              : str  — same ``query_text`` (passed through)
    step_idx                : int  — 0 = seed retrieval; 1 … N = per-loop retrieve
                              calls; ``len(retrieval_log)`` = synthetic final step
    doc_id                  : str  — retrieved document identifier
    text                    : str  — document text
    rank                    : int  — 1-indexed rank within this step
    has_valid_final_results : bool — True iff the agent returned a non-empty final list
    is_final_result         : bool — True only on the synthetic final-step rows

    Parameters
    ----------
    invoke_url : str
        LLM endpoint. Forwarded as the LLM config's ``base_url``.
    llm_model : str
        Model identifier forwarded verbatim to the backend (litellm
        provider-prefix transform is deferred).
    retriever_fn : callable
        By default, ``(query_text, top_k) → [{doc_id, text, score}]``. When
        ``retriever_fn_accepts_query_id`` is true, the operator also passes the
        current input ``query_id`` as a keyword argument.
        Wrapped by ``create_retrieve_tool`` after renaming ``doc_id`` → ``id``.
    retriever_fn_accepts_query_id : bool
        Pass ``query_id=...`` to ``retriever_fn``. Defaults to false for backward
        compatibility with two-argument callbacks.
    retriever_top_k : int
        Default number of documents requested per retrieve call (the tool's
        ``default_top_k``). Defaults to ``500``.
    target_top_k : int
        Number of final documents the agent targets. Defaults to ``10``.
    max_steps : int
        Maximum agent LLM steps per query. Defaults to ``200``.
    num_concurrent : int
        Number of queries processed concurrently via ``ThreadPoolExecutor``.
    api_key : str, optional
        Literal API key **or** an ``"os.environ/VAR_NAME"`` reference (resolved
        by the LLM backend).
    max_tokens : int, optional
        Per-request completion budget (the LLM config's ``max_completion_tokens``).
    parallel_tool_calls : bool, optional
        Forwarded as the LLM config's ``parallel_tool_calls`` (sent to the
        provider only when set).
    reasoning_effort : str, optional
        Forwarded as the LLM config's ``reasoning_effort``.
    temperature : float, optional
        Forwarded as the LLM config's ``temperature`` (sent to the provider
        only when set).
    backend : {"callable", "litellm"}
        LLM backend to build. ``"callable"`` (default) drives an OpenAI-compatible
        completion callable: ``chat_completion_fn`` when supplied (the in-process
        vLLM adapter), otherwise the shared ``invoke_chat_completion_step`` HTTP
        client against ``invoke_url``.
    chat_completion_fn : callable, optional
        OpenAI-compatible completion callable (e.g. the local in-process vLLM
        adapter). When set, forwarded to the ``"callable"`` LLM backend.

    Notes
    -----
    ``retriever_fn`` must be picklable when used with ``RayDataExecutor``. The
    :class:`~nemo_retriever._agentic.nemo_agent.Agent` and LLM backend are built lazily on
    first ``process`` call (not stored as constructor kwargs) so the operator
    stays reconstructable via ``get_constructor_kwargs``.

    ``run_sync`` performs one ``asyncio.run`` per query; it must not be called
    from inside a running event loop. This is safe because the graph is executed
    synchronously (single query in the calling thread, batches in worker threads).
    """

    _NVIDIA_BUILD_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        *,
        invoke_url: Optional[str] = None,
        llm_model: str,
        retriever_fn: Callable[..., List[Dict[str, Any]]],
        retriever_fn_accepts_query_id: bool = False,
        retriever_top_k: int = 500,
        target_top_k: int = 10,
        max_steps: int = 200,
        num_concurrent: int = 8,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        backend: str = "callable",
        chat_completion_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self._invoke_url = invoke_url or self._NVIDIA_BUILD_ENDPOINT
        self._llm_model = llm_model
        self._retriever_fn = retriever_fn
        self._retriever_fn_accepts_query_id = retriever_fn_accepts_query_id
        self._retriever_top_k = retriever_top_k
        self._target_top_k = target_top_k
        self._max_steps = max_steps
        self._num_concurrent = num_concurrent
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._parallel_tool_calls = parallel_tool_calls
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._backend = backend
        # When set, an OpenAI-compatible completion callable (e.g. the in-process
        # local vLLM adapter). Forwarded to the "callable" LLM backend by _build_llm.
        self._chat_completion_fn = chat_completion_fn
        # Built lazily on first process() so the live Agent/LLM (which hold a
        # litellm client + lock) are never part of the picklable ctor state.
        self._agent: Optional[Agent] = None

    # ------------------------------------------------------------------
    # private agent construction (lazy, memoized)
    # ------------------------------------------------------------------

    def _build_llm(self) -> Any:
        config = create_llm_config(
            self._backend,
            model=str(self._llm_model),
            base_url=self._invoke_url,
            api_key=self._api_key,
            reasoning_effort=self._reasoning_effort or None,
            temperature=self._temperature,
            parallel_tool_calls=self._parallel_tool_calls,
            max_completion_tokens=self._max_tokens,
        )
        completion_fn = self._chat_completion_fn
        if self._backend == "callable" and completion_fn is None:
            # Remote run on the default backend: supply the shared chat-completions
            # client. Imported HERE and injected, never imported by the agent
            # library, which must not depend on the rest of nemo_retriever.
            from nemo_retriever.models.nim.chat_completions import invoke_chat_completion_step

            completion_fn = invoke_chat_completion_step
        kwargs = {"completion_fn": completion_fn} if completion_fn is not None else {}
        return create_llm(config, **kwargs)

    def _retrieve_adapter(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Adapt ``retriever_fn`` output to the private agent's ``id``/``score``/``text`` contract."""
        top_k = min(top_k, 1_000)
        out: List[Dict[str, Any]] = []
        if self._retriever_fn_accepts_query_id:
            query_id = _ACTIVE_QUERY_ID.get()
            if query_id is None:
                raise RuntimeError("ReAct retrieval callback ran outside an active query context.")
            docs = self._retriever_fn(query, top_k, query_id=query_id)
        else:
            docs = self._retriever_fn(query, top_k)
        for doc in docs:
            doc_id = str(doc.get("doc_id", doc.get("id", "")))
            if not doc_id:
                continue
            out.append(
                {
                    "id": doc_id,
                    "score": float(doc.get("score", 0.0)),
                    "text": str(doc.get("text", "")),
                }
            )
        return out

    def _ensure_agent(self) -> Agent:
        if self._agent is None:
            retrieve_tool = create_retrieve_tool(
                "default",
                self._retrieve_adapter,
                name="retrieve",
                default_top_k=int(self._retriever_top_k),
            )
            self._agent = Agent(
                config=AgentConfig(
                    mode="select",
                    target_top_k=int(self._target_top_k),
                    enforce_top_k=True,
                    user_msg_type="with_results",
                    extended_relevance=True,
                    enable_think=False,
                    ensure_new_docs=True,
                    end_tool_with_msg=False,
                    max_steps=int(self._max_steps),
                    on_error="never_raise",
                ),
                llm=self._build_llm(),
                retrieve_tool=retrieve_tool,
            )
        return self._agent

    def pop_query_usage(self, query_id: str) -> Dict[str, Any]:
        """Remove and return provider-reported LLM usage for one query."""
        if self._agent is None:
            return {}
        return self._agent.llm.pop_query_usage(str(query_id))

    # ------------------------------------------------------------------
    # AbstractOperator interface
    # ------------------------------------------------------------------

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"ReActAgentOperator expects a pd.DataFrame, got {type(data).__name__!r}.")
        required = {"query_id", "query_text"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required column(s): {sorted(missing)}. " f"Expected: {sorted(required)}."
            )
        return data[["query_id", "query_text"]].copy()

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Run the agent for each query, concurrently up to num_concurrent."""
        self._ensure_agent()
        rows: List[Dict[str, Any]] = []

        query_rows = [(str(r["query_id"]), str(r["query_text"])) for _, r in data.iterrows()]

        if not query_rows:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)
        if len(query_rows) == 1:
            # Fast path: single query, no threading overhead.
            rows.extend(self._run_single_query(*query_rows[0]))
        else:
            # Collect per-query results keyed by query_id, then re-emit in the
            # ORIGINAL input order so downstream groupby(sort=False) output is
            # deterministic regardless of thread completion order.
            results_by_qid: Dict[str, List[Dict[str, Any]]] = {}
            with ThreadPoolExecutor(max_workers=min(self._num_concurrent, len(query_rows))) as executor:
                futures = {executor.submit(self._run_single_query, qid, qtxt): qid for qid, qtxt in query_rows}
                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        results_by_qid[qid] = future.result()
                    except _FatalAgentError:
                        raise
                    except Exception as exc:  # production: one bad query must not kill the batch
                        logger.warning("ReActAgentOperator: query %r failed: %s", qid, exc, exc_info=True)
            for qid, _qtxt in query_rows:
                rows.extend(results_by_qid.get(qid, []))

        if not rows:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)

        return pd.DataFrame(rows)

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data

    # ------------------------------------------------------------------
    # Internal: single query
    # ------------------------------------------------------------------

    def _run_single_query(self, query_id: str, query_text: str) -> List[Dict[str, Any]]:
        """Run one query, raising fatal agent errors while preserving recoverable fallback rows."""
        agent = self._ensure_agent()
        logger.info(
            "ReActAgentOperator: query=%s start max_steps=%d target_top_k=%d query=%r",
            query_id,
            int(self._max_steps),
            int(self._target_top_k),
            _preview_text(query_text),
        )
        query_id_token = _ACTIVE_QUERY_ID.set(str(query_id))
        try:
            result = agent.run_sync(str(query_text), query_id=str(query_id), raw_log_dir=None)
        finally:
            _ACTIVE_QUERY_ID.reset(query_id_token)
        persist_atif_trajectory(result.atif_trace)

        if result.error is not None and result.error.category in _FATAL_AGENT_ERROR_CATEGORIES:
            raise _FatalAgentError(
                f"Agentic retrieval failed ({result.error.category}): {result.error.message} "
                "Check the configured agent LLM, embedding, vector database, and reranker settings and connectivity."
            )

        # Private agent retrieval_log entries are {"input", "tool_name",
        # "query_type", "output": [ {id, score, text|note, ...} ]}. The exploded
        # schema needs one score-desc ranked list per step (the private agent already
        # sorts each output by score descending).
        step_lists = [entry.get("output", []) or [] for entry in result.retrieval_log]
        # Empty final_doc_ids (failure / no valid final_results) maps to None so
        # has_valid_final_results and the synthetic final step stay consistent
        # (both driven by the same truthiness). Never index end_payload here.
        final_doc_ids = result.final_doc_ids or None

        logger.info(
            "ReActAgentOperator: query=%s done retrieval_steps=%d succeeded=%s final_doc_ids=%s",
            query_id,
            len(step_lists),
            result.succeeded,
            (final_doc_ids or [])[:_LOG_DOC_ID_LIMIT],
        )
        return _build_output_rows(str(query_id), str(query_text), step_lists, final_doc_ids)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_output_rows(
    query_id: str,
    query_text: str,
    retrieval_log: List[List[Dict[str, Any]]],
    final_doc_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Convert the retrieval log to one row per (step_idx, rank, doc_id).

    ``retrieval_log`` is a list of per-step document lists (the private agent's
    ``retrieval_log[*]["output"]``); each document carries an ``id`` (private-agent
    key) — ``doc_id`` is also accepted for robustness.
    """
    rows: List[Dict[str, Any]] = []
    for step_idx, step_docs in enumerate(retrieval_log):
        for rank, doc in enumerate(step_docs, 1):
            rows.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "step_idx": step_idx,
                    "doc_id": str(doc.get("id", doc.get("doc_id", ""))),
                    "text": str(doc.get("text", "")),
                    "has_valid_final_results": final_doc_ids is not None,
                    "is_final_result": False,
                    "rank": rank,
                }
            )

    # If final_results was returned, also emit those as a synthetic final step
    # (step_idx = len(retrieval_log)) so RRF/selection can recover the agent's
    # final ranking. Gated on final_doc_ids truthiness — the same predicate that
    # drives has_valid_final_results above.
    if final_doc_ids:
        first_doc_by_id: Dict[str, Dict[str, Any]] = {}
        for step_docs in retrieval_log:
            for doc in step_docs:
                doc_id = str(doc.get("id", doc.get("doc_id", "")))
                if doc_id and doc_id not in first_doc_by_id:
                    first_doc_by_id[doc_id] = doc

        emitted: set[str] = set()
        final_step_idx = len(retrieval_log)
        for rank, doc_id in enumerate(final_doc_ids, 1):
            doc_id = str(doc_id)
            if not doc_id or doc_id in emitted:
                continue
            emitted.add(doc_id)
            doc = first_doc_by_id.get(doc_id, {})
            rows.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "step_idx": final_step_idx,
                    "doc_id": doc_id,
                    "text": str(doc.get("text", "")),
                    "has_valid_final_results": True,
                    "is_final_result": True,
                    "rank": rank,
                }
            )
    return rows
