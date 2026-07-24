# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator that re-ranks retrieved documents using an LLM-based selection agent.

The selection logic lives in the private
:class:`~nemo_retriever._agentic.nemo_agent.SelectionAgent`. This operator adapts the
RRF-stage DataFrame into that library's inputs and applies a three-tier gate per
query: pass through the ReAct agent's ``final_results`` when present, otherwise
run the selection agent, otherwise fall back to the RRF ranking.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from nemo_retriever._agentic.nemo_agent import SelectionAgent, SelectionAgentConfig
from nemo_retriever._agentic.nemo_agent.llm import create_llm, create_llm_config
from nemo_retriever._agentic.nemo_agent.results import AgentRunResult
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator

logger = logging.getLogger(__name__)

_LOG_PREVIEW_CHARS = 300
_LOG_DOC_ID_LIMIT = 20

#: Max LLM steps for the selection sub-agent (mirrors the reference workflow).
_SELECTION_MAX_STEPS = 10


def _preview_text(value: Any, *, limit: int = _LOG_PREVIEW_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


class SelectionAgentOperator(AbstractOperator, CPUOperator):
    """Re-rank retrieved documents using an LLM-based selection agent.

    For each ``query_id`` group produced by :class:`RRFAggregatorOperator`, the
    operator applies a three-tier gate:

    1. **final_results** — if the ReAct agent produced a non-empty final list
       (recovered from ``react_final_rank``), pass those doc ids through.
    2. **selection_agent** — otherwise run
       :class:`~nemo_retriever._agentic.nemo_agent.SelectionAgent` over the RRF-ranked
       candidates (with the RRF scores arming its context-overflow shrink retry).
    3. **rrf** — if selection produced nothing (failure / empty), fall back to
       the top RRF-ranked candidates.

    Input DataFrame schema
    ----------------------
    query_id   : str  — unique query identifier
    query_text : str  — original query text shown to the LLM
    doc_id     : str  — candidate document identifier
    text       : str  — document text content shown to the LLM
    rrf_score        : float, optional — used to order candidates and as the
                       selection shrink-retry priority / RRF fallback
    react_final_rank : int, optional — ReAct final ordering (drives tier 1)
    (any additional columns are ignored)

    Output DataFrame schema
    -----------------------
    query_id      : str  — same ``query_id`` as the input
    doc_id        : str  — selected document ID
    rank          : int  — 1-indexed rank (1 = most relevant)
    message       : str  — always empty (retained for schema compatibility)
    result_source : str  — one of ``{"final_results", "selection_agent", "rrf"}``

    Parameters
    ----------
    llm_model : str
        Model identifier forwarded verbatim to the backend.
    invoke_url : str
        LLM endpoint. Forwarded as the LLM config's ``base_url``.
    top_k : int
        Number of documents to select per query. Defaults to ``10``.
    api_key : str, optional
        Literal API key **or** an ``"os.environ/VAR_NAME"`` reference.
    max_tokens : int, optional
        Per-request completion budget (the LLM config's ``max_completion_tokens``).
    max_steps : int
        Maximum selection-agent LLM steps per query. Defaults to ``10``.
    system_prompt_override : str, optional
        Forwarded as the selection agent's ``system_prompt`` (``None`` selects
        the packaged default selection prompt).
    text_truncation : int
        Maximum characters of each document's text passed to the agent.
        ``0`` disables truncation. Defaults to ``0``.
    parallel_tool_calls : bool, optional
        Forwarded as the LLM config's ``parallel_tool_calls`` (sent to the
        provider only when set).
    base_url : str, optional
        Deprecated alias for ``invoke_url``.
    reasoning_effort : str, optional
        Forwarded as the LLM config's ``reasoning_effort``.
    temperature : float, optional
        Forwarded as the LLM config's ``temperature`` (sent to the provider
        only when set).
    backend : {"litellm", "callable"}
        LLM backend to build. ``"litellm"`` (default) is supported; ``"callable"``
        wraps ``chat_completion_fn`` and is selected automatically for in-process
        runs.
    chat_completion_fn : callable, optional
        OpenAI-compatible completion callable (e.g. the local in-process vLLM
        adapter). When set, forwarded to the ``"callable"`` LLM backend.

    Notes
    -----
    The :class:`~nemo_retriever._agentic.nemo_agent.SelectionAgent` and LLM backend are
    built lazily on first ``process`` call so the operator stays reconstructable
    via ``get_constructor_kwargs``. ``select_sync`` performs one ``asyncio.run``
    per query and must not be called from inside a running event loop.
    """

    _NVIDIA_BUILD_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        *,
        llm_model: str,
        invoke_url: Optional[str] = None,
        top_k: int = 10,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_steps: int = _SELECTION_MAX_STEPS,
        system_prompt_override: Optional[str] = None,
        text_truncation: int = 0,
        parallel_tool_calls: Optional[bool] = None,
        base_url: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        backend: str = "litellm",
        chat_completion_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self._llm_model = llm_model
        self._top_k = top_k
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._max_steps = max_steps
        self._system_prompt_override = system_prompt_override
        self._text_truncation = text_truncation
        self._parallel_tool_calls = parallel_tool_calls
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._backend = backend
        # When set, an OpenAI-compatible completion callable (e.g. the in-process
        # local vLLM adapter). Forwarded to the "callable" LLM backend by _build_llm.
        self._chat_completion_fn = chat_completion_fn

        if invoke_url is not None:
            self._invoke_url = invoke_url
        elif base_url is not None:
            import warnings

            warnings.warn(
                "SelectionAgentOperator: 'base_url' is deprecated, use 'invoke_url' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._invoke_url = base_url.rstrip("/") + "/v1/chat/completions"
        else:
            self._invoke_url = self._NVIDIA_BUILD_ENDPOINT

        # Built lazily on first process(); never part of the picklable ctor state.
        self._sel: Optional[SelectionAgent] = None

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
            drop_params=True,
        )
        kwargs = {"completion_fn": self._chat_completion_fn} if self._chat_completion_fn is not None else {}
        return create_llm(config, **kwargs)

    def _ensure_agent(self) -> SelectionAgent:
        if self._sel is None:
            self._sel = SelectionAgent(
                config=SelectionAgentConfig(
                    target_top_k=int(self._top_k),
                    extended_relevance=True,
                    enable_think=False,
                    end_tool_with_msg=False,
                    shrink_attempts=2,
                    max_steps=int(self._max_steps),
                    on_error="never_raise",
                    system_prompt=self._system_prompt_override,
                ),
                llm=self._build_llm(),
            )
        return self._sel

    # ------------------------------------------------------------------
    # AbstractOperator interface
    # ------------------------------------------------------------------

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        """Validate that *data* is a DataFrame with the required columns."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"SelectionAgentOperator expects a pd.DataFrame, got {type(data).__name__!r}.")
        required = {"query_id", "query_text", "doc_id", "text"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required column(s): {sorted(missing)}. " f"Expected: {sorted(required)}."
            )
        return data.copy()

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Apply the three-tier selection gate for each query group."""
        self._ensure_agent()
        rows: List[Dict[str, Any]] = []

        for query_id, group in data.groupby("query_id", sort=False):
            query_text = str(group["query_text"].iloc[0]) if "query_text" in group.columns else ""
            ordered = group.sort_values("rrf_score", ascending=False) if "rrf_score" in group.columns else group

            # Candidate documents in RRF-descending order, deduplicated (first wins),
            # plus the {doc_id: rrf_score} priority side-table (covers every candidate).
            documents: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for _, row in ordered.iterrows():
                doc_id = str(row["doc_id"])
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                text = str(row["text"])
                if self._text_truncation and int(self._text_truncation) > 0:
                    text = text[: int(self._text_truncation)]
                documents.append({"id": doc_id, "text": text})
            scores: Optional[Dict[str, float]] = None
            if "rrf_score" in ordered.columns:
                scores = {str(row["doc_id"]): float(row["rrf_score"]) for _, row in ordered.iterrows()}

            logger.info(
                "SelectionAgentOperator: query=%s candidates=%d query=%r",
                query_id,
                len(documents),
                _preview_text(query_text),
            )

            # Tier 1: ReAct produced a final list (success or salvage) -> pass through.
            react_final = self._react_final_doc_ids(ordered)
            if react_final:
                doc_ids = list(react_final)
                result_source = "final_results"
            else:
                doc_ids = []
                result_source = ""
                # Tier 2: run the selection agent over the RRF candidates.
                if documents:
                    result = self._run_selection(query_text, documents, scores, str(query_id))
                    if result is not None and result.succeeded and result.final_doc_ids:
                        doc_ids = [str(d) for d in result.final_doc_ids][: int(self._top_k)]
                        result_source = "selection_agent"
                # Tier 3: fall back to the RRF ranking.
                if not doc_ids:
                    doc_ids = ordered["doc_id"].astype(str).drop_duplicates().head(int(self._top_k)).tolist()
                    result_source = "rrf"

            logger.info(
                "SelectionAgentOperator: query=%s result_source=%s selected=%s",
                query_id,
                result_source,
                doc_ids[:_LOG_DOC_ID_LIMIT],
            )
            for rank, doc_id in enumerate(doc_ids, 1):
                rows.append(
                    {
                        "query_id": query_id,
                        "doc_id": str(doc_id),
                        "rank": rank,
                        "message": "",
                        "result_source": result_source,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["query_id", "doc_id", "rank", "message", "result_source"])

        return pd.DataFrame(rows)

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _react_final_doc_ids(self, ordered_group: pd.DataFrame) -> List[str]:
        """Recover the ReAct agent's final ranked doc ids from ``react_final_rank``.

        Returns an empty list when the ReAct agent produced no final results — the
        gate branches on non-emptiness, so the None-vs-empty distinction that the
        old code relied on is irrelevant here.
        """
        if "react_final_rank" not in ordered_group.columns:
            return []
        final_rows = ordered_group[ordered_group["react_final_rank"].notna()]
        if final_rows.empty:
            return []
        final_rows = final_rows.copy()
        final_rows["react_final_rank"] = final_rows["react_final_rank"].astype(int)
        doc_ids: List[str] = []
        for doc_id in final_rows.sort_values("react_final_rank")["doc_id"].astype(str):
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
            if len(doc_ids) >= int(self._top_k):
                break
        return doc_ids

    def _run_selection(
        self,
        query_text: str,
        documents: List[Dict[str, Any]],
        scores: Optional[Dict[str, float]],
        query_id: str,
    ) -> Optional[AgentRunResult]:
        """Run the selection agent, returning None on an unexpected failure."""
        try:
            return self._ensure_agent().select_sync(
                query_text,
                documents,
                scores=scores,
                query_id=query_id,
                raw_log_dir=None,
            )
        except Exception as exc:  # production: fall back to RRF rather than crash
            logger.warning("SelectionAgentOperator: selection failed for query %r: %s", query_id, exc, exc_info=True)
            return None
