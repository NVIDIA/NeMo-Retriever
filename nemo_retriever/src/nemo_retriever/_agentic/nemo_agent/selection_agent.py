# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone selection agent: pick the top-k most relevant candidate documents.

The agent takes a query plus a list of candidate documents and runs an LLM
loop (on the shared engine in ``loop.py``) that ends with a
``log_selected_documents`` call naming the ``target_top_k`` best candidates.
It knows nothing about retrieval, RRF, or the main agent.

Two modes, selected explicitly by the ``scores`` argument of :meth:`SelectionAgent.select`:

- ``scores=None`` — single attempt; a context-window overflow surfaces per
  ``config.on_error`` like any other known LLM error.
- ``scores={doc_id: priority}`` — passing a priority ranking is the permission
  to shrink: on a context-window overflow the agent drops the lowest-priority
  quarter of the candidates and retries (up to ``config.shrink_attempts``
  total attempts).

The mode is never inferred from the documents themselves: candidate docs that
came out of retrieval carry per-subquery ``score`` keys which are NOT
comparable across subqueries — truncating by them would be wrong, so they are
ignored (and never shown to the LLM). The ``scores`` side table is a globally
comparable ranking the caller owns (e.g. RRF fusion scores).
"""

from __future__ import annotations

import asyncio
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import Field

from .cache_propagation import PropagationPacer
from .llm import BaseLLMBackend, bind_query_id
from .loop import (
    BaseAgentLoopConfig,
    _awrite_json,
    _BaseAgentLoop,
    _RunState,
    build_auto_continue_msg,
)
from .prompts import render_system_prompt
from .results import ERROR_CONTEXT_LIMIT, AgentRunResult
from .tools import BaseTool, LogSelectedDocs, SelectionThinkTool

# Fraction of candidates dropped (lowest priority first) before a retry after
# a context-window overflow. Deliberately a constant, not a config knob:
# 0.25 drops a quarter of the candidates (``floor(n * 0.25)``) per shrink step.
_SHRINK_DROP_FRACTION = 0.25

_DEFAULT_SELECTION_PROMPT = "selection/01_v0.j2"
_END_TOOL_NAME = "log_selected_documents"
_END_PAYLOAD_PHRASE = "with your selected doc_ids"


class SelectionAgentConfig(BaseAgentLoopConfig):
    """Configuration for the selection agent.

    Pure data, like ``AgentConfig``; the LLM instance is a constructor
    argument. The loop-policy fields (``max_steps``, ``on_error``, logging and
    pacing knobs) are inherited from
    :class:`~nemo_agent.loop.BaseAgentLoopConfig`.

    Attributes
    ----------
    system_prompt:
        Packaged prompt name or filesystem path; ``None`` selects the default
        selection prompt. Render variables: ``extended_relevance`` and the
        per-run ``top_k``.
    target_top_k:
        How many documents a run selects — one k per agent instance. Runs with
        fewer candidates than ``target_top_k`` self-adjust to the candidate
        count.
    extended_relevance:
        Render the extended relevance-definition guidance into the system
        prompt (and the think tool's description).
    enable_think:
        Register the selection think tool. Off by default, consistent with
        ``AgentConfig``.
    end_tool_with_msg:
        Require (and pass through) the end tool's diagnostic ``message``
        argument.
    shrink_attempts:
        Total attempts (including the first) when ``scores`` are provided and
        a context-window overflow occurs. Defaults to 2: one shrink retry
        after the initial attempt.
    """

    system_prompt: Optional[str] = None
    target_top_k: int = Field(default=10, ge=1)
    extended_relevance: bool = True
    enable_think: bool = False
    end_tool_with_msg: bool = True
    shrink_attempts: int = Field(default=2, ge=1)


class SelectionAgent(_BaseAgentLoop):
    """LLM agent that selects the top-k candidates for a query from a given list.

    Construction wires the static pieces (stage label, auto-continue message,
    optional think tool); everything that depends on a run's candidate set —
    the system prompt (``top_k`` baked in), the
    :class:`~nemo_agent.tools.LogSelectedDocs` end tool (count + allowed ids),
    and the documents user message — is assembled per attempt inside
    :meth:`select`. The instance is immutable after construction and safely
    serves many (including concurrent) runs.

    Results reuse :class:`~nemo_agent.results.AgentRunResult`: the selected
    ids land in ``final_doc_ids`` (with the end tool's ``message`` in
    ``end_payload``), ``retrieval_log`` stays empty, and when the shrink retry
    fired, ``extra_data["context_shrink"]`` records what was dropped.

    Token usage is attributed to the stage ``f"top{target_top_k}_agent"`` —
    the *configured* k, even when a run's feasible k is smaller.
    """

    def __init__(self, config: SelectionAgentConfig, llm: BaseLLMBackend) -> None:
        if not isinstance(config, SelectionAgentConfig):
            raise TypeError(f"config must be a SelectionAgentConfig, got {type(config).__name__}.")
        super().__init__(config=config, llm=llm)
        self._stage = f"top{config.target_top_k}_agent"
        self._auto_user_msg = build_auto_continue_msg(_END_TOOL_NAME, _END_PAYLOAD_PHRASE)
        self._prompt_name = config.system_prompt or _DEFAULT_SELECTION_PROMPT
        self._think_tool: Optional[SelectionThinkTool] = (
            SelectionThinkTool(extended_relevance=config.extended_relevance) if config.enable_think else None
        )

    # ------------------------------------------------------------------
    # Entry points.
    # ------------------------------------------------------------------

    async def select(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        *,
        scores: Optional[Dict[str, float]] = None,
        query_id: Optional[str] = None,
        task_info: Optional[Any] = None,
        raw_log_dir: Optional[Union[str, Path]] = None,
    ) -> AgentRunResult:
        """Select the ``config.target_top_k`` most relevant documents for ``query``.

        Parameters
        ----------
        query:
            The user's question.
        documents:
            Candidate documents: dicts with a required ``id`` (coerced to
            ``str``) and optional ``text`` / ``image`` strings. Duplicate ids
            are deduplicated (first occurrence wins); any other keys —
            including retrieval ``score`` keys — are ignored and never shown
            to the LLM.
        scores:
            Optional ``{doc_id: priority}`` ranking covering every candidate
            id (a superset is fine). Passing it arms the context-overflow
            shrink retry: intermediate context-limit failures are consumed by
            the retry loop regardless of ``config.on_error`` — dropping the
            lowest-priority quarter each time — and only the final attempt's
            outcome is subject to the configured policy. Every failure that is
            not a context-window overflow follows the policy immediately, on
            any attempt. Without ``scores`` there is exactly one attempt.
        query_id:
            Optional id bound for the run, exactly like ``Agent.run``: token
            usage lands under ``llm.get_usage(query_id)`` and logs are
            labeled. An ambient caller-side ``bind_query_id`` also applies.
        task_info:
            Arbitrary JSON-serializable info written once to
            ``extra_info.json`` when ``raw_log_dir`` is set.
        raw_log_dir:
            Per-call directory for raw LLM IO artifacts (caller-built;
            ``None`` discards them). The first attempt writes at the root —
            identical layout to an ``Agent`` run — and shrink retries write
            under ``attempt_{i}/`` subdirectories (``i`` >= 2), so nothing
            overwrites.
        """
        candidates = _validated_unique_documents(documents)
        if scores is not None:
            _validate_scores_cover(scores, candidates)

        binding = bind_query_id(query_id) if query_id is not None else nullcontext()
        with binding:
            base_log_dir = Path(raw_log_dir) if raw_log_dir is not None else None
            if base_log_dir is not None and task_info is not None:
                await _awrite_json(task_info, base_log_dir, "extra_info.json")

            if scores is None:
                return await self._run_attempt(
                    query=str(query),
                    candidates=candidates,
                    raw_log_dir=base_log_dir,
                    record_context_limit=False,
                )

            # Scored mode: retry on context overflow, shrinking the candidates.
            initial_count = len(candidates)
            dropped_ids: List[str] = []
            attempt = 1
            while True:
                final = attempt >= self.config.shrink_attempts or len(candidates) == 1
                attempt_dir = base_log_dir
                if base_log_dir is not None and attempt > 1:
                    attempt_dir = base_log_dir / f"attempt_{attempt}"
                result = await self._run_attempt(
                    query=str(query),
                    candidates=candidates,
                    raw_log_dir=attempt_dir,
                    record_context_limit=not final,
                )
                if final or result.error is None or result.error.category != ERROR_CONTEXT_LIMIT:
                    break
                candidates, newly_dropped = _drop_lowest_priority(candidates, scores)
                dropped_ids.extend(newly_dropped)
                attempt += 1
            if dropped_ids:
                result.extra_data["context_shrink"] = {
                    "attempts_run": attempt,
                    "dropped_doc_ids": dropped_ids,
                    "initial_candidates": initial_count,
                    "final_candidates": len(candidates),
                }
            return result

    def select_sync(self, query: str, documents: List[Dict[str, Any]], **kwargs: Any) -> AgentRunResult:
        """Synchronous facade over :meth:`select` (one ``asyncio.run`` per call).

        For thread-based callers; must not be invoked from inside a running
        event loop — ``await select(...)`` there instead.
        """
        return asyncio.run(self.select(query, documents, **kwargs))

    # ------------------------------------------------------------------
    # Per-attempt assembly.
    # ------------------------------------------------------------------

    async def _run_attempt(
        self,
        *,
        query: str,
        candidates: List[Dict[str, Any]],
        raw_log_dir: Optional[Path],
        record_context_limit: bool,
    ) -> AgentRunResult:
        feasible_topk = min(self.config.target_top_k, len(candidates))
        system_prompt = render_system_prompt(
            self._prompt_name,
            extended_relevance=self.config.extended_relevance,
            top_k=feasible_topk,
        )
        end_tool = LogSelectedDocs(
            top_k=feasible_topk,
            candidate_docids=[d["id"] for d in candidates],
            include_msg=self.config.end_tool_with_msg,
        )
        tool_map: Dict[str, BaseTool] = {}
        if self._think_tool is not None:
            tool_map[self._think_tool.name] = self._think_tool
        tool_map[end_tool.name] = end_tool

        state = _RunState(
            query=query,
            raw_log_dir=raw_log_dir,
            exclude_docs=set(),
            pacer=PropagationPacer(target_s=self.config.cache_propagation_target_s),
            tool_map=tool_map,
            tool_specs=[t.spec for t in tool_map.values()],
            auto_user_msg=self._auto_user_msg,
            stage=self._stage,
            record_context_limit=record_context_limit,
            message_history=[
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                _documents_user_message(query, candidates),
            ],
        )
        return await self._run_state_to_result(state)


# ---------------------------------------------------------------------------
# Input validation + per-run message building.
# ---------------------------------------------------------------------------


def _validated_unique_documents(documents: Any) -> List[Dict[str, Any]]:
    """Validate the candidate documents and dedup by id (first occurrence wins).

    Returns copies with ``id`` coerced to ``str``; never mutates the caller's
    dicts. Violations raise ``ValueError`` naming the offending item — these
    are caller bugs, not run outcomes.
    """
    if not isinstance(documents, list) or len(documents) == 0:
        raise ValueError("documents must be a non-empty list of document dicts.")
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for idx, doc in enumerate(documents):
        if not isinstance(doc, dict):
            raise ValueError(f"documents[{idx}] must be a dict, got {type(doc).__name__}.")
        if doc.get("id") is None:
            raise ValueError(f"documents[{idx}] is missing the required 'id' key.")
        doc_id = str(doc["id"])
        for key in ("text", "image"):
            val = doc.get(key)
            if val is not None and not isinstance(val, str):
                raise ValueError(
                    f"documents[{idx}] (id {doc_id!r}) has a non-string {key!r}: {type(val).__name__}."
                )
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append({**doc, "id": doc_id})
    return out


def _validate_scores_cover(scores: Any, candidates: List[Dict[str, Any]]) -> None:
    """Require a numeric score for every candidate id (supersets tolerated)."""
    if not isinstance(scores, dict):
        raise ValueError(f"scores must be a dict of doc_id -> priority, got {type(scores).__name__}.")
    missing = [d["id"] for d in candidates if d["id"] not in scores]
    if missing:
        raise ValueError(
            "scores must cover every candidate document id; missing: "
            + ", ".join(repr(i) for i in missing)
            + "."
        )
    bad = [d["id"] for d in candidates if not isinstance(scores[d["id"]], (int, float))]
    if bad:
        raise ValueError(
            "scores values must be numbers; non-numeric score for id(s): "
            + ", ".join(repr(i) for i in bad)
            + "."
        )


def _drop_lowest_priority(
    candidates: List[Dict[str, Any]], scores: Dict[str, float]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Drop the lowest-priority quarter (at least 1 doc, never all of them).

    Deterministic: ascending score, original position as the tiebreak (ties
    drop the earlier doc first, via a stable sort over first-seen order). The
    kept documents preserve their original relative order, so the retry's user
    message renders them in the same sequence minus the dropped ones.
    """
    n = len(candidates)
    drop_n = min(max(1, math.floor(n * _SHRINK_DROP_FRACTION)), n - 1)
    by_priority = sorted(range(n), key=lambda i: (scores[candidates[i]["id"]], i))
    drop_idx = set(by_priority[:drop_n])
    dropped_ids = [candidates[i]["id"] for i in by_priority[:drop_n]]
    kept = [candidates[i] for i in range(n) if i not in drop_idx]
    return kept, dropped_ids


def _documents_user_message(query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the candidates user message.

    Only ``id`` / ``text`` / ``image`` are rendered; other keys (including
    retrieval ``score``s) are never shown to the LLM.
    """
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Query:\n{query}"},
        {"type": "text", "text": "Candidate Documents:"},
    ]
    for doc in candidates:
        content.append({"type": "text", "text": f"Doc ID: {doc['id']}"})
        text = doc.get("text")
        if isinstance(text, str) and text.strip() != "":
            content.append({"type": "text", "text": f"Doc Text: {text}"})
        image = doc.get("image")
        if image is not None and str(image).strip() != "":
            content.append({"type": "image_url", "image_url": {"url": image}})
    return {"role": "user", "content": content}
