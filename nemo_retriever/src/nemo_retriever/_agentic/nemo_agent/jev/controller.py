# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieval loop whose actions are chosen by a System One model instead of the LLM.

The LLM agent spends one full-context completion per step just to decide what
to do next. Here a System One model (Jev) makes that decision from a compact
state — the query, the best evidence so far, the searches run and pending —
and the LLM is called only when text must be written:

1. bootstrap: retrieve with the original query;
2. plan (LLM): propose an initial set of sub-queries;
3. loop (Jev): pick one of ``retrieve`` (run the next pending sub-query, also
   chosen by Jev), ``expand`` (LLM writes more sub-queries), ``think`` (LLM
   writes an analysis note, optionally with sub-queries), or ``finish``.

With ``batch_retrieve`` (the default) every pending sub-query is run as soon as
it is written, and Jev only chooses between ``expand``, ``think``, and
``finish`` once nothing is pending: one decision per round instead of one per
search.

``finish`` is only accepted when Jev's "evidence is sufficient" probability
clears ``jev_stop_threshold``; otherwise the loop falls back to searching. The
run never produces a ranked final list: the downstream selection stage ranks
the retrieved candidates, so loop savings and selection savings stay separable.
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..agent import Agent, AgentConfig
from ..cache_propagation import PropagationPacer
from ..llm import BaseLLMBackend, bind_query_id, bind_stage
from ..loop import _RunState
from ..prompts import render_system_prompt
from ..results import AgentRunResult
from ..tools import BaseRetrieveTool
from .client import JevCallError, JevClient, JevContextLengthError, choice, noul

logger = logging.getLogger(__name__)

STAGE_CONTROLLER = "jev_controller"
STAGE_PLAN = "jev_plan"
STAGE_EXPAND = "jev_expand"
STAGE_THINK = "jev_think"

ACTION_RETRIEVE = "retrieve"
ACTION_EXPAND = "expand"
ACTION_THINK = "think"
ACTION_FINISH = "finish"

# Context-overflow retries for a controller decision, halving the evidence budget each time.
_CONTEXT_RETRIES = 2
_FALLBACK_EVIDENCE_BUDGET = 2000

_PROPOSE_SUBQUERIES_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_subqueries",
        "description": "Submit the new search queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "subqueries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "New search queries, most promising first.",
                }
            },
            "required": ["subqueries"],
        },
    },
}

_RECORD_ANALYSIS_TOOL = {
    "type": "function",
    "function": {
        "name": "record_analysis",
        "description": "Submit your analysis of the retrieval progress.",
        "parameters": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "description": "What is covered, missing, or conflicting."},
                "subqueries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional new search queries, most promising first.",
                },
            },
            "required": ["analysis"],
        },
    },
}

_ACTION_CRITERIA = {
    ACTION_RETRIEVE: (
        "Run one of the searches in `pending_searches`. Right when `evidence` lacks information that "
        "`query` needs and a pending search could find it."
    ),
    ACTION_EXPAND: (
        "Write new searches. Right when `evidence` lacks information that `query` needs and no search in "
        "`pending_searches` targets that information, or `pending_searches` is empty."
    ),
    ACTION_THINK: (
        "Analyze the situation before searching further. Right when `evidence` is contradictory, or when it "
        "is unclear what information `query` is asking for."
    ),
    ACTION_FINISH: "Stop searching. Right when `evidence` already contains the documents that `query` needs.",
}

_ACTION_INSTRUCTIONS = (
    "A retrieval agent is collecting documents for `query`. `evidence` holds the most relevant documents found "
    "so far, `searches_run` the searches already executed, and `pending_searches` planned searches not yet run. "
    "Which action should the agent take next?"
)
_SUFFICIENT_INSTRUCTIONS = "Does `evidence` contain the information needed to answer `query` completely?"
_NEXT_SEARCH_INSTRUCTIONS = (
    "Which search in `pending_searches` is most likely to find information that `query` needs and `evidence` lacks?"
)


class JevControllerConfig(AgentConfig):
    """Configuration for :class:`JevControlledAgent` (``mode`` must stay ``"select"``).

    Attributes
    ----------
    jev_stop_threshold:
        Minimum P(evidence is sufficient) for Jev's ``finish`` to be accepted.
    min_retrievals:
        Sub-query retrievals required (beyond the bootstrap) before stopping.
    max_actions:
        Upper bound on Jev decisions per run.
    initial_subqueries / expand_subqueries:
        How many sub-queries the plan and each expansion may propose.
    max_expansions / max_thinks:
        Budgets for the LLM-backed ``expand`` and ``think`` actions.
    evidence_docs / evidence_chars:
        How many documents, and how many characters of each, Jev sees.
    evidence_char_budget:
        Total evidence characters per Jev request, so the state fits the
        endpoint's context window (``0`` = no limit; the best document is
        always included).
    batch_retrieve:
        Run all pending sub-queries without asking Jev, so Jev decides only
        between rounds (``expand`` / ``think`` / ``finish``). ``False`` lets Jev
        pick each search.
    """

    jev_stop_threshold: float = 0.7
    min_retrievals: int = 1
    max_actions: int = 12
    initial_subqueries: int = 4
    expand_subqueries: int = 3
    max_expansions: int = 2
    max_thinks: int = 1
    evidence_docs: int = 10
    evidence_chars: int = 500
    evidence_char_budget: int = 4000
    batch_retrieve: bool = True
    plan_prompt: str = "jev/plan_v0.j2"
    expand_prompt: str = "jev/expand_v0.j2"
    think_prompt: str = "jev/think_v0.j2"


@dataclass
class _Progress:
    """Per-run controller bookkeeping on top of the shared run state."""

    pending: List[str] = field(default_factory=list)
    searches_run: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    evidence: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    retrievals: int = 0
    expansions: int = 0
    thinks: int = 0
    seen_queries: Set[str] = field(default_factory=set)


class JevControlledAgent(Agent):
    """Select-mode retrieval agent whose loop decisions are made by Jev.

    Shares the :class:`~nemo_agent.agent.Agent` retrieve bookkeeping (over-fetch,
    dedup, exclusion, retrieval log) and result building; ``llm`` is used only
    for the plan, expand, and think actions.
    """

    def __init__(
        self,
        config: JevControllerConfig,
        llm: BaseLLMBackend,
        retrieve_tool: BaseRetrieveTool,
        jev: JevClient,
    ) -> None:
        if not isinstance(config, JevControllerConfig):
            raise TypeError(f"config must be a JevControllerConfig, got {type(config).__name__}.")
        if config.mode != "select":
            raise ValueError("JevControlledAgent only supports mode='select'.")
        super().__init__(config=config, llm=llm, retrieve_tool=retrieve_tool)
        self.jev = jev
        self._plan_prompt = render_system_prompt(config.plan_prompt, max_subqueries=config.initial_subqueries)
        self._expand_prompt = render_system_prompt(config.expand_prompt, max_subqueries=config.expand_subqueries)
        self._think_prompt = render_system_prompt(config.think_prompt, max_subqueries=config.expand_subqueries)

    @property
    def jev_config(self) -> JevControllerConfig:
        assert isinstance(self.config, JevControllerConfig)
        return self.config

    async def run(
        self,
        query: str,
        *,
        query_id: Optional[str] = None,
        task_instruction: Optional[str] = None,
        task_info: Optional[Any] = None,
        exclude_docids: Optional[Set[str]] = None,
        raw_log_dir: Optional[Union[str, Path]] = None,
    ) -> AgentRunResult:
        """Run one query; the result has an empty ``final_doc_ids`` by design."""
        binding = bind_query_id(query_id) if query_id is not None else nullcontext()
        with binding:
            state = _RunState(
                query=str(query),
                raw_log_dir=Path(raw_log_dir) if raw_log_dir is not None else None,
                exclude_docs=set(exclude_docids) if exclude_docids is not None else set(),
                pacer=PropagationPacer(target_s=0.0),
                tool_map=self._tool_map,
                tool_specs=self._tool_specs,
                auto_user_msg="",
                stage=STAGE_CONTROLLER,
                message_history=[{"role": "user", "content": [{"type": "text", "text": f"Query:\n{query}"}]}],
            )
            state.extra_data["jev_decisions"] = []
            progress = _Progress(seen_queries={_normalize(query)})
            try:
                await self._retrieve(state, progress, str(query), query_type="main")
                await self._plan(state, progress)
                await self._control(state, progress)
            except Exception as e:
                if not self._handle_run_exception(state, e):
                    raise
            finally:
                await self._flush_deferred_logs(state)
            return self._build_result(state)

    # ------------------------------------------------------------------
    # Control loop.
    # ------------------------------------------------------------------

    async def _control(self, state: _RunState, progress: _Progress) -> None:
        cfg = self.jev_config
        for step in range(cfg.max_actions):
            if cfg.batch_retrieve and progress.pending:
                await self._retrieve_round(state, progress, step)
            actions = self._available_actions(progress)
            if actions == [ACTION_FINISH]:
                self._log_decision(state, step, None, ACTION_FINISH, "no other action available")
                return
            try:
                answers = await self._decide(state, progress, actions)
            except JevCallError as e:
                logger.warning(
                    "JevControlledAgent: Jev decision failed (%s); running pending searches and finishing", e
                )
                if progress.pending:
                    await self._retrieve_round(state, progress, step)
                self._log_decision(state, step, None, ACTION_FINISH, "jev call failed")
                return
            picked = str(answers["action"]["choice"])
            sufficient = float(answers["sufficient"]["noul"])
            action, reason = self._gate(progress, actions, picked, sufficient)
            self._log_decision(state, step, answers, action, reason)
            if action == ACTION_FINISH:
                return
            if action == ACTION_RETRIEVE:
                index = _pending_index(answers.get("next_search"), len(progress.pending))
                subquery = progress.pending.pop(index)
                await self._retrieve(state, progress, subquery, query_type="agent")
            elif action == ACTION_EXPAND:
                await self._expand(state, progress)
            else:
                await self._think(state, progress)
        self._log_decision(state, cfg.max_actions, None, ACTION_FINISH, "max_actions reached")

    async def _retrieve_round(self, state: _RunState, progress: _Progress, step: int) -> None:
        searches, progress.pending = progress.pending, []
        for subquery in searches:
            await self._retrieve(state, progress, subquery, query_type="agent")
        self._log_decision(state, step, None, ACTION_RETRIEVE, f"round of {len(searches)} searches")

    def _available_actions(self, progress: _Progress) -> List[str]:
        cfg = self.jev_config
        actions: List[str] = []
        if progress.pending:
            actions.append(ACTION_RETRIEVE)
        if progress.expansions < cfg.max_expansions:
            actions.append(ACTION_EXPAND)
        if progress.thinks < cfg.max_thinks:
            actions.append(ACTION_THINK)
        actions.append(ACTION_FINISH)
        return actions

    async def _decide(self, state: _RunState, progress: _Progress, actions: List[str]) -> Dict[str, Dict[str, Any]]:
        questions: Dict[str, Dict[str, Any]] = {
            "action": choice(_ACTION_INSTRUCTIONS, {a: _ACTION_CRITERIA[a] for a in actions}),
            "sufficient": noul(_SUFFICIENT_INSTRUCTIONS),
        }
        if len(progress.pending) >= 2:
            questions["next_search"] = choice(
                _NEXT_SEARCH_INSTRUCTIONS, {f"s{i}": text for i, text in enumerate(progress.pending)}
            )
        budget = self.jev_config.evidence_char_budget
        with bind_stage(STAGE_CONTROLLER):
            for attempt in range(_CONTEXT_RETRIES + 1):
                try:
                    return await self.jev.aevaluate(self._jev_state(state, progress, budget), questions)
                except JevContextLengthError:
                    if attempt == _CONTEXT_RETRIES:
                        raise
                    budget = max(budget // 2, 1) if budget > 0 else _FALLBACK_EVIDENCE_BUDGET
        raise AssertionError("unreachable")

    def _gate(self, progress: _Progress, actions: List[str], picked: str, sufficient: float) -> tuple[str, str]:
        """Turn Jev's pick into the action to execute, guarding against premature stops."""
        cfg = self.jev_config
        may_stop = progress.retrievals >= cfg.min_retrievals
        if picked not in actions:
            picked = actions[0]
        if picked != ACTION_FINISH:
            return picked, "jev choice"
        if may_stop and sufficient >= cfg.jev_stop_threshold:
            return ACTION_FINISH, "jev choice"
        for fallback in (ACTION_RETRIEVE, ACTION_EXPAND):
            if fallback in actions:
                return fallback, "finish rejected: evidence not sufficient or too few retrievals"
        return ACTION_FINISH, "finish accepted: no search budget left"

    # ------------------------------------------------------------------
    # Actions.
    # ------------------------------------------------------------------

    async def _retrieve(self, state: _RunState, progress: _Progress, query: str, *, query_type: str) -> None:
        content = await self._execute_retrieve(
            state, tool=self._retrieve_tool, llm_kwargs={"query": query}, query_type=query_type
        )
        progress.searches_run.append(query)
        if query_type != "main":
            progress.retrievals += 1
        if state.retrieval_log and state.retrieval_log[-1]["input"]["query"] == query:
            for rank, doc in enumerate(state.retrieval_log[-1]["output"], 1):
                entry = progress.evidence.setdefault(
                    str(doc["id"]), {"text": str(doc.get("text") or ""), "best_rank": rank}
                )
                entry["best_rank"] = min(entry["best_rank"], rank)
        self._append_trace(state, ACTION_RETRIEVE, {"query": query}, content)

    async def _plan(self, state: _RunState, progress: _Progress) -> None:
        args = await self._llm_tool_call(
            state, STAGE_PLAN, self._plan_prompt, self._llm_context(state, progress), _PROPOSE_SUBQUERIES_TOOL
        )
        added = self._add_subqueries(progress, args.get("subqueries"), self.jev_config.initial_subqueries)
        self._append_trace(state, "plan", {"subqueries": added}, None)

    async def _expand(self, state: _RunState, progress: _Progress) -> None:
        progress.expansions += 1
        args = await self._llm_tool_call(
            state, STAGE_EXPAND, self._expand_prompt, self._llm_context(state, progress), _PROPOSE_SUBQUERIES_TOOL
        )
        added = self._add_subqueries(progress, args.get("subqueries"), self.jev_config.expand_subqueries)
        if not added:
            # An expansion that proposes nothing new will not do better next time.
            progress.expansions = self.jev_config.max_expansions
        self._append_trace(state, ACTION_EXPAND, {"subqueries": added}, None)

    async def _think(self, state: _RunState, progress: _Progress) -> None:
        progress.thinks += 1
        args = await self._llm_tool_call(
            state, STAGE_THINK, self._think_prompt, self._llm_context(state, progress), _RECORD_ANALYSIS_TOOL
        )
        analysis = str(args.get("analysis") or "").strip()
        if analysis:
            progress.notes.append(analysis)
        added = self._add_subqueries(progress, args.get("subqueries"), self.jev_config.expand_subqueries)
        self._append_trace(state, ACTION_THINK, {"analysis": analysis, "subqueries": added}, None)

    async def _llm_tool_call(
        self,
        state: _RunState,
        stage: str,
        system_prompt: str,
        user_text: str,
        tool_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """One LLM completion offering a single tool; returns that tool's parsed arguments (``{}`` if absent)."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
        with bind_stage(stage):
            result = await self.llm.acompletion(messages=messages, tools=[tool_spec])
        state.steps += 1
        name = tool_spec["function"]["name"]
        for call in result.message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != name:
                continue
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                continue
            if isinstance(args, dict):
                return args
        content = str(result.message.get("content") or "")
        if tool_spec is _RECORD_ANALYSIS_TOOL:
            return {"analysis": content}
        return _subqueries_from_text(content)

    # ------------------------------------------------------------------
    # State rendering and bookkeeping.
    # ------------------------------------------------------------------

    def _top_evidence(self, progress: _Progress, char_budget: Optional[int] = None) -> List[Dict[str, str]]:
        cfg = self.jev_config
        budget = cfg.evidence_char_budget if char_budget is None else char_budget
        ranked = sorted(progress.evidence.items(), key=lambda item: item[1]["best_rank"])
        evidence: List[Dict[str, str]] = []
        used = 0
        for doc_id, entry in ranked[: cfg.evidence_docs]:
            text = _truncate(entry["text"], min(cfg.evidence_chars, budget) if budget > 0 else cfg.evidence_chars)
            if evidence and budget > 0 and used + len(text) > budget:
                break
            evidence.append({"id": doc_id, "text": text})
            used += len(text)
        return evidence

    def _jev_state(self, state: _RunState, progress: _Progress, char_budget: Optional[int] = None) -> Dict[str, Any]:
        jev_state: Dict[str, Any] = {
            "query": state.query,
            "evidence": self._top_evidence(progress, char_budget),
            "searches_run": list(progress.searches_run),
            "pending_searches": list(progress.pending),
        }
        if progress.notes:
            jev_state["notes"] = list(progress.notes)
        return jev_state

    def _llm_context(self, state: _RunState, progress: _Progress) -> str:
        parts = [
            f"Query:\n{state.query}",
            "Searches already run:\n" + "\n".join(f"- {q}" for q in progress.searches_run),
        ]
        if progress.pending:
            parts.append("Planned searches not yet run:\n" + "\n".join(f"- {q}" for q in progress.pending))
        if progress.notes:
            parts.append("Analysis notes:\n" + "\n".join(f"- {n}" for n in progress.notes))
        parts.append(
            "Most relevant evidence so far:\n" + "\n".join(json.dumps(d) for d in self._top_evidence(progress))
        )
        return "\n\n".join(parts)

    @staticmethod
    def _add_subqueries(progress: _Progress, raw: Any, limit: int) -> List[str]:
        added: List[str] = []
        for item in raw if isinstance(raw, list) else []:
            text = " ".join(str(item).split())
            key = _normalize(text)
            if not text or key in progress.seen_queries:
                continue
            progress.seen_queries.add(key)
            progress.pending.append(text)
            added.append(text)
            if len(added) >= limit:
                break
        return added

    @staticmethod
    def _append_trace(
        state: _RunState, action: str, args: Dict[str, Any], content: Optional[List[Dict[str, Any]]]
    ) -> None:
        call_id = f"jev-{len(state.message_history)}"
        state.message_history.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": action, "arguments": json.dumps(args)}}
                ],
            }
        )
        if content is not None:
            state.message_history.append({"role": "tool", "tool_call_id": call_id, "name": action, "content": content})

    @staticmethod
    def _log_decision(
        state: _RunState, step: int, answers: Optional[Dict[str, Dict[str, Any]]], action: str, reason: str
    ) -> None:
        record: Dict[str, Any] = {"step": step, "action": action, "reason": reason}
        if answers is not None:
            record["picked"] = answers["action"].get("choice")
            record["probabilities"] = answers["action"].get("probabilities")
            record["sufficient"] = answers["sufficient"].get("noul")
        state.extra_data["jev_decisions"].append(record)


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _truncate(text: str, limit: int) -> str:
    return text if limit <= 0 or len(text) <= limit else text[:limit].rstrip() + "..."


def _pending_index(answer: Optional[Dict[str, Any]], num_pending: int) -> int:
    if answer is None:
        return 0
    picked = str(answer.get("choice") or "")
    if picked.startswith("s") and picked[1:].isdigit() and int(picked[1:]) < num_pending:
        return int(picked[1:])
    return 0


def _subqueries_from_text(text: str) -> Dict[str, Any]:
    """Best-effort recovery when the model listed sub-queries in text instead of calling the tool."""
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    lines = [line.strip().lstrip("-*0123456789.) ").strip() for line in text.splitlines()]
    return {"subqueries": [line for line in lines if line]}
