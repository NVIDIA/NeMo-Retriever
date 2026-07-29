# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal agent-loop engine shared by :class:`~nemo_agent.agent.Agent` and
:class:`~nemo_agent.selection_agent.SelectionAgent`.

This module owns everything both agents have in common: the per-run state, the
LLM step loop (finish-reason branching, auto-continue, reasoning annotation),
tool-call dispatch (unknown-tool and malformed-arguments recovery, structural
:class:`~nemo_agent.tools.BaseEndTool` termination), the ``on_error`` policy,
raw-IO capture/flush, progress logging, and result building.

Subclasses provide assembly, not loop mechanics: they seed a :class:`_RunState`
(system/user messages, per-run tool map and specs, auto-continue text, usage
stage label) and call :meth:`_BaseAgentLoop._run_state_to_result`. The one
dispatch hook is :meth:`_BaseAgentLoop._dispatch_tool_call`, which
:class:`~nemo_agent.agent.Agent` overrides to intercept retrieve tools.

Everything here is private implementation detail within
:mod:`nemo_retriever._agentic.nemo_agent`; names used by sibling modules are not
part of the supported external interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict

from .cache_propagation import PropagationPacer
from .llm import (
    BaseLLMBackend,
    CompletionResult,
    ContentPolicyError,
    ContextLimitError,
    LLMCallError,
    bind_stage,
    get_query_id,
)
from .results import (
    ERROR_BAD_FINISH_REASON,
    ERROR_CONTENT_POLICY,
    ERROR_CONTEXT_LIMIT,
    ERROR_LLM_CALL_FAILED,
    ERROR_MAX_STEPS,
    ERROR_TOOL_FAILED,
    ERROR_UNEXPECTED,
    AgentError,
    AgentRunResult,
)
from .tools import BaseEndTool, BaseTool

logger = logging.getLogger(__name__)

# LLM call failures that are expected, model-input-dependent, and
# non-actionable. Under ``on_error="raise_unknown"`` ONLY these end the run
# with an error record; everything else (bare LLMCallError — auth/timeout/5xx,
# a RateLimitError that exhausted the backend's retries, tool crashes, bugs)
# raises so it gets seen and fixed.
KNOWN_LLM_ERRORS: Tuple[type, ...] = (ContextLimitError, ContentPolicyError)


class ToolExecutionError(Exception):
    """A tool raised an unexpected exception while the agent executed it.

    The original exception is chained on ``__cause__``. Never sent to the LLM;
    the agent's error policy maps it to the ``tool_failed`` category.
    """

    def __init__(self, tool_name: str, original: BaseException) -> None:
        super().__init__(f"Tool '{tool_name}' failed. {type(original).__name__}: {original}")
        self.tool_name = tool_name


def build_auto_continue_msg(end_tool_name: str, end_payload_phrase: str) -> str:
    """The user message appended when the model stops without calling a tool.

    One template for every agent, interpolating the actual end tool's name so
    the instruction can never point at a tool that doesn't exist in the run.
    """
    return (
        "Please continue on whatever approach you think is suitable.\n"
        f"If you think you have solved the task, you MUST call the {end_tool_name} tool "
        f"{end_payload_phrase}. Saying the task is complete in text only does NOT "
        f"end the interaction; you must call {end_tool_name}.\n"
        "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN RESPONSE.\n"
    )


class BaseAgentLoopConfig(BaseModel):
    """Config fields the shared loop engine reads; both agent configs inherit it.

    Pure data — see the subclasses (``AgentConfig``, ``SelectionAgentConfig``)
    for the agent-specific fields and the field semantics they add.

    Attributes
    ----------
    max_steps:
        Maximum number of LLM calls per run; ``None`` = unlimited (callers set
        their own limit). An end-tool call made on the last allowed step still
        counts as success (the limit is checked before each LLM call).
    on_error:
        Error policy. ``raise_unknown`` (default): expected
        model-input-dependent failures (:data:`KNOWN_LLM_ERRORS`) become a
        terminal error record; everything else raises. ``never_raise``
        (production): every ``Exception`` becomes an error record (full
        traceback logged). ``raise_all`` (debugging): everything raises except
        the normal terminations (max-steps, bad finish_reason).
    write_all_llm_io_logs:
        When a run gets a ``raw_log_dir``: ``True`` writes every step's
        prompt/response pair immediately; ``False`` (default) writes only the
        most recent pair, once, at run end.
    verbose_progress_logs:
        Per-step INFO progress line (step counter + rate-limit info), labeled
        with the bound query id.
    cache_propagation_target_s:
        Minimum wall-time gap between consecutive LLM calls in a run, for
        Anthropic prompt-cache propagation. ``0.0`` (default) disables pacing.
    """

    model_config = ConfigDict(extra="forbid")

    max_steps: Optional[int] = None
    on_error: Literal["never_raise", "raise_unknown", "raise_all"] = "raise_unknown"
    write_all_llm_io_logs: bool = False
    verbose_progress_logs: bool = True
    cache_propagation_target_s: float = 0.0


@dataclass
class _RunState:
    """All per-run mutable state plus the per-run inputs the loop consumes.

    One instance per run (for the selection agent: per shrink attempt). Agent
    instances themselves are immutable after construction, so a single agent
    safely serves many (including concurrent) runs. ``tool_map`` /
    ``tool_specs`` / ``auto_user_msg`` / ``stage`` are per-run *inputs* seeded
    by the agent: static copies for ``Agent``, built per call for
    ``SelectionAgent`` (whose end tool and prompt depend on the candidates).
    """

    query: str
    raw_log_dir: Optional[Path]
    exclude_docs: Set[str]
    pacer: PropagationPacer
    tool_map: Dict[str, BaseTool]
    tool_specs: List[Dict[str, Any]]
    auto_user_msg: str
    stage: str
    # When True, a ContextLimitError is recorded as an error result even under
    # on_error="raise_all" — set only by SelectionAgent's non-final shrink
    # attempts, whose retry loop must inspect the typed result.
    record_context_limit: bool = False
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    retrieved_docs: Set[str] = field(default_factory=set)
    retrieval_log: List[Dict[str, Any]] = field(default_factory=list)
    extra_data: Dict[str, Any] = field(default_factory=dict)
    extra_response_infos: List[Dict[str, Any]] = field(default_factory=list)
    last_reasoning: Optional[str] = None
    last_raw_io: Optional[Tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = None
    end_payload: Optional[Dict[str, Any]] = None
    # Best-effort payload from the most recent INVALID end-tool call, kept so a
    # run that fails without ever ending validly can still surface the agent's
    # last attempt. The last attempt wins.
    last_end_attempt: Optional[Dict[str, Any]] = None
    error: Optional[AgentError] = None
    warned_missing_raw_io: bool = False


class _BaseAgentLoop:
    """The loop engine. Subclasses assemble runs; this class executes them.

    Mirrors the ``BaseLLMBackend`` template pattern: the run/step/dispatch
    methods here are the templates — subclasses must not override them, with
    the single documented exception of :meth:`_dispatch_tool_call` (extend by
    intercepting a tool kind, then defer to ``super()``).
    """

    def __init__(self, config: BaseAgentLoopConfig, llm: BaseLLMBackend) -> None:
        if not isinstance(config, BaseAgentLoopConfig):
            raise TypeError(f"config must be a BaseAgentLoopConfig, got {type(config).__name__}.")
        if not isinstance(llm, BaseLLMBackend):
            raise TypeError(f"llm must be a BaseLLMBackend, got {type(llm).__name__}.")
        self.config = config
        self.llm = llm

    # ------------------------------------------------------------------
    # Run skeleton.
    # ------------------------------------------------------------------

    async def _run_state_to_result(
        self,
        state: _RunState,
        *,
        prologue: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> AgentRunResult:
        """Drive a seeded run state through the loop under the error policy.

        ``prologue`` (when given) runs inside the try block, so its failures —
        e.g. the main agent's bootstrap retrieve — hit the same error policy
        as loop failures. Deferred log artifacts flush in ``finally`` so they
        land on error and raise paths too.
        """
        try:
            if prologue is not None:
                await prologue()
            await self._loop(state)
        except Exception as e:
            if not self._handle_run_exception(state, e):
                raise
        finally:
            await self._flush_deferred_logs(state)
        return self._build_result(state)

    # ------------------------------------------------------------------
    # Loop.
    # ------------------------------------------------------------------

    async def _loop(self, state: _RunState) -> None:
        while True:
            if self.config.max_steps is not None and state.steps >= self.config.max_steps:
                self._record_error(
                    state,
                    AgentError(category=ERROR_MAX_STEPS, message="Agent reached maximum allowed iterations"),
                )
                return
            await self._step(state)
            if state.error is not None:
                return
            tool_calls = state.message_history[-1].get("tool_calls") or []
            if len(tool_calls) == 0:
                state.message_history.append(
                    {"role": "user", "content": [{"type": "text", "text": state.auto_user_msg}]}
                )
                continue
            ended = await self._process_tool_calls(state)
            if ended:
                return

    async def _step(self, state: _RunState) -> None:
        """One LLM call: append the assistant message or record a terminal error."""
        await state.pacer.await_propagation()
        with bind_stage(state.stage):
            result = await self.llm.acompletion(messages=state.message_history, tools=state.tool_specs)
        state.pacer.mark()
        step_idx = state.steps
        state.steps += 1
        state.extra_response_infos.append(result.extra_response_info)
        await self._capture_raw_io(state, step_idx, result)
        self._log_progress(state, result)

        # Overwritten every step — None when this turn exposed no reasoning,
        # so stale reasoning never leaks into the next retrieve.
        state.last_reasoning = result.reasoning

        if result.finish_reason not in ("stop", "tool_calls"):
            self._record_error(
                state,
                AgentError(
                    category=ERROR_BAD_FINISH_REASON,
                    message=f"LLM failed with finish_reason '{result.finish_reason}'",
                ),
            )
            return
        message = dict(result.message)
        if result.reasoning is not None:
            message["__reasoning__"] = result.reasoning  # backends strip __-prefixed keys
        state.message_history.append(message)

    async def _process_tool_calls(self, state: _RunState) -> bool:
        """Execute the last assistant message's tool calls; True when the run ended."""
        ended = False
        tool_messages: List[Dict[str, Any]] = []
        for call_info in state.message_history[-1]["tool_calls"]:
            fn_name = str((call_info.get("function") or {}).get("name"))
            content: Optional[List[Dict[str, Any]]] = None
            fn_kwargs: Optional[Dict[str, Any]] = None
            try:
                fn_kwargs = json.loads(call_info["function"]["arguments"])
                if not isinstance(fn_kwargs, dict):
                    raise TypeError("tool arguments must decode to an object")
            except Exception:
                content = [
                    {"type": "text", "text": "Error parsing tool arguments. Tool arguments not correctly formatted."}
                ]
            if content is None:
                assert fn_kwargs is not None
                content, call_ended = await self._dispatch_tool_call(state, fn_name, fn_kwargs)
                ended = ended or call_ended
            tool_messages.append(
                {"content": content, "role": "tool", "tool_call_id": call_info.get("id"), "name": fn_name}
            )
        state.message_history.extend(tool_messages)
        return ended

    async def _dispatch_tool_call(
        self, state: _RunState, fn_name: str, fn_kwargs: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Return ``(tool_message_content, run_ended)`` for one tool call.

        Handles unknown tools, end tools, and generic tools. ``Agent``
        overrides this to intercept retrieve tools first, then defers here.
        """
        tool = state.tool_map.get(fn_name)
        if tool is None:
            # LLMs occasionally hallucinate tool names; an error result lets
            # the model self-correct instead of aborting the whole run.
            available = sorted(state.tool_map)
            state.extra_data.setdefault("unknown_tool_calls", []).append(
                {"requested": str(fn_name), "available": available}
            )
            text = (
                f"Error: tool '{fn_name}' is not available. "
                f"Available tools: {', '.join(available) if available else '(none)'}. "
                "Please retry using one of the available tool names exactly as listed."
            )
            return [{"type": "text", "text": text}], False
        try:
            if isinstance(tool, BaseEndTool):
                payload, text = tool.try_end(**fn_kwargs)
                if payload is not None:
                    state.end_payload = payload
                    return [{"type": "text", "text": text}], True
                # Invalid end call: the agent retries (error text below), but keep
                # its best-effort attempt so a run that later fails (e.g. max
                # steps) can still surface the model's last answer/doc_ids
                # instead of nothing. Overwrite so the LAST attempt wins.
                salvaged = tool.salvage_payload(fn_kwargs)
                if salvaged is not None:
                    state.last_end_attempt = salvaged
                return [{"type": "text", "text": text}], False
            output = await tool.acall(**fn_kwargs)
            if not isinstance(output, str):
                output = json.dumps(output)
            return [{"type": "text", "text": output}], False
        except Exception as e:
            raise ToolExecutionError(fn_name, e) from e

    # ------------------------------------------------------------------
    # Errors.
    # ------------------------------------------------------------------

    def _handle_run_exception(self, state: _RunState, exc: Exception) -> bool:
        """Apply ``config.on_error``; True when recorded (caller must not re-raise)."""
        category, exc_class = _classify_exception(exc)
        if self.config.on_error == "never_raise":
            record = True
        elif self.config.on_error == "raise_unknown":
            record = isinstance(exc, KNOWN_LLM_ERRORS)
        else:  # raise_all
            record = False
        if state.record_context_limit and isinstance(exc, ContextLimitError):
            record = True
        if not record:
            return False
        if isinstance(exc, KNOWN_LLM_ERRORS):
            logger.warning("Agent run (query_id=%r) ended with expected error (%s): %s", get_query_id(), category, exc)
        else:
            logger.error(
                "Agent run (query_id=%r) ended with error (%s): %s",
                get_query_id(),
                category,
                exc,
                exc_info=True,
            )
        self._record_error(state, AgentError(category=category, message=str(exc), exception_class=exc_class))
        return True

    @staticmethod
    def _record_error(state: _RunState, error: AgentError) -> None:
        state.error = error
        state.message_history.append({"role": "agent_error", "content": f"[{error.category}] {error.message}"})

    # ------------------------------------------------------------------
    # Raw-IO logging (paths come from the caller; the LLM only captures).
    # ------------------------------------------------------------------

    async def _capture_raw_io(self, state: _RunState, step_idx: int, result: CompletionResult) -> None:
        if state.raw_log_dir is None:
            return
        if result.raw_request is None and result.raw_response is None:
            if not state.warned_missing_raw_io:
                logger.warning(
                    "raw_log_dir=%s was provided but the LLM backend returned no raw IO; construct "
                    "the backend with capture_raw_io=True to get per-step prompt/response logs.",
                    state.raw_log_dir,
                )
                state.warned_missing_raw_io = True
            return
        if self.config.write_all_llm_io_logs:
            await _write_raw_pair(state.raw_log_dir, step_idx, result.raw_request, result.raw_response)
        else:
            state.last_raw_io = (step_idx, result.raw_request, result.raw_response)

    async def _flush_deferred_logs(self, state: _RunState) -> None:
        """Write deferred artifacts at run end (also on error/raise paths).

        Best-effort: a failing log write must never mask the run's outcome.
        """
        if state.raw_log_dir is None:
            return
        try:
            if not self.config.write_all_llm_io_logs and state.last_raw_io is not None:
                step_idx, raw_request, raw_response = state.last_raw_io
                await _write_raw_pair(state.raw_log_dir, step_idx, raw_request, raw_response)
            if state.extra_response_infos:
                await _awrite_json(state.extra_response_infos, state.raw_log_dir, "api_response_extras.json")
        except Exception:
            logger.exception("Failed to write LLM IO logs under %s.", state.raw_log_dir)

    # ------------------------------------------------------------------
    # Progress + result.
    # ------------------------------------------------------------------

    def _log_progress(self, state: _RunState, result: CompletionResult) -> None:
        if not self.config.verbose_progress_logs:
            return
        parts = [f"S: {state.steps}"]
        ratelimit = result.extra_response_info.get("ratelimit")
        if isinstance(ratelimit, dict):
            parts.extend(f"{k}: {v}" for k, v in ratelimit.items())
        qid = get_query_id()
        prefix = f"[{qid}] " if qid else ""
        logger.info("%s%s", prefix, " ".join(parts))

    def _build_result(self, state: _RunState) -> AgentRunResult:
        payload = state.end_payload
        if payload is None and state.last_end_attempt is not None:
            # The run ended in error without a valid end call. Fall back to the
            # agent's last (invalid) end-tool attempt so callers still get its
            # best-effort answer/doc_ids. The run still counts as failed
            # (``error`` set, ``succeeded`` False).
            payload = state.last_end_attempt
        final_doc_ids: List[str] = []
        answer: Optional[str] = None
        if payload is not None:
            doc_ids = payload.get("doc_ids")
            if isinstance(doc_ids, list):
                final_doc_ids = [str(d) for d in doc_ids]
            ans = payload.get("answer")
            if isinstance(ans, str):
                answer = ans
        return AgentRunResult(
            final_doc_ids=final_doc_ids,
            answer=answer,
            end_payload=payload,
            error=state.error,
            trajectory=state.message_history,
            retrieval_log=state.retrieval_log,
            extra_data=state.extra_data,
        )


def _classify_exception(exc: Exception) -> Tuple[str, str]:
    """Map an exception to an ``AgentError`` category + exception class name."""
    if isinstance(exc, ContextLimitError):
        return ERROR_CONTEXT_LIMIT, type(exc).__name__
    if isinstance(exc, ContentPolicyError):
        return ERROR_CONTENT_POLICY, type(exc).__name__
    if isinstance(exc, LLMCallError):
        return ERROR_LLM_CALL_FAILED, type(exc).__name__
    if isinstance(exc, ToolExecutionError):
        cause = exc.__cause__
        return ERROR_TOOL_FAILED, type(cause).__name__ if cause is not None else type(exc).__name__
    return ERROR_UNEXPECTED, type(exc).__name__


async def _write_raw_pair(
    log_dir: Path, step_idx: int, raw_request: Optional[Dict[str, Any]], raw_response: Optional[Dict[str, Any]]
) -> None:
    if raw_request is not None:
        await _awrite_json(raw_request, log_dir, f"{step_idx}_prompt.json")
    if raw_response is not None:
        await _awrite_json(raw_response, log_dir, f"{step_idx}_response.json")


async def _awrite_json(obj: Any, log_dir: Union[str, Path], filename: str) -> None:
    def _write() -> None:
        path = Path(log_dir, filename)
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    await asyncio.to_thread(_write)
