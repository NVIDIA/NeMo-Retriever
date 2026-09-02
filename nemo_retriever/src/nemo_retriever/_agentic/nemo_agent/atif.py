# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight ATIF trajectory construction and persistence for agentic runs."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from nemo_retriever.version import __version__

logger = logging.getLogger(__name__)

ATIF_SCHEMA_VERSION = "ATIF-v1.7"
DEFAULT_TRACE_DIR = "agentic-traces"
MAX_OBSERVATION_CHARS = 20_000
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def utc_timestamp() -> str:
    """Return an ATIF-compatible UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def llm_trace_record(usage: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Capture the per-call fields needed when message history becomes ATIF."""
    return {
        "timestamp": utc_timestamp(),
        "usage": dict(usage) if isinstance(usage, Mapping) else {},
    }


def build_atif_trajectory(
    *,
    query: str,
    query_id: Optional[str],
    stage: str,
    model_name: str,
    message_history: Sequence[Mapping[str, Any]],
    llm_records: Sequence[Mapping[str, Any]],
    retrieval_log: Sequence[Mapping[str, Any]],
    error: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert one completed inner-agent run into a lightweight ATIF trajectory."""
    steps: List[Dict[str, Any]] = []
    llm_idx = 0
    message_idx = 0

    while message_idx < len(message_history):
        raw_message = message_history[message_idx]
        role = str(raw_message.get("role", ""))
        if role in {"system", "user"}:
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "timestamp": utc_timestamp(),
                    "source": role,
                    "message": _bounded_text(_content_text(raw_message.get("content"))),
                }
            )
            message_idx += 1
            continue

        if role == "assistant":
            record = llm_records[llm_idx] if llm_idx < len(llm_records) else {}
            llm_idx += 1
            tool_calls = _tool_calls(raw_message.get("tool_calls"))
            step: Dict[str, Any] = {
                "step_id": len(steps) + 1,
                "timestamp": str(record.get("timestamp") or utc_timestamp()),
                "source": "agent",
                "model_name": model_name,
                "message": _bounded_text(_content_text(raw_message.get("content"))),
                "llm_call_count": 1,
                "extra": {"stage": stage},
            }
            reasoning = raw_message.get("__reasoning__")
            if reasoning is not None:
                step["reasoning_content"] = _bounded_text(str(reasoning))
            if tool_calls:
                step["tool_calls"] = tool_calls

            observations: List[Dict[str, Any]] = []
            lookahead = message_idx + 1
            while lookahead < len(message_history) and message_history[lookahead].get("role") == "tool":
                tool_message = message_history[lookahead]
                observations.append(
                    {
                        "source_call_id": str(tool_message.get("tool_call_id") or ""),
                        "content": _bounded_text(_content_text(tool_message.get("content"))),
                    }
                )
                lookahead += 1
            if observations:
                step["observation"] = {"results": observations}
            metrics = _atif_metrics(record.get("usage"))
            if metrics:
                step["metrics"] = metrics
            steps.append(step)
            message_idx = lookahead
            continue

        if role == "agent_error":
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "timestamp": utc_timestamp(),
                    "source": "agent",
                    "message": _bounded_text(_content_text(raw_message.get("content"))),
                    "llm_call_count": 0,
                    "extra": {"stage": stage, "event": "agent_error"},
                }
            )
        message_idx += 1

    _insert_bootstrap_retrieval(steps, retrieval_log, stage=stage)
    _renumber_steps(steps)
    final_metrics = _final_metrics(steps)
    trace_extra: Dict[str, Any] = {"stage": stage}
    if query_id is not None:
        trace_extra["query_id"] = str(query_id)
    trace_extra["query"] = query
    if error:
        trace_extra["error"] = dict(error)

    return {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": str(uuid.uuid4()),
        "agent": {
            "name": "nemo-retriever-agentic",
            "version": __version__,
            "model_name": model_name,
            "extra": trace_extra,
        },
        "steps": steps,
        "final_metrics": final_metrics,
        "notes": "Lightweight agentic retrieval trace; large message and observation content is bounded.",
    }


def persist_atif_trajectory(trace: Optional[Mapping[str, Any]]) -> Optional[Path]:
    """Best-effort write under ``./agentic-traces``; never fail retrieval."""
    if not trace:
        return None
    try:
        trace_dir = Path.cwd() / DEFAULT_TRACE_DIR
        trace_dir.mkdir(parents=True, exist_ok=True)
        agent = trace.get("agent")
        agent_extra = agent.get("extra") if isinstance(agent, Mapping) else {}
        query_id = agent_extra.get("query_id", "query") if isinstance(agent_extra, Mapping) else "query"
        stage = agent_extra.get("stage", "agent") if isinstance(agent_extra, Mapping) else "agent"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        filename = (
            f"{timestamp}-{_safe_filename(str(query_id))}-{_safe_filename(str(stage))}-{uuid.uuid4().hex[:8]}.json"
        )
        path = trace_dir / filename
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        logger.info("Saved agentic ATIF trace to %s", path)
        return path
    except Exception:
        logger.warning("Failed to persist agentic ATIF trace", exc_info=True)
        return None


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, Mapping):
                text = block.get("text")
                if text is not None:
                    parts.append(str(text))
                elif block.get("type") == "image_url":
                    parts.append("[image]")
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _bounded_text(value: str, *, limit: int = MAX_OBSERVATION_CHARS) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit].rstrip()}\n...[{omitted} characters omitted]"


def _tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_calls, list):
        return []
    calls: List[Dict[str, Any]] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            continue
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            continue
        raw_arguments = function.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": _bounded_text(raw_arguments)}
        else:
            arguments = raw_arguments
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        calls.append(
            {
                "tool_call_id": str(raw_call.get("id") or ""),
                "function_name": str(function.get("name") or ""),
                "arguments": arguments,
            }
        )
    return calls


def _atif_metrics(raw_usage: Any) -> Dict[str, Any]:
    if not isinstance(raw_usage, Mapping) or not raw_usage:
        return {}

    prompt = _integer(raw_usage, "prompt_tokens")
    if prompt is None:
        prompt = _integer(raw_usage, "input_tokens")
        if prompt is not None:
            prompt += sum(
                value or 0
                for value in (
                    _integer(raw_usage, "cache_creation_input_tokens"),
                    _integer(raw_usage, "cache_read_input_tokens"),
                )
            )
    completion = _integer(raw_usage, "completion_tokens", "output_tokens")
    cached = _integer(raw_usage, "cached_tokens", "cached_input_tokens", "cache_read_input_tokens")
    if cached is None:
        prompt_details = raw_usage.get("prompt_tokens_details")
        if isinstance(prompt_details, Mapping):
            cached = _integer(prompt_details, "cached_tokens")
    metrics: Dict[str, Any] = {}
    if prompt is not None:
        metrics["prompt_tokens"] = prompt
    if completion is not None:
        metrics["completion_tokens"] = completion
    if cached is not None:
        metrics["cached_tokens"] = cached
    metrics["extra"] = {"provider_usage": dict(raw_usage)}
    return metrics


def _integer(values: Mapping[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = values.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return None


def _insert_bootstrap_retrieval(
    steps: List[Dict[str, Any]], retrieval_log: Sequence[Mapping[str, Any]], *, stage: str
) -> None:
    bootstrap = next((entry for entry in retrieval_log if entry.get("query_type") == "main"), None)
    if bootstrap is None:
        return
    call_id = f"bootstrap-{uuid.uuid4().hex[:12]}"
    tool_name = str(bootstrap.get("tool_name") or "retrieve")
    arguments = bootstrap.get("input")
    if not isinstance(arguments, dict):
        arguments = {}
    output = bootstrap.get("output")
    summary = _retrieval_summary(output)
    step = {
        "step_id": 0,
        "timestamp": utc_timestamp(),
        "source": "agent",
        "message": f"Executed bootstrap {tool_name}",
        "tool_calls": [
            {
                "tool_call_id": call_id,
                "function_name": tool_name,
                "arguments": arguments,
            }
        ],
        "observation": {"results": [{"source_call_id": call_id, "content": summary}]},
        "llm_call_count": 0,
        "extra": {"stage": stage, "query_type": "main"},
    }
    insert_at = next((idx + 1 for idx, candidate in enumerate(steps) if candidate.get("source") == "user"), 1)
    steps.insert(insert_at, step)


def _retrieval_summary(raw_output: Any) -> str:
    if not isinstance(raw_output, list):
        return _bounded_text(str(raw_output or ""))
    documents: List[Dict[str, Any]] = []
    for raw_document in raw_output:
        if not isinstance(raw_document, Mapping):
            continue
        document: Dict[str, Any] = {"id": str(raw_document.get("id") or "")}
        score = raw_document.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            document["score"] = score
        if raw_document.get("note") is not None:
            document["note"] = _bounded_text(str(raw_document["note"]), limit=1_000)
        elif raw_document.get("text") is not None:
            document["text_preview"] = _bounded_text(str(raw_document["text"]), limit=1_000)
        documents.append(document)
    return _bounded_text(json.dumps({"documents": documents}, ensure_ascii=False, default=str))


def _final_metrics(steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prompt = 0
    completion = 0
    cached = 0
    has_prompt = False
    has_completion = False
    has_cached = False
    llm_calls = 0
    for step in steps:
        metrics = step.get("metrics")
        if isinstance(metrics, Mapping):
            value = _integer(metrics, "prompt_tokens")
            if value is not None:
                prompt += value
                has_prompt = True
            value = _integer(metrics, "completion_tokens")
            if value is not None:
                completion += value
                has_completion = True
            value = _integer(metrics, "cached_tokens")
            if value is not None:
                cached += value
                has_cached = True
        value = step.get("llm_call_count")
        if isinstance(value, int) and not isinstance(value, bool):
            llm_calls += value

    final: Dict[str, Any] = {
        "total_steps": len(steps),
        "extra": {"llm_call_count": llm_calls},
    }
    if has_prompt:
        final["total_prompt_tokens"] = prompt
    if has_completion:
        final["total_completion_tokens"] = completion
    if has_cached:
        final["total_cached_tokens"] = cached
    return final


def _renumber_steps(steps: Sequence[Dict[str, Any]]) -> None:
    for step_id, step in enumerate(steps, start=1):
        step["step_id"] = step_id


def _safe_filename(value: str) -> str:
    safe = _FILENAME_SAFE_RE.sub("_", value).strip("._")
    return safe[:80] or "unknown"
