# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project ATIF trajectories into documents for standard text ingestion."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote


@dataclass(frozen=True)
class TrajectoryDocument:
    """One text document and its retrieval metadata."""

    filename: str
    text: str
    metadata: dict[str, Any]


def project_agent_trajectory(
    trajectory: Mapping[str, Any],
    *,
    exclude_tool_names: Sequence[str] = (),
    tool_output_char_limit: int | None = None,
) -> list[TrajectoryDocument]:
    """Convert one ATIF trajectory into independently ingestible text documents."""
    session_id, steps = _validate_trajectory(trajectory)
    excluded = _validate_options(exclude_tool_names, tool_output_char_limit)
    documents: list[TrajectoryDocument] = []
    seen_step_ids: set[str] = set()

    for step_index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise TypeError(f"trajectory.steps[{step_index}] must be a mapping")
        raw_step_id = step.get("step_id", step_index)
        if raw_step_id is None or str(raw_step_id) == "":
            raise ValueError(
                f"trajectory.steps[{step_index}].step_id must not be empty"
            )
        step_id = str(raw_step_id)
        if step_id in seen_step_ids:
            raise ValueError(
                f"trajectory step_id values must be unique; found {step_id!r}"
            )
        seen_step_ids.add(step_id)

        source = str(step.get("source") or "")
        if source == "user":
            _append_text_event(
                documents,
                trajectory,
                step,
                session_id,
                step_id,
                "message",
                step.get("message"),
                role="user",
            )
        elif source in {"agent", "assistant"}:
            _append_text_event(
                documents,
                trajectory,
                step,
                session_id,
                step_id,
                "message",
                step.get("message"),
                role="assistant",
            )
            _append_text_event(
                documents,
                trajectory,
                step,
                session_id,
                step_id,
                "reasoning",
                step.get("reasoning_content"),
                role="assistant",
            )
            _append_observations(
                documents,
                trajectory,
                step,
                session_id,
                step_id,
                excluded,
                tool_output_char_limit,
            )

    return [
        TrajectoryDocument(
            filename=f"trajectory-{index:06d}.txt",
            text=document.text,
            metadata=document.metadata,
        )
        for index, document in enumerate(documents)
    ]


def decode_trajectory(content: bytes) -> Mapping[str, Any]:
    """Decode an uploaded UTF-8 JSON trajectory."""
    try:
        raw = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("trajectory is not valid UTF-8") from exc
    try:
        trajectory = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("trajectory is not valid JSON") from exc
    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory JSON must decode to a mapping")
    return trajectory


def _validate_trajectory(
    trajectory: Mapping[str, Any],
) -> tuple[str, Sequence[Any]]:
    if not isinstance(trajectory, Mapping):
        raise TypeError(
            f"trajectory must be a mapping, got {type(trajectory).__name__}"
        )
    session_id = trajectory.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("trajectory.session_id must be a non-empty string")
    steps = trajectory.get("steps")
    if not isinstance(steps, Sequence) or isinstance(
        steps, (str, bytes, bytearray)
    ):
        raise TypeError("trajectory.steps must be a sequence")
    return session_id, steps


def _validate_options(
    exclude_tool_names: Sequence[str],
    tool_output_char_limit: int | None,
) -> frozenset[str]:
    if isinstance(exclude_tool_names, (str, bytes)) or not isinstance(
        exclude_tool_names, Sequence
    ):
        raise TypeError("exclude_tool_names must be a sequence of strings")
    if not all(isinstance(name, str) for name in exclude_tool_names):
        raise TypeError("exclude_tool_names must contain only strings")
    if tool_output_char_limit is not None and (
        isinstance(tool_output_char_limit, bool)
        or not isinstance(tool_output_char_limit, int)
        or tool_output_char_limit < 0
    ):
        raise ValueError(
            "tool_output_char_limit must be a non-negative integer or None"
        )
    return frozenset(exclude_tool_names)


def _text_content(value: Any, location: str) -> str | None:
    """Normalize ATIF string or content-part-array text; ignore non-text parts."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{location} must be a string or content-part array")

    text_parts: list[str] = []
    for index, part in enumerate(value):
        if not isinstance(part, Mapping):
            raise TypeError(f"{location}[{index}] must be a mapping")
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if not isinstance(text, str):
            raise TypeError(f"{location}[{index}].text must be a string")
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def _append_text_event(
    documents: list[TrajectoryDocument],
    trajectory: Mapping[str, Any],
    step: Mapping[str, Any],
    session_id: str,
    step_id: str,
    event_type: str,
    value: Any,
    *,
    role: str,
) -> None:
    text = _text_content(value, f"trajectory step {step_id!r} {event_type}")
    if text is None or not text.strip():
        return
    documents.append(
        _document(
            trajectory,
            step,
            session_id,
            step_id,
            event_type,
            text,
            role=role,
        )
    )


def _append_observations(
    documents: list[TrajectoryDocument],
    trajectory: Mapping[str, Any],
    step: Mapping[str, Any],
    session_id: str,
    step_id: str,
    excluded: frozenset[str],
    char_limit: int | None,
) -> None:
    observation = step.get("observation")
    if observation is None:
        return
    if not isinstance(observation, Mapping):
        raise TypeError(
            f"trajectory step {step_id!r} observation must be a mapping"
        )
    results = observation.get("results", ())
    if not isinstance(results, Sequence) or isinstance(
        results, (str, bytes, bytearray)
    ):
        raise TypeError(
            f"trajectory step {step_id!r} observation.results must be a sequence"
        )

    tool_calls = _tool_calls_by_call_id(step, step_id)
    for result_index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise TypeError(
                f"trajectory step {step_id!r} observation result "
                f"{result_index} must be a mapping"
            )
        call_id = str(result.get("source_call_id") or "")
        tool_name, tool_arguments = tool_calls.get(call_id, ("", None))
        if tool_name in excluded:
            continue
        content = _text_content(
            result.get("content"),
            f"trajectory step {step_id!r} observation result "
            f"{result_index} content",
        )
        if content is None:
            continue
        original_chars = len(content)
        if char_limit is not None:
            content = content[:char_limit]
        if not content.strip():
            continue
        truncated = len(content) < original_chars
        if tool_name:
            arguments = json.dumps(
                {} if tool_arguments is None else tool_arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            content = f"{tool_name}({arguments}) -> {content}"
        event_id = f"observation:{call_id or result_index}:{result_index}"
        documents.append(
            _document(
                trajectory,
                step,
                session_id,
                step_id,
                event_id,
                content,
                role="tool",
                tool_name=tool_name,
                tool_call_id=call_id or None,
                truncated=truncated,
                original_chars=original_chars,
            )
        )


def _tool_calls_by_call_id(
    step: Mapping[str, Any], step_id: str
) -> dict[str, tuple[str, Any]]:
    raw_calls = step.get("tool_calls", ())
    if not isinstance(raw_calls, Sequence) or isinstance(
        raw_calls, (str, bytes, bytearray)
    ):
        raise TypeError(
            f"trajectory step {step_id!r} tool_calls must be a sequence"
        )
    calls: dict[str, tuple[str, Any]] = {}
    for index, call in enumerate(raw_calls):
        if not isinstance(call, Mapping):
            raise TypeError(
                f"trajectory step {step_id!r} tool call {index} must be a mapping"
            )
        call_id = str(call.get("tool_call_id") or "")
        if call_id:
            calls[call_id] = (
                str(call.get("function_name") or ""),
                call.get("arguments"),
            )
    return calls


def _document(
    trajectory: Mapping[str, Any],
    step: Mapping[str, Any],
    session_id: str,
    step_id: str,
    event_id: str,
    text: str,
    **event_metadata: Any,
) -> TrajectoryDocument:
    source_path = (
        f"agent-trajectory://session/{quote(session_id, safe='')}"
        f"/step/{quote(step_id, safe='')}/event/{quote(event_id, safe='')}"
    )
    metadata: dict[str, Any] = {
        "subtype": "agent_trajectory",
        "source_path": source_path,
        "session_id": session_id,
        "step_id": step.get("step_id", step_id),
        "event_id": event_id,
        "event_type": event_id.split(":", 1)[0],
    }
    if step.get("timestamp") is not None:
        metadata["timestamp"] = step["timestamp"]
    metadata.update(
        {key: value for key, value in event_metadata.items() if value is not None}
    )
    metadata.update(
        {
            key: trajectory[key]
            for key in ("schema_version", "agent")
            if key in trajectory
        }
    )
    metadata.update(
        {
            key: step[key]
            for key in ("source", "metrics", "extra")
            if key in step
        }
    )
    return TrajectoryDocument(filename="", text=text, metadata=metadata)
