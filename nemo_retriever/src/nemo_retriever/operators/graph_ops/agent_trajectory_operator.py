# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project canonical ATIF agent trajectories into text-document rows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

import pandas as pd

from nemo_retriever.graph.designer import designer_component
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator

_OUTPUT_COLUMNS = ["path", "bytes", "text", "metadata"]
_ASSISTANT_SOURCES = frozenset(("agent", "assistant"))


def project_agent_trajectory(
    trajectory: Mapping[str, Any],
    *,
    exclude_tool_names: Sequence[str] = (),
    tool_output_char_limit: int | None = None,
) -> pd.DataFrame:
    """Project one ATIF-style trajectory into independently addressable text rows.

    User and assistant messages, assistant reasoning, and tool observations
    become UTF-8 documents. Empty textual events are omitted.
    """
    session_id, steps = _validate_trajectory(trajectory)
    excluded = _validate_options(exclude_tool_names, tool_output_char_limit)
    rows: list[dict[str, Any]] = []
    seen_step_ids: set[str] = set()

    for step_index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise TypeError(f"trajectory.steps[{step_index}] must be a mapping")
        raw_step_id = step.get("step_id", step_index)
        if raw_step_id is None or str(raw_step_id) == "":
            raise ValueError(f"trajectory.steps[{step_index}].step_id must not be empty")
        step_id = str(raw_step_id)
        if step_id in seen_step_ids:
            raise ValueError(f"trajectory step_id values must be unique; found {step_id!r}")
        seen_step_ids.add(step_id)

        source = str(step.get("source") or "")
        if source == "user":
            _append_text_event(
                rows,
                trajectory,
                step,
                session_id,
                step_id,
                "message",
                step.get("message"),
                role="user",
            )
        elif source in _ASSISTANT_SOURCES:
            _append_text_event(
                rows,
                trajectory,
                step,
                session_id,
                step_id,
                "message",
                step.get("message"),
                role="assistant",
            )
            _append_text_event(
                rows,
                trajectory,
                step,
                session_id,
                step_id,
                "reasoning",
                step.get("reasoning_content"),
                role="assistant",
            )
            _append_observations(
                rows,
                trajectory,
                step,
                session_id,
                step_id,
                excluded,
                tool_output_char_limit,
            )

    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)


@designer_component(
    name="Agent Trajectory Projection",
    category="Text & Content",
    compute="cpu",
    description="Projects ATIF agent sessions into text documents",
    category_color="#42d6a4",
)
class AgentTrajectoryProjectionOperator(AbstractOperator, CPUOperator):
    """CPU operator for projecting ATIF mappings or uploaded JSON rows."""

    def __init__(
        self,
        *,
        exclude_tool_names: Sequence[str] = (),
        tool_output_char_limit: int | None = None,
    ) -> None:
        excluded = _validate_options(exclude_tool_names, tool_output_char_limit)
        super().__init__(
            exclude_tool_names=tuple(sorted(excluded)),
            tool_output_char_limit=tool_output_char_limit,
        )
        self.exclude_tool_names = tuple(sorted(excluded))
        self.tool_output_char_limit = tool_output_char_limit

    def preprocess(self, data: Any, **kwargs: Any) -> Mapping[str, Any] | pd.DataFrame:
        if not isinstance(data, (Mapping, pd.DataFrame)):
            raise TypeError(f"data must be an ATIF trajectory mapping or DataFrame, got {type(data).__name__}")
        return data

    def process(self, data: Mapping[str, Any] | pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        trajectories = [data] if isinstance(data, Mapping) else self._decode_dataframe(data)
        projected = [
            project_agent_trajectory(
                trajectory,
                exclude_tool_names=self.exclude_tool_names,
                tool_output_char_limit=self.tool_output_char_limit,
            )
            for trajectory in trajectories
        ]
        nonempty = [frame for frame in projected if not frame.empty]
        if not nonempty:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)
        return pd.concat(nonempty, ignore_index=True)

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data

    @staticmethod
    def _decode_dataframe(data: pd.DataFrame) -> list[Mapping[str, Any]]:
        trajectories: list[Mapping[str, Any]] = []
        for row_index, row in data.iterrows():
            raw = row.get("bytes")
            if isinstance(raw, (bytes, bytearray)):
                try:
                    raw = bytes(raw).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"trajectory row {row_index!r} is not valid UTF-8") from exc
            if raw is None:
                raw = row.get("text")
            if not isinstance(raw, str):
                raise TypeError(f"trajectory row {row_index!r} must contain bytes or text")
            try:
                trajectory = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"trajectory row {row_index!r} is not valid JSON") from exc
            if not isinstance(trajectory, Mapping):
                raise TypeError(f"trajectory row {row_index!r} JSON must decode to a mapping")
            trajectories.append(trajectory)
        return trajectories


def _validate_trajectory(trajectory: Mapping[str, Any]) -> tuple[str, Sequence[Any]]:
    if not isinstance(trajectory, Mapping):
        raise TypeError(f"trajectory must be a mapping, got {type(trajectory).__name__}")
    session_id = trajectory.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("trajectory.session_id must be a non-empty string")
    steps = trajectory.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)):
        raise TypeError("trajectory.steps must be a sequence")
    return session_id, steps


def _validate_options(exclude_tool_names: Sequence[str], tool_output_char_limit: int | None) -> frozenset[str]:
    if isinstance(exclude_tool_names, (str, bytes)) or not isinstance(exclude_tool_names, Sequence):
        raise TypeError("exclude_tool_names must be a sequence of strings")
    if not all(isinstance(name, str) for name in exclude_tool_names):
        raise TypeError("exclude_tool_names must contain only strings")
    if tool_output_char_limit is not None and (
        isinstance(tool_output_char_limit, bool)
        or not isinstance(tool_output_char_limit, int)
        or tool_output_char_limit < 0
    ):
        raise ValueError("tool_output_char_limit must be a non-negative integer or None")
    return frozenset(exclude_tool_names)


def _append_text_event(
    rows: list[dict[str, Any]],
    trajectory: Mapping[str, Any],
    step: Mapping[str, Any],
    session_id: str,
    step_id: str,
    event_type: str,
    value: Any,
    *,
    role: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"trajectory step {step_id!r} {event_type} must be a string")
    if not value.strip():
        return
    rows.append(
        _document_row(
            trajectory,
            step,
            session_id,
            step_id,
            event_type,
            value,
            role=role,
        )
    )


def _append_observations(
    rows: list[dict[str, Any]],
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
        raise TypeError(f"trajectory step {step_id!r} observation must be a mapping")
    results = observation.get("results", ())
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes, bytearray)):
        raise TypeError(f"trajectory step {step_id!r} observation.results must be a sequence")

    tool_names = _tool_names_by_call_id(step, step_id)
    for result_index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise TypeError(f"trajectory step {step_id!r} observation result {result_index} must be a mapping")
        call_id = str(result.get("source_call_id") or "")
        tool_name = tool_names.get(call_id)
        if tool_name in excluded:
            continue
        content = result.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            raise TypeError(f"trajectory step {step_id!r} observation result {result_index} content must be a string")
        original_chars = len(content)
        if char_limit is not None:
            content = content[:char_limit]
        if not content.strip():
            continue
        event_id = f"observation:{call_id or result_index}:{result_index}"
        rows.append(
            _document_row(
                trajectory,
                step,
                session_id,
                step_id,
                event_id,
                content,
                role="tool",
                tool_name=tool_name,
                tool_call_id=call_id or None,
                truncated=len(content) < original_chars,
                original_chars=original_chars,
            )
        )


def _tool_names_by_call_id(step: Mapping[str, Any], step_id: str) -> dict[str, str]:
    raw_calls = step.get("tool_calls", ())
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes, bytearray)):
        raise TypeError(f"trajectory step {step_id!r} tool_calls must be a sequence")
    names: dict[str, str] = {}
    for index, call in enumerate(raw_calls):
        if not isinstance(call, Mapping):
            raise TypeError(f"trajectory step {step_id!r} tool call {index} must be a mapping")
        call_id = str(call.get("tool_call_id") or "")
        if call_id:
            names[call_id] = str(call.get("function_name") or "")
    return names


def _document_row(
    trajectory: Mapping[str, Any],
    step: Mapping[str, Any],
    session_id: str,
    step_id: str,
    event_id: str,
    text: str,
    **event_metadata: Any,
) -> dict[str, Any]:
    path = (
        f"agent-trajectory://session/{quote(session_id, safe='')}"
        f"/step/{quote(step_id, safe='')}/event/{quote(event_id, safe='')}"
    )
    content_metadata = {
        "type": "text",
        "subtype": "agent_trajectory",
        "session_id": session_id,
        "step_id": step.get("step_id", step_id),
        "event_type": event_id.split(":", 1)[0],
    }
    content_metadata.update({key: value for key, value in event_metadata.items() if value is not None})
    trajectory_metadata = {key: trajectory[key] for key in ("schema_version", "agent") if key in trajectory}
    trajectory_metadata.update({key: step[key] for key in ("timestamp", "source", "metrics", "extra") if key in step})
    return {
        "path": path,
        "bytes": text.encode("utf-8"),
        "text": text,
        "metadata": {
            "source_path": path,
            "content_metadata": content_metadata,
            "agent_trajectory": trajectory_metadata,
        },
    }
