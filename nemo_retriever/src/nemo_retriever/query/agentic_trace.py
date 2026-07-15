# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Mapping
from uuid import uuid4

logger = logging.getLogger(__name__)

TRACE_FILE_NAME = "agentic_trace.jsonl"
_TRACE_ROOT = Path.cwd() / "artifacts" / "agentic_traces"
_TRACE_TEXT_LIMIT = 4096
_TRACE_COLLECTION_LIMIT = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def make_agentic_trace_path(root: str | Path | None = None) -> Path:
    """Return a unique artifact trace path for one root CLI agentic query run."""
    trace_root = Path(root).expanduser() if root is not None else _TRACE_ROOT
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = f"{stamp}_{os.getpid()}_{uuid4().hex[:8]}"
    return trace_root / run_dir / TRACE_FILE_NAME


class AgenticTraceWriter:
    """Append best-effort, structured agentic events as JSONL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._lock = threading.Lock()
        self._sequence = 0
        self._warned = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, payload: Mapping[str, Any]) -> None:
        try:
            self._append(payload)
        except Exception as exc:  # trace failures should not fail retrieval
            if not self._warned:
                logger.warning("Failed to write agentic trace event to %s: %s", self.path, exc, exc_info=True)
                self._warned = True

    def _append(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            event = {
                "time": utc_now(),
                "sequence": self._sequence,
                **dict(payload),
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_jsonable(event), sort_keys=False) + "\n")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= _TRACE_TEXT_LIMIT:
            return value
        omitted = len(value) - _TRACE_TEXT_LIMIT
        return f"{value[:_TRACE_TEXT_LIMIT]}...[truncated {omitted} chars]"
    if isinstance(value, Mapping):
        items = list(value.items())
        out = {str(key): _jsonable(nested) for key, nested in items[:_TRACE_COLLECTION_LIMIT]}
        if len(items) > _TRACE_COLLECTION_LIMIT:
            out["_truncated_items"] = len(items) - _TRACE_COLLECTION_LIMIT
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        out = [_jsonable(item) for item in items[:_TRACE_COLLECTION_LIMIT]]
        if len(items) > _TRACE_COLLECTION_LIMIT:
            out.append({"_truncated_items": len(items) - _TRACE_COLLECTION_LIMIT})
        return out
    return str(value)
