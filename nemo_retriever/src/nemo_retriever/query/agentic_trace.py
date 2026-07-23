# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

logger = logging.getLogger(__name__)

TRACE_FILE_NAME = "agentic_trace.jsonl"
_TRACE_ROOT = Path.cwd() / "artifacts" / "agentic_traces"
_TRACE_TEXT_LIMIT = 4096
_TRACE_COLLECTION_LIMIT = 100


def make_agentic_trace_path(root: str | Path | None = None) -> Path:
    """Return a unique artifact trace path for one root CLI agentic query run."""
    trace_root = Path(root).expanduser() if root is not None else _TRACE_ROOT
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = f"{stamp}_{os.getpid()}_{uuid4().hex[:8]}"
    return trace_root / run_dir / TRACE_FILE_NAME


class AgenticTraceJsonlFormatter(logging.Formatter):
    """Format one structured agentic trace event as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "agentic_trace_payload", {})
        event = {
            "time": datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "sequence": getattr(record, "agentic_trace_sequence", 0),
            **dict(payload),
        }
        return json.dumps(_jsonable(event), sort_keys=False)


class AgenticTraceJsonlHandler(logging.Handler):
    """Write formatted trace records to one JSONL artifact."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(level=logging.INFO)
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        self._warned = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sequence += 1
            record.agentic_trace_sequence = self._sequence
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(self.format(record) + "\n")
        except Exception as exc:  # tracing must not fail retrieval
            if not self._warned:
                logger.warning("Failed to write agentic trace event to %s: %s", self.path, exc, exc_info=True)
                self._warned = True


def make_agentic_trace_logger(path: str | Path) -> logging.Logger:
    """Create an isolated logger that writes structured events to ``path``."""
    trace_logger = logging.Logger(f"{__name__}.{uuid4().hex}", level=logging.INFO)
    trace_logger.propagate = False
    handler = AgenticTraceJsonlHandler(path)
    handler.setFormatter(AgenticTraceJsonlFormatter())
    trace_logger.addHandler(handler)
    return trace_logger


def log_agentic_trace(trace_logger: logging.Logger, payload: Mapping[str, Any]) -> None:
    """Submit one structured event to an agentic trace logger."""
    trace_logger.info(
        str(payload.get("event", "agentic_trace")),
        extra={"agentic_trace_payload": dict(payload)},
    )


def bind_trace_emitter(
    trace_event: Callable[[Mapping[str, Any]], None] | None,
    *,
    operator: str,
) -> Callable[..., None]:
    """Bind shared best-effort emission logic to one operator."""

    def emit(event: str, **payload: Any) -> None:
        if trace_event is None:
            return
        try:
            trace_event({"event": event, "operator": operator, **payload})
        except Exception as exc:
            logger.warning("%s trace event failed: %s", operator, exc, exc_info=True)

    return emit


def trace_documents(
    docs: Sequence[Mapping[str, Any]],
    *,
    id_fields: tuple[str, ...] = ("doc_id", "id"),
) -> list[dict[str, Any]]:
    """Shape documents consistently for structured trace events."""
    return [
        {
            "rank": rank,
            "doc_id": str(next((doc.get(field) for field in id_fields if doc.get(field) is not None), "")),
            "score": doc.get("score"),
            "text": str(doc.get("text", "")),
        }
        for rank, doc in enumerate(docs, start=1)
    ]


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
        out = {str(key): _jsonable(nested) for key, nested in items[:_TRACE_COLLECTION_LIMIT] if nested is not None}
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
