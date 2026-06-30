# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

from nemo_retriever.harness.contracts import FailurePayload, PHASE_VALUES, STATUS_VALUES
from nemo_retriever.harness.json_io import jsonable, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=False) + "\n")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


@contextlib.contextmanager
def capture_output_to_log(path: Path, *, label: str):
    """Capture noisy stdout/stderr to a persistent run log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    append_text(path, f"\n## {utc_now()} {label} start\n")
    try:
        stdout_fd, stderr_fd = sys.stdout.fileno(), sys.stderr.fileno()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        append_text(path, "stdio capture unavailable in this runtime\n")
        yield
        append_text(path, f"## {utc_now()} {label} complete\n")
        return

    saved_stdout = saved_stderr = buf = None
    failed = False
    try:
        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)
        buf = tempfile.TemporaryFile(mode="w+b")
        try:
            try:
                os.dup2(buf.fileno(), stdout_fd)
                os.dup2(buf.fileno(), stderr_fd)
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_stdout, stdout_fd)
                os.dup2(saved_stderr, stderr_fd)
        except BaseException:
            failed = True
            raise
        finally:
            if buf is not None:
                buf.seek(0)
                captured = buf.read()
                with path.open("ab") as handle:
                    if captured:
                        handle.write(captured)
                        if not captured.endswith(b"\n"):
                            handle.write(b"\n")
                    handle.write(f"## {utc_now()} {label} {'failed' if failed else 'complete'}\n".encode("utf-8"))
                if failed and captured:
                    sys.stderr.buffer.write(captured)
                    sys.stderr.flush()
    finally:
        if buf is not None:
            buf.close()
        if saved_stderr is not None:
            os.close(saved_stderr)
        if saved_stdout is not None:
            os.close(saved_stdout)


_SENSITIVE_KEY_PARTS = ("api_key", "password", "secret", "credential", "token")


def _is_sensitive_key(value: Any) -> bool:
    normalized = str(value).lower()
    return any(token in normalized for token in _SENSITIVE_KEY_PARTS)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            if _is_sensitive_key(key):
                out[key] = "<redacted>" if nested else nested
            else:
                out[key] = redact(nested)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if isinstance(value, str) and "=" in value:
        key, _separator, _raw = value.partition("=")
        if _is_sensitive_key(key):
            return f"{key}=<redacted>"
    return value


class ArtifactWriter:
    def __init__(self, *, artifact_dir: Path, run_id: str, benchmark: str) -> None:
        self.artifact_dir = artifact_dir.expanduser().resolve()
        self.run_id = run_id
        self.benchmark = benchmark
        self.started_at = utc_now()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.artifact_dir / "events.jsonl"
        if self.events_path.exists():
            self.events_path.unlink()

    def path(self, name: str) -> Path:
        return self.artifact_dir / name

    def event(self, phase: str, event: str, message: str, data: Mapping[str, Any] | None = None) -> None:
        append_jsonl(
            self.events_path,
            {
                "time": utc_now(),
                "run_id": self.run_id,
                "benchmark": self.benchmark,
                "phase": phase,
                "event": event,
                "message": message,
                "data": dict(data or {}),
            },
        )

    def status(
        self,
        *,
        status: str,
        phase: str,
        failure: FailurePayload | None = None,
        summary_metrics_path: Path | None = None,
    ) -> dict[str, Any]:
        if status not in STATUS_VALUES:
            raise ValueError(f"Invalid status: {status}")
        if phase not in PHASE_VALUES:
            raise ValueError(f"Invalid phase: {phase}")
        payload = {
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "status": status,
            "phase": phase,
            "started_at": self.started_at,
            "updated_at": utc_now(),
            "artifact_dir": str(self.artifact_dir),
            "summary_metrics_path": str(summary_metrics_path) if summary_metrics_path is not None else None,
            "failure": failure.to_dict() if failure is not None else None,
        }
        write_json(self.path("status.json"), payload)
        self.event(phase, f"status_{status}", f"status={status} phase={phase}")
        return payload


def artifact_paths(writer: ArtifactWriter) -> dict[str, str]:
    names = (
        "status.json",
        "events.jsonl",
        "resolved_benchmark.json",
        "ingest_plan.json",
        "query_plan.json",
        "summary_metrics.json",
        "environment.json",
        "runfile.json",
        "results.json",
        "run.log",
        "beir_metrics.json",
        "beir_run.trec",
        "query_results.jsonl",
    )
    return {name: str(writer.path(name)) for name in names if writer.path(name).exists() or name == "results.json"}
