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
import re
import shutil
import sys
import tempfile
import traceback
from typing import Any, Mapping

from nemo_retriever.harness.contracts import FailurePayload, PHASE_VALUES, STATUS_VALUES
from nemo_retriever.harness.json_io import artifact_write_error, jsonable, write_json

_ARTIFACT_NAMES = {
    "status": "status.json",
    "events": "events.jsonl",
    "environment": "environment.json",
    "runfile": "runfile.json",
    "resolved_benchmark": "resolved_benchmark.json",
    "ingest_plan": "ingest_plan.json",
    "query_plan": "query_plan.json",
    "log": "run.log",
    "query_results": "query_results.jsonl",
    "beir_metrics": "beir_metrics.json",
    "beir_run": "beir_run.trec",
    "lancedb": "lancedb",
}
_LEGACY_ARTIFACT_NAMES = ("summary_metrics.json",)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(payload), sort_keys=False) + "\n")
    except Exception as exc:
        raise artifact_write_error(exc) from exc


def append_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:
        raise artifact_write_error(exc) from exc


@contextlib.contextmanager
def capture_output_to_log(path: Path, *, label: str):
    """Capture noisy stdout/stderr to a persistent run log."""
    append_text(path, f"\n## {utc_now()} {label} start\n")
    try:
        stdout_fd, stderr_fd = sys.stdout.fileno(), sys.stderr.fileno()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        append_text(path, "stdio capture unavailable in this runtime\n")
        try:
            yield
        except BaseException:
            append_text(path, traceback.format_exc())
            append_text(path, f"## {utc_now()} {label} failed\n")
            raise
        else:
            append_text(path, f"## {utc_now()} {label} complete\n")
        return

    saved_stdout = saved_stderr = buf = None
    failed = False
    failure_traceback: str | None = None
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
            failure_traceback = traceback.format_exc()
            raise
        finally:
            if buf is not None:
                buf.seek(0)
                captured = buf.read()
                try:
                    with path.open("ab") as handle:
                        if captured:
                            handle.write(captured)
                            if not captured.endswith(b"\n"):
                                handle.write(b"\n")
                        if failure_traceback:
                            handle.write(failure_traceback.encode("utf-8", errors="replace"))
                            if not failure_traceback.endswith("\n"):
                                handle.write(b"\n")
                        handle.write(f"## {utc_now()} {label} {'failed' if failed else 'complete'}\n".encode("utf-8"))
                except OSError as exc:
                    raise artifact_write_error(exc) from exc
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


_SENSITIVE_KEY_MARKERS = ("api_key", "password", "secret", "credential", "webhook")
_TOKEN_CREDENTIAL_QUALIFIERS = {"access", "api", "auth", "bearer", "oauth", "refresh", "service", "session"}


def _is_sensitive_key(value: Any) -> bool:
    snake_case = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    normalized = re.sub(r"[^a-z0-9]+", "_", snake_case.lower()).strip("_")
    if any(marker in normalized for marker in _SENSITIVE_KEY_MARKERS):
        return True
    parts = normalized.split("_")
    if "token" in parts:
        return True
    return any(
        part == "tokens" and index > 0 and parts[index - 1] in _TOKEN_CREDENTIAL_QUALIFIERS
        for index, part in enumerate(parts)
    )


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
        key, _separator, raw = value.partition("=")
        if _is_sensitive_key(key):
            return f"{key}=<redacted>"
        try:
            structured_value = json.loads(raw)
        except json.JSONDecodeError:
            return value
        redacted_value = redact(structured_value)
        if redacted_value != structured_value:
            return f"{key}={json.dumps(redacted_value, sort_keys=True)}"
    return value


class ArtifactWriter:
    def __init__(self, *, artifact_dir: Path, run_id: str, benchmark: str) -> None:
        self.artifact_dir = artifact_dir.expanduser().resolve()
        self.run_id = run_id
        self.benchmark = benchmark
        self.started_at = utc_now()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        for name in (*_ARTIFACT_NAMES.values(), *_LEGACY_ARTIFACT_NAMES, "results.json"):
            path = self.artifact_dir / name
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
        self.events_path = self.artifact_dir / "events.jsonl"

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
        results_path: Path | None = None,
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
            "results_path": str(results_path.relative_to(self.artifact_dir)) if results_path is not None else None,
            "failure": failure.to_dict() if failure is not None else None,
        }
        write_json(self.path("status.json"), payload)
        self.event(phase, f"status_{status}", f"status={status} phase={phase}")
        return payload


def artifact_paths(writer: ArtifactWriter) -> dict[str, str]:
    return {key: name for key, name in _ARTIFACT_NAMES.items() if writer.path(name).exists()}
