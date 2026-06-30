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
from nemo_retriever.harness.json_io import jsonable, redact, write_json


ARTIFACT_NAMES = (
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


class ArtifactWriteError(RuntimeError):
    """An artifact directory could not be initialized or written safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(redact(jsonable(payload)), sort_keys=False) + "\n")


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


def initialize_artifact_dir(path: Path) -> Path:
    """Create an artifact directory, refusing to reuse non-empty output."""
    artifact_dir = path.expanduser().resolve()
    try:
        artifact_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        if not artifact_dir.is_dir():
            raise ArtifactWriteError(f"Artifact path exists and is not a directory: {artifact_dir}") from exc
        try:
            next(artifact_dir.iterdir())
        except StopIteration:
            pass
        except OSError as list_exc:
            raise ArtifactWriteError(f"Cannot inspect artifact directory {artifact_dir}: {list_exc}") from list_exc
        else:
            raise ArtifactWriteError(
                f"Artifact directory is not empty: {artifact_dir}. Choose a new --output-dir."
            ) from exc
    except OSError as exc:
        raise ArtifactWriteError(f"Cannot create artifact directory {artifact_dir}: {exc}") from exc
    return artifact_dir


class ArtifactWriter:
    def __init__(self, *, artifact_dir: Path, run_id: str, benchmark: str) -> None:
        self.artifact_dir = initialize_artifact_dir(artifact_dir)
        self.run_id = run_id
        self.benchmark = benchmark
        self.started_at = utc_now()
        self.events_path = self.artifact_dir / "events.jsonl"
        self._written_artifacts: set[str] = set()

    def path(self, name: str) -> Path:
        return self.artifact_dir / name

    def _write_error(self, name: str, exc: Exception) -> ArtifactWriteError:
        return ArtifactWriteError(f"Failed to write artifact {self.path(name)}: {exc}")

    def write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        try:
            write_json(self.path(name), payload)
        except Exception as exc:
            raise self._write_error(name, exc) from exc
        self._written_artifacts.add(name)
        return self.path(name)

    def append_jsonl(self, name: str, payload: Mapping[str, Any]) -> Path:
        try:
            append_jsonl(self.path(name), payload)
        except Exception as exc:
            raise self._write_error(name, exc) from exc
        self._written_artifacts.add(name)
        return self.path(name)

    def write_text(self, name: str, text: str) -> Path:
        try:
            path = self.path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except Exception as exc:
            raise self._write_error(name, exc) from exc
        self._written_artifacts.add(name)
        return self.path(name)

    def register_existing(self, name: str) -> None:
        if self.path(name).exists():
            self._written_artifacts.add(name)

    @contextlib.contextmanager
    def capture_output(self, name: str, *, label: str):
        body_error: BaseException | None = None
        try:
            with capture_output_to_log(self.path(name), label=label):
                try:
                    yield
                except BaseException as exc:
                    body_error = exc
                    raise
        except OSError as exc:
            if exc is body_error:
                raise
            raise self._write_error(name, exc) from exc
        finally:
            self.register_existing(name)

    def artifact_paths(self, *, pending: tuple[str, ...] = ()) -> dict[str, str]:
        current_run = self._written_artifacts | set(pending)
        return {name: str(self.path(name)) for name in ARTIFACT_NAMES if name in current_run}

    def event(self, phase: str, event: str, message: str, data: Mapping[str, Any] | None = None) -> None:
        self.append_jsonl(
            "events.jsonl",
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
        self.write_json("status.json", payload)
        self.event(phase, f"status_{status}", f"status={status} phase={phase}")
        return payload


def artifact_paths(writer: ArtifactWriter, *, pending: tuple[str, ...] = ()) -> dict[str, str]:
    return writer.artifact_paths(pending=pending)
