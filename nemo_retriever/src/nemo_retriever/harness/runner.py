# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner agent that registers with a harness portal manager and sends heartbeats."""

from __future__ import annotations

import collections
import json as json_module
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import typer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_json(url: str, data: dict[str, Any] | None, method: str, timeout: int = 10) -> dict[str, Any]:
    body = json_module.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json_module.loads(resp.read().decode("utf-8"))


def _post_json(url: str, data: dict[str, Any], timeout: int = 10) -> dict[str, Any]:
    return _http_json(url, data, "POST", timeout)


def _put_json(url: str, data: dict[str, Any], timeout: int = 10) -> dict[str, Any]:
    return _http_json(url, data, "PUT", timeout)


def _get_json(url: str, timeout: int = 10) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json_module.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path | None:
    """Walk up from this file to find the nearest .git directory."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _git_checkout_commit(commit: str, ref: str | None = None) -> str | None:
    """Fetch the latest refs and check out a specific commit.

    Returns the previous HEAD SHA so we can restore it afterwards,
    or ``None`` if the checkout failed.
    """
    repo_root = _find_repo_root()
    if repo_root is None:
        logger.warning("Cannot find git repo root — skipping checkout")
        return None

    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"

    def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=120,
            check=check,
            env=env,
        )

    try:
        prev = _run_git("rev-parse", "HEAD").stdout.strip()
    except Exception:
        prev = None

    try:
        _run_git("fetch", "--all", "--prune", check=False)
        logger.info("Checking out commit %s in %s", commit[:12], repo_root)
        _run_git("checkout", commit)
        actual = _run_git("rev-parse", "HEAD").stdout.strip()
        logger.info("HEAD is now at %s", actual[:12])
        return prev
    except Exception as exc:
        logger.error("Git checkout of %s failed: %s", commit, exc)
        return None


def _git_restore(prev_ref: str | None) -> None:
    """Restore the working tree to the previous HEAD after a job finishes."""
    if prev_ref is None:
        return
    repo_root = _find_repo_root()
    if repo_root is None:
        return
    try:
        subprocess.run(
            ["git", "checkout", prev_ref],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        logger.info("Restored HEAD to %s", prev_ref[:12])
    except Exception as exc:
        logger.warning("Failed to restore HEAD to %s: %s", prev_ref, exc)


# ---------------------------------------------------------------------------
# Job tracker — shared state between heartbeat loop and job thread
# ---------------------------------------------------------------------------

_LOG_TAIL_MAX = 500


class _JobTracker:
    """Thread-safe tracker for the currently executing job."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.job_id: str | None = None
        self.log_lines: collections.deque[str] = collections.deque(maxlen=_LOG_TAIL_MAX)
        self.cancel_requested: bool = False

    def start_job(self, job_id: str) -> None:
        with self._lock:
            self.job_id = job_id
            self.log_lines.clear()
            self.cancel_requested = False

    def finish_job(self) -> None:
        with self._lock:
            self.job_id = None
            self.cancel_requested = False

    def request_cancel(self) -> None:
        with self._lock:
            self.cancel_requested = True

    def is_cancel_requested(self) -> bool:
        with self._lock:
            return self.cancel_requested

    def add_log(self, text: str) -> None:
        with self._lock:
            for line in text.splitlines():
                stripped = line.rstrip()
                if stripped:
                    self.log_lines.append(stripped)

    def get_log_tail(self, count: int = 200) -> list[str]:
        with self._lock:
            items = list(self.log_lines)
            return items[-count:]

    def get_current_job_id(self) -> str | None:
        with self._lock:
            return self.job_id


_job_tracker = _JobTracker()


class _TeeWriter:
    """Wraps a file-like writer, teeing output to the job tracker log buffer."""

    def __init__(self, original: Any) -> None:
        self._original = original

    def write(self, text: str) -> int:
        result = self._original.write(text)
        _job_tracker.add_log(text)
        return result if result is not None else len(text)

    def flush(self) -> None:
        self._original.flush()

    def fileno(self) -> int:
        return self._original.fileno()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def _kill_child_processes() -> None:
    """Send SIGTERM to all child processes of the current process."""
    try:
        import psutil

        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        _, alive = psutil.wait_procs(children, timeout=10)
        for child in alive:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        logger.warning("psutil not available; falling back to pkill for child cleanup")
        try:
            subprocess.run(
                ["pkill", "-TERM", "-P", str(os.getpid())],
                capture_output=True,
                timeout=5,
                check=False,
            )
            time.sleep(5)
            subprocess.run(
                ["pkill", "-KILL", "-P", str(os.getpid())],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            pass
    except Exception as exc:
        logger.warning("Failed to kill child processes: %s", exc)


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def _validate_dataset_path(job: dict[str, Any]) -> str | None:
    """Check if the dataset directory exists locally.

    Returns an error message if the path is missing, or ``None`` if OK.
    """
    dataset_path = job.get("dataset_path")
    if not dataset_path:
        overrides = job.get("dataset_overrides") or {}
        dataset_path = overrides.get("dataset_dir")
    if dataset_path and os.path.isabs(dataset_path) and not os.path.isdir(dataset_path):
        return f"Dataset directory does not exist: {dataset_path}"
    return None


def _execute_job_on_runner(base_url: str, job: dict[str, Any], runner_id: int = 0) -> None:
    """Claim a job, execute it locally, and report results back."""
    job_id = job["id"]

    dataset_error = _validate_dataset_path(job)
    if dataset_error:
        logger.warning("Rejecting job %s — %s", job_id, dataset_error)
        try:
            reject_rid = job.get("assigned_runner_id") or runner_id
            _post_json(
                f"{base_url}/api/jobs/{job_id}/reject",
                {"runner_id": reject_rid, "reason": dataset_error},
            )
        except Exception as exc:
            logger.error("Failed to reject job %s: %s", job_id, exc)
        return

    try:
        _post_json(f"{base_url}/api/jobs/{job_id}/claim", {})
    except Exception as exc:
        logger.warning("Failed to claim job %s: %s", job_id, exc)
        return

    _job_tracker.start_job(job_id)

    git_commit = job.get("git_commit")
    git_ref = job.get("git_ref")
    prev_head: str | None = None

    if not git_commit and job.get("trigger_source") == "scheduled":
        git_commit = "origin/main"
        git_ref = "main"
        logger.info("Job %s is scheduled — will pull latest main before running", job_id)

    if git_commit:
        logger.info("Job %s requests commit %s (ref=%s) — pulling latest code", job_id, git_commit[:12], git_ref)
        prev_head = _git_checkout_commit(git_commit, git_ref)

    dataset_value = job.get("dataset_path") or job["dataset"]
    overrides = job.get("dataset_overrides") or {}
    logger.info(
        "Executing job %s (dataset=%s, path=%s, preset=%s)",
        job_id,
        job.get("dataset"),
        dataset_value,
        job.get("preset"),
    )

    original_stdout = sys.stdout
    sys.stdout = _TeeWriter(original_stdout)
    try:
        from nemo_retriever.harness.run import _run_entry

        result = _run_entry(
            run_name=None,
            config_file=job.get("config"),
            session_dir=None,
            dataset=dataset_value,
            preset=job.get("preset"),
            sweep_overrides=overrides if overrides else None,
            tags=job.get("tags"),
            skip_local_history=True,
        )

        if _job_tracker.is_cancel_requested():
            _post_json(
                f"{base_url}/api/jobs/{job_id}/complete",
                {"success": False, "error": "Cancelled by user", "result": result},
            )
            logger.info("Job %s cancelled by user", job_id)
        else:
            success = bool(result.get("success"))
            _post_json(f"{base_url}/api/jobs/{job_id}/complete", {"success": success, "result": result})
            logger.info("Job %s completed (success=%s)", job_id, success)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Job %s failed: %s\n%s", job_id, exc, tb)
        try:
            error_msg = f"{exc}\n\n{tb}"
            if _job_tracker.is_cancel_requested():
                error_msg = "Cancelled by user"
            _post_json(f"{base_url}/api/jobs/{job_id}/complete", {"success": False, "error": error_msg})
        except Exception:
            pass
    finally:
        sys.stdout = original_stdout
        _job_tracker.finish_job()
        if prev_head:
            _git_restore(prev_head)


# ---------------------------------------------------------------------------
# Runner main loop
# ---------------------------------------------------------------------------


def _build_registration_payload(
    runner_name: str, meta: dict[str, Any], tags: list[str], heartbeat_interval: int = 30,
) -> dict[str, Any]:
    """Build the JSON payload used to register (or re-register) with the portal."""
    return {
        "name": runner_name,
        "hostname": meta.get("host"),
        "gpu_type": meta.get("gpu_type"),
        "gpu_count": meta.get("gpu_count"),
        "cpu_count": meta.get("cpu_count"),
        "memory_gb": meta.get("memory_gb"),
        "status": "online",
        "tags": tags,
        "heartbeat_interval": heartbeat_interval,
        "metadata": {
            "cuda_driver": meta.get("cuda_driver"),
            "ray_version": meta.get("ray_version"),
            "python_version": meta.get("python_version"),
        },
    }


def _register_with_portal(base_url: str, payload: dict[str, Any]) -> int | None:
    """Register this runner with the portal and return the assigned runner ID."""
    try:
        result = _post_json(f"{base_url}/api/runners", payload)
        return result.get("id")
    except Exception as exc:
        logger.warning("Registration failed: %s", exc)
        return None


def runner_start_command(
    name: str | None = typer.Option(None, "--name", help="Runner name. Defaults to hostname."),
    manager_url: str | None = typer.Option(None, "--manager-url", help="Portal URL to register this runner with."),
    heartbeat_interval: int = typer.Option(30, "--heartbeat-interval", help="Heartbeat interval in seconds."),
    tag: list[str] = typer.Option([], "--tag", help="Runner tags. Repeatable."),
) -> None:
    """Start a harness runner and optionally register with a portal manager."""
    from nemo_retriever.harness.run import _collect_run_metadata

    meta = _collect_run_metadata()
    runner_name = name or meta.get("host", "unknown")

    typer.echo(f"Runner: {runner_name}")
    typer.echo(f"  Hostname : {meta.get('host')}")
    typer.echo(f"  CPU      : {meta.get('cpu_count') or 'N/A'} cores")
    typer.echo(f"  Memory   : {meta.get('memory_gb') or 'N/A'} GB")
    typer.echo(f"  GPU      : {meta.get('gpu_type') or 'N/A'} (x{meta.get('gpu_count') or 0})")
    typer.echo(f"  Python   : {meta.get('python_version')}")

    runner_id: int | None = None
    base_url: str | None = None
    reg_payload: dict[str, Any] | None = None

    if manager_url:
        base_url = manager_url.rstrip("/")
        reg_payload = _build_registration_payload(runner_name, meta, tag or [], heartbeat_interval)
        typer.echo(f"\nRegistering with {base_url} ...")
        runner_id = _register_with_portal(base_url, reg_payload)
        if runner_id is not None:
            typer.echo(f"Registered as runner #{runner_id}")
        else:
            typer.echo("Warning: Failed to register — will retry on next heartbeat.", err=True)
    else:
        typer.echo("\nNo --manager-url provided; running in standalone mode.")

    typer.echo(f"\nRunner is active (heartbeat every {heartbeat_interval}s). Press Ctrl+C to stop.\n")

    stop = False
    active_job_thread: threading.Thread | None = None

    def _handle_signal(sig: int, frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop:
            time.sleep(heartbeat_interval)
            if not base_url:
                continue

            # If we don't have a runner_id yet, try to register.
            if runner_id is None and reg_payload:
                runner_id = _register_with_portal(base_url, reg_payload)
                if runner_id is not None:
                    logger.info("Registered with portal as runner #%s", runner_id)
                continue

            if runner_id is None:
                continue

            heartbeat_job = None

            hb_payload: dict[str, Any] = {}
            current_jid = _job_tracker.get_current_job_id()
            if current_jid:
                hb_payload["current_job_id"] = current_jid
                hb_payload["log_tail"] = _job_tracker.get_log_tail(200)

            try:
                hb_resp = _post_json(f"{base_url}/api/runners/{runner_id}/heartbeat", hb_payload)
                if hb_resp and hb_resp.get("job"):
                    heartbeat_job = hb_resp["job"]
                cancel_id = hb_resp.get("cancel_job_id") if hb_resp else None
                if cancel_id and cancel_id == current_jid:
                    logger.info("Cancel requested for job %s — killing child processes", cancel_id)
                    _job_tracker.request_cancel()
                    _kill_child_processes()
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and reg_payload:
                    logger.warning(
                        "Portal returned 404 for runner #%s — re-registering",
                        runner_id,
                    )
                    runner_id = _register_with_portal(base_url, reg_payload)
                    if runner_id is not None:
                        logger.info("Re-registered as runner #%s", runner_id)
                else:
                    logger.debug("Heartbeat HTTP error %s — portal may be restarting", exc.code)
                continue
            except Exception as exc:
                logger.debug("Heartbeat failed (%s) — portal may be restarting", exc)
                continue

            if active_job_thread is None or not active_job_thread.is_alive():
                active_job_thread = None
                work = heartbeat_job
                if not work:
                    try:
                        work = _get_json(f"{base_url}/api/runners/{runner_id}/work")
                    except urllib.error.HTTPError:
                        work = None
                    except Exception as exc:
                        logger.debug("Work poll error: %s", exc)
                        work = None
                if work and work.get("id"):
                    active_job_thread = threading.Thread(
                        target=_execute_job_on_runner,
                        args=(base_url, work, runner_id),
                        daemon=True,
                    )
                    active_job_thread.start()
    finally:
        if base_url and runner_id:
            typer.echo("\nDeregistering runner...")
            try:
                _put_json(f"{base_url}/api/runners/{runner_id}", {"status": "offline"})
            except Exception:
                pass
        typer.echo("Runner stopped.")
