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


_ARTIFACT_UPLOAD_EXCLUDES = {"lancedb"}


def _upload_artifacts(base_url: str, run_id: int, artifact_dir: str, timeout: int = 120) -> None:
    """Zip the artifact directory (excluding large data like lancedb) and upload to the portal."""
    import io as _io
    import zipfile as _zipfile

    art_path = Path(artifact_dir)
    if not art_path.is_dir():
        logger.warning("Artifact directory %s does not exist — skipping upload", artifact_dir)
        return

    buf = _io.BytesIO()
    with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(art_path.rglob("*")):
            if fp.is_file() and not any(excl in fp.parts for excl in _ARTIFACT_UPLOAD_EXCLUDES):
                zf.write(fp, fp.relative_to(art_path))
    raw = buf.getvalue()
    logger.info("Uploading %d bytes of artifacts for run %d", len(raw), run_id)

    boundary = f"----RunnerUpload{run_id}"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="artifacts.zip"\r\n'
            f"Content-Type: application/zip\r\n\r\n"
        ).encode("utf-8")
        + raw
        + f"\r\n--{boundary}--\r\n".encode("utf-8")
    )

    url = f"{base_url}/api/runs/{run_id}/upload-artifacts"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json_module.loads(resp.read().decode("utf-8"))
    logger.info("Artifact upload complete for run %d: %s", run_id, result)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _get_routable_ip() -> str:
    """Return this machine's routable IP address (not 127.0.0.1)."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def _resolve_ray_address(addr: str | None) -> str | None:
    """Rewrite a Ray address so loopback/localhost becomes the routable IP."""
    if not addr:
        return addr
    raw = addr.strip()
    if raw.lower() in ("auto", "local"):
        return raw

    prefix = ""
    rest = raw
    if rest.lower().startswith("ray://"):
        prefix = "ray://"
        rest = rest[6:]

    host, _, port = rest.partition(":")
    if host.lower() in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
        host = _get_routable_ip()

    return f"{prefix}{host}:{port}" if port else f"{prefix}{host}"


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
        if commit.startswith("refs/pull/"):
            # PR ref from GitHub (refs/pull/<n>/head) — fetch via origin into a
            # temporary local ref then checkout by the resolved SHA.
            _run_git("fetch", "origin", commit, check=True)
            sha = _run_git("rev-parse", "FETCH_HEAD").stdout.strip()
            logger.info("Checking out PR ref %s (%s) in %s", commit, sha[:12], repo_root)
            _run_git("checkout", sha)
        else:
            if "/" in commit and not commit.startswith("origin/"):
                remote_name = commit.split("/")[0]
                _run_git("fetch", remote_name, "--prune", check=False)
            _run_git("fetch", "--all", "--prune", check=False)
            logger.info("Checking out %s in %s", commit, repo_root)
            _run_git("checkout", commit)
        actual = _run_git("rev-parse", "HEAD").stdout.strip()
        logger.info("HEAD is now at %s", actual[:12])
        return prev
    except Exception as exc:
        logger.error("Git checkout of %s failed: %s", commit, exc)
        return None


def _get_current_git_commit() -> str | None:
    """Return the full SHA of the current HEAD, or None if not in a git repo."""
    repo_root = _find_repo_root()
    if repo_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


_UPDATE_MARKER_FILE = Path("/tmp/.nemo_runner_update_marker")


def _write_update_marker(previous_commit: str, new_commit: str) -> None:
    """Write a marker file so the restarted process knows it came from a portal update.

    Also persists runtime state (ray_address, run_code_ref) that may have been
    configured via the portal and would otherwise be lost across the restart.
    """
    try:
        _UPDATE_MARKER_FILE.write_text(
            json_module.dumps(
                {
                    "previous_commit": previous_commit,
                    "new_commit": new_commit,
                    "ts": time.time(),
                    "ray_address": _runner_ray_address,
                    "run_code_ref": _runner_run_code_ref,
                    "num_gpus": _runner_num_gpus,
                }
            ),
        )
    except Exception as exc:
        logger.warning("Failed to write update marker: %s", exc)


def _read_and_clear_update_marker() -> dict[str, Any] | None:
    """Read the update marker if present and delete it. Returns the marker dict or None."""
    try:
        if _UPDATE_MARKER_FILE.exists():
            data = json_module.loads(_UPDATE_MARKER_FILE.read_text())
            _UPDATE_MARKER_FILE.unlink(missing_ok=True)
            return data
    except Exception as exc:
        logger.warning("Failed to read update marker: %s", exc)
        _UPDATE_MARKER_FILE.unlink(missing_ok=True)
    return None


def _report_update_to_portal(base_url: str, runner_id: int, marker: dict[str, Any]) -> None:
    """Notify the portal that this runner restarted after a code update."""
    try:
        _post_json(
            f"{base_url}/api/runners/{runner_id}/update-complete",
            {
                "previous_commit": marker.get("previous_commit"),
                "new_commit": marker.get("new_commit"),
            },
        )
        logger.info(
            "Reported successful update to portal: %s → %s",
            (marker.get("previous_commit") or "?")[:12],
            (marker.get("new_commit") or "?")[:12],
        )
    except Exception as exc:
        logger.warning("Failed to report update to portal: %s", exc)


def _self_update_and_restart(commit: str, base_url: str, runner_id: int, reg_payload: dict[str, Any]) -> None:
    """Checkout the requested commit, reinstall the package, and restart this process."""
    repo_root = _find_repo_root()
    if repo_root is None:
        logger.error("Cannot find git repo root — skipping self-update")
        return

    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"

    def _run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            **kwargs,
        )

    previous_commit = _get_current_git_commit() or "unknown"
    logger.info("Self-update requested: updating to %s (from %s)", commit[:12], previous_commit[:12])

    try:
        _run(["git", "fetch", "--all", "--prune"], check=False)
        result = _run(["git", "checkout", commit], check=True)
        logger.info("Checked out %s", commit[:12])
    except Exception as exc:
        logger.error("Git checkout failed: %s", exc)
        return

    nemo_retriever_dir = repo_root / "nemo_retriever"
    if not nemo_retriever_dir.exists():
        logger.error("nemo_retriever directory not found at %s", nemo_retriever_dir)
        return

    logger.info("Running uv pip install -e ./nemo_retriever ...")
    try:
        result = _run(["uv", "pip", "install", "-e", "./nemo_retriever"], check=True)
        if result.stdout:
            logger.info("pip install stdout: %s", result.stdout[:500])
        if result.stderr:
            logger.info("pip install stderr: %s", result.stderr[:500])
    except FileNotFoundError:
        logger.info("uv not found, falling back to pip install -e ./nemo_retriever")
        try:
            _run([sys.executable, "-m", "pip", "install", "-e", "./nemo_retriever"], check=True)
        except Exception as exc:
            logger.error("pip install failed: %s", exc)
            return
    except Exception as exc:
        logger.error("uv pip install failed: %s", exc)
        return

    _write_update_marker(previous_commit, commit)
    logger.info("Self-update complete. Restarting runner process...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


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
_runner_ray_address: str | None = None
_runner_run_code_ref: str | None = None
_runner_num_gpus: int | None = None


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


def _is_playground_job(job: dict[str, Any]) -> bool:
    return job.get("trigger_source") == "playground"


def _download_playground_files(base_url: str, job: dict[str, Any]) -> str | None:
    """Download playground session files from the portal and return the local directory path.

    Returns the local path on success, or None on failure.
    """
    dataset_name = job.get("dataset") or ""
    if not dataset_name.startswith("playground_"):
        return None
    session_id = dataset_name[len("playground_") :]
    if not session_id:
        return None

    import tempfile
    import zipfile

    local_dir = Path(tempfile.gettempdir()) / "harness_playground_uploads" / session_id
    if local_dir.is_dir() and any(local_dir.iterdir()):
        logger.info("Playground session %s already cached at %s", session_id, local_dir)
        return str(local_dir)

    url = f"{base_url}/api/playground/sessions/{session_id}/download"
    logger.info("Downloading playground files for session %s from %s", session_id, url)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=120) as resp:
            zip_bytes = resp.read()
    except Exception as exc:
        logger.error("Failed to download playground session %s: %s", session_id, exc)
        return None

    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        import io

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(local_dir)
        logger.info("Extracted %d files to %s", len(list(local_dir.iterdir())), local_dir)
        return str(local_dir)
    except Exception as exc:
        logger.error("Failed to extract playground zip for session %s: %s", session_id, exc)
        return None


def _validate_dataset_path(job: dict[str, Any]) -> str | None:
    """Check if the dataset directory exists locally.

    Returns an error message if the path is missing, or ``None`` if OK.
    Playground jobs are skipped since their files are downloaded separately.
    """
    if _is_playground_job(job):
        return None
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

    if not git_commit and _runner_run_code_ref:
        git_commit = _runner_run_code_ref
        git_ref = _runner_run_code_ref.split("/")[-1] if "/" in _runner_run_code_ref else _runner_run_code_ref
        logger.info("Job %s — using portal run_code_ref: %s", job_id, _runner_run_code_ref)
    elif not git_commit and job.get("trigger_source") == "scheduled":
        git_commit = "origin/main"
        git_ref = "main"
        logger.info("Job %s is scheduled — will pull latest main before running", job_id)

    if git_commit:
        logger.info("Job %s requests commit %s (ref=%s) — pulling latest code", job_id, git_commit[:12], git_ref)
        prev_head = _git_checkout_commit(git_commit, git_ref)

    extra_packages = job.get("extra_packages") or []
    if extra_packages:
        logger.info("Job %s — installing extra packages: %s", job_id, extra_packages)
        try:
            import shutil
            uv_bin = shutil.which("uv")
            if uv_bin:
                cmd = [uv_bin, "pip", "install", "--python", sys.executable, *extra_packages]
            else:
                cmd = [sys.executable, "-m", "pip", "install", *extra_packages]
            subprocess.run(cmd, check=True, timeout=300)
        except subprocess.CalledProcessError as exc:
            logger.error("Job %s — extra_packages install failed: %s", job_id, exc)
            try:
                _post_json(
                    f"{base_url}/api/jobs/{job_id}/complete",
                    {"success": False, "error": f"extra_packages install failed: {exc}"},
                )
            except Exception:
                pass
            _job_tracker.finish_job()
            if prev_head:
                _git_restore(prev_head)
            return

    execution_commit = _get_current_git_commit()

    dataset_value = job.get("dataset_path") or job["dataset"]
    overrides = job.get("dataset_overrides") or {}

    if _is_playground_job(job):
        local_dir = _download_playground_files(base_url, job)
        if local_dir:
            dataset_value = local_dir
            overrides["dataset_dir"] = local_dir
            logger.info("Playground job %s: using local dataset dir %s", job_id, local_dir)
        else:
            _post_json(
                f"{base_url}/api/jobs/{job_id}/complete",
                {"success": False, "error": "Failed to download playground files from portal"},
            )
            _job_tracker.finish_job()
            return

    if _runner_ray_address and "ray_address" not in overrides:
        overrides["ray_address"] = _runner_ray_address
        logger.info("Injecting runner ray_address=%s into job overrides", _runner_ray_address)
    overrides.setdefault("write_detection_file", True)
    logger.info(
        "Executing job %s (dataset=%s, path=%s, preset=%s, ray=%s)",
        job_id,
        job.get("dataset"),
        dataset_value,
        job.get("preset"),
        overrides.get("ray_address", "local"),
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
            complete_resp = _post_json(
                f"{base_url}/api/jobs/{job_id}/complete",
                {
                    "success": False,
                    "error": "Cancelled by user",
                    "result": result,
                    "execution_commit": execution_commit,
                    "num_gpus": _runner_num_gpus,
                },
            )
            logger.info("Job %s cancelled by user", job_id)
        else:
            success = bool(result.get("success"))
            complete_resp = _post_json(
                f"{base_url}/api/jobs/{job_id}/complete",
                {
                    "success": success,
                    "result": result,
                    "execution_commit": execution_commit,
                    "num_gpus": _runner_num_gpus,
                },
            )
            logger.info("Job %s completed (success=%s)", job_id, success)

        resp_run_id = complete_resp.get("run_id") if isinstance(complete_resp, dict) else None
        if resp_run_id and result:
            artifacts = result.get("artifacts") or {}
            art_dir = artifacts.get("runtime_metrics_dir")
            if art_dir:
                art_dir = str(Path(art_dir).parent)
            else:
                cmd_file = artifacts.get("command_file", "")
                art_dir = str(Path(cmd_file).parent) if cmd_file else None
            if art_dir:
                try:
                    _upload_artifacts(base_url, resp_run_id, art_dir)
                except Exception as upload_exc:
                    logger.warning("Failed to upload artifacts for run %d: %s", resp_run_id, upload_exc)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Job %s failed: %s\n%s", job_id, exc, tb)
        try:
            error_msg = f"{exc}\n\n{tb}"
            if _job_tracker.is_cancel_requested():
                error_msg = "Cancelled by user"
            _post_json(
                f"{base_url}/api/jobs/{job_id}/complete",
                {
                    "success": False,
                    "error": error_msg,
                    "execution_commit": execution_commit,
                    "num_gpus": _runner_num_gpus,
                },
            )
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
    runner_name: str,
    meta: dict[str, Any],
    tags: list[str],
    heartbeat_interval: int = 30,
    ray_address: str | None = None,
    num_gpus: int | None = None,
) -> dict[str, Any]:
    """Build the JSON payload used to register (or re-register) with the portal."""
    return {
        "name": runner_name,
        "hostname": meta.get("host"),
        "gpu_type": meta.get("gpu_type"),
        "gpu_count": num_gpus if num_gpus is not None else meta.get("gpu_count"),
        "cpu_count": meta.get("cpu_count"),
        "memory_gb": meta.get("memory_gb"),
        "status": "online",
        "tags": tags,
        "heartbeat_interval": heartbeat_interval,
        "git_commit": _get_current_git_commit(),
        "ray_address": _resolve_ray_address(ray_address),
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
    ray_address: str | None = typer.Option(
        None,
        "--ray-address",
        help="Ray cluster address for this runner (e.g. 'auto', 'ray://host:10001'). Omit for local Ray.",
    ),
    num_gpus: int | None = typer.Option(
        None,
        "--num-gpus",
        help="Number of GPUs to report for this runner. Overrides auto-detected count.",
    ),
) -> None:
    """Start a harness runner and optionally register with a portal manager."""
    from nemo_retriever.harness.run import _collect_run_metadata

    meta = _collect_run_metadata()
    runner_name = name or meta.get("host", "unknown")

    current_commit = _get_current_git_commit()

    typer.echo(f"Runner: {runner_name}")
    typer.echo(f"  Hostname : {meta.get('host')}")
    typer.echo(f"  CPU      : {meta.get('cpu_count') or 'N/A'} cores")
    typer.echo(f"  Memory   : {meta.get('memory_gb') or 'N/A'} GB")
    typer.echo(f"  GPU      : {meta.get('gpu_type') or 'N/A'} (x{meta.get('gpu_count') or 0})")
    typer.echo(f"  Python   : {meta.get('python_version')}")
    typer.echo(f"  Git      : {current_commit[:12] if current_commit else 'unknown'}")
    typer.echo(f"  Ray      : {ray_address or 'local (embedded)'}")

    global _runner_ray_address  # noqa: PLW0603
    _runner_ray_address = _resolve_ray_address(ray_address)

    global _runner_run_code_ref  # noqa: PLW0603

    global _runner_num_gpus  # noqa: PLW0603
    _runner_num_gpus = num_gpus if num_gpus is not None else meta.get("gpu_count")
    typer.echo(f"  Num GPUs : {_runner_num_gpus or 'auto'}")

    update_marker = _read_and_clear_update_marker()
    if update_marker:
        saved_ray = update_marker.get("ray_address")
        if saved_ray:
            _runner_ray_address = saved_ray
            typer.echo(f"  Ray (restored from update): {_runner_ray_address}")
        saved_ref = update_marker.get("run_code_ref")
        if saved_ref:
            _runner_run_code_ref = saved_ref
        saved_gpus = update_marker.get("num_gpus")
        if saved_gpus is not None:
            _runner_num_gpus = saved_gpus

    runner_id: int | None = None
    base_url: str | None = None
    reg_payload: dict[str, Any] | None = None

    if manager_url:
        base_url = manager_url.rstrip("/")
        reg_payload = _build_registration_payload(
            runner_name,
            meta,
            tag or [],
            heartbeat_interval,
            ray_address=_runner_ray_address,
            num_gpus=_runner_num_gpus,
        )
        typer.echo(f"\nRegistering with {base_url} ...")
        runner_id = _register_with_portal(base_url, reg_payload)
        if runner_id is not None:
            typer.echo(f"Registered as runner #{runner_id}")
            if update_marker:
                typer.echo(
                    f"  Restarted after portal-triggered update: "
                    f"{(update_marker.get('previous_commit') or '?')[:12]} → "
                    f"{(update_marker.get('new_commit') or '?')[:12]}"
                )
                _report_update_to_portal(base_url, runner_id, update_marker)
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

            hb_payload: dict[str, Any] = {
                "git_commit": _get_current_git_commit(),
            }
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
                    reg_payload["git_commit"] = _get_current_git_commit()
                    runner_id = _register_with_portal(base_url, reg_payload)
                    if runner_id is not None:
                        logger.info("Re-registered as runner #%s", runner_id)
                else:
                    logger.debug("Heartbeat HTTP error %s — portal may be restarting", exc.code)
                continue
            except Exception as exc:
                logger.debug("Heartbeat failed (%s) — portal may be restarting", exc)
                continue

            if hb_resp and "ray_address" in hb_resp:
                portal_ray_addr = _resolve_ray_address(hb_resp["ray_address"])
                if portal_ray_addr != _runner_ray_address:
                    _runner_ray_address = portal_ray_addr
                    logger.info("Ray address updated from portal: %s", _runner_ray_address or "local")

            if hb_resp and "run_code_ref" in hb_resp:
                new_ref = hb_resp["run_code_ref"]
                if new_ref != _runner_run_code_ref:
                    _runner_run_code_ref = new_ref
                    logger.info("Run code ref updated from portal: %s", _runner_run_code_ref)

            update_commit = hb_resp.get("update_to_commit") if hb_resp else None
            if update_commit:
                current_sha = _get_current_git_commit()
                if current_sha and current_sha.startswith(update_commit[:7]):
                    logger.info("Already at requested commit %s — skipping update", update_commit[:12])
                elif active_job_thread is not None and active_job_thread.is_alive():
                    logger.info("Update to %s pending — waiting for current job to finish", update_commit[:12])
                else:
                    _self_update_and_restart(update_commit, base_url, runner_id, reg_payload)

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
