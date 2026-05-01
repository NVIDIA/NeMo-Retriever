# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner agent that registers with a harness portal manager and sends heartbeats."""

from __future__ import annotations

import collections
import json as json_module
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
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


_ARTIFACT_MAX_UPLOAD_MB = 500


def _upload_artifacts(base_url: str, run_id: int, artifact_dir: str, timeout: int = 120) -> None:
    """Zip the artifact directory (excluding large data like lancedb) and upload to the portal."""
    import io as _io
    import zipfile as _zipfile

    art_path = Path(artifact_dir)
    if not art_path.is_dir():
        logger.warning("Artifact directory %s does not exist — skipping upload", artifact_dir)
        return

    has_nsys = any(art_path.rglob("*.nsys-rep"))

    buf = _io.BytesIO()
    with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(art_path.rglob("*")):
            if fp.is_file() and not any(excl in fp.parts for excl in _ARTIFACT_UPLOAD_EXCLUDES):
                file_mb = fp.stat().st_size / (1024 * 1024)
                if file_mb > _ARTIFACT_MAX_UPLOAD_MB:
                    logger.warning(
                        "Skipping large file %s (%.1f MB > %d MB limit)",
                        fp.name,
                        file_mb,
                        _ARTIFACT_MAX_UPLOAD_MB,
                    )
                    continue
                compress = _zipfile.ZIP_STORED if fp.suffix == ".nsys-rep" else _zipfile.ZIP_DEFLATED
                zf.write(fp, fp.relative_to(art_path), compress_type=compress)
    raw = buf.getvalue()
    zip_mb = len(raw) / (1024 * 1024)
    logger.info("Uploading %.1f MB of artifacts for run %d", zip_mb, run_id)

    effective_timeout = max(timeout, 300) if has_nsys else timeout

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
    with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
        resp_data = json_module.loads(resp.read().decode("utf-8"))
    logger.info("Artifact upload complete for run %d: %s", run_id, resp_data)


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
# Per-job virtual environment
# ---------------------------------------------------------------------------

_VENV_BASE_DIR = Path("/tmp/.nemo_runner_venvs")

_JOB_WRAPPER_SCRIPT = """\
import json, sys, traceback, inspect

with open(sys.argv[1]) as f:
    args = json.load(f)

try:
    from nemo_retriever.harness.run import _run_entry
    kwargs = dict(
        run_name=args.get("run_name"),
        config_file=args.get("config_file"),
        session_dir=args.get("session_dir"),
        dataset=args.get("dataset"),
        preset=args.get("preset"),
        sweep_overrides=args.get("sweep_overrides"),
        tags=args.get("tags"),
        skip_local_history=args.get("skip_local_history", True),
        graph_code=args.get("graph_code"),
    )
    sig = inspect.signature(_run_entry)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    result = _run_entry(**kwargs)
except Exception:
    traceback.print_exc()
    result = {
        "success": False,
        "failure_reason": traceback.format_exc().splitlines()[-1],
        "return_code": 1,
    }

with open(sys.argv[2], "w") as f:
    json.dump(result, f)
"""

_GRAPH_WRAPPER_SCRIPT = """\
import json, sys, os, traceback, time
from nemo_retriever.utils.hf_cache import collect_hf_runtime_env

graph_code_file = sys.argv[1]
result_file = sys.argv[2]
ray_address = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "__none__" else None

def _root_cause(exc):
    \"\"\"Walk the exception chain to find the original cause.\"\"\"
    seen = set()
    while exc.__cause__ is not None and id(exc.__cause__) not in seen:
        seen.add(id(exc))
        exc = exc.__cause__
    return exc

try:
    print(f"[diag] Python executable: {sys.executable}")
    print(f"[diag] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")

    import subprocess as _sp
    def _detect_gpu_count():
        try:
            out = _sp.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True, timeout=10,
            )
            return len([l for l in out.strip().splitlines() if l.strip()])
        except Exception:
            return 0

    import ray

    effective_ray = ray_address or os.environ.get("RAY_ADDRESS")
    is_local = effective_ray in ("auto", "local", None, "")

    ray.shutdown()

    venv = os.path.dirname(os.path.dirname(sys.executable))
    venv_bin = os.path.join(venv, "bin")
    pypath = os.pathsep.join(p for p in sys.path if p)
    ray_env_vars: dict[str, str] = {
        "VIRTUAL_ENV": venv,
        "PATH": venv_bin + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONPATH": pypath,
    }
    ray_env_vars.update(collect_hf_runtime_env())
    runtime_env = {"env_vars": ray_env_vars}

    if is_local:
        os.environ.pop("RAY_ADDRESS", None)
        detected_gpus = _detect_gpu_count()
        print(f"[diag] Starting fresh local Ray cluster (nvidia-smi detected {detected_gpus} GPU(s))")

        try:
            ray.init(
                num_gpus=detected_gpus if detected_gpus > 0 else None,
                runtime_env=runtime_env,
            )
        except ValueError as _ve:
            if "existing cluster" in str(_ve):
                print("[diag] Detected running Ray cluster — stopping it to start a fresh one")
                try:
                    _sp.run(["ray", "stop", "--force"], capture_output=True, timeout=30)
                except Exception:
                    pass
                ray.shutdown()
                try:
                    ray.init(
                        num_gpus=detected_gpus if detected_gpus > 0 else None,
                        runtime_env=runtime_env,
                    )
                except ValueError:
                    print("[diag] Still cannot start fresh cluster — connecting to existing one instead")
                    ray.init(runtime_env=runtime_env)
            else:
                raise
    else:
        print(f"[diag] Connecting to existing Ray cluster: {effective_ray}")
        ray.init(address=effective_ray, runtime_env=runtime_env)

    cluster_res = ray.cluster_resources()
    print(f"[diag] Ray cluster resources: {cluster_res}")
    print(f"[diag] Ray GPU count: {cluster_res.get('GPU', 0)}")

    import torch
    print(f"[diag] torch.version.cuda = {getattr(torch.version, 'cuda', 'N/A')}")
    print(f"[diag] torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"[diag] torch.cuda.device_count() = {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"[diag]   GPU {i}: {torch.cuda.get_device_name(i)}")

    with open(graph_code_file) as f:
        code = f.read()

    ns = {"__name__": "__graph_runner__", "__file__": graph_code_file}

    wall_start = time.perf_counter()
    exec(compile(code, graph_code_file, "exec"), ns)

    result_ds = ns.get("result")
    graph = ns.get("graph")
    _requested_plan = ns.get("requested_plan")

    ray_stats_str = None

    if result_ds is not None:
        elapsed = round(time.perf_counter() - wall_start, 2)
        try:
            import ray.data as _rd
            if isinstance(result_ds, _rd.Dataset):
                row_count = result_ds.count()
                print(f"Pipeline complete: {row_count} rows in {elapsed}s")
                try:
                    ray_stats_str = result_ds.stats()
                    print(f"\\n=== Ray Data Execution Stats ===\\n{ray_stats_str}")
                except Exception:
                    pass
                result = {"success": True, "return_code": 0, "rows": row_count, "elapsed_secs": elapsed}
            else:
                print(f"Pipeline complete in {elapsed}s (result type: {type(result_ds).__name__})")
                result = {"success": True, "return_code": 0, "elapsed_secs": elapsed}
        except Exception:
            print(f"Pipeline complete in {elapsed}s")
            result = {"success": True, "return_code": 0, "elapsed_secs": elapsed}
    elif graph is not None:
        outputs = graph.execute(None)
        elapsed = round(time.perf_counter() - wall_start, 2)
        print(f"Graph.execute complete: {len(outputs)} output(s) in {elapsed}s")
        result = {"success": True, "return_code": 0, "outputs": len(outputs), "elapsed_secs": elapsed}
    else:
        raise RuntimeError("Generated code did not produce a 'result' (Ray Data) or 'graph' variable")

    if _requested_plan is not None:
        result["requested_plan"] = _requested_plan
    if ray_stats_str is not None:
        result["ray_stats"] = ray_stats_str

except Exception as exc:
    full_tb = traceback.format_exc()
    print(full_tb, file=sys.stderr)
    print(full_tb)

    root = _root_cause(exc)
    root_msg = f"{type(root).__name__}: {root}"
    root_tb_lines = traceback.format_exception(type(root), root, root.__traceback__)
    root_tb = "".join(root_tb_lines)

    if root is not exc:
        print(f"\\n=== Root cause ===\\n{root_tb}")

    failure_lines = [root_msg]
    if len(full_tb) <= 4000:
        failure_lines.append(full_tb)
    else:
        failure_lines.append(full_tb[-4000:])

    result = {
        "success": False,
        "failure_reason": root_msg,
        "error_detail": "\\n".join(failure_lines),
        "return_code": 1,
    }

with open(result_file, "w") as f:
    json.dump(result, f)
"""


def _nsys_available() -> bool:
    """Return True if ``nsys`` is on PATH and report its version."""
    try:
        result = subprocess.run(["nsys", "--version"], capture_output=True, text=True, timeout=5)
        version = (result.stdout or result.stderr or "").strip()
        logger.info("nsys found: %s", version)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("nsys not available: %s", exc)
        return False


def _nsys_prefix(output_path: str) -> list[str]:
    """Return the command prefix to wrap a subprocess with nsys profile.

    Traces only CUDA kernels and NVTX annotations (no OS-runtime tracing,
    which was the main source of multi-GB reports).  The ``gpu_inference_range``
    NVTX markers placed around model invocations will appear as named regions
    in the Nsight Systems timeline for easy identification.

    Note: ``--capture-range=nvtx`` is intentionally NOT used because Ray Data
    runs model inference in separate worker processes that nsys cannot follow
    via fork-tracing.  Full-process capture with ``-t cuda,nvtx`` keeps report
    sizes manageable (typically tens of MB) while capturing all GPU activity.
    """
    return [
        "nsys",
        "profile",
        "-o",
        output_path,
        "--force-overwrite=true",
        "-t",
        "cuda,nvtx",
    ]


def _collect_nsys_report_info(nsys_output_dir: Path | None) -> dict[str, Any]:
    """Check for nsys report files and return diagnostic info.

    Returns a dict with ``found`` (bool), ``files`` (list of dicts with
    name/size_mb), and ``error`` (str or None) keys.
    """
    info: dict[str, Any] = {"found": False, "files": [], "error": None}
    if nsys_output_dir is None or not nsys_output_dir.exists():
        info["error"] = "nsys output directory does not exist"
        return info

    nsys_files = list(nsys_output_dir.glob("*.nsys-rep"))
    if not nsys_files:
        all_files = list(nsys_output_dir.iterdir())
        if all_files:
            info["error"] = (
                f"No .nsys-rep files found in {nsys_output_dir}. " f"Files present: {[f.name for f in all_files[:10]]}"
            )
        else:
            info["error"] = (
                "nsys produced no output files. The profiled process may have "
                "exited before nsys could finalize the report, or nsys encountered "
                "an internal error. Check the job logs for nsys warnings."
            )
        return info

    info["found"] = True
    for fp in nsys_files:
        size_mb = round(fp.stat().st_size / (1024 * 1024), 2)
        info["files"].append({"name": fp.name, "size_mb": size_mb})
        logger.info("nsys report: %s (%.2f MB)", fp.name, size_mb)
    return info


def _copy_nsys_profiles(src_dir: Path, dest_dir: Path) -> list[str]:
    """Copy any .nsys-rep files from *src_dir* into *dest_dir*.

    Returns list of copied file names.
    """
    copied: list[str] = []
    if not dest_dir.is_dir():
        logger.warning("nsys copy destination %s is not a directory", dest_dir)
        return copied
    for fp in src_dir.glob("*.nsys-rep"):
        try:
            shutil.copy2(fp, dest_dir / fp.name)
            logger.info("Copied nsys profile %s -> %s", fp, dest_dir / fp.name)
            copied.append(fp.name)
        except Exception as exc:
            logger.warning("Failed to copy nsys profile %s: %s", fp, exc)
    return copied


def _create_job_venv(job_id: str, repo_root: Path) -> Path | None:
    """Create a uv venv for *job_id* and install nemo_retriever into it.

    Uses ``uv sync`` so that ``[tool.uv.sources]`` (e.g. the CUDA torch
    index) are respected.  Falls back to ``uv pip install`` if the project
    has no lock-file or ``uv sync`` is unavailable.

    Returns the venv directory on success, or ``None`` on failure.
    """
    venv_dir = _VENV_BASE_DIR / job_id
    _VENV_BASE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[venv] Creating isolated uv venv for job {job_id} at {venv_dir} …")
        result = subprocess.run(
            ["uv", "venv", str(venv_dir), "--python", sys.executable],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        print(f"[venv] Created venv for job {job_id}")
        logger.info("Created venv for job %s at %s", job_id, venv_dir)
    except FileNotFoundError:
        logger.error("uv is not installed — cannot create job venv")
        return None
    except Exception as exc:
        logger.error("Failed to create venv for job %s: %s", job_id, exc)
        return None

    nemo_dir = repo_root / "nemo_retriever"
    use_sync = (nemo_dir / "pyproject.toml").exists()

    if use_sync:
        env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv_dir)}
        try:
            print("[venv] Running uv sync --all-extras (respects [tool.uv.sources] for CUDA torch) …")
            result = subprocess.run(
                ["uv", "sync", "--no-dev", "--all-extras"],
                cwd=str(nemo_dir),
                capture_output=True,
                text=True,
                check=True,
                timeout=900,
                env=env,
            )
            if result.stdout:
                for line in result.stdout.strip().splitlines()[-5:]:
                    logger.info("  sync: %s", line)
            print(f"[venv] uv sync complete for job {job_id}")
            logger.info("uv sync complete for job venv %s", job_id)
            return venv_dir
        except Exception as exc:
            stderr_text = getattr(exc, "stderr", "") or ""
            logger.warning(
                "uv sync failed for job %s, falling back to uv pip install: %s\n%s",
                job_id,
                exc,
                stderr_text[:1000],
            )
            print("[venv] uv sync failed, falling back to uv pip install …")

    venv_python = str(venv_dir / "bin" / "python")
    try:
        print("[venv] Running uv pip install -e './nemo_retriever[all]' …")
        result = subprocess.run(
            ["uv", "pip", "install", "-e", "./nemo_retriever[all]", "--python", venv_python],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        if result.stdout:
            for line in result.stdout.strip().splitlines()[-5:]:
                logger.info("  pip: %s", line)
        print(f"[venv] Installed nemo_retriever into job venv {job_id}")
        logger.info("Installed nemo_retriever into job venv %s", job_id)
    except Exception as exc:
        logger.error("Failed to install into job venv %s: %s", job_id, exc)
        if hasattr(exc, "stderr") and exc.stderr:
            logger.error("stderr: %s", exc.stderr[:1000])
        shutil.rmtree(venv_dir, ignore_errors=True)
        return None

    return venv_dir


def _capture_pip_list(job_id: str) -> str:
    """Run ``uv pip list`` in the job's venv and return the output as a string."""
    venv_dir = _VENV_BASE_DIR / job_id
    venv_python = str(venv_dir / "bin" / "python")
    if not venv_dir.exists():
        return ""
    try:
        result = subprocess.run(
            ["uv", "pip", "list", "--python", venv_python],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() if result.stdout else ""
    except Exception as exc:
        logger.warning("Failed to capture pip list for job %s: %s", job_id, exc)
        return ""


def _destroy_job_venv(job_id: str) -> None:
    """Remove the venv for a specific job (safe to call even if it doesn't exist)."""
    venv_dir = _VENV_BASE_DIR / job_id
    if venv_dir.exists():
        try:
            print(f"[venv] Deactivating and removing venv for job {job_id} at {venv_dir} …")
            shutil.rmtree(venv_dir)
            print(f"[venv] Removed venv for job {job_id}")
            logger.info("Removed venv for job %s", job_id)
        except Exception as exc:
            logger.warning("Failed to remove venv for job %s: %s", job_id, exc)


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

DATASET_CACHE_DIR: Path = Path(
    os.environ.get("HARNESS_DATASET_CACHE_DIR", str(Path.home() / ".cache" / "harness" / "datasets"))
)


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


def _build_cache_rewrites(cache_dir: Path, query_csv_bundled: bool) -> dict[str, Any]:
    """Build dataset_overrides pointing to a local cache directory."""
    rewrites: dict[str, Any] = {"dataset_dir": str(cache_dir)}
    if query_csv_bundled:
        bundled_dir = cache_dir / "_query_csv"
        if bundled_dir.is_dir():
            csv_files = list(bundled_dir.iterdir())
            if csv_files:
                rewrites["query_csv"] = str(csv_files[0])
    return rewrites


def _ensure_dataset_cached(base_url: str, job: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Download a distributable dataset from the portal if not already cached.

    Uses ``dataset_config_hash`` from the job payload for exact cache
    invalidation (covers file changes *and* config changes).  Falls back
    to the portal ``/hash`` endpoint for legacy jobs without the field.

    Returns ``(local_dir, override_rewrites)`` on success, or ``None`` if the
    dataset is not distributable (so the caller can fall through to the
    existing path-must-exist behaviour).
    """
    if _is_playground_job(job):
        return None

    dataset_name = job.get("dataset")
    if not dataset_name:
        return None

    cache_dir = DATASET_CACHE_DIR / dataset_name
    meta_file = cache_dir / ".dataset_meta.json"

    job_hash = job.get("dataset_config_hash")

    if cache_dir.is_dir() and meta_file.is_file():
        try:
            local_meta = json_module.loads(meta_file.read_text())
            cached_hash = local_meta.get("hash")
            if cached_hash:
                if job_hash:
                    if cached_hash == job_hash:
                        logger.info(
                            "Dataset %s cache hit (config hash match %s)",
                            dataset_name,
                            cached_hash[:12],
                        )
                        return str(cache_dir), _build_cache_rewrites(
                            cache_dir, local_meta.get("query_csv_bundled", False)
                        )
                    logger.info(
                        "Dataset %s cache stale (local %s != job %s) — re-downloading",
                        dataset_name,
                        cached_hash[:12],
                        job_hash[:12],
                    )
                else:
                    logger.info(
                        "Dataset %s cache hit at %s (hash %s, no job hash to compare)",
                        dataset_name,
                        cache_dir,
                        cached_hash[:12],
                    )
                    return str(cache_dir), _build_cache_rewrites(cache_dir, local_meta.get("query_csv_bundled", False))
        except Exception:
            pass

    if not job_hash:
        hash_url = f"{base_url}/api/managed-datasets/by-name/{urllib.request.quote(dataset_name, safe='')}/hash"
        try:
            hash_resp = _get_json(hash_url, timeout=15)
        except Exception:
            return None
        if not hash_resp or not isinstance(hash_resp, dict) or "hash" not in hash_resp:
            return None
        job_hash = hash_resp["hash"]

    dl_url = f"{base_url}/api/managed-datasets/by-name/{urllib.request.quote(dataset_name, safe='')}/download"
    logger.info("Downloading dataset %s from %s", dataset_name, dl_url)
    try:
        req = urllib.request.Request(dl_url)
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp_hash = resp.headers.get("X-Dataset-Hash", job_hash)
            query_csv_bundled = resp.headers.get("X-Query-Csv-Bundled", "false") == "true"
            content_length = resp.headers.get("Content-Length")
            total_size = int(content_length) if content_length else None

            chunks: list[bytes] = []
            downloaded = 0
            chunk_size = 256 * 1024  # 256 KB
            last_pct = -1
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
                downloaded += len(chunk)
                if total_size and total_size > 0:
                    pct = int(downloaded * 100 / total_size)
                    if pct >= last_pct + 5:
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        bar_len = 30
                        filled = int(bar_len * downloaded / total_size)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        logger.info(
                            "  %s  %3d%%  %.1f / %.1f MB  [%s]",
                            dataset_name,
                            pct,
                            mb_done,
                            mb_total,
                            bar,
                        )
                        last_pct = pct
                else:
                    mb_done = downloaded / (1024 * 1024)
                    if downloaded == len(chunks[-1]) or downloaded % (5 * 1024 * 1024) < chunk_size:
                        logger.info("  %s  %.1f MB downloaded ...", dataset_name, mb_done)

            zip_bytes = b"".join(chunks)
            if total_size:
                logger.info(
                    "Download complete: %s (%.1f MB)",
                    dataset_name,
                    len(zip_bytes) / (1024 * 1024),
                )
    except Exception as exc:
        logger.error("Failed to download dataset %s: %s", dataset_name, exc)
        return None

    import io
    import zipfile

    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(cache_dir)
    except Exception as exc:
        logger.error("Failed to extract dataset zip for %s: %s", dataset_name, exc)
        shutil.rmtree(cache_dir, ignore_errors=True)
        return None

    from datetime import datetime, timezone

    effective_hash = job.get("dataset_config_hash") or resp_hash
    meta = {
        "dataset_name": dataset_name,
        "hash": effective_hash,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "portal_url": base_url,
        "query_csv_bundled": query_csv_bundled,
    }
    meta_file.write_text(json_module.dumps(meta, indent=2))

    file_count = sum(1 for _ in cache_dir.rglob("*") if _.is_file())
    logger.info(
        "Cached dataset %s at %s (%d files, hash %s)",
        dataset_name,
        cache_dir,
        file_count,
        effective_hash[:12],
    )

    return str(cache_dir), _build_cache_rewrites(cache_dir, query_csv_bundled)


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


def _enrich_standalone_graph_result(result: dict[str, Any], job: dict[str, Any]) -> None:
    """Add harness-format fields to a standalone graph result so it appears
    properly in the Runs view with basic metrics."""
    if result.get("timestamp"):
        return
    from datetime import datetime, timezone

    rows = result.get("rows", 0)
    elapsed = result.get("elapsed_secs", 0)
    pps = round(rows / elapsed, 2) if rows and elapsed and elapsed > 0 else None

    result["timestamp"] = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    result.setdefault(
        "test_config",
        {
            "dataset_label": job.get("dataset", "graph-run"),
            "preset": job.get("preset"),
            "input_type": "graph",
            "graph_pipeline": True,
        },
    )
    result.setdefault(
        "metrics",
        {
            "pages": rows,
            "files": rows,
            "ingest_secs": elapsed,
            "pages_per_sec_ingest": pps,
            "rows_processed": rows,
        },
    )
    result.setdefault(
        "summary_metrics",
        {
            "pages": rows,
            "files": rows,
            "ingest_secs": elapsed,
            "pages_per_sec_ingest": pps,
        },
    )
    try:
        import socket as _sock

        host = _sock.gethostname().strip() or "unknown"
    except Exception:
        host = "unknown"
    result.setdefault("run_metadata", {"host": host})
    result.setdefault("artifacts", {})
    if job.get("tags"):
        result.setdefault("tags", job["tags"])


def _execute_job_on_runner(base_url: str, job: dict[str, Any], runner_id: int = 0) -> None:
    """Claim a job, execute it locally, and report results back."""
    job_id = job["id"]

    _payload_dump = json_module.dumps(job, indent=2, default=str)
    print(f"\n===== RAW JOB PAYLOAD (job {job_id}) =====")
    print(_payload_dump)
    print("===== END RAW JOB PAYLOAD =====\n", flush=True)

    # Try to download a distributable dataset before validating paths.
    # If the dataset is cached/downloaded, rewrite the job's paths so
    # _validate_dataset_path sees the local cache dir instead of the
    # portal's path (which doesn't exist on this runner).
    _dist_cached = None
    if not _is_playground_job(job):
        _dist_cached = _ensure_dataset_cached(base_url, job)
        if _dist_cached is not None:
            cached_path, cached_rewrites = _dist_cached
            job = dict(job)
            job["dataset_path"] = cached_path
            job_overrides = job.get("dataset_overrides") or {}
            if isinstance(job_overrides, str):
                try:
                    job_overrides = json_module.loads(job_overrides)
                except (json_module.JSONDecodeError, TypeError):
                    job_overrides = {}
            job_overrides.update(cached_rewrites)
            job["dataset_overrides"] = job_overrides
            logger.info("Job %s: dataset %s cached at %s", job_id, job.get("dataset"), cached_path)

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

    # --- Create an isolated venv for this job ---
    repo_root = _find_repo_root()
    venv_dir: Path | None = None
    if repo_root is not None:
        logger.info("Job %s — creating isolated venv …", job_id)
        venv_dir = _create_job_venv(job_id, repo_root)
        if venv_dir is not None:
            logger.info("Job %s — venv ready at %s", job_id, venv_dir)
        else:
            logger.warning("Job %s — venv creation failed, falling back to current environment", job_id)

    is_graph_job = job.get("trigger_source") == "graph"
    graph_code = job.get("graph_code") or ""
    graph_meta: dict[str, Any] = {}
    if is_graph_job and job.get("config"):
        try:
            graph_meta = json_module.loads(job["config"])
        except (json_module.JSONDecodeError, TypeError):
            pass

    if is_graph_job and graph_meta.get("ray_address") and "ray_address" not in overrides:
        overrides["ray_address"] = graph_meta["ray_address"]

    if is_graph_job and not graph_code.strip():
        logger.error("Job %s is a graph job but has no graph_code — completing as failed", job_id)
        _post_json(
            f"{base_url}/api/jobs/{job_id}/complete",
            {"success": False, "error": "Graph job has no graph_code. Save the graph and retry."},
        )
        _job_tracker.finish_job()
        if prev_head:
            _git_restore(prev_head)
        _destroy_job_venv(job_id)
        return

    nsys_profile = bool(job.get("nsys_profile"))
    use_nsys = nsys_profile and _nsys_available()
    nsys_diag: dict[str, Any] = {"requested": nsys_profile, "enabled": use_nsys}
    if nsys_profile and not use_nsys:
        nsys_diag["error"] = "nsys is not installed or not on PATH"
        logger.warning("Job %s requested nsys profiling but nsys is not on PATH — proceeding without", job_id)
    nsys_output_dir = Path(tempfile.mkdtemp(prefix=f"nsys_{job_id}_")) if use_nsys else None
    if nsys_output_dir:
        nsys_diag["output_dir"] = str(nsys_output_dir)

    result: dict[str, Any] | None = None

    original_stdout = sys.stdout
    sys.stdout = _TeeWriter(original_stdout)
    try:
        is_harness_graph = is_graph_job and job.get("graph_id")

        if is_harness_graph:
            # ---- Graph-as-preset: route through _run_entry for full metrics ----
            if venv_dir is not None:
                args_file = venv_dir / "job_args.json"
                result_file = venv_dir / "job_result.json"
                wrapper_file = venv_dir / "job_wrapper.py"

                job_args = {
                    "run_name": None,
                    "config_file": None,
                    "session_dir": None,
                    "dataset": dataset_value,
                    "preset": job.get("preset"),
                    "sweep_overrides": overrides if overrides else None,
                    "tags": job.get("tags"),
                    "skip_local_history": True,
                    "graph_code": graph_code,
                }
                args_file.write_text(json_module.dumps(job_args))
                wrapper_file.write_text(_JOB_WRAPPER_SCRIPT)

                venv_python = str(venv_dir / "bin" / "python")
                cmd = [venv_python, str(wrapper_file), str(args_file), str(result_file)]
                if use_nsys:
                    cmd = _nsys_prefix(str(nsys_output_dir / "profile")) + cmd
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(repo_root),
                )

                for line in proc.stdout:
                    sys.stdout.write(line)

                proc.wait()

                if result_file.exists():
                    result = json_module.loads(result_file.read_text())
                else:
                    result = {
                        "success": False,
                        "failure_reason": f"Graph harness process terminated (exit code {proc.returncode})",
                        "return_code": proc.returncode,
                    }
            else:
                from nemo_retriever.harness.run import _run_entry

                result = _run_entry(
                    run_name=None,
                    config_file=None,
                    session_dir=None,
                    dataset=dataset_value,
                    preset=job.get("preset"),
                    sweep_overrides=overrides if overrides else None,
                    tags=job.get("tags"),
                    skip_local_history=True,
                    graph_code=graph_code,
                )

        elif is_graph_job:
            # ---- Standalone graph run (from Designer) — direct wrapper ----
            run_dir = venv_dir or Path(tempfile.mkdtemp(prefix=f"graph_{job_id}_"))
            code_file = run_dir / "graph_pipeline.py"
            result_file = run_dir / "job_result.json"
            wrapper_file = run_dir / "graph_wrapper.py"

            code_file.write_text(graph_code)
            wrapper_file.write_text(_GRAPH_WRAPPER_SCRIPT)

            ray_addr = graph_meta.get("ray_address") or overrides.get("ray_address") or "__none__"

            python_bin = str(venv_dir / "bin" / "python") if venv_dir else sys.executable
            cmd = [python_bin, str(wrapper_file), str(code_file), str(result_file), ray_addr]
            if use_nsys:
                cmd = _nsys_prefix(str(nsys_output_dir / "profile")) + cmd
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(repo_root) if repo_root else None,
            )

            for line in proc.stdout:
                sys.stdout.write(line)

            proc.wait()

            if result_file.exists():
                result = json_module.loads(result_file.read_text())
            else:
                result = {
                    "success": False,
                    "failure_reason": f"Graph process terminated (exit code {proc.returncode})",
                    "return_code": proc.returncode,
                }

            _enrich_standalone_graph_result(result, job)

            if not venv_dir and run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

        elif venv_dir is not None:
            # ---- Run in isolated subprocess using the job venv ----
            args_file = venv_dir / "job_args.json"
            result_file = venv_dir / "job_result.json"
            wrapper_file = venv_dir / "job_wrapper.py"

            job_args = {
                "run_name": None,
                "config_file": job.get("config"),
                "session_dir": None,
                "dataset": dataset_value,
                "preset": job.get("preset"),
                "sweep_overrides": overrides if overrides else None,
                "tags": job.get("tags"),
                "skip_local_history": True,
            }
            args_file.write_text(json_module.dumps(job_args))
            wrapper_file.write_text(_JOB_WRAPPER_SCRIPT)

            venv_python = str(venv_dir / "bin" / "python")
            cmd = [venv_python, str(wrapper_file), str(args_file), str(result_file)]
            if use_nsys:
                cmd = _nsys_prefix(str(nsys_output_dir / "profile")) + cmd
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(repo_root),
            )

            for line in proc.stdout:
                sys.stdout.write(line)

            proc.wait()

            if result_file.exists():
                result = json_module.loads(result_file.read_text())
            else:
                result = {
                    "success": False,
                    "failure_reason": f"Job process terminated (exit code {proc.returncode})",
                    "return_code": proc.returncode,
                }
        else:
            # ---- Fallback: run in the current process ----
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

        # --- Collect nsys profiling diagnostics ---
        if use_nsys:
            report_info = _collect_nsys_report_info(nsys_output_dir)
            nsys_diag.update(report_info)
            if report_info["found"]:
                total_mb = sum(f["size_mb"] for f in report_info["files"])
                print(f"[nsys] Profile captured: {len(report_info['files'])} file(s), {total_mb:.1f} MB total")
            else:
                print(f"[nsys] WARNING: No profile captured. {report_info.get('error', 'Unknown reason')}")

        if isinstance(result, dict):
            result["nsys_status"] = nsys_diag

        pip_list_output = _capture_pip_list(job_id)

        final_log_tail = _job_tracker.get_log_tail(_LOG_TAIL_MAX)
        if _job_tracker.is_cancel_requested():
            complete_resp = _post_json(
                f"{base_url}/api/jobs/{job_id}/complete",
                {
                    "success": False,
                    "error": "Cancelled by user",
                    "result": result,
                    "execution_commit": execution_commit,
                    "num_gpus": _runner_num_gpus,
                    "log_tail": final_log_tail,
                    "pip_list": pip_list_output,
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
                    "log_tail": final_log_tail,
                    "pip_list": pip_list_output,
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

            if use_nsys and nsys_output_dir:
                if art_dir and Path(art_dir).is_dir():
                    copied = _copy_nsys_profiles(nsys_output_dir, Path(art_dir))
                    if copied:
                        logger.info("Copied %d nsys file(s) to artifact dir", len(copied))
                    else:
                        logger.warning("No nsys profiles to copy to artifact dir")

            if art_dir and Path(art_dir).is_dir():
                try:
                    _upload_artifacts(base_url, resp_run_id, art_dir)
                except Exception as upload_exc:
                    logger.warning("Failed to upload artifacts for run %d: %s", resp_run_id, upload_exc)
            elif use_nsys and nsys_output_dir:
                nsys_files = list(nsys_output_dir.glob("*.nsys-rep"))
                if nsys_files:
                    logger.info(
                        "No artifact dir — uploading nsys files directly (%d file(s), %.1f MB)",
                        len(nsys_files),
                        sum(f.stat().st_size for f in nsys_files) / (1024 * 1024),
                    )
                    try:
                        _upload_artifacts(base_url, resp_run_id, str(nsys_output_dir))
                    except Exception as upload_exc:
                        logger.warning("Failed to upload nsys artifacts for run %d: %s", resp_run_id, upload_exc)
                else:
                    logger.warning("nsys profiling was enabled but no .nsys-rep files found to upload")
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Job %s failed: %s\n%s", job_id, exc, tb)
        if isinstance(result, dict):
            result.setdefault("nsys_status", nsys_diag)
        try:
            error_msg = f"{exc}\n\n{tb}"
            if _job_tracker.is_cancel_requested():
                error_msg = "Cancelled by user"
            err_pip_list = _capture_pip_list(job_id)
            _post_json(
                f"{base_url}/api/jobs/{job_id}/complete",
                {
                    "success": False,
                    "error": error_msg,
                    "result": result if isinstance(result, dict) else None,
                    "execution_commit": execution_commit,
                    "num_gpus": _runner_num_gpus,
                    "log_tail": _job_tracker.get_log_tail(_LOG_TAIL_MAX),
                    "pip_list": err_pip_list,
                },
            )
        except Exception:
            pass
    finally:
        sys.stdout = original_stdout
        _job_tracker.finish_job()
        if prev_head:
            _git_restore(prev_head)
        _destroy_job_venv(job_id)
        if nsys_output_dir and nsys_output_dir.exists():
            shutil.rmtree(nsys_output_dir, ignore_errors=True)


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
    dataset_cache_dir: str | None = typer.Option(
        None,
        "--dataset-cache-dir",
        envvar="HARNESS_DATASET_CACHE_DIR",
        help="Directory for caching downloaded datasets. Defaults to ~/.cache/harness/datasets.",
    ),
) -> None:
    """Start a harness runner and optionally register with a portal manager."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    global DATASET_CACHE_DIR  # noqa: PLW0603
    if dataset_cache_dir:
        DATASET_CACHE_DIR = Path(dataset_cache_dir)

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
    typer.echo(f"  Dataset  : {DATASET_CACHE_DIR}")
    _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    typer.echo(f"  HF_TOKEN : {'set (' + _hf_token[:4] + '...' + _hf_token[-4:] + ')' if _hf_token else 'NOT SET'}")

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
                hb_payload["log_tail"] = _job_tracker.get_log_tail(_LOG_TAIL_MAX)

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
