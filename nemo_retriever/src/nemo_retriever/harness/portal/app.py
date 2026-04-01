# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI portal for viewing and triggering nemo_retriever harness runs."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json as json_module
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apscheduler.triggers.cron import CronTrigger

from nemo_retriever.harness import history
from nemo_retriever.harness import scheduler as sched_module

mimetypes.add_type("text/javascript", ".jsx")

STATIC_DIR = Path(__file__).resolve().parent / "static"
if not STATIC_DIR.is_dir():
    import importlib.resources as _pkg_res

    _pkg_ref = _pkg_res.files("nemo_retriever.harness.portal").joinpath("static")
    if hasattr(_pkg_ref, "_path"):
        _candidate = Path(str(_pkg_ref._path))
    else:
        _candidate = Path(str(_pkg_ref))
    if _candidate.is_dir():
        STATIC_DIR = _candidate

GITHUB_WEBHOOK_SECRET = os.environ.get("RETRIEVER_HARNESS_GITHUB_SECRET", "")
GITHUB_REPO_URL_OVERRIDE = os.environ.get("RETRIEVER_HARNESS_GITHUB_REPO_URL", "")


def _url_to_github_web(raw_url: str) -> str:
    """Convert a git remote URL to a GitHub web URL, or return empty string."""
    m = re.match(r"git@github\.com:(.+?)(?:\.git)?$", raw_url)
    if m:
        return f"https://github.com/{m.group(1)}"
    m = re.match(r"https?://github\.com/(.+?)(?:\.git)?$", raw_url)
    if m:
        return f"https://github.com/{m.group(1)}"
    return ""


@lru_cache(maxsize=1)
def _detect_nvidia_remote() -> tuple[str, str]:
    """Find the git remote that points to the official NVIDIA repo.

    Returns (remote_name, github_web_url).  Falls back to any available
    remote if no NVIDIA remote is found.
    """
    try:
        raw = subprocess.check_output(
            ["git", "remote", "-v"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return ("", "")

    remotes: list[tuple[str, str]] = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 2 and "(fetch)" in line:
            remotes.append((parts[0], parts[1]))

    for name, url in remotes:
        if "NVIDIA/" in url or "nvidia/" in url:
            return (name, _url_to_github_web(url))

    for preferred in ("nvidia", "upstream", "origin"):
        for name, url in remotes:
            if name == preferred:
                return (name, _url_to_github_web(url))

    if remotes:
        return (remotes[0][0], _url_to_github_web(remotes[0][1]))
    return ("", "")


@lru_cache(maxsize=1)
def _detect_github_repo_url() -> str:
    """Return the GitHub web URL for the official NVIDIA remote."""
    if GITHUB_REPO_URL_OVERRIDE:
        return GITHUB_REPO_URL_OVERRIDE.rstrip("/")
    _, url = _detect_nvidia_remote()
    return url


_runner_health_task: asyncio.Task | None = None


async def _runner_health_check_loop():
    """Background loop that marks stale runners offline and fires alerts."""
    while True:
        await asyncio.sleep(15)
        try:
            newly_offline = history.mark_stale_runners_offline()
            for runner in newly_offline:
                hostname = runner.get("hostname") or runner.get("name") or f"Runner #{runner['id']}"
                logger.warning("Runner %s (id=%s) went offline — missed heartbeats", hostname, runner["id"])
                history.create_system_alert_event(
                    {
                        "metric": "runner_status",
                        "metric_value": 0,
                        "threshold": 0,
                        "operator": "system",
                        "message": f"Runner '{hostname}' is offline — missed {history.RUNNER_MISSED_HEARTBEATS_THRESHOLD} consecutive heartbeats",
                        "hostname": hostname,
                    }
                )
        except Exception:
            logger.exception("Error in runner health check loop")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _seed_default_run_code_ref()
    sched_module.start_scheduler()
    global _runner_health_task
    _runner_health_task = asyncio.create_task(_runner_health_check_loop())
    yield
    if _runner_health_task:
        _runner_health_task.cancel()
    sched_module.stop_scheduler()


def _seed_default_run_code_ref() -> None:
    """Set run_code_ref default based on the detected NVIDIA remote if not already explicitly saved."""
    nvidia_remote, _ = _detect_nvidia_remote()
    if not nvidia_remote:
        return
    default_ref = f"{nvidia_remote}/main"
    current = history.get_portal_setting("run_code_ref")
    if current is None or current in ("upstream/main", "origin/main", "nvidia/main"):
        if current != default_ref:
            history.set_portal_setting("run_code_ref", default_ref)
            logger.info("Auto-configured run_code_ref to %s", default_ref)


app = FastAPI(title="Harness Portal", docs_url="/api/docs", redoc_url=None, lifespan=_lifespan)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    logger.warning("Static directory not found at %s — portal UI will not be served", STATIC_DIR)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TriggerRequest(BaseModel):
    dataset: str
    preset: str | None = None
    config: str | None = None
    tags: list[str] | None = None
    runner_id: int | None = None
    git_ref: str | None = None
    git_commit: str | None = None
    nsys_profile: bool = False
    graph_id: int | None = None
    extra_packages: list[str] | None = None


class TriggerResponse(BaseModel):
    job_id: str
    status: str


class RerunRequest(BaseModel):
    runner_id: int | None = None


class RunnerCreateRequest(BaseModel):
    name: str
    hostname: str | None = None
    url: str | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None
    cpu_count: int | None = None
    memory_gb: float | None = None
    status: str = "online"
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    heartbeat_interval: int = 30
    git_commit: str | None = None
    ray_address: str | None = None


class RunnerUpdateRequest(BaseModel):
    name: str | None = None
    hostname: str | None = None
    url: str | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None
    cpu_count: int | None = None
    memory_gb: float | None = None
    status: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    ray_address: str | None = None


class ScheduleCreateRequest(BaseModel):
    name: str
    description: str | None = None
    dataset: str = ""
    preset: str | None = None
    preset_matrix: str | None = None
    config: str | None = None
    trigger_type: str = "cron"
    cron_expression: str | None = None
    github_repo: str | None = None
    github_branch: str | None = None
    min_gpu_count: int | None = None
    gpu_type_pattern: str | None = None
    min_cpu_count: int | None = None
    min_memory_gb: float | None = None
    preferred_runner_id: int | None = None
    preferred_runner_ids: list[int] | None = None
    enabled: bool = True
    tags: list[str] | None = None


class ScheduleUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    dataset: str | None = None
    preset: str | None = None
    preset_matrix: str | None = None
    config: str | None = None
    trigger_type: str | None = None
    cron_expression: str | None = None
    github_repo: str | None = None
    github_branch: str | None = None
    min_gpu_count: int | None = None
    gpu_type_pattern: str | None = None
    min_cpu_count: int | None = None
    min_memory_gb: float | None = None
    preferred_runner_id: int | None = None
    preferred_runner_ids: list[int] | None = None
    enabled: bool | None = None
    tags: list[str] | None = None


class JobCompleteRequest(BaseModel):
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    execution_commit: str | None = None
    num_gpus: int | None = None
    log_tail: list[str] | None = None
    pip_list: str | None = None


class PresetCreateRequest(BaseModel):
    name: str
    description: str | None = None
    config: dict[str, Any] = {}
    tags: list[str] | None = None
    overrides: dict[str, Any] = {}


class PresetUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    config: dict[str, Any] | None = None
    tags: list[str] | None = None
    overrides: dict[str, Any] | None = None


class PresetMatrixCreateRequest(BaseModel):
    name: str
    description: str | None = None
    dataset_names: list[str]
    preset_names: list[str]
    tags: list[str] | None = None
    preferred_runner_id: int | None = None
    gpu_type_filter: str | None = None
    git_ref: str | None = None
    git_commit: str | None = None
    nsys_profile: bool = False


class PresetMatrixUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    dataset_names: list[str] | None = None
    preset_names: list[str] | None = None
    tags: list[str] | None = None
    preferred_runner_id: int | None = None
    gpu_type_filter: str | None = None
    git_ref: str | None = None
    git_commit: str | None = None
    nsys_profile: bool | None = None


class MatrixTriggerRequest(BaseModel):
    git_ref: str | None = None
    git_commit: str | None = None
    nsys_profile: bool | None = None


class MatrixTriggerResponse(BaseModel):
    matrix_id: int
    matrix_name: str
    matrix_run_id: str
    job_ids: list[str]
    job_count: int


class DatasetCreateRequest(BaseModel):
    name: str
    path: str
    query_csv: str | None = None
    input_type: str = "pdf"
    recall_required: bool = False
    recall_match_mode: str = "pdf_page"
    recall_adapter: str = "none"
    evaluation_mode: str = "recall"
    beir_loader: str | None = None
    beir_dataset_name: str | None = None
    beir_split: str = "test"
    beir_query_language: str | None = None
    beir_doc_id_field: str = "pdf_basename"
    beir_ks: list[int] | None = None
    embed_model_name: str | None = None
    embed_modality: str = "text"
    embed_granularity: str = "element"
    extract_page_as_image: bool = False
    extract_infographics: bool = False
    description: str | None = None
    tags: list[str] | None = None
    runner_ids: list[int] | None = None


class DatasetUpdateRequest(BaseModel):
    name: str | None = None
    path: str | None = None
    query_csv: str | None = None
    input_type: str | None = None
    recall_required: bool | None = None
    recall_match_mode: str | None = None
    recall_adapter: str | None = None
    evaluation_mode: str | None = None
    beir_loader: str | None = None
    beir_dataset_name: str | None = None
    beir_split: str | None = None
    beir_query_language: str | None = None
    beir_doc_id_field: str | None = None
    beir_ks: list[int] | None = None
    embed_model_name: str | None = None
    embed_modality: str | None = None
    embed_granularity: str | None = None
    extract_page_as_image: bool | None = None
    extract_infographics: bool | None = None
    description: str | None = None
    tags: list[str] | None = None
    runner_ids: list[int] | None = None


class AlertRuleCreateRequest(BaseModel):
    name: str
    description: str | None = None
    metric: str
    operator: str
    threshold: float
    dataset_filter: str | None = None
    preset_filter: str | None = None
    enabled: bool = True


class AlertRuleUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    metric: str | None = None
    operator: str | None = None
    threshold: float | None = None
    dataset_filter: str | None = None
    preset_filter: str | None = None
    enabled: bool | None = None


# ---------------------------------------------------------------------------
# Static / index
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Portal UI not found. Expected at {index_path}. "
            "Reinstall the package with: uv pip install -e ./nemo_retriever",
        )
    return FileResponse(str(index_path))


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


@app.get("/api/version")
async def get_version():
    from nemo_retriever.version import get_version_info

    return get_version_info()


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@app.get("/api/runs")
async def list_runs(
    dataset: str | None = Query(None),
    commit: str | None = Query(None),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    return history.get_runs(dataset=dataset, commit=commit, limit=limit, offset=offset)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: int):
    row = history.get_run_by_id(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return row


@app.get("/api/runs/{run_id}/download/json")
async def download_run_json(run_id: int):
    row = history.get_run_by_id(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    content = json_module.dumps(row, indent=2, default=str)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id}.json"'},
    )


@app.get("/api/runs/{run_id}/download/zip")
async def download_run_zip(run_id: int):
    row = history.get_run_by_id(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")

    dataset = row.get("dataset", "unknown")

    uploaded_zip = history.portal_artifacts_dir() / f"run_{run_id}.zip"
    if uploaded_zip.is_file():
        return FileResponse(
            path=str(uploaded_zip),
            media_type="application/zip",
            filename=f"run_{run_id}_{dataset}.zip",
        )

    artifact_dir = row.get("artifact_dir")
    if not artifact_dir or not Path(artifact_dir).is_dir():
        raise HTTPException(status_code=404, detail="Artifact directory not found")

    buf = io.BytesIO()
    artifact_path = Path(artifact_dir)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(artifact_path.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(artifact_path))
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id}_{dataset}.zip"'},
    )


@app.get("/api/runs/{run_id}/download/nsys-profile")
async def download_nsys_profile(run_id: int):
    row = history.get_run_by_id(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")

    dataset = row.get("dataset", "unknown")

    # Check the raw_json for nsys_status diagnostics to give a helpful error
    raw = row.get("raw_json") or {}
    nsys_status = raw.get("nsys_status") or {}

    uploaded_zip = history.portal_artifacts_dir() / f"run_{run_id}.zip"
    if uploaded_zip.is_file():
        try:
            with zipfile.ZipFile(uploaded_zip, "r") as zf:
                nsys_files = [n for n in zf.namelist() if n.endswith(".nsys-rep")]
                if not nsys_files:
                    detail = "No .nsys-rep file found in uploaded artifact zip."
                    if nsys_status.get("error"):
                        detail += f" Runner reported: {nsys_status['error']}"
                    raise HTTPException(status_code=404, detail=detail)
                nsys_name = nsys_files[0]
                buf = io.BytesIO(zf.read(nsys_name))
                download_name = f"run_{run_id}_{dataset}.nsys-rep"
                return StreamingResponse(
                    buf,
                    media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
                )
        except zipfile.BadZipFile:
            raise HTTPException(status_code=500, detail="Corrupt artifact zip")

    artifact_dir = row.get("artifact_dir")
    if artifact_dir and Path(artifact_dir).is_dir():
        for fp in Path(artifact_dir).rglob("*.nsys-rep"):
            download_name = f"run_{run_id}_{dataset}.nsys-rep"
            return FileResponse(
                path=str(fp),
                media_type="application/octet-stream",
                filename=download_name,
            )

    detail = "No nsys profile found for this run."
    if not nsys_status.get("requested"):
        detail = "Nsys profiling was not requested for this run."
    elif not nsys_status.get("enabled"):
        detail += f" {nsys_status.get('error', 'nsys was not available on the runner.')}"
    elif not nsys_status.get("found"):
        detail += f" {nsys_status.get('error', 'nsys ran but produced no report file.')}"
    else:
        detail += " The report file was generated but could not be uploaded to the portal."
    raise HTTPException(status_code=404, detail=detail)


@app.post("/api/runs/{run_id}/upload-artifacts")
async def upload_run_artifacts(run_id: int, file: UploadFile = File(...)):
    row = history.get_run_by_id(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")

    dest = history.portal_artifacts_dir() / f"run_{run_id}.zip"
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as exc:
        if dest.is_file():
            dest.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to save artifacts: {exc}")

    return {"ok": True, "size_bytes": dest.stat().st_size}


@app.get("/api/runs/{run_id}/command")
async def get_run_command(run_id: int):
    """Return the shell command that was executed for this run."""
    row = history.get_run_by_id(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    raw = row.get("raw_json") or {}
    command_file = (raw.get("artifacts") or {}).get("command_file")
    if command_file:
        p = Path(command_file)
        if p.is_file():
            return {"command": p.read_text(encoding="utf-8").strip()}
    return {"command": None}


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: int):
    if not history.delete_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return {"ok": True}


class BulkDeleteRunsRequest(BaseModel):
    run_ids: list[int]


@app.post("/api/runs/delete-bulk")
async def delete_runs_bulk(req: BulkDeleteRunsRequest):
    """Delete multiple runs in one request."""
    count = history.delete_runs_bulk(req.run_ids)
    return {"ok": True, "deleted": count}


@app.get("/api/runs/{run_id}/rerun-info")
async def get_rerun_info(run_id: int):
    """Return the information needed to re-run: original hostname and runner match status."""
    row = history.get_run_by_id(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")

    raw = row.get("raw_json") or {}
    run_meta = raw.get("run_metadata") or {}
    original_hostname = run_meta.get("host") or row.get("hostname")
    original_commit = row.get("execution_commit") or row.get("git_commit") or (raw.get("latest_commit"))

    runners = history.get_runners()
    online_runners = [r for r in runners if r.get("status") == "online"]

    original_runner = None
    if original_hostname:
        for r in online_runners:
            if (r.get("hostname") or "").lower() == original_hostname.lower():
                original_runner = r
                break

    return {
        "original_hostname": original_hostname,
        "original_commit": original_commit,
        "original_runner": original_runner,
        "online_runners": [
            {"id": r["id"], "name": r["name"], "hostname": r.get("hostname"), "gpu_type": r.get("gpu_type")}
            for r in online_runners
        ],
    }


@app.post("/api/runs/{run_id}/rerun")
async def rerun_run(run_id: int, req: RerunRequest | None = None):
    """Create a new job that reproduces the exact configuration of a previous run."""
    row = history.get_run_by_id(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")

    raw = row.get("raw_json") or {}
    test_config = raw.get("test_config") or {}
    run_meta = raw.get("run_metadata") or {}

    original_hostname = run_meta.get("host") or row.get("hostname")
    original_commit = row.get("execution_commit") or row.get("git_commit") or raw.get("latest_commit")

    overrides: dict[str, Any] = {}
    if test_config.get("dataset_dir"):
        overrides["dataset_dir"] = test_config["dataset_dir"]
    if test_config.get("query_csv"):
        overrides["query_csv"] = test_config["query_csv"]
    if test_config.get("input_type"):
        overrides["input_type"] = test_config["input_type"]
    if test_config.get("recall_required") is not None:
        overrides["recall_required"] = test_config["recall_required"]
    if test_config.get("recall_match_mode"):
        overrides["recall_match_mode"] = test_config["recall_match_mode"]
    if test_config.get("recall_adapter"):
        overrides["recall_adapter"] = test_config["recall_adapter"]
    if test_config.get("ray_address"):
        overrides["ray_address"] = test_config["ray_address"]
    if test_config.get("hybrid") is not None:
        overrides["hybrid"] = test_config["hybrid"]
    if test_config.get("evaluation_mode"):
        overrides["evaluation_mode"] = test_config["evaluation_mode"]
    if test_config.get("beir_loader"):
        overrides["beir_loader"] = test_config["beir_loader"]
    if test_config.get("beir_dataset_name"):
        overrides["beir_dataset_name"] = test_config["beir_dataset_name"]
    if test_config.get("beir_split"):
        overrides["beir_split"] = test_config["beir_split"]
    if test_config.get("beir_query_language"):
        overrides["beir_query_language"] = test_config["beir_query_language"]
    if test_config.get("beir_doc_id_field"):
        overrides["beir_doc_id_field"] = test_config["beir_doc_id_field"]
    if test_config.get("beir_ks"):
        overrides["beir_ks"] = test_config["beir_ks"]
    if test_config.get("embed_model_name"):
        overrides["embed_model_name"] = test_config["embed_model_name"]
    if test_config.get("embed_modality"):
        overrides["embed_modality"] = test_config["embed_modality"]
    if test_config.get("embed_granularity"):
        overrides["embed_granularity"] = test_config["embed_granularity"]
    if test_config.get("extract_page_as_image") is not None:
        overrides["extract_page_as_image"] = test_config["extract_page_as_image"]
    if test_config.get("extract_infographics") is not None:
        overrides["extract_infographics"] = test_config["extract_infographics"]
    if test_config.get("write_detection_file") is not None:
        overrides["write_detection_file"] = test_config["write_detection_file"]
    tuning = test_config.get("tuning") or {}
    for k, v in tuning.items():
        overrides[k] = v

    runner_id = req.runner_id if req else None
    if not runner_id and original_hostname:
        runners = history.get_runners()
        for r in runners:
            if r.get("status") == "online" and (r.get("hostname") or "").lower() == original_hostname.lower():
                runner_id = r["id"]
                break

    original_tags = row.get("tags") or []
    rerun_tags = [t for t in original_tags if not t.startswith("rerun:")]
    rerun_tags.append(f"rerun:of_run_{run_id}")

    job = history.create_job(
        {
            "trigger_source": "rerun",
            "dataset": row.get("dataset") or test_config.get("dataset_label", "unknown"),
            "dataset_path": test_config.get("dataset_dir"),
            "dataset_overrides": overrides if overrides else None,
            "preset": None,
            "assigned_runner_id": runner_id,
            "git_commit": original_commit,
            "git_ref": original_commit,
            "tags": rerun_tags,
        }
    )

    return {
        "job_id": job["id"],
        "status": "pending",
        "assigned_runner_id": runner_id,
        "git_commit": original_commit,
    }


# ---------------------------------------------------------------------------
# Retrieval Playground
# ---------------------------------------------------------------------------

LANCEDB_TABLE = "nv-ingest"


def _get_lancedb_uri_for_run(run: dict[str, Any]) -> str | None:
    """Extract a valid LanceDB URI from a run's raw_json or artifact_dir."""
    raw = run.get("raw_json") or {}
    tc = raw.get("test_config") or {}
    uri = tc.get("lancedb_uri")
    if uri and Path(uri).is_dir():
        return uri
    artifact_dir = run.get("artifact_dir")
    if artifact_dir:
        candidate = Path(artifact_dir) / "lancedb"
        if candidate.is_dir():
            return str(candidate)
    return None


@app.get("/api/runs/{run_id}/lancedb-info")
async def get_run_lancedb_info(run_id: int):
    """Check whether a run has a usable LanceDB database and return metadata."""
    row = history.get_run_by_id(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    uri = _get_lancedb_uri_for_run(row)
    if not uri:
        return {"available": False, "uri": None, "row_count": 0}
    try:
        import lancedb  # type: ignore

        db = lancedb.connect(uri)
        table = db.open_table(LANCEDB_TABLE)
        count = int(table.count_rows())
        return {"available": True, "uri": uri, "row_count": count, "table": LANCEDB_TABLE}
    except Exception as exc:
        logger.debug("LanceDB probe failed for run %s: %s", run_id, exc)
        return {"available": False, "uri": uri, "row_count": 0, "error": str(exc)}


class RetrievalQueryRequest(BaseModel):
    query: str
    top_k: int = 10


@app.post("/api/runs/{run_id}/retrieval")
async def run_retrieval_query(run_id: int, req: RetrievalQueryRequest):
    """Execute a retrieval query against a run's LanceDB database."""
    row = history.get_run_by_id(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    uri = _get_lancedb_uri_for_run(row)
    if not uri:
        raise HTTPException(status_code=404, detail="LanceDB not available for this run")
    try:
        import lancedb  # type: ignore
        import numpy as np  # type: ignore

        raw = row.get("raw_json") or {}
        tc = raw.get("test_config") or {}
        embed_model = tc.get("embed_model_name", "nvidia/llama-nemotron-embed-1b-v2")

        from nemo_retriever.retriever import Retriever

        retriever = Retriever(
            lancedb_uri=uri,
            lancedb_table=LANCEDB_TABLE,
            embedder=embed_model,
            top_k=req.top_k,
        )
        hits = retriever.query(req.query)
        results = []
        for hit in hits:
            entry: dict[str, Any] = {}
            for key in ("text", "source", "page_number", "_distance", "_rerank_score"):
                if key in hit:
                    val = hit[key]
                    if hasattr(val, "item"):
                        val = val.item()
                    entry[key] = val
            metadata = hit.get("metadata")
            if metadata and isinstance(metadata, dict):
                entry["metadata"] = {
                    k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool, type(None)))
                }
            elif isinstance(metadata, str):
                entry["metadata"] = metadata
            results.append(entry)
        return {
            "query": req.query,
            "top_k": req.top_k,
            "embed_model": embed_model,
            "result_count": len(results),
            "results": results,
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="lancedb is not installed on this server")
    except Exception as exc:
        logger.error("Retrieval query failed for run %s: %s", run_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Ingestion Playground
# ---------------------------------------------------------------------------

PLAYGROUND_DIR = Path(tempfile.gettempdir()) / "harness_playground_uploads"
PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/api/playground/upload")
async def playground_upload(files: list[UploadFile] = File(...)):
    """Upload documents to a temporary directory for playground ingestion.

    Returns the session_id and list of uploaded file names.
    """
    session_id = uuid.uuid4().hex[:12]
    session_dir = PLAYGROUND_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    total_size = 0
    for f in files:
        safe_name = Path(f.filename).name if f.filename else f"upload_{len(saved)}"
        dest = session_dir / safe_name
        content = await f.read()
        total_size += len(content)
        dest.write_bytes(content)
        saved.append(safe_name)
    return {
        "session_id": session_id,
        "files": saved,
        "file_count": len(saved),
        "total_bytes": total_size,
        "upload_dir": str(session_dir),
    }


class PlaygroundIngestRequest(BaseModel):
    session_id: str
    preset: str | None = None
    runner_id: int | None = None
    input_type: str = "pdf"


@app.post("/api/playground/ingest")
async def playground_ingest(req: PlaygroundIngestRequest):
    """Trigger a harness run using uploaded playground documents."""
    session_dir = PLAYGROUND_DIR / req.session_id
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Upload session not found. Please upload files first.")
    file_count = sum(1 for f in session_dir.iterdir() if f.is_file())
    if file_count == 0:
        raise HTTPException(status_code=400, detail="Upload session contains no files.")

    dataset_name = f"playground_{req.session_id}"
    dataset_path = str(session_dir)
    overrides: dict[str, Any] = {
        "dataset_dir": dataset_path,
        "input_type": req.input_type,
        "recall_required": False,
        "query_csv": None,
    }

    job = history.create_job(
        {
            "trigger_source": "playground",
            "dataset": dataset_name,
            "dataset_path": dataset_path,
            "dataset_overrides": overrides,
            "preset": req.preset,
            "assigned_runner_id": req.runner_id,
            "tags": ["playground"],
        }
    )
    return {
        "job_id": job["id"],
        "status": "pending",
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "file_count": file_count,
    }


@app.get("/api/playground/sessions")
async def list_playground_sessions():
    """List existing playground upload sessions."""
    if not PLAYGROUND_DIR.is_dir():
        return []
    sessions = []
    for d in sorted(PLAYGROUND_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir():
            files = [f.name for f in d.iterdir() if f.is_file()]
            total_bytes = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            sessions.append(
                {
                    "session_id": d.name,
                    "files": files,
                    "file_count": len(files),
                    "total_bytes": total_bytes,
                    "path": str(d),
                }
            )
    return sessions


@app.get("/api/playground/sessions/{session_id}/download")
async def download_playground_session(session_id: str):
    """Download all files in a playground session as a zip archive."""
    import zipfile

    session_dir = PLAYGROUND_DIR / session_id
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in session_dir.iterdir():
            if f.is_file():
                zf.write(f, f.name)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=playground_{session_id}.zip"},
    )


@app.delete("/api/playground/sessions/{session_id}")
async def delete_playground_session(session_id: str):
    """Delete a playground upload session and its files."""
    session_dir = PLAYGROUND_DIR / session_id
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")
    shutil.rmtree(session_dir, ignore_errors=True)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Models Playground
# ---------------------------------------------------------------------------

_AVAILABLE_MODELS = [
    {
        "id": "nvidia/llama-nemotron-embed-1b-v2",
        "name": "Llama Nemotron Embed 1B v2",
        "type": "embedding",
        "category": "Retrieval",
        "description": "Dense text embedding model for retrieval. Produces 4096-dim vectors.",
        "input_type": "text",
        "max_length": 8192,
    },
    {
        "id": "nvidia/llama-nemotron-embed-vl-1b-v2",
        "name": "Llama Nemotron Embed VL 1B v2",
        "type": "embedding",
        "category": "Retrieval",
        "description": "Vision-language embedding model for multimodal retrieval.",
        "input_type": "text",
        "max_length": 8192,
    },
    {
        "id": "nvidia/llama-nemotron-rerank-1b-v2",
        "name": "Llama Nemotron Rerank 1B v2",
        "type": "reranker",
        "category": "Retrieval",
        "description": "Cross-encoder reranker. Scores query-document relevance (higher = better).",
        "input_type": "text",
        "max_length": 8192,
    },
    {
        "id": "nemotron-ocr-v1",
        "name": "Nemotron OCR v1",
        "type": "ocr",
        "category": "Document AI",
        "description": "End-to-end OCR: text detection, recognition, and reading-order analysis.",
        "input_type": "image",
        "output_classes": ["word", "sentence", "paragraph"],
    },
    {
        "id": "page_element_v3",
        "name": "Nemotron Page Elements v3",
        "type": "object-detection",
        "category": "Document AI",
        "description": "Detects document elements: tables, charts, titles, infographics, text regions, headers/footers.",
        "input_type": "image",
        "output_classes": ["table", "chart", "title", "infographic", "text", "header_footer"],
    },
    {
        "id": "table_structure_v1",
        "name": "Nemotron Table Structure v1",
        "type": "object-detection",
        "category": "Document AI",
        "description": "Detects table structure: cells (including merged), rows, and columns from cropped table images.",
        "input_type": "image",
        "output_classes": ["cell", "row", "column"],
    },
    {
        "id": "graphic_elements_v1",
        "name": "Nemotron Graphic Elements v1",
        "type": "object-detection",
        "category": "Document AI",
        "description": "Detects chart elements: axis titles/labels, legends, markers, value labels.",
        "input_type": "image",
        "output_classes": [
            "chart_title",
            "x_axis_title",
            "y_axis_title",
            "legend_title",
            "legend_label",
            "marker_label",
            "value_label",
        ],
    },
    {
        "id": "nvidia/NVIDIA-Nemotron-Parse-v1.2",
        "name": "Nemotron Parse v1.2",
        "type": "document-parser",
        "category": "Document AI",
        "description": "Image-to-structured-text model. Converts document images to Markdown with bounding boxes.",
        "input_type": "image",
    },
    {
        "id": "nvidia/parakeet-ctc-1.1b",
        "name": "Parakeet CTC 1.1B",
        "type": "asr",
        "category": "Audio",
        "description": "Automatic speech recognition model. Transcribes 16 kHz mono audio to text.",
        "input_type": "audio",
    },
]


@app.get("/api/models")
async def list_models():
    """Return the list of available HuggingFace models."""
    return _AVAILABLE_MODELS


class EmbedTestRequest(BaseModel):
    model_id: str = "nvidia/llama-nemotron-embed-1b-v2"
    texts: list[str]
    prefix: str = "query: "
    batch_size: int = 64


@app.post("/api/models/embed")
async def test_embed_model(req: EmbedTestRequest):
    """Send texts to an embedding model and return vectors + metadata."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    try:
        import time as _time

        from nemo_retriever.model import create_local_embedder

        prefixed = [f"{req.prefix}{t}" for t in req.texts] if req.prefix else list(req.texts)
        t0 = _time.perf_counter()
        embedder = create_local_embedder(req.model_id)
        load_time = _time.perf_counter() - t0

        t1 = _time.perf_counter()
        vecs = embedder.embed(prefixed, batch_size=req.batch_size)
        embed_time = _time.perf_counter() - t1

        results = []
        for i, text in enumerate(req.texts):
            vec = vecs[i].tolist()
            results.append(
                {
                    "text": text,
                    "embedding_dim": len(vec),
                    "embedding_preview": vec[:8],
                    "embedding_norm": round(sum(v * v for v in vec) ** 0.5, 6),
                }
            )

        return {
            "model_id": req.model_id,
            "prefix": req.prefix,
            "count": len(results),
            "embedding_dim": results[0]["embedding_dim"] if results else 0,
            "model_load_ms": round(load_time * 1000, 1),
            "embed_ms": round(embed_time * 1000, 1),
            "results": results,
        }
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class RerankTestRequest(BaseModel):
    model_id: str = "nvidia/llama-nemotron-rerank-1b-v2"
    query: str
    documents: list[str]
    max_length: int = 512
    batch_size: int = 32


@app.post("/api/models/rerank")
async def test_rerank_model(req: RerankTestRequest):
    """Score query-document relevance pairs using a cross-encoder reranker."""
    if not req.query:
        raise HTTPException(status_code=400, detail="query cannot be empty")
    if not req.documents:
        raise HTTPException(status_code=400, detail="documents list cannot be empty")
    try:
        import time as _time

        from nemo_retriever.model.local.nemotron_rerank_v2 import NemotronRerankV2

        t0 = _time.perf_counter()
        reranker = NemotronRerankV2(model_name=req.model_id)
        load_time = _time.perf_counter() - t0

        t1 = _time.perf_counter()
        scores = reranker.score(
            req.query,
            req.documents,
            max_length=req.max_length,
            batch_size=req.batch_size,
        )
        score_time = _time.perf_counter() - t1

        results = []
        for i, (doc, score) in enumerate(zip(req.documents, scores)):
            results.append(
                {
                    "rank": i + 1,
                    "document": doc,
                    "score": round(float(score), 4),
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return {
            "model_id": req.model_id,
            "query": req.query,
            "count": len(results),
            "model_load_ms": round(load_time * 1000, 1),
            "score_ms": round(score_time * 1000, 1),
            "results": results,
        }
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class OCRTestRequest(BaseModel):
    model_id: str = "nemotron-ocr-v1"
    image_b64: str
    merge_level: str = "paragraph"


@app.post("/api/models/ocr")
async def test_ocr_model(req: OCRTestRequest):
    """Run OCR on a base64-encoded image and return extracted text."""
    if not req.image_b64:
        raise HTTPException(status_code=400, detail="image_b64 cannot be empty")
    try:
        import time as _time

        from nemo_retriever.model.local.nemotron_ocr_v1 import NemotronOCRV1

        t0 = _time.perf_counter()
        model = NemotronOCRV1()
        load_time = _time.perf_counter() - t0

        img_data = req.image_b64
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]

        t1 = _time.perf_counter()
        raw = model.invoke(img_data, merge_level=req.merge_level)
        infer_time = _time.perf_counter() - t1

        text = NemotronOCRV1._extract_text(raw)

        return {
            "model_id": req.model_id,
            "merge_level": req.merge_level,
            "model_load_ms": round(load_time * 1000, 1),
            "inference_ms": round(infer_time * 1000, 1),
            "text": text,
            "raw_output_type": type(raw).__name__,
        }
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class ParseTestRequest(BaseModel):
    model_id: str = "nvidia/NVIDIA-Nemotron-Parse-v1.2"
    image_b64: str


@app.post("/api/models/parse")
async def test_parse_model(req: ParseTestRequest):
    """Run Nemotron Parse on a base64-encoded image and return structured Markdown."""
    if not req.image_b64:
        raise HTTPException(status_code=400, detail="image_b64 cannot be empty")
    try:
        import base64
        import io
        import time as _time

        from PIL import Image

        from nemo_retriever.model.local.nemotron_parse_v1_2 import NemotronParseV12

        img_data = req.image_b64
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]

        img_bytes = base64.b64decode(img_data)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        t0 = _time.perf_counter()
        model = NemotronParseV12()
        load_time = _time.perf_counter() - t0

        t1 = _time.perf_counter()
        result = model.invoke(pil_img)
        infer_time = _time.perf_counter() - t1

        text = result if isinstance(result, str) else str(result)

        return {
            "model_id": req.model_id,
            "model_load_ms": round(load_time * 1000, 1),
            "inference_ms": round(infer_time * 1000, 1),
            "markdown": text,
        }
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class DetectTestRequest(BaseModel):
    model_id: str = "page_element_v3"
    image_b64: str
    score_threshold: float = 0.25


@app.post("/api/models/detect")
async def test_detect_model(req: DetectTestRequest):
    """Run an object-detection model on a base64-encoded image.

    Returns detected boxes drawn on the original image (as base64) plus a
    structured list of detections.
    """
    if not req.image_b64:
        raise HTTPException(status_code=400, detail="image_b64 cannot be empty")
    try:
        import base64
        import io
        import time as _time

        import numpy as np
        import torch
        from PIL import Image, ImageDraw, ImageFont

        img_data = req.image_b64
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]

        img_bytes = base64.b64decode(img_data)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_w, orig_h = pil_img.size

        img_np = np.array(pil_img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

        t0 = _time.perf_counter()

        if req.model_id == "page_element_v3":
            from nemo_retriever.model.local.nemotron_page_elements_v3 import NemotronPageElementsV3

            model = NemotronPageElementsV3()
            label_names = ["table", "chart", "title", "infographic", "text", "header_footer"]
        elif req.model_id == "table_structure_v1":
            from nemo_retriever.model.local.nemotron_table_structure_v1 import NemotronTableStructureV1

            model = NemotronTableStructureV1()
            label_names = ["cell", "merged_cell", "row", "column"]
        elif req.model_id == "graphic_elements_v1":
            from nemo_retriever.model.local.nemotron_graphic_elements_v1 import NemotronGraphicElementsV1

            model = NemotronGraphicElementsV1()
            label_names = [
                "chart_title",
                "x_axis_title",
                "y_axis_title",
                "x_tick_label",
                "y_tick_label",
                "legend_title",
                "legend_label",
                "marker_label",
                "value_label",
                "other_label",
            ]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown detection model: {req.model_id}")

        load_time = _time.perf_counter() - t0

        preprocessed = model.preprocess(img_tensor)
        if hasattr(preprocessed, "ndim") and preprocessed.ndim == 3:
            preprocessed = preprocessed.unsqueeze(0)

        device = next(model.model.parameters()).device
        preprocessed = preprocessed.to(device)

        t1 = _time.perf_counter()
        raw_preds = model.invoke(preprocessed, (orig_h, orig_w))
        boxes_t, labels_t, scores_t = model.postprocess(raw_preds)
        infer_time = _time.perf_counter() - t1

        if isinstance(boxes_t, list):
            boxes_t, labels_t, scores_t = boxes_t[0], labels_t[0], scores_t[0]

        boxes_np = boxes_t.cpu().numpy() if hasattr(boxes_t, "cpu") else np.array(boxes_t)
        labels_np = labels_t.cpu().numpy() if hasattr(labels_t, "cpu") else np.array(labels_t)
        scores_np = scores_t.cpu().numpy() if hasattr(scores_t, "cpu") else np.array(scores_t)

        CLASS_COLORS = [
            (118, 185, 0),
            (100, 180, 255),
            (255, 140, 0),
            (187, 134, 252),
            (255, 80, 80),
            (0, 212, 170),
            (252, 211, 77),
            (255, 105, 180),
            (0, 191, 255),
            (144, 238, 144),
        ]

        draw_img = pil_img.copy()
        draw = ImageDraw.Draw(draw_img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(12, min(orig_h, orig_w) // 60)
            )
        except Exception:
            font = ImageFont.load_default()

        detections = []
        for i in range(len(boxes_np)):
            score = float(scores_np[i])
            if score < req.score_threshold:
                continue
            label_idx = int(labels_np[i])
            label_str = label_names[label_idx] if label_idx < len(label_names) else f"class_{label_idx}"
            box = boxes_np[i]
            x1 = float(box[0]) * orig_w
            y1 = float(box[1]) * orig_h
            x2 = float(box[2]) * orig_w
            y2 = float(box[3]) * orig_h

            color = CLASS_COLORS[label_idx % len(CLASS_COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, min(orig_h, orig_w) // 300))
            txt = f"{label_str} {score:.0%}"
            text_bbox = draw.textbbox((x1, y1), txt, font=font)
            draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
            draw.text((x1, y1), txt, fill=(0, 0, 0), font=font)

            detections.append(
                {
                    "label": label_str,
                    "score": round(score, 4),
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "box_normalized": [round(float(box[j]), 4) for j in range(4)],
                }
            )

        detections.sort(key=lambda d: d["score"], reverse=True)

        buf = io.BytesIO()
        draw_img.save(buf, format="PNG")
        annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "model_id": req.model_id,
            "model_load_ms": round(load_time * 1000, 1),
            "inference_ms": round(infer_time * 1000, 1),
            "image_size": [orig_w, orig_h],
            "detection_count": len(detections),
            "score_threshold": req.score_threshold,
            "detections": detections,
            "annotated_image": f"data:image/png;base64,{annotated_b64}",
        }
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        logger.error("Detection model error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/datasets")
async def list_datasets():
    """Return distinct dataset names from run history (legacy)."""
    return history.get_datasets()


@app.get("/api/config")
async def get_config():
    """Return dataset and preset names from the managed database entries."""
    managed_dataset_names = history.get_dataset_names()
    managed_preset_names = history.get_preset_names()
    matrices = history.get_all_preset_matrices()
    return {
        "datasets": sorted(managed_dataset_names),
        "presets": sorted(managed_preset_names),
        "preset_matrices": [{"id": m["id"], "name": m["name"]} for m in matrices],
        "github_repo_url": _detect_github_repo_url(),
    }


@app.get("/api/yaml-config")
async def get_yaml_config():
    """Return empty config — all datasets and presets are managed via the Portal UI."""
    return {"datasets": {}, "presets": {}, "active": {}}


# ---------------------------------------------------------------------------
# Managed Dataset CRUD
# ---------------------------------------------------------------------------


@app.get("/api/managed-datasets")
async def list_managed_datasets():
    return history.get_all_datasets()


@app.post("/api/managed-datasets")
async def create_managed_dataset(req: DatasetCreateRequest):
    runner_ids = req.runner_ids
    data = req.model_dump(exclude_none=True)
    data.pop("runner_ids", None)
    try:
        ds = history.create_dataset(data)
    except Exception as exc:
        if "UNIQUE constraint" in str(exc):
            raise HTTPException(status_code=409, detail=f"Dataset '{req.name}' already exists")
        raise HTTPException(status_code=400, detail=str(exc))
    if runner_ids is not None:
        history.set_dataset_runners(ds["id"], runner_ids)
        ds["runner_ids"] = runner_ids
    return ds


@app.get("/api/managed-datasets/export.yaml")
async def export_datasets_yaml():
    """Export all managed datasets as a downloadable YAML file."""
    import yaml as _yaml

    datasets = history.get_all_datasets()
    export: dict[str, Any] = {}
    _SKIP = {"id", "created_at", "updated_at", "runner_ids"}
    for ds in datasets:
        entry: dict[str, Any] = {}
        for k, v in ds.items():
            if k in _SKIP or k == "name":
                continue
            if v is None or v == "" or v == [] or v is False:
                continue
            if k == "recall_required" and v is True:
                entry[k] = True
            elif v is not True and v is not False:
                entry[k] = v
            else:
                entry[k] = v
        export[ds["name"]] = entry

    content = _yaml.dump(export, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return Response(
        content=content,
        media_type="application/x-yaml",
        headers={"Content-Disposition": "attachment; filename=datasets.yaml"},
    )


@app.post("/api/managed-datasets/import")
async def import_datasets_yaml(file: UploadFile = File(...)):
    """Import datasets from an uploaded YAML file.

    Each top-level key is a dataset name.  Existing datasets with the same
    name are updated; new names are inserted.
    """
    import yaml as _yaml

    raw = await file.read()
    try:
        data = _yaml.safe_load(raw)
    except _yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="YAML root must be a mapping of dataset names to their config")

    created = 0
    updated = 0
    for name, body in data.items():
        if not isinstance(name, str) or not name.strip():
            continue
        body = body if isinstance(body, dict) else {}
        payload = {"name": name.strip(), **body}
        payload.setdefault("path", "")
        existing = history.get_dataset_by_name(name.strip())
        if existing:
            history.update_dataset(existing["id"], payload)
            updated += 1
        else:
            history.create_dataset(payload)
            created += 1

    return {"ok": True, "created": created, "updated": updated}


@app.get("/api/managed-datasets/{dataset_id}")
async def get_managed_dataset(dataset_id: int):
    row = history.get_dataset_by_id(dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return row


@app.put("/api/managed-datasets/{dataset_id}")
async def update_managed_dataset(dataset_id: int, req: DatasetUpdateRequest):
    runner_ids = req.runner_ids
    data = {k: v for k, v in req.model_dump().items() if v is not None}
    data.pop("runner_ids", None)
    row = history.update_dataset(dataset_id, data)
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if runner_ids is not None:
        history.set_dataset_runners(dataset_id, runner_ids)
        row["runner_ids"] = runner_ids
    return row


@app.delete("/api/managed-datasets/{dataset_id}")
async def delete_managed_dataset(dataset_id: int):
    if not history.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Managed Preset CRUD
# ---------------------------------------------------------------------------


@app.get("/api/managed-presets")
async def list_managed_presets():
    return history.get_all_presets()


@app.post("/api/managed-presets")
async def create_managed_preset(req: PresetCreateRequest):
    data = req.model_dump(exclude_none=True)
    try:
        return history.create_preset(data)
    except Exception as exc:
        if "UNIQUE constraint" in str(exc):
            raise HTTPException(status_code=409, detail=f"Preset '{req.name}' already exists")
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/managed-presets/export.yaml")
async def export_presets_yaml():
    """Export all managed presets as a downloadable YAML file."""
    import yaml as _yaml

    presets = history.get_all_presets()
    export: dict[str, Any] = {}
    for p in presets:
        entry: dict[str, Any] = {}
        if p.get("description"):
            entry["description"] = p["description"]
        cfg = p.get("config")
        if isinstance(cfg, dict) and cfg:
            entry["config"] = cfg
        ovr = p.get("overrides")
        if isinstance(ovr, dict) and ovr:
            entry["overrides"] = ovr
        tags = p.get("tags")
        if isinstance(tags, list) and tags:
            entry["tags"] = tags
        export[p["name"]] = entry

    content = _yaml.dump(export, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return Response(
        content=content,
        media_type="application/x-yaml",
        headers={"Content-Disposition": "attachment; filename=presets.yaml"},
    )


@app.post("/api/managed-presets/import")
async def import_presets_yaml(file: UploadFile = File(...)):
    """Import presets from an uploaded YAML file.

    Each top-level key is a preset name.  Existing presets with the same name
    are updated; new names are inserted.
    """
    import yaml as _yaml

    raw = await file.read()
    try:
        data = _yaml.safe_load(raw)
    except _yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="YAML root must be a mapping of preset names to their config")

    created = 0
    updated = 0
    for name, body in data.items():
        if not isinstance(name, str) or not name.strip():
            continue
        body = body if isinstance(body, dict) else {}
        existing = history.get_preset_by_name(name.strip())
        payload = {
            "name": name.strip(),
            "description": body.get("description"),
            "config": body.get("config") or {},
            "tags": body.get("tags") or [],
            "overrides": body.get("overrides") or {},
        }
        if existing:
            history.update_preset(existing["id"], payload)
            updated += 1
        else:
            history.create_preset(payload)
            created += 1

    return {"ok": True, "created": created, "updated": updated}


@app.get("/api/managed-presets/{preset_id}")
async def get_managed_preset(preset_id: int):
    row = history.get_preset_by_id(preset_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    return row


@app.put("/api/managed-presets/{preset_id}")
async def update_managed_preset(preset_id: int, req: PresetUpdateRequest):
    data = {k: v for k, v in req.model_dump().items() if v is not None}
    row = history.update_preset(preset_id, data)
    if row is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    return row


@app.delete("/api/managed-presets/{preset_id}")
async def delete_managed_preset(preset_id: int):
    if not history.delete_preset(preset_id):
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Preset Matrix CRUD
# ---------------------------------------------------------------------------


@app.get("/api/preset-matrices")
async def list_preset_matrices():
    return history.get_all_preset_matrices()


@app.post("/api/preset-matrices")
async def create_preset_matrix(req: PresetMatrixCreateRequest):
    data = req.model_dump(exclude_none=True)
    try:
        return history.create_preset_matrix(data)
    except Exception as exc:
        if "UNIQUE constraint" in str(exc):
            raise HTTPException(status_code=409, detail=f"Preset matrix '{req.name}' already exists")
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/preset-matrices/{matrix_id}")
async def get_preset_matrix(matrix_id: int):
    row = history.get_preset_matrix_by_id(matrix_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Preset matrix not found")
    return row


@app.put("/api/preset-matrices/{matrix_id}")
async def update_preset_matrix(matrix_id: int, req: PresetMatrixUpdateRequest):
    data = {k: v for k, v in req.model_dump().items() if v is not None}
    row = history.update_preset_matrix(matrix_id, data)
    if row is None:
        raise HTTPException(status_code=404, detail="Preset matrix not found")
    return row


@app.delete("/api/preset-matrices/{matrix_id}")
async def delete_preset_matrix(matrix_id: int):
    if not history.delete_preset_matrix(matrix_id):
        raise HTTPException(status_code=404, detail="Preset matrix not found")
    return {"ok": True}


@app.post("/api/preset-matrices/{matrix_id}/trigger", response_model=MatrixTriggerResponse)
async def trigger_preset_matrix(matrix_id: int, req: MatrixTriggerRequest | None = None):
    from nemo_retriever.harness.scheduler import match_runner

    matrix = history.get_preset_matrix_by_id(matrix_id)
    if matrix is None:
        raise HTTPException(status_code=404, detail="Preset matrix not found")

    dataset_names: list[str] = matrix.get("dataset_names") or []
    preset_names: list[str] = matrix.get("preset_names") or []
    if not dataset_names or not preset_names:
        raise HTTPException(status_code=400, detail="Matrix must have at least one dataset and one preset")

    req_ref = req.git_ref if req else None
    req_commit = req.git_commit if req else None
    if not req_ref and not req_commit:
        req_ref = matrix.get("git_ref")
        req_commit = matrix.get("git_commit")
    pinned_sha, pinned_ref = _resolve_git_override(req_ref, req_commit)

    matrix_run_id = str(uuid.uuid4())
    preferred_runner_id = matrix.get("preferred_runner_id")
    gpu_type_filter = matrix.get("gpu_type_filter")
    matrix_tags = matrix.get("tags") or []

    nsys_flag = req.nsys_profile if (req and req.nsys_profile is not None) else bool(matrix.get("nsys_profile"))
    nsys_val = int(bool(nsys_flag))

    job_ids: list[str] = []
    for ds_name in dataset_names:
        dataset_path, dataset_overrides = _resolve_dataset_config(ds_name)
        for pr_name in preset_names:
            runner = match_runner(
                gpu_type_pattern=gpu_type_filter,
                preferred_runner_id=preferred_runner_id,
                dataset_name=ds_name,
            )
            preset_overrides = _resolve_preset_overrides(pr_name)
            merged_overrides = {**(dataset_overrides or {}), **preset_overrides}
            job = history.create_job(
                {
                    "trigger_source": "matrix",
                    "dataset": ds_name,
                    "dataset_path": dataset_path,
                    "dataset_overrides": merged_overrides if merged_overrides else None,
                    "preset": pr_name,
                    "assigned_runner_id": runner["id"] if runner else None,
                    "git_commit": pinned_sha,
                    "git_ref": pinned_ref,
                    "tags": matrix_tags,
                    "matrix_run_id": matrix_run_id,
                    "matrix_name": matrix["name"],
                    "nsys_profile": nsys_val,
                }
            )
            job_ids.append(job["id"])

    return MatrixTriggerResponse(
        matrix_id=matrix["id"],
        matrix_name=matrix["name"],
        matrix_run_id=matrix_run_id,
        job_ids=job_ids,
        job_count=len(job_ids),
    )


# ---------------------------------------------------------------------------
# Trigger / Jobs (persistent)
# ---------------------------------------------------------------------------


def _resolve_dataset_config(dataset_name: str) -> tuple[str | None, dict[str, Any] | None]:
    """Look up the filesystem path and full config overrides for a dataset.

    Checks managed datasets first, then falls back to the YAML config.
    Returns (dataset_path, overrides_dict) — either or both may be ``None``.
    """
    managed = history.get_dataset_by_name(dataset_name)
    if managed and managed.get("path"):
        overrides: dict[str, Any] = {"dataset_dir": managed["path"]}
        overrides["query_csv"] = managed.get("query_csv") or None
        if managed.get("input_type"):
            overrides["input_type"] = managed["input_type"]
        if managed.get("recall_required") is not None:
            overrides["recall_required"] = managed["recall_required"]
        else:
            overrides["recall_required"] = bool(overrides["query_csv"])
        if managed.get("recall_match_mode"):
            overrides["recall_match_mode"] = managed["recall_match_mode"]
        if managed.get("recall_adapter"):
            overrides["recall_adapter"] = managed["recall_adapter"]

        eval_mode = managed.get("evaluation_mode")
        if eval_mode:
            overrides["evaluation_mode"] = eval_mode
        if managed.get("beir_loader"):
            overrides["beir_loader"] = managed["beir_loader"]
        if managed.get("beir_dataset_name"):
            overrides["beir_dataset_name"] = managed["beir_dataset_name"]
        if managed.get("beir_split"):
            overrides["beir_split"] = managed["beir_split"]
        if managed.get("beir_query_language"):
            overrides["beir_query_language"] = managed["beir_query_language"]
        if managed.get("beir_doc_id_field"):
            overrides["beir_doc_id_field"] = managed["beir_doc_id_field"]
        beir_ks = managed.get("beir_ks")
        if beir_ks and isinstance(beir_ks, list):
            overrides["beir_ks"] = beir_ks
        if managed.get("embed_model_name"):
            overrides["embed_model_name"] = managed["embed_model_name"]
        if managed.get("embed_modality"):
            overrides["embed_modality"] = managed["embed_modality"]
        if managed.get("embed_granularity"):
            overrides["embed_granularity"] = managed["embed_granularity"]
        if managed.get("extract_page_as_image") is not None:
            overrides["extract_page_as_image"] = managed["extract_page_as_image"]
        if managed.get("extract_infographics") is not None:
            overrides["extract_infographics"] = managed["extract_infographics"]

        return managed["path"], overrides

    return None, None


def _resolve_preset_overrides(preset_name: str | None) -> dict[str, Any]:
    """Look up a managed preset and return its config + overrides merged together.

    The managed preset's tuning config fields and custom overrides are combined
    into a single dict that can be merged into job overrides / sweep_overrides.
    """
    if not preset_name:
        return {}
    managed = history.get_preset_by_name(preset_name)
    if not managed:
        return {}
    result: dict[str, Any] = {}
    cfg = managed.get("config")
    if isinstance(cfg, dict):
        result.update(cfg)
    ovr = managed.get("overrides")
    if isinstance(ovr, dict):
        result.update(ovr)
    return result


@app.post("/api/runs/trigger", response_model=TriggerResponse)
async def trigger_run(req: TriggerRequest):
    dataset_path, dataset_overrides = _resolve_dataset_config(req.dataset)
    preset_overrides = _resolve_preset_overrides(req.preset)
    merged_overrides = {**(dataset_overrides or {}), **preset_overrides}
    pinned_sha, pinned_ref = _resolve_git_override(req.git_ref, req.git_commit)

    if req.graph_id is not None:
        graph = history.get_graph(req.graph_id)
        if not graph:
            raise HTTPException(404, "Graph not found")
        code = (graph.get("generated_code") or "").strip()
        if not code:
            raise HTTPException(400, "Graph has no generated code. Save the graph first.")

        graph_name = graph.get("name") or f"graph-{req.graph_id}"
        job = history.create_job(
            {
                "trigger_source": "graph",
                "dataset": req.dataset,
                "dataset_path": dataset_path,
                "dataset_overrides": merged_overrides if merged_overrides else None,
                "preset": req.preset,
                "config": req.config,
                "assigned_runner_id": req.runner_id,
                "git_commit": pinned_sha,
                "git_ref": pinned_ref,
                "tags": req.tags or ["graph-run", graph_name],
                "nsys_profile": int(req.nsys_profile),
                "graph_code": code,
                "graph_id": req.graph_id,
            }
        )
    else:
        job = history.create_job(
            {
                "trigger_source": "manual",
                "dataset": req.dataset,
                "dataset_path": dataset_path,
                "dataset_overrides": merged_overrides if merged_overrides else None,
                "preset": req.preset,
                "config": req.config,
                "assigned_runner_id": req.runner_id,
                "git_commit": pinned_sha,
                "git_ref": pinned_ref,
                "tags": req.tags or [],
                "nsys_profile": int(req.nsys_profile),
                "extra_packages": req.extra_packages or [],
            }
        )

    return TriggerResponse(job_id=job["id"], status="pending")


@app.get("/api/jobs")
async def list_jobs():
    return history.get_jobs()


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = history.get_job_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/diagnose")
async def diagnose_job(job_id: str):
    """Explain why a pending job has not yet been picked up by a runner."""
    job = history.get_job_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "pending":
        return {
            "job_id": job_id,
            "status": job["status"],
            "reasons": [],
            "summary": f"Job is {job['status']}, not pending.",
        }

    runners = history.get_runners()
    reasons: list[dict[str, Any]] = []
    eligible_count = 0

    assigned_id = job.get("assigned_runner_id")
    dataset_name = job.get("dataset")
    allowed_runner_ids = history.get_runner_ids_for_dataset_name(dataset_name) if dataset_name else None
    rejected_runners = job.get("rejected_runners") or []
    rejected_set = {str(rid) for rid in rejected_runners}

    for r in runners:
        rid = r["id"]
        rname = r.get("name") or r.get("hostname") or f"Runner #{rid}"
        blockers = []

        if r.get("status") == "offline":
            blockers.append("Runner is offline")
        elif r.get("status") == "paused":
            blockers.append("Runner is paused (maintenance mode)")

        if assigned_id is not None and assigned_id != rid:
            blockers.append(f"Job is assigned to runner #{assigned_id}, not this runner")

        if allowed_runner_ids is not None and rid not in allowed_runner_ids:
            blockers.append(
                f"Dataset '{dataset_name}' restricts eligible runners — this runner is not in the allowed list"
            )

        if str(rid) in rejected_set:
            blockers.append("Runner previously rejected this job (e.g. missing dataset on disk)")

        if history.runner_has_running_job(rid):
            blockers.append("Runner is already executing another job")

        pending_update = r.get("pending_update_commit")
        if pending_update:
            blockers.append(f"Runner has a pending code update to {pending_update[:12]}")

        if not blockers:
            eligible_count += 1

        reasons.append(
            {
                "runner_id": rid,
                "runner_name": rname,
                "status": r.get("status", "unknown"),
                "eligible": len(blockers) == 0,
                "blockers": blockers,
            }
        )

    if not runners:
        summary = "No runners are registered with the portal."
    elif eligible_count == 0:
        summary = f"No eligible runners out of {len(runners)} registered. All runners have blocking conditions — expand each runner below for details."
    else:
        summary = f"{eligible_count} of {len(runners)} runner(s) are eligible. The job should be picked up on the next heartbeat cycle."

    return {
        "job_id": job_id,
        "status": "pending",
        "dataset": dataset_name,
        "assigned_runner_id": assigned_id,
        "allowed_runner_ids": allowed_runner_ids,
        "rejected_runners": rejected_runners,
        "runner_count": len(runners),
        "eligible_count": eligible_count,
        "summary": summary,
        "runners": reasons,
    }


@app.post("/api/jobs/{job_id}/claim")
async def claim_job(job_id: str):
    if not history.claim_job(job_id):
        raise HTTPException(status_code=409, detail="Job not claimable (already running or completed)")
    return {"ok": True}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job_endpoint(job_id: str):
    """Request cancellation of a pending or running job."""
    if not history.request_job_cancel(job_id):
        raise HTTPException(status_code=409, detail="Job cannot be cancelled (not pending or running)")
    return {"ok": True}


@app.post("/api/matrix-runs/{matrix_run_id}/cancel")
async def cancel_matrix_run(matrix_run_id: str):
    """Cancel all pending and running jobs that belong to a matrix run."""
    count = history.cancel_jobs_by_matrix_run_id(matrix_run_id)
    return {"ok": True, "cancelled_count": count}


@app.delete("/api/jobs/{job_id}")
async def force_delete_job(job_id: str):
    """Permanently delete a job regardless of its current status."""
    if not history.force_delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}


@app.post("/api/jobs/{job_id}/reject")
async def reject_job_endpoint(job_id: str, req: JobRejectRequest):
    """A runner reports it cannot execute this job (e.g. missing dataset).

    The runner is added to the job's rejected list so it won't be offered
    again, and a system alert is created so the operator can resolve the issue.
    """
    job = history.get_job_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    history.reject_job_by_runner(job_id, req.runner_id, reason=req.reason)

    runner = history.get_runner_by_id(req.runner_id)
    runner_label = (
        (runner.get("name") or runner.get("hostname") or f"#{req.runner_id}") if runner else f"#{req.runner_id}"
    )
    dataset_label = job.get("dataset", "unknown")
    dataset_path = job.get("dataset_path") or "N/A"

    try:
        rule = history.get_or_create_system_alert_rule(
            "Dataset Not Found on Runner",
            description="Fired when a runner cannot find a configured dataset directory on its filesystem.",
        )
        history.create_alert_event(
            {
                "rule_id": rule["id"],
                "run_id": 0,
                "metric": "system",
                "metric_value": None,
                "threshold": 0,
                "operator": "!=",
                "message": f'Dataset "{dataset_label}" (path: {dataset_path}) not found on runner {runner_label}',
                "git_commit": job.get("git_commit"),
                "dataset": dataset_label,
                "hostname": runner.get("hostname") if runner else None,
            }
        )
        logger.warning(
            "Runner %s rejected job %s — dataset '%s' not found at %s",
            runner_label,
            job_id,
            dataset_label,
            dataset_path,
        )
    except Exception as exc:
        logger.error("Failed to create alert for rejected job %s: %s", job_id, exc)

    return {"ok": True}


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    """Return the stored log tail for a job."""
    job = history.get_job_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job.get("status"), "log_tail": job.get("log_tail", [])}


@app.post("/api/jobs/{job_id}/complete")
async def complete_job_endpoint(job_id: str, req: JobCompleteRequest):
    job_before = history.get_job_by_id(job_id)
    was_cancelling = job_before and job_before.get("status") == "cancelling"

    if req.log_tail:
        history.update_job_log(job_id, req.log_tail)

    if req.pip_list:
        history.update_job_pip_list(job_id, req.pip_list)

    if was_cancelling and not req.success:
        history.complete_job(job_id, success=False, result=req.result, error=req.error or "Cancelled by user")
        history.update_job_status(job_id, "cancelled", error=req.error or "Cancelled by user")
    else:
        history.complete_job(job_id, success=req.success, result=req.result, error=req.error)

    job = history.get_job_by_id(job_id)
    effective_success = req.success and not was_cancelling
    effective_error = req.error or ("Cancelled by user" if was_cancelling else None)
    run_id = _record_run_from_job(
        job,
        effective_success,
        req.result,
        effective_error,
        execution_commit=req.execution_commit,
        num_gpus=req.num_gpus,
    )

    return {"ok": True, "run_id": run_id}


def _normalize_runner_result(
    result: dict[str, Any],
    job: dict[str, Any],
    success: bool,
    error: str | None,
    execution_commit: str | None,
) -> dict[str, Any]:
    """Normalise a runner result dict into the canonical ``record_run`` format.

    The runner may send either the full ``_run_entry`` payload (which already
    has ``timestamp``, ``test_config``, ``summary_metrics``, ``run_metadata``,
    etc.) or a compact/legacy payload with top-level ``dataset``, ``preset``,
    and a flat ``metrics`` dict.  This function fills in the structural keys
    that ``record_run`` relies on so metrics are never silently discarded.
    """
    run_result = dict(result)

    if not run_result.get("timestamp"):
        run_result["timestamp"] = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")

    run_result.setdefault("latest_commit", execution_commit)
    run_result.setdefault("success", success)
    run_result.setdefault("failure_reason", error or (None if success else "Job failed"))

    if "test_config" not in run_result:
        run_result["test_config"] = {
            "dataset_label": result.get("dataset") or job.get("dataset", "unknown"),
            "preset": result.get("preset") or job.get("preset"),
        }

    if "summary_metrics" not in run_result:
        run_result["summary_metrics"] = dict(result.get("metrics") or {})

    if "run_metadata" not in run_result:
        runner_id = job.get("assigned_runner_id")
        runner = history.get_runner_by_id(runner_id) if runner_id else None
        run_result["run_metadata"] = {
            "host": (runner or {}).get("hostname"),
            "gpu_type": (runner or {}).get("gpu_type"),
        }

    if "artifacts" not in run_result and result.get("artifact_dir"):
        art = str(result["artifact_dir"])
        run_result["artifacts"] = {
            "runtime_metrics_dir": str(Path(art) / "runtime_metrics"),
        }

    run_result.setdefault("tags", job.get("tags"))
    return run_result


def _record_run_from_job(
    job: dict[str, Any] | None,
    success: bool,
    result: dict[str, Any] | None,
    error: str | None,
    execution_commit: str | None = None,
    num_gpus: int | None = None,
) -> int | None:
    """Create a run record in the runs table from a completed job.

    When the runner sends back a ``result`` dict it is normalised into the
    canonical format expected by ``record_run``.  If no result is available at
    all, a minimal stub is synthesised so the job still appears in the Runs
    view.

    Returns the newly created run row id, or ``None`` on failure.
    """
    if job is None:
        return None

    if result and isinstance(result, dict):
        run_result = _normalize_runner_result(result, job, success, error, execution_commit)
    else:
        logger.warning(
            "Job %s completed with no result dict — metrics will be empty in the Runs view",
            job.get("id"),
        )
        now_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
        run_result = {
            "timestamp": now_ts,
            "latest_commit": execution_commit,
            "success": success,
            "return_code": None,
            "failure_reason": error or (None if success else "Job failed (no result returned)"),
            "test_config": {
                "dataset_label": job.get("dataset", "unknown"),
                "preset": job.get("preset"),
            },
            "metrics": {},
            "summary_metrics": {},
            "run_metadata": {},
            "artifacts": {},
            "tags": job.get("tags"),
        }

    trigger_source = job.get("trigger_source")
    schedule_id = job.get("schedule_id")
    artifacts = run_result.get("artifacts") or {}
    runtime_metrics_dir = artifacts.get("runtime_metrics_dir", "")
    if runtime_metrics_dir:
        artifact_dir = str(Path(runtime_metrics_dir).parent)
    else:
        command_file = artifacts.get("command_file", "")
        artifact_dir = str(Path(command_file).parent) if command_file else run_result.get("artifact_dir", "")

    try:
        run_row_id = history.record_run(
            run_result,
            artifact_dir=artifact_dir,
            trigger_source=trigger_source,
            schedule_id=schedule_id,
            execution_commit=execution_commit,
            num_gpus=num_gpus,
            job_id=job.get("id"),
            nsys_profile=int(bool(job.get("nsys_profile"))),
        )
        if run_row_id:
            run_row = history.get_run_by_id(run_row_id)
            if run_row:
                try:
                    history.evaluate_alerts_for_run(run_row)
                except Exception as alert_exc:
                    logger.error("Alert evaluation failed for run %s: %s", run_row_id, alert_exc)
        return run_row_id
    except Exception as exc:
        logger.error("Failed to record run for job %s: %s", job.get("id"), exc)
        return None


# ---------------------------------------------------------------------------
# Runner endpoints
# ---------------------------------------------------------------------------


@app.get("/api/runners")
async def list_runners():
    return history.get_runners()


@app.get("/api/runners/gpu-types")
async def list_gpu_types():
    runners = history.get_runners()
    gpu_types: set[str] = set()
    for r in runners:
        gt = r.get("gpu_type")
        if gt:
            gpu_types.add(gt)
    return sorted(gpu_types)


@app.post("/api/runners")
async def create_runner(req: RunnerCreateRequest):
    data = req.model_dump(exclude_unset=True)
    return history.register_runner(data)


@app.get("/api/runners/{runner_id}")
async def get_runner(runner_id: int):
    runner = history.get_runner_by_id(runner_id)
    if runner is None:
        raise HTTPException(status_code=404, detail="Runner not found")
    return runner


@app.put("/api/runners/{runner_id}")
async def update_runner_endpoint(runner_id: int, req: RunnerUpdateRequest):
    if history.get_runner_by_id(runner_id) is None:
        raise HTTPException(status_code=404, detail="Runner not found")
    data = req.model_dump(exclude_unset=True)
    return history.update_runner(runner_id, data)


@app.delete("/api/runners/{runner_id}")
async def delete_runner_endpoint(runner_id: int):
    if not history.delete_runner(runner_id):
        raise HTTPException(status_code=404, detail="Runner not found")
    return {"ok": True}


@app.post("/api/runners/{runner_id}/pause")
async def pause_runner_endpoint(runner_id: int):
    """Temporarily pause a runner so no new jobs are dispatched to it."""
    if not history.pause_runner(runner_id):
        raise HTTPException(status_code=404, detail="Runner not found")
    return {"ok": True, "status": "paused"}


@app.post("/api/runners/{runner_id}/resume")
async def resume_runner_endpoint(runner_id: int):
    """Resume a paused runner so it can receive jobs again."""
    if not history.resume_runner(runner_id):
        raise HTTPException(status_code=404, detail="Runner not found")
    return {"ok": True, "status": "online"}


class JobRejectRequest(BaseModel):
    runner_id: int
    reason: str = "Dataset not found on runner"


class HeartbeatRequest(BaseModel):
    current_job_id: str | None = None
    log_tail: list[str] | None = None
    git_commit: str | None = None


@app.post("/api/runners/{runner_id}/heartbeat")
async def runner_heartbeat(runner_id: int, req: HeartbeatRequest | None = None):
    runner_status = history.heartbeat_runner(
        runner_id,
        git_commit=req.git_commit if req else None,
    )
    if runner_status is None:
        raise HTTPException(status_code=404, detail="Runner not found")

    cancel_job_id: str | None = None

    if req and req.current_job_id:
        if req.log_tail:
            history.update_job_log(req.current_job_id, req.log_tail)
        current_job = history.get_job_by_id(req.current_job_id)
        if current_job and current_job.get("status") == "cancelling":
            cancel_job_id = req.current_job_id

    next_job = None
    if runner_status != "paused":
        jobs = history.get_pending_jobs_for_runner(runner_id)
        next_job = _pick_job_for_runner(jobs, runner_id)

    runner_record = history.get_runner_by_id(runner_id)
    update_to = runner_record.get("pending_update_commit") if runner_record else None
    ray_addr = runner_record.get("ray_address") if runner_record else None
    run_code_ref = history.get_portal_setting("run_code_ref") or "upstream/main"

    return {
        "ok": True,
        "job": next_job,
        "cancel_job_id": cancel_job_id,
        "status": runner_status,
        "update_to_commit": update_to,
        "ray_address": ray_addr,
        "run_code_ref": run_code_ref,
    }


class RunnerUpdateCompleteRequest(BaseModel):
    previous_commit: str | None = None
    new_commit: str | None = None


@app.post("/api/runners/{runner_id}/update-complete")
async def runner_update_complete(runner_id: int, req: RunnerUpdateCompleteRequest):
    """Called by a runner after it restarts from a portal-triggered code update."""
    runner = history.get_runner_by_id(runner_id)
    if not runner:
        raise HTTPException(status_code=404, detail="Runner not found")

    rname = runner.get("name") or runner.get("hostname") or f"Runner #{runner_id}"
    prev_short = (req.previous_commit or "unknown")[:12]
    new_short = (req.new_commit or "unknown")[:12]

    logger.info("Runner #%s (%s) completed update: %s → %s", runner_id, rname, prev_short, new_short)

    history.clear_pending_update(runner_id)

    try:
        history.create_alert_event(
            {
                "rule_id": None,
                "run_id": None,
                "metric": "runner_update",
                "metric_value": None,
                "threshold": 0,
                "operator": "info",
                "message": f"Runner '{rname}' (#{runner_id}) restarted with updated code: {prev_short} → {new_short}",
                "git_commit": req.new_commit,
                "dataset": None,
                "preset": None,
                "hostname": runner.get("hostname"),
            }
        )
    except Exception as exc:
        logger.warning("Failed to create alert event for runner update: %s", exc)

    return {"ok": True, "message": f"Update acknowledged: {prev_short} → {new_short}"}


@app.get("/api/runners/{runner_id}/work")
async def runner_get_work(runner_id: int):
    """Return the next pending job for this runner (assigned or unassigned), or 204 if none."""
    runner = history.get_runner_by_id(runner_id)
    if not runner:
        raise HTTPException(status_code=404, detail="Runner not found")
    if runner.get("status") in ("offline", "paused"):
        return Response(status_code=204)
    jobs = history.get_pending_jobs_for_runner(runner_id)
    job = _pick_job_for_runner(jobs, runner_id)
    if not job:
        return Response(status_code=204)
    return job


def _pick_job_for_runner(jobs: list[dict[str, Any]], runner_id: int) -> dict[str, Any] | None:
    """Select the first pending job this runner is allowed to run.

    Respects dataset→runner associations: if a dataset restricts which
    runners may use it, only those runners can pick up jobs for it.
    """
    for job in jobs:
        dataset_name = job.get("dataset")
        if dataset_name:
            allowed_runner_ids = history.get_runner_ids_for_dataset_name(dataset_name)
            if allowed_runner_ids is not None and runner_id not in allowed_runner_ids:
                continue
        if job.get("assigned_runner_id") is None:
            history.assign_job_to_runner(job["id"], runner_id)
        return job
    return None


# ---------------------------------------------------------------------------
# Schedule endpoints
# ---------------------------------------------------------------------------


def _compute_next_run(cron_expression: str, count: int = 1) -> list[str]:
    """Compute the next ``count`` fire times for a cron expression.

    Returns ISO-8601 UTC strings.
    """
    try:
        cron_kwargs = sched_module._parse_cron_expression(cron_expression)
        trigger = CronTrigger(**cron_kwargs)
        now = datetime.now(timezone.utc)
        times: list[str] = []
        for _ in range(count):
            nxt = trigger.get_next_fire_time(None, now)
            if nxt is None:
                break
            times.append(nxt.strftime("%Y-%m-%dT%H:%M:%SZ"))
            now = nxt + timedelta(seconds=1)
        return times
    except Exception:
        return []


def _enrich_schedule_next_run(schedule: dict[str, Any]) -> dict[str, Any]:
    """Add ``next_run_at`` and ``pending_jobs`` to a schedule dict."""
    if schedule.get("trigger_type") == "cron" and schedule.get("enabled") and schedule.get("cron_expression"):
        times = _compute_next_run(schedule["cron_expression"], 1)
        schedule["next_run_at"] = times[0] if times else None
    else:
        schedule["next_run_at"] = None
    pending = history.get_pending_jobs_for_schedule(schedule["id"])
    schedule["pending_jobs"] = len(pending)
    return schedule


@app.get("/api/schedules")
async def list_schedules():
    schedules = history.get_schedules()
    return [_enrich_schedule_next_run(s) for s in schedules]


@app.get("/api/schedules/upcoming")
async def list_upcoming(count: int = Query(10, ge=1, le=50)):
    """Return the next ``count`` scheduled fire times across all enabled cron schedules."""
    schedules = history.get_enabled_schedules(trigger_type="cron")
    entries: list[dict[str, Any]] = []
    for sched in schedules:
        expr = sched.get("cron_expression")
        if not expr:
            continue
        pending = history.get_pending_jobs_for_schedule(sched["id"])
        times = _compute_next_run(expr, count)
        for t in times:
            entries.append(
                {
                    "schedule_id": sched["id"],
                    "schedule_name": sched.get("name", ""),
                    "dataset": sched.get("dataset", ""),
                    "preset": sched.get("preset"),
                    "cron_expression": expr,
                    "fire_at": t,
                    "pending_jobs": len(pending),
                }
            )
    entries.sort(key=lambda e: e["fire_at"])
    return entries[:count]


@app.post("/api/schedules")
async def create_schedule(req: ScheduleCreateRequest):
    data = req.model_dump(exclude_unset=True)
    schedule = history.create_schedule(data)
    sched_module.sync_schedule(schedule["id"])
    return schedule


@app.get("/api/schedules/{schedule_id}")
async def get_schedule(schedule_id: int):
    schedule = history.get_schedule_by_id(schedule_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return schedule


@app.put("/api/schedules/{schedule_id}")
async def update_schedule_endpoint(schedule_id: int, req: ScheduleUpdateRequest):
    if history.get_schedule_by_id(schedule_id) is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    data = req.model_dump(exclude_unset=True)
    schedule = history.update_schedule(schedule_id, data)
    sched_module.sync_schedule(schedule_id)
    return schedule


@app.delete("/api/schedules/{schedule_id}")
async def delete_schedule_endpoint(schedule_id: int):
    if not history.delete_schedule(schedule_id):
        raise HTTPException(status_code=404, detail="Schedule not found")
    sched_module.sync_schedule(schedule_id)
    return {"ok": True}


@app.post("/api/schedules/{schedule_id}/trigger")
async def trigger_schedule(schedule_id: int):
    """Manually fire a schedule now, bypassing the cron timer."""
    job = sched_module.trigger_schedule_now(schedule_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return job


# ---------------------------------------------------------------------------
# GitHub Webhook
# ---------------------------------------------------------------------------


@app.post("/api/webhooks/github")
async def github_webhook(request: Request):
    """Receive GitHub push events and dispatch matching schedules."""
    body = await request.body()

    if GITHUB_WEBHOOK_SECRET:
        signature = request.headers.get("X-Hub-Signature-256", "")
        expected = "sha256=" + hmac.new(GITHUB_WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise HTTPException(status_code=403, detail="Invalid signature")

    event = request.headers.get("X-GitHub-Event", "")
    if event != "push":
        return {"ok": True, "skipped": True, "reason": f"event={event}"}

    try:
        payload = json_module.loads(body)
    except json_module.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    ref = payload.get("ref", "")
    branch = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ref
    repo_full = (payload.get("repository") or {}).get("full_name", "")
    commit_sha = payload.get("after", "")

    if not repo_full or not branch:
        return {"ok": True, "skipped": True, "reason": "missing repo or branch"}

    dispatched = sched_module.handle_github_webhook(repo_full, branch, commit_sha)
    return {"ok": True, "dispatched": len(dispatched), "jobs": [j["id"] for j in dispatched]}


# ---------------------------------------------------------------------------
# Alert Rule endpoints
# ---------------------------------------------------------------------------


@app.get("/api/alert-rules")
async def list_alert_rules():
    return history.get_alert_rules()


@app.post("/api/alert-rules")
async def create_alert_rule(req: AlertRuleCreateRequest):
    if req.metric not in history.VALID_ALERT_METRICS:
        raise HTTPException(
            status_code=400, detail=f"Invalid metric '{req.metric}'. Valid: {history.VALID_ALERT_METRICS}"
        )
    if req.operator not in history.VALID_ALERT_OPERATORS:
        raise HTTPException(
            status_code=400, detail=f"Invalid operator '{req.operator}'. Valid: {history.VALID_ALERT_OPERATORS}"
        )
    data = req.model_dump(exclude_none=True)
    return history.create_alert_rule(data)


@app.get("/api/alert-rules/{rule_id}")
async def get_alert_rule(rule_id: int):
    rule = history.get_alert_rule_by_id(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return rule


@app.put("/api/alert-rules/{rule_id}")
async def update_alert_rule_endpoint(rule_id: int, req: AlertRuleUpdateRequest):
    data = {k: v for k, v in req.model_dump().items() if v is not None}
    if "metric" in data and data["metric"] not in history.VALID_ALERT_METRICS:
        raise HTTPException(status_code=400, detail=f"Invalid metric. Valid: {history.VALID_ALERT_METRICS}")
    if "operator" in data and data["operator"] not in history.VALID_ALERT_OPERATORS:
        raise HTTPException(status_code=400, detail=f"Invalid operator. Valid: {history.VALID_ALERT_OPERATORS}")
    rule = history.update_alert_rule(rule_id, data)
    if rule is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return rule


@app.delete("/api/alert-rules/{rule_id}")
async def delete_alert_rule_endpoint(rule_id: int):
    if not history.delete_alert_rule(rule_id):
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Alert Event endpoints
# ---------------------------------------------------------------------------


@app.get("/api/alert-events")
async def list_alert_events(
    limit: int = Query(200, ge=1, le=5000),
    rule_id: int | None = Query(None),
    acknowledged: bool | None = Query(None),
):
    return history.get_alert_events(limit=limit, rule_id=rule_id, acknowledged=acknowledged)


@app.post("/api/alert-events/{event_id}/acknowledge")
async def acknowledge_alert_event_endpoint(event_id: int):
    if not history.acknowledge_alert_event(event_id):
        raise HTTPException(status_code=404, detail="Alert event not found")
    return {"ok": True}


@app.post("/api/alert-events/acknowledge-all")
async def acknowledge_all_alerts():
    count = history.acknowledge_all_alert_events()
    return {"ok": True, "acknowledged": count}


@app.get("/api/alert-metrics")
async def get_alert_metrics():
    """Return valid metric names for alert rules."""
    return {"metrics": history.VALID_ALERT_METRICS, "operators": history.VALID_ALERT_OPERATORS}


# ---------------------------------------------------------------------------
# System / Settings
# ---------------------------------------------------------------------------


def _git_run(*args: str, cwd: str | None = None, timeout: int = 30) -> str:
    """Run a git command and return stripped stdout."""
    return subprocess.check_output(
        ["git", *args],
        cwd=cwd,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    ).strip()


def _resolve_run_code_ref_sha() -> tuple[str | None, str | None]:
    """Resolve the portal ``run_code_ref`` setting to a concrete SHA.

    Fetches the latest refs and runs ``git rev-parse`` so that callers
    get a pinned commit hash instead of a symbolic ref like
    ``nvidia/main``.  Returns ``(sha, ref)`` — both may be ``None`` if
    the setting is missing or the resolve fails.
    """
    ref = history.get_portal_setting("run_code_ref")
    if not ref:
        return None, None
    try:
        if "/" in ref and not ref.startswith("origin/"):
            remote_name = ref.split("/")[0]
            _git_run("fetch", remote_name, "--prune")
        else:
            _git_run("fetch", "--all", "--prune")
        sha = _git_run("rev-parse", ref)
        return sha, ref
    except Exception as exc:
        logger.warning("Failed to resolve run_code_ref '%s' to SHA: %s", ref, exc)
        return ref, ref


def _resolve_git_override(git_ref: str | None, git_commit: str | None) -> tuple[str | None, str | None]:
    """Resolve explicit per-trigger git overrides, falling back to the global setting.

    Returns ``(sha, ref)``.

    * If *git_commit* is provided it is used as-is (exact SHA checkout).
      *git_ref* is stored alongside it for display purposes.
    * If only *git_ref* is provided the latest SHA for that ref is resolved
      via ``git fetch`` + ``git rev-parse``.
    * If neither is provided the global ``run_code_ref`` portal setting is
      used (existing behaviour).
    """
    if git_commit:
        return git_commit, git_ref or git_commit
    if git_ref:
        try:
            if "/" in git_ref and not git_ref.startswith("origin/"):
                remote_name = git_ref.split("/")[0]
                _git_run("fetch", remote_name, "--prune")
            else:
                _git_run("fetch", "--all", "--prune")
            sha = _git_run("rev-parse", git_ref)
            return sha, git_ref
        except Exception as exc:
            logger.warning("Failed to resolve git_ref '%s' to SHA: %s", git_ref, exc)
            return git_ref, git_ref
    return _resolve_run_code_ref_sha()


# ---------------------------------------------------------------------------
# Portal settings (key/value config)
# ---------------------------------------------------------------------------


@app.get("/api/portal-settings")
async def get_portal_settings():
    settings = history.get_all_portal_settings()
    nvidia_remote, nvidia_url = _detect_nvidia_remote()
    settings["_nvidia_remote_name"] = nvidia_remote
    settings["_nvidia_remote_url"] = nvidia_url
    return settings


class PortalSettingsUpdateRequest(BaseModel):
    run_code_ref: str | None = None


@app.put("/api/portal-settings")
async def update_portal_settings(req: PortalSettingsUpdateRequest):
    if req.run_code_ref is not None:
        history.set_portal_setting("run_code_ref", req.run_code_ref)
    return history.get_all_portal_settings()


# ---------------------------------------------------------------------------
# Settings — Git info & deploy
# ---------------------------------------------------------------------------


@app.get("/api/settings/git-info")
async def get_git_info():
    """Return information about the current git state of the portal codebase."""
    try:
        repo_root = _git_run("rev-parse", "--show-toplevel")
    except Exception:
        return {"available": False, "error": "Not a git repository or git not installed"}

    try:
        current_branch = _git_run("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root)
        current_sha = _git_run("rev-parse", "HEAD", cwd=repo_root)
        current_short = _git_run("rev-parse", "--short", "HEAD", cwd=repo_root)

        remote_names_raw = _git_run("remote", cwd=repo_root)
        remotes: list[dict[str, str]] = []
        for rname in remote_names_raw.splitlines():
            rname = rname.strip()
            if not rname:
                continue
            try:
                rurl = _git_run("remote", "get-url", rname, cwd=repo_root)
            except Exception:
                rurl = ""
            remotes.append({"name": rname, "url": rurl})

        nvidia_remote_name, _ = _detect_nvidia_remote()
        if nvidia_remote_name:
            remotes.sort(key=lambda r: (0 if r["name"] == nvidia_remote_name else 1, r["name"]))

        try:
            _git_run("fetch", "--all", "--prune", cwd=repo_root, timeout=15)
        except Exception:
            pass

        branches_raw = _git_run("branch", "-r", "--no-color", cwd=repo_root)
        remote_branches: list[str] = []
        for line in branches_raw.splitlines():
            b = line.strip()
            if "->" in b:
                continue
            remote_branches.append(b)
        remote_branches.sort()

        local_branches_raw = _git_run("branch", "--no-color", cwd=repo_root)
        local_branches: list[str] = []
        for line in local_branches_raw.splitlines():
            b = line.strip().lstrip("* ").strip()
            if b:
                local_branches.append(b)
        local_branches.sort()

        is_dirty = bool(_git_run("status", "--porcelain", cwd=repo_root))

        last_commits_raw = _git_run("log", "--oneline", "-10", "--format=%H|%h|%s|%ci", cwd=repo_root)
        recent_commits: list[dict[str, str]] = []
        for line in last_commits_raw.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                recent_commits.append(
                    {
                        "sha": parts[0],
                        "short_sha": parts[1],
                        "message": parts[2],
                        "date": parts[3],
                    }
                )

        return {
            "available": True,
            "repo_root": repo_root,
            "current_branch": current_branch,
            "current_sha": current_sha,
            "current_short_sha": current_short,
            "is_dirty": is_dirty,
            "remotes": remotes,
            "remote_branches": remote_branches,
            "local_branches": local_branches,
            "recent_commits": recent_commits,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


class DeployRequest(BaseModel):
    branch: str = "main"
    remote: str = ""
    update_runners: bool = True


@app.post("/api/settings/deploy")
async def deploy_latest(req: DeployRequest):
    """Pull the latest code from a remote branch and restart the portal.

    Steps:
    1. ``git fetch <remote>``
    2. ``git checkout <remote>/<branch>`` (or ``<branch>`` for local)
    3. ``git pull <remote> <branch>`` (if on a tracking branch)
    4. Restart the process via ``os.execv`` to pick up code changes.

    The HTTP response is sent *before* the restart so the client receives
    confirmation. A short delay gives the response time to flush.
    """
    import signal
    import sys
    import threading

    try:
        repo_root = _git_run("rev-parse", "--show-toplevel")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Git not available: {exc}")

    if not req.remote:
        nvidia_name, _ = _detect_nvidia_remote()
        req.remote = nvidia_name or "origin"

    log_lines: list[str] = []

    def _step(label: str, *args: str, **kwargs: Any) -> str:
        log_lines.append(f"$ git {' '.join(args)}")
        try:
            out = _git_run(*args, cwd=repo_root, **kwargs)
            if out:
                log_lines.append(out)
            return out
        except subprocess.CalledProcessError as exc:
            output = exc.output if exc.output else str(exc)
            log_lines.append(f"ERROR: {output}")
            raise HTTPException(
                status_code=500,
                detail=f"{label} failed: {output}\n\n" + "\n".join(log_lines),
            )

    try:
        _step("stash", "stash", "--include-untracked")
    except HTTPException:
        pass

    _step("fetch", "fetch", req.remote, timeout=30)

    current_branch = _git_run("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root)
    if current_branch == req.branch:
        _step("pull", "pull", req.remote, req.branch, timeout=60)
    else:
        try:
            _step("checkout", "checkout", req.branch)
            _step("pull", "pull", req.remote, req.branch, timeout=60)
        except HTTPException:
            _step("checkout remote", "checkout", f"{req.remote}/{req.branch}")

    new_sha_short = _git_run("rev-parse", "--short", "HEAD", cwd=repo_root)
    new_sha_full = _git_run("rev-parse", "HEAD", cwd=repo_root)
    log_lines.append(f"Now at {new_sha_short} on {req.branch}")

    updated_count = 0
    if req.update_runners:
        updated_count = history.set_pending_update_all_runners(new_sha_full)
        if updated_count:
            log_lines.append(f"Signalled {updated_count} runner(s) to update to {new_sha_short}")
    else:
        log_lines.append("Runner update skipped (portal-only deploy)")

    def _restart_after_delay():
        import time

        time.sleep(2)
        logger.info("Restarting portal process after deploy…")
        try:
            sched_module.stop_scheduler()
        except Exception:
            pass
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_restart_after_delay, daemon=True).start()

    runners_msg = f" {updated_count} runner(s) will update." if req.update_runners else " Runners were not updated."
    return {
        "ok": True,
        "new_sha": new_sha_short,
        "branch": req.branch,
        "remote": req.remote,
        "update_runners": req.update_runners,
        "runners_updated": updated_count,
        "log": log_lines,
        "message": f"Deployed {req.branch} ({new_sha_short}). Portal will restart in ~2 seconds.{runners_msg}",
    }


class UpdateRunnersRequest(BaseModel):
    branch: str = "main"
    remote: str = ""


@app.post("/api/settings/update-runners")
async def update_runners_only(req: UpdateRunnersRequest):
    """Signal all online/paused runners to update to the latest commit on a branch.

    This does NOT restart the portal — it only resolves the branch to a SHA
    and sets ``pending_update_commit`` on every eligible runner.
    """
    try:
        repo_root = _git_run("rev-parse", "--show-toplevel")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Git not available: {exc}")

    if not req.remote:
        nvidia_name, _ = _detect_nvidia_remote()
        req.remote = nvidia_name or "origin"

    try:
        _git_run("fetch", req.remote, "--prune", cwd=repo_root, timeout=30)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"git fetch failed: {exc}")

    ref = f"{req.remote}/{req.branch}"
    try:
        sha = _git_run("rev-parse", ref, cwd=repo_root)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not resolve {ref}: {exc}")

    short_sha = sha[:12]
    updated_count = history.set_pending_update_all_runners(sha)

    return {
        "ok": True,
        "sha": sha,
        "short_sha": short_sha,
        "ref": ref,
        "runners_updated": updated_count,
        "message": f"Signalled {updated_count} runner(s) to update to {short_sha} ({ref}).",
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


class ReportBundleImage(BaseModel):
    filename: str
    data_url: str


class ReportBundleRequest(BaseModel):
    images: list[ReportBundleImage]


@app.post("/api/reports/bundle")
async def bundle_report_images(req: ReportBundleRequest):
    """Accept base64 PNG data-URLs from the client and return a ZIP archive."""
    import base64

    if not req.images:
        raise HTTPException(status_code=400, detail="No images provided")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in req.images:
            parts = img.data_url.split(",", 1)
            raw = base64.b64decode(parts[-1])
            zf.writestr(img.filename, raw)
    buf.seek(0)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="harness_report_{ts}.zip"'},
    )


@app.get("/api/reports/export")
async def export_runs_json(
    dataset: str | None = Query(None),
    preset: str | None = Query(None),
    status: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    include_raw: bool = Query(True),
    limit: int = Query(5000, ge=1, le=50000),
):
    """Export historical run data as a downloadable JSON file.

    Supports the same filters as the reporting UI. Each run includes full
    detail and optionally the raw result JSON for offline analysis.
    """
    all_runs = history.get_runs(dataset=dataset, limit=limit)

    filtered: list[dict[str, Any]] = []
    for r in all_runs:
        if preset and r.get("preset") != preset:
            continue
        if status == "pass" and r.get("success") != 1:
            continue
        if status == "fail" and r.get("success") != 0:
            continue
        if date_from or date_to:
            ts = r.get("timestamp", "")
            if date_from and ts < date_from:
                continue
            if date_to and ts[:10] > date_to:
                continue
        filtered.append(r)

    export_runs = []
    for r in filtered:
        if include_raw:
            full = history.get_run_by_id(r["id"])
            if full:
                export_runs.append(full)
                continue
        export_runs.append(r)

    export_payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "dataset": dataset,
            "preset": preset,
            "status": status,
            "date_from": date_from,
            "date_to": date_to,
        },
        "total_runs": len(export_runs),
        "runs": export_runs,
    }

    content = json_module.dumps(export_payload, indent=2, default=str)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="harness_runs_export_{ts}.json"'},
    )


# ---------------------------------------------------------------------------
# Graph Designer API
# ---------------------------------------------------------------------------


_RAY_DATA_SOURCES: list[dict[str, Any]] = [
    {
        "class_name": "ReadBinaryFiles",
        "display_name": "Read Binary Files",
        "module": "ray.data",
        "import_path": "import ray.data",
        "type": "ray_data_source",
        "ray_fn": "ray.data.read_binary_files",
        "params": [
            {"name": "paths", "type": "str", "label": "Paths"},
            {"name": "include_paths", "type": "bool", "default": True, "label": "Include Paths"},
        ],
        "category": "Data Sources",
        "category_color": "#ff9f43",
        "compute": "cpu",
        "description": "Read binary files from a directory into a Ray Dataset",
    },
    {
        "class_name": "ReadCSV",
        "display_name": "Read CSV",
        "module": "ray.data",
        "import_path": "import ray.data",
        "type": "ray_data_source",
        "ray_fn": "ray.data.read_csv",
        "params": [
            {"name": "paths", "type": "str", "label": "Paths"},
        ],
        "category": "Data Sources",
        "category_color": "#ff9f43",
        "compute": "cpu",
        "description": "Read CSV files into a Ray Dataset",
    },
    {
        "class_name": "ReadParquet",
        "display_name": "Read Parquet",
        "module": "ray.data",
        "import_path": "import ray.data",
        "type": "ray_data_source",
        "ray_fn": "ray.data.read_parquet",
        "params": [
            {"name": "paths", "type": "str", "label": "Paths"},
        ],
        "category": "Data Sources",
        "category_color": "#ff9f43",
        "compute": "cpu",
        "description": "Read Parquet files into a Ray Dataset",
    },
    {
        "class_name": "ReadJSON",
        "display_name": "Read JSON",
        "module": "ray.data",
        "import_path": "import ray.data",
        "type": "ray_data_source",
        "ray_fn": "ray.data.read_json",
        "params": [
            {"name": "paths", "type": "str", "label": "Paths"},
        ],
        "category": "Data Sources",
        "category_color": "#ff9f43",
        "compute": "cpu",
        "description": "Read JSON files into a Ray Dataset",
    },
    {
        "class_name": "ReadImages",
        "display_name": "Read Images",
        "module": "ray.data",
        "import_path": "import ray.data",
        "type": "ray_data_source",
        "ray_fn": "ray.data.read_images",
        "params": [
            {"name": "paths", "type": "str", "label": "Paths"},
            {"name": "mode", "type": "str", "default": "RGB", "label": "Mode"},
        ],
        "category": "Data Sources",
        "category_color": "#ff9f43",
        "compute": "cpu",
        "description": "Read image files into a Ray Dataset",
    },
    {
        "class_name": "ReadText",
        "display_name": "Read Text",
        "module": "ray.data",
        "import_path": "import ray.data",
        "type": "ray_data_source",
        "ray_fn": "ray.data.read_text",
        "params": [
            {"name": "paths", "type": "str", "label": "Paths"},
        ],
        "category": "Data Sources",
        "category_color": "#ff9f43",
        "compute": "cpu",
        "description": "Read text files into a Ray Dataset",
    },
]


def _introspect_pydantic_fields(model_cls: type) -> list[dict[str, Any]] | None:
    """If *model_cls* is a Pydantic BaseModel, return its scalar field descriptors."""
    try:
        from pydantic import BaseModel as _BM
        from pydantic_core import PydanticUndefined as _PU
    except ImportError:
        return None
    if not (isinstance(model_cls, type) and issubclass(model_cls, _BM)):
        return None
    fields: list[dict[str, Any]] = []
    for fname, finfo in model_cls.model_fields.items():
        ftype = finfo.annotation
        try:
            if isinstance(ftype, type) and issubclass(ftype, _BM):
                continue
        except TypeError:
            pass
        if finfo.default_factory is not None:
            continue
        f_data: dict[str, Any] = {"name": fname}
        if finfo.default is not _PU:
            default_val = finfo.default
            if default_val is None:
                f_data["default"] = None
            else:
                try:
                    json_module.dumps(default_val)
                    f_data["default"] = default_val
                except (TypeError, ValueError):
                    f_data["default"] = str(default_val)
        else:
            f_data["required"] = True
        if finfo.annotation is not None:
            f_data["type"] = str(finfo.annotation)
        fields.append(f_data)
    return fields


_operators_cache: list[dict[str, Any]] | None = None


def _scan_nemo_retriever_package() -> None:
    """Walk all ``nemo_retriever`` submodules to trigger ``@designer_component``
    registrations.  Errors from individual modules are silently skipped."""
    import importlib
    import pkgutil

    try:
        import nemo_retriever as _pkg
    except ImportError:
        return
    if not hasattr(_pkg, "__path__"):
        return
    for _importer, modname, _ispkg in pkgutil.walk_packages(_pkg.__path__, prefix="nemo_retriever."):
        try:
            importlib.import_module(modname)
        except Exception:
            pass


def _extract_param_annotation(resolved_type):
    """If *resolved_type* is ``Annotated[T, Param(...), ...]``, return the
    ``Param`` instance and the unwrapped base type.  Otherwise ``(None, None)``."""
    import typing as _typing

    origin = getattr(resolved_type, "__origin__", None)
    if origin is not None:
        type_name = getattr(origin, "__name__", "") or getattr(origin, "_name", "") or ""
    else:
        type_name = ""
    args = getattr(resolved_type, "__metadata__", None)
    if args is None:
        args_full = _typing.get_args(resolved_type)
        if args_full and len(args_full) >= 2:
            from nemo_retriever.graph.designer import Param as _Param

            for a in args_full[1:]:
                if isinstance(a, _Param):
                    return a, args_full[0]
    else:
        from nemo_retriever.graph.designer import Param as _Param

        for a in args:
            if isinstance(a, _Param):
                base_args = _typing.get_args(resolved_type)
                base_type = base_args[0] if base_args else resolved_type
                return a, base_type
    return None, None


def _discover_operators() -> list[dict[str, Any]]:
    """Discover all ``@designer_component``-decorated classes/functions and
    build the operator list for the Designer palette."""
    global _operators_cache
    if _operators_cache is not None:
        return _operators_cache

    import inspect as _inspect
    import typing as _typing

    _scan_nemo_retriever_package()

    from nemo_retriever.graph.designer import get_registry

    registry: list[dict[str, Any]] = []

    for _key, meta in get_registry().items():
        target = meta["target"]
        class_name = target.__name__
        module_path = target.__module__

        try:
            init_fn = target.__init__ if isinstance(target, type) else target
            sig = _inspect.signature(init_fn)
        except (ValueError, TypeError):
            sig = None

        try:
            resolved_hints = _typing.get_type_hints(init_fn, include_extras=True)
        except Exception:
            resolved_hints = {}

        params: list[dict[str, Any]] = []
        if sig is not None:
            for pname, param in sig.parameters.items():
                if pname == "self" or param.kind in (
                    _inspect.Parameter.VAR_POSITIONAL,
                    _inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                p_info: dict[str, Any] = {"name": pname}

                resolved_type = resolved_hints.get(pname)

                param_meta, base_type = _extract_param_annotation(resolved_type) if resolved_type else (None, None)
                if param_meta is not None:
                    if param_meta.label:
                        p_info["label"] = param_meta.label
                    if param_meta.description:
                        p_info["description"] = param_meta.description
                    if param_meta.choices:
                        p_info["choices"] = param_meta.choices
                    if param_meta.min_val is not None:
                        p_info["min_val"] = param_meta.min_val
                    if param_meta.max_val is not None:
                        p_info["max_val"] = param_meta.max_val
                    if param_meta.hidden:
                        p_info["hidden"] = True
                    if param_meta.placeholder:
                        p_info["placeholder"] = param_meta.placeholder
                    effective_type = base_type
                else:
                    effective_type = resolved_type

                if effective_type is not None:
                    pydantic_fields = _introspect_pydantic_fields(effective_type)
                    if pydantic_fields is not None:
                        p_info["pydantic"] = True
                        p_info["pydantic_class"] = effective_type.__name__
                        p_info["pydantic_module"] = effective_type.__module__
                        p_info["pydantic_import"] = f"from {effective_type.__module__} import {effective_type.__name__}"
                        p_info["fields"] = pydantic_fields

                if param.default is not _inspect.Parameter.empty:
                    try:
                        json_module.dumps(param.default)
                        p_info["default"] = param.default
                    except (TypeError, ValueError):
                        p_info["default"] = str(param.default)
                if param.annotation is not _inspect.Parameter.empty:
                    p_info["type"] = str(param.annotation)
                params.append(p_info)

        concurrency_param: dict[str, Any] = {
            "name": "concurrency",
            "label": "Concurrency",
            "description": "Number of parallel actor instances for this stage",
            "type": "int",
        }
        if meta["compute"] == "gpu":
            concurrency_param["default"] = 1
        params.append(concurrency_param)

        entry: dict[str, Any] = {
            "class_name": class_name,
            "module": module_path,
            "import_path": f"from {module_path} import {class_name}",
            "params": params,
            "category": meta["category"],
            "display_name": meta["name"],
            "compute": meta["compute"],
            "description": meta.get("description", ""),
        }
        if meta.get("category_color"):
            entry["category_color"] = meta["category_color"]
        if meta.get("component_type"):
            entry["type"] = meta["component_type"]
        registry.append(entry)

    registry.extend(_RAY_DATA_SOURCES)
    _operators_cache = registry
    return registry


@app.get("/api/operators")
async def list_operators():
    """Return all available pipeline operators with their constructor parameters."""
    return _discover_operators()


class GraphCreateRequest(BaseModel):
    name: str
    description: str = ""
    graph_json: Any
    generated_code: str = ""


class GraphUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    graph_json: Any = None
    generated_code: str | None = None


@app.get("/api/graphs")
async def list_graphs_endpoint():
    return history.list_graphs()


@app.get("/api/graphs/{graph_id}")
async def get_graph_endpoint(graph_id: int):
    g = history.get_graph(graph_id)
    if not g:
        raise HTTPException(404, "Graph not found")
    return g


@app.post("/api/graphs")
async def create_graph_endpoint(req: GraphCreateRequest):
    return history.create_graph(req.model_dump())


@app.put("/api/graphs/{graph_id}")
async def update_graph_endpoint(graph_id: int, req: GraphUpdateRequest):
    data = {k: v for k, v in req.model_dump().items() if v is not None}
    g = history.update_graph(graph_id, data)
    if not g:
        raise HTTPException(404, "Graph not found")
    return g


@app.delete("/api/graphs/{graph_id}")
async def delete_graph_endpoint(graph_id: int):
    ok = history.delete_graph(graph_id)
    if not ok:
        raise HTTPException(404, "Graph not found")
    return {"ok": True}


class GraphRunRequest(BaseModel):
    runner_id: int | None = None
    git_ref: str | None = None
    git_commit: str | None = None
    ray_address: str | None = None


class GraphRunResponse(BaseModel):
    job_id: str
    status: str


@app.post("/api/graphs/{graph_id}/run", response_model=GraphRunResponse)
async def run_graph_endpoint(graph_id: int, req: GraphRunRequest):
    graph = history.get_graph(graph_id)
    if not graph:
        raise HTTPException(404, "Graph not found")

    code = graph.get("generated_code") or ""
    if not code.strip():
        raise HTTPException(400, "Graph has no generated code. Save the graph first.")

    pinned_sha, pinned_ref = _resolve_git_override(req.git_ref, req.git_commit)

    graph_meta = {
        "ray_address": req.ray_address,
    }

    graph_name = graph.get("name") or f"graph-{graph_id}"

    input_path = graph_name
    try:
        raw_json = graph.get("graph_json") or "{}"
        gj = json_module.loads(raw_json) if isinstance(raw_json, str) else raw_json
        for node in gj.get("nodes") or []:
            op = node.get("operator") or {}
            if op.get("type") == "ray_data_source":
                paths_val = (node.get("config") or {}).get("paths", "")
                if paths_val:
                    input_path = paths_val
                break
    except Exception:
        pass

    job = history.create_job(
        {
            "trigger_source": "graph",
            "dataset": input_path,
            "preset": graph_name,
            "assigned_runner_id": req.runner_id,
            "git_commit": pinned_sha,
            "git_ref": pinned_ref,
            "graph_code": code,
            "graph_id": graph_id,
            "tags": ["graph-run", graph_name],
            "config": json_module.dumps(graph_meta),
        }
    )
    return GraphRunResponse(job_id=job["id"], status="pending")
