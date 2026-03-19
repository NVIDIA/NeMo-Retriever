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

STATIC_DIR = Path(__file__).parent / "static"

GITHUB_WEBHOOK_SECRET = os.environ.get("RETRIEVER_HARNESS_GITHUB_SECRET", "")
GITHUB_REPO_URL_OVERRIDE = os.environ.get("RETRIEVER_HARNESS_GITHUB_REPO_URL", "")


@lru_cache(maxsize=1)
def _detect_github_repo_url() -> str:
    """Derive the GitHub web URL from the git remote origin, or use the env override."""
    if GITHUB_REPO_URL_OVERRIDE:
        return GITHUB_REPO_URL_OVERRIDE.rstrip("/")
    try:
        out = subprocess.check_output(
            ["git", "remote", "get-url", "nvidia"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        try:
            out = subprocess.check_output(
                ["git", "remote", "get-url", "origin"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
        except Exception:
            return ""
    m = re.match(r"git@github\.com:(.+?)(?:\.git)?$", out)
    if m:
        return f"https://github.com/{m.group(1)}"
    m = re.match(r"https?://github\.com/(.+?)(?:\.git)?$", out)
    if m:
        return f"https://github.com/{m.group(1)}"
    return ""


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
                history.create_system_alert_event({
                    "metric": "runner_status",
                    "metric_value": 0,
                    "threshold": 0,
                    "operator": "system",
                    "message": f"Runner '{hostname}' is offline — missed {history.RUNNER_MISSED_HEARTBEATS_THRESHOLD} consecutive heartbeats",
                    "hostname": hostname,
                })
        except Exception:
            logger.exception("Error in runner health check loop")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _import_yaml_datasets_on_startup()
    sched_module.start_scheduler()
    global _runner_health_task
    _runner_health_task = asyncio.create_task(_runner_health_check_loop())
    yield
    if _runner_health_task:
        _runner_health_task.cancel()
    sched_module.stop_scheduler()


def _import_yaml_datasets_on_startup() -> None:
    """Seed the managed datasets table with entries from test_configs.yaml."""
    try:
        from nemo_retriever.harness.config import DEFAULT_TEST_CONFIG_PATH, _read_yaml_mapping

        cfg = _read_yaml_mapping(DEFAULT_TEST_CONFIG_PATH)
        yaml_datasets = cfg.get("datasets") or {}
        if yaml_datasets:
            count = history.import_yaml_datasets(yaml_datasets)
            if count:
                logger.info("Imported %d YAML dataset(s) into managed datasets", count)
    except Exception as exc:
        logger.warning("Failed to import YAML datasets on startup: %s", exc)


app = FastAPI(title="Harness Portal", docs_url="/api/docs", redoc_url=None, lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TriggerRequest(BaseModel):
    dataset: str
    preset: str | None = None
    config: str | None = None
    tags: list[str] | None = None
    runner_id: int | None = None


class TriggerResponse(BaseModel):
    job_id: str
    status: str


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


class ScheduleCreateRequest(BaseModel):
    name: str
    description: str | None = None
    dataset: str
    preset: str | None = None
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
    enabled: bool = True
    tags: list[str] | None = None


class ScheduleUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    dataset: str | None = None
    preset: str | None = None
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
    enabled: bool | None = None
    tags: list[str] | None = None


class JobCompleteRequest(BaseModel):
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None


class PresetCreateRequest(BaseModel):
    name: str
    description: str | None = None
    config: dict[str, Any] = {}
    tags: list[str] | None = None


class PresetUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    config: dict[str, Any] | None = None
    tags: list[str] | None = None


class DatasetCreateRequest(BaseModel):
    name: str
    path: str
    query_csv: str | None = None
    input_type: str = "pdf"
    recall_required: bool = False
    recall_match_mode: str = "pdf_page"
    recall_adapter: str = "none"
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
    return FileResponse(str(STATIC_DIR / "index.html"))


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

    dataset = row.get("dataset", "unknown")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id}_{dataset}.zip"'},
    )


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
                    k: v for k, v in metadata.items()
                    if isinstance(v, (str, int, float, bool, type(None)))
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
            sessions.append({
                "session_id": d.name,
                "files": files,
                "file_count": len(files),
                "total_bytes": total_bytes,
                "path": str(d),
            })
    return sessions


@app.delete("/api/playground/sessions/{session_id}")
async def delete_playground_session(session_id: str):
    """Delete a playground upload session and its files."""
    session_dir = PLAYGROUND_DIR / session_id
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")
    shutil.rmtree(session_dir, ignore_errors=True)
    return {"ok": True}


@app.get("/api/datasets")
async def list_datasets():
    """Return distinct dataset names from run history (legacy)."""
    return history.get_datasets()


@app.get("/api/config")
async def get_config():
    """Return merged dataset and preset names from YAML config + managed entries."""
    yaml_datasets: list[str] = []
    yaml_presets: list[str] = []
    try:
        from nemo_retriever.harness.config import DEFAULT_TEST_CONFIG_PATH, _read_yaml_mapping

        cfg = _read_yaml_mapping(DEFAULT_TEST_CONFIG_PATH)
        yaml_datasets = list((cfg.get("datasets") or {}).keys())
        yaml_presets = list((cfg.get("presets") or {}).keys())
    except Exception:
        pass

    managed_dataset_names = history.get_dataset_names()
    managed_preset_names = history.get_preset_names()
    all_datasets = sorted(set(yaml_datasets + managed_dataset_names))
    all_presets = sorted(set(yaml_presets + managed_preset_names))
    return {
        "datasets": all_datasets,
        "presets": all_presets,
        "github_repo_url": _detect_github_repo_url(),
    }


@app.get("/api/yaml-config")
async def get_yaml_config():
    """Return the full dataset and preset definitions from test_configs.yaml."""
    try:
        from nemo_retriever.harness.config import DEFAULT_TEST_CONFIG_PATH, _read_yaml_mapping

        cfg = _read_yaml_mapping(DEFAULT_TEST_CONFIG_PATH)
        return {
            "datasets": cfg.get("datasets") or {},
            "presets": cfg.get("presets") or {},
            "active": cfg.get("active") or {},
        }
    except Exception:
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
        if managed.get("query_csv"):
            overrides["query_csv"] = managed["query_csv"]
        if managed.get("input_type"):
            overrides["input_type"] = managed["input_type"]
        if managed.get("recall_required") is not None:
            overrides["recall_required"] = managed["recall_required"]
        if managed.get("recall_match_mode"):
            overrides["recall_match_mode"] = managed["recall_match_mode"]
        if managed.get("recall_adapter"):
            overrides["recall_adapter"] = managed["recall_adapter"]
        return managed["path"], overrides

    try:
        from nemo_retriever.harness.config import DEFAULT_TEST_CONFIG_PATH, _read_yaml_mapping

        cfg = _read_yaml_mapping(DEFAULT_TEST_CONFIG_PATH)
        ds_cfg = (cfg.get("datasets") or {}).get(dataset_name)
        if ds_cfg and isinstance(ds_cfg, dict) and ds_cfg.get("path"):
            yaml_overrides: dict[str, Any] = {"dataset_dir": str(ds_cfg["path"])}
            if ds_cfg.get("query_csv"):
                yaml_overrides["query_csv"] = str(ds_cfg["query_csv"])
            if ds_cfg.get("input_type"):
                yaml_overrides["input_type"] = str(ds_cfg["input_type"])
            if ds_cfg.get("recall_required") is not None:
                yaml_overrides["recall_required"] = ds_cfg["recall_required"]
            if ds_cfg.get("recall_match_mode"):
                yaml_overrides["recall_match_mode"] = str(ds_cfg["recall_match_mode"])
            if ds_cfg.get("recall_adapter"):
                yaml_overrides["recall_adapter"] = str(ds_cfg["recall_adapter"])
            return str(ds_cfg["path"]), yaml_overrides
    except Exception:
        pass
    return None, None


@app.post("/api/runs/trigger", response_model=TriggerResponse)
async def trigger_run(req: TriggerRequest):
    dataset_path, dataset_overrides = _resolve_dataset_config(req.dataset)
    job = history.create_job(
        {
            "trigger_source": "manual",
            "dataset": req.dataset,
            "dataset_path": dataset_path,
            "dataset_overrides": dataset_overrides,
            "preset": req.preset,
            "config": req.config,
            "assigned_runner_id": req.runner_id,
            "tags": req.tags or [],
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

    if was_cancelling and not req.success:
        history.complete_job(job_id, success=False, result=req.result, error=req.error or "Cancelled by user")
        history.update_job_status(job_id, "cancelled", error=req.error or "Cancelled by user")
    else:
        history.complete_job(job_id, success=req.success, result=req.result, error=req.error)

    job = history.get_job_by_id(job_id)
    effective_success = req.success and not was_cancelling
    effective_error = req.error or ("Cancelled by user" if was_cancelling else None)
    _record_run_from_job(job, effective_success, req.result, effective_error)

    return {"ok": True}


def _record_run_from_job(
    job: dict[str, Any] | None,
    success: bool,
    result: dict[str, Any] | None,
    error: str | None,
) -> None:
    """Create a run record in the runs table from a completed job.

    When the runner sends back a full ``result`` dict (from ``_run_entry``),
    use that directly.  Otherwise synthesise a minimal result so that failed
    jobs still appear in the Runs view.
    """
    if job is None:
        return

    if result and isinstance(result, dict) and result.get("timestamp"):
        run_result = result
    else:
        now_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
        run_result = {
            "timestamp": now_ts,
            "latest_commit": None,
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
    artifact_dir = (run_result.get("artifacts") or {}).get("runtime_metrics_dir", "")

    try:
        run_row_id = history.record_run(
            run_result,
            artifact_dir=artifact_dir,
            trigger_source=trigger_source,
            schedule_id=schedule_id,
        )
        if run_row_id:
            run_row = history.get_run_by_id(run_row_id)
            if run_row:
                try:
                    history.evaluate_alerts_for_run(run_row)
                except Exception as alert_exc:
                    logger.error("Alert evaluation failed for run %s: %s", run_row_id, alert_exc)
    except Exception as exc:
        logger.error("Failed to record run for job %s: %s", job.get("id"), exc)


# ---------------------------------------------------------------------------
# Runner endpoints
# ---------------------------------------------------------------------------


@app.get("/api/runners")
async def list_runners():
    return history.get_runners()


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


class JobRejectRequest(BaseModel):
    runner_id: int
    reason: str = "Dataset not found on runner"


class HeartbeatRequest(BaseModel):
    current_job_id: str | None = None
    log_tail: list[str] | None = None


@app.post("/api/runners/{runner_id}/heartbeat")
async def runner_heartbeat(runner_id: int, req: HeartbeatRequest | None = None):
    if not history.heartbeat_runner(runner_id):
        raise HTTPException(status_code=404, detail="Runner not found")

    cancel_job_id: str | None = None

    if req and req.current_job_id:
        if req.log_tail:
            history.update_job_log(req.current_job_id, req.log_tail)
        current_job = history.get_job_by_id(req.current_job_id)
        if current_job and current_job.get("status") == "cancelling":
            cancel_job_id = req.current_job_id

    jobs = history.get_pending_jobs_for_runner(runner_id)
    next_job = _pick_job_for_runner(jobs, runner_id)

    return {"ok": True, "job": next_job, "cancel_job_id": cancel_job_id}


@app.get("/api/runners/{runner_id}/work")
async def runner_get_work(runner_id: int):
    """Return the next pending job for this runner (assigned or unassigned), or 204 if none."""
    runner = history.get_runner_by_id(runner_id)
    if not runner:
        raise HTTPException(status_code=404, detail="Runner not found")
    if runner.get("status") == "offline":
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

        remotes_raw = _git_run("remote", "-v", cwd=repo_root)
        remotes: list[dict[str, str]] = []
        seen: set[str] = set()
        for line in remotes_raw.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] not in seen:
                seen.add(parts[0])
                remotes.append({"name": parts[0], "url": parts[1]})

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

        last_commits_raw = _git_run(
            "log", "--oneline", "-10", "--format=%H|%h|%s|%ci", cwd=repo_root
        )
        recent_commits: list[dict[str, str]] = []
        for line in last_commits_raw.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                recent_commits.append({
                    "sha": parts[0], "short_sha": parts[1],
                    "message": parts[2], "date": parts[3],
                })

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
    remote: str = "origin"


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

    new_sha = _git_run("rev-parse", "--short", "HEAD", cwd=repo_root)
    log_lines.append(f"Now at {new_sha} on {req.branch}")

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

    return {
        "ok": True,
        "new_sha": new_sha,
        "branch": req.branch,
        "remote": req.remote,
        "log": log_lines,
        "message": f"Deployed {req.branch} ({new_sha}). Portal will restart in ~2 seconds.",
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
