# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP tool definitions for the Harness Portal.

Each function decorated with ``@portal_tool`` is automatically discovered at
import time and registered with the MCP server on portal startup.  Developers
can add new tools simply by adding a decorated function here (or in any other
module under ``nemo_retriever``).
"""

from __future__ import annotations

from typing import Any

from nemo_retriever.harness import history
from nemo_retriever.harness.portal.mcp_registry import portal_tool


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@portal_tool(
    name="trigger_benchmark_run",
    category="Jobs",
    description=(
        "Trigger a benchmark run on a dataset with an optional preset. "
        "Returns the job ID and status. Use list_datasets and list_presets "
        "first to discover valid names."
    ),
    tags=["write", "jobs"],
)
def trigger_benchmark_run(
    dataset: str,
    preset: str | None = None,
    runner_id: int | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Trigger a benchmark run."""
    from nemo_retriever.harness.portal.app import (
        _resolve_dataset_config,
        _resolve_git_override,
        _resolve_preset_overrides,
    )

    dataset_path, dataset_overrides = _resolve_dataset_config(dataset)
    preset_overrides = _resolve_preset_overrides(preset)
    merged_overrides = {**(dataset_overrides or {}), **preset_overrides}
    pinned_sha, pinned_ref = _resolve_git_override(None, None)

    job = history.create_job(
        {
            "trigger_source": "mcp",
            "dataset": dataset,
            "dataset_path": dataset_path,
            "dataset_overrides": merged_overrides if merged_overrides else None,
            "preset": preset,
            "assigned_runner_id": runner_id,
            "git_commit": pinned_sha,
            "git_ref": pinned_ref,
            "tags": tags or ["mcp-triggered"],
        }
    )
    return {"job_id": job["id"], "status": "pending"}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@portal_tool(
    name="list_runs",
    category="Runs",
    description=(
        "List recent benchmark runs with optional dataset and commit filters. "
        "Returns run metadata including success status, pages/sec, and recall."
    ),
    tags=["read", "runs"],
)
def list_runs(
    dataset: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List recent benchmark runs."""
    runs = history.get_runs(dataset=dataset, limit=min(limit, 200))
    return [
        {
            "id": r["id"],
            "dataset": r.get("dataset"),
            "preset": r.get("preset"),
            "success": r.get("success"),
            "pages": r.get("pages"),
            "pages_per_sec": r.get("pages_per_sec"),
            "recall_5": r.get("recall_5"),
            "timestamp": r.get("timestamp"),
            "hostname": r.get("hostname"),
            "gpu_type": r.get("gpu_type"),
        }
        for r in runs
    ]


@portal_tool(
    name="get_run_details",
    category="Runs",
    description="Get detailed results for a specific benchmark run by ID.",
    tags=["read", "runs"],
)
def get_run_details(run_id: int) -> dict[str, Any]:
    """Get detailed results for a specific run."""
    row = history.get_run_by_id(run_id)
    if row is None:
        return {"error": f"Run {run_id} not found"}
    return row


@portal_tool(
    name="compare_runs",
    category="Runs",
    description=(
        "Compare metrics between two benchmark runs. Provide two run IDs "
        "and get a side-by-side comparison of key metrics."
    ),
    tags=["read", "runs", "analysis"],
)
def compare_runs(run_id_a: int, run_id_b: int) -> dict[str, Any]:
    """Compare metrics between two runs."""
    a = history.get_run_by_id(run_id_a)
    b = history.get_run_by_id(run_id_b)
    if a is None:
        return {"error": f"Run {run_id_a} not found"}
    if b is None:
        return {"error": f"Run {run_id_b} not found"}

    metrics = ["pages_per_sec", "recall_1", "recall_5", "recall_10", "ingest_secs", "pages"]

    def _pick(run: dict, keys: list[str]) -> dict:
        return {k: run.get(k) for k in keys}

    comparison: dict[str, Any] = {
        "run_a": {"id": run_id_a, "dataset": a.get("dataset"), "preset": a.get("preset"), **_pick(a, metrics)},
        "run_b": {"id": run_id_b, "dataset": b.get("dataset"), "preset": b.get("preset"), **_pick(b, metrics)},
        "deltas": {},
    }
    for m in metrics:
        va, vb = a.get(m), b.get(m)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            comparison["deltas"][m] = round(vb - va, 6)
    return comparison


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


@portal_tool(
    name="list_runners",
    category="Runners",
    description="Show all registered runners and their current status, GPU type, and resource info.",
    tags=["read", "runners"],
)
def list_runners() -> list[dict[str, Any]]:
    """List registered runners."""
    runners = history.get_runners()
    return [
        {
            "id": r["id"],
            "name": r.get("name"),
            "hostname": r.get("hostname"),
            "status": r.get("status"),
            "gpu_type": r.get("gpu_type"),
            "gpu_count": r.get("gpu_count"),
            "cpu_count": r.get("cpu_count"),
            "memory_gb": r.get("memory_gb"),
            "last_heartbeat": r.get("last_heartbeat"),
        }
        for r in runners
    ]


# ---------------------------------------------------------------------------
# Data (datasets & presets)
# ---------------------------------------------------------------------------


@portal_tool(
    name="list_datasets",
    category="Data",
    description="List all managed benchmark datasets with their configuration.",
    tags=["read", "data"],
)
def list_datasets() -> list[dict[str, Any]]:
    """List managed benchmark datasets."""
    datasets = history.get_all_datasets()
    return [
        {
            "id": d["id"],
            "name": d.get("name"),
            "path": d.get("path"),
            "input_type": d.get("input_type"),
            "description": d.get("description"),
            "evaluation_mode": d.get("evaluation_mode"),
        }
        for d in datasets
    ]


@portal_tool(
    name="list_presets",
    category="Data",
    description="List all managed presets (pipeline configurations).",
    tags=["read", "data"],
)
def list_presets() -> list[dict[str, Any]]:
    """List managed presets."""
    presets = history.get_all_presets()
    return [
        {
            "id": p["id"],
            "name": p.get("name"),
            "description": p.get("description"),
        }
        for p in presets
    ]


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


@portal_tool(
    name="create_schedule",
    category="Scheduling",
    description=(
        "Create a new cron-based benchmark schedule. Requires a name, "
        "dataset, cron expression, and optionally a preset."
    ),
    tags=["write", "scheduling"],
)
def create_schedule(
    name: str,
    dataset: str,
    cron_expression: str,
    preset: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Create a cron schedule for recurring benchmark runs."""
    schedule = history.create_schedule(
        {
            "name": name,
            "dataset": dataset,
            "preset": preset,
            "description": description,
            "trigger_type": "cron",
            "cron_expression": cron_expression,
        }
    )
    return {"id": schedule["id"], "name": schedule["name"], "status": "created"}


@portal_tool(
    name="list_schedules",
    category="Scheduling",
    description="List all benchmark schedules (cron and GitHub-triggered).",
    tags=["read", "scheduling"],
)
def list_schedules() -> list[dict[str, Any]]:
    """List all schedules."""
    schedules = history.get_schedules()
    return [
        {
            "id": s["id"],
            "name": s.get("name"),
            "dataset": s.get("dataset"),
            "preset": s.get("preset"),
            "trigger_type": s.get("trigger_type"),
            "cron_expression": s.get("cron_expression"),
            "enabled": s.get("enabled"),
        }
        for s in schedules
    ]


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


@portal_tool(
    name="get_portal_status",
    category="System",
    description="Get portal version, git SHA, runner health summary, and job queue status.",
    tags=["read", "system"],
)
def get_portal_status() -> dict[str, Any]:
    """Get portal health and version information."""
    from nemo_retriever.version import get_version_info

    version = get_version_info()
    runners = history.get_runners()
    online = sum(1 for r in runners if r.get("status") == "online")
    jobs = history.get_jobs()
    pending = sum(1 for j in jobs if j.get("status") == "pending")
    running = sum(1 for j in jobs if j.get("status") == "running")

    return {
        "version": version.get("version"),
        "git_sha": version.get("git_sha"),
        "runners_total": len(runners),
        "runners_online": online,
        "jobs_pending": pending,
        "jobs_running": running,
    }


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


@portal_tool(
    name="list_alert_events",
    category="Monitoring",
    description="Get recent alert events from the portal monitoring system.",
    tags=["read", "monitoring"],
)
def list_alert_events(limit: int = 50) -> list[dict[str, Any]]:
    """List recent alert events."""
    events = history.get_alert_events(limit=min(limit, 200))
    return [
        {
            "id": e["id"],
            "metric": e.get("metric"),
            "message": e.get("message"),
            "dataset": e.get("dataset"),
            "acknowledged": e.get("acknowledged"),
            "created_at": e.get("created_at"),
        }
        for e in events
    ]
