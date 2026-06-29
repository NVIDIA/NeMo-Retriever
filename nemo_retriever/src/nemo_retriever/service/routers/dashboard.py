# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dashboard UI router — gateway-only.

Serves the SPA shell and provides REST/SSE API endpoints consumed by the
React frontend for the Overview, Job Tracker, and VDB Explorer views.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])

_STATIC_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "static"


# ── Request models ───────────────────────────────────────────────────


class VdbQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=1000)


# ── SPA shell ────────────────────────────────────────────────────────


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
async def index():
    index_path = _STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(500, f"Dashboard UI not found at {index_path}")
    return FileResponse(str(index_path), media_type="text/html")


# ── Overview API ─────────────────────────────────────────────────────


async def _fetch_pool_stats(client: httpx.AsyncClient, base_url: str) -> dict:
    """Best-effort fetch of ``GET /v1/admin/pool_stats`` from a backend.

    Returns ``{}`` on any error so the overview never fails to render
    just because one worker pod is briefly unhealthy.
    """
    try:
        resp = await client.get(f"{base_url}/v1/admin/pool_stats", timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.debug("pool_stats fetch failed for %s: %s", base_url, exc)
    return {}


@router.get("/api/overview")
async def overview(request: Request) -> JSONResponse:
    """Aggregate cluster status for the overview panel.

    On gateway pods, this fans out to each worker Service to collect
    live pool stats (queue depth, queue ratio, processed counts) so
    the dashboard can surface scaling pressure without forcing the
    operator to open Grafana. On standalone pods the local pool is
    read in-process via :func:`get_pipeline_pool`.
    """
    config = request.app.state.config

    backends = {}
    pool_stats: dict[str, dict] = {}
    try:
        from nemo_retriever.service.services.proxy import get_proxy
        from nemo_retriever.service.services.pipeline_pool import (
            PoolType,
            get_pipeline_pool,
        )

        proxy = get_proxy()
        if proxy is not None:
            backends["realtime"] = await proxy.check_backend(PoolType.REALTIME)
            backends["batch"] = await proxy.check_backend(PoolType.BATCH)
            # H6: fan out to each backend for live queue depth. The
            # gateway has no local pool, so this is the only way the
            # overview page can show "realtime queue 50% full" without
            # going through Prometheus.
            gateway_cfg = getattr(config, "gateway", None)
            if gateway_cfg is not None:
                async with httpx.AsyncClient() as client:
                    rt_task = _fetch_pool_stats(client, gateway_cfg.realtime_url)
                    bt_task = _fetch_pool_stats(client, gateway_cfg.batch_url)
                    rt_stats, bt_stats = await asyncio.gather(rt_task, bt_task)
                # Each worker's response carries its own pools dict; the
                # realtime pod returns {"realtime": {...}} and batch
                # returns {"batch": {...}}. Merge for the consumer.
                for stats in (rt_stats, bt_stats):
                    for pool_name, pool_data in (stats.get("pools") or {}).items():
                        pool_stats[pool_name] = pool_data
        else:
            # Standalone (or worker) pod — pull stats from the local
            # singleton directly to avoid an HTTP round trip to ourselves.
            local_pool = get_pipeline_pool()
            if local_pool is not None:
                for pt in (PoolType.REALTIME, PoolType.BATCH):
                    p = local_pool.pool_for(pt)
                    if p is None:
                        continue
                    depth = p.queue_depth
                    max_qs = max(1, p.max_queue_size)
                    pool_stats[pt.value] = {
                        "queue_depth": depth,
                        "queue_depth_ratio": round(depth / max_qs, 4),
                        "max_queue_size": p.max_queue_size,
                        "num_workers": p.num_workers,
                        "processed": p.processed,
                        "is_running": p.is_running,
                    }
    except Exception as exc:
        logger.debug("Could not check backends / pool stats: %s", exc)

    vdb_status = None
    vdb_url = getattr(config, "vectordb", None)
    if vdb_url and getattr(vdb_url, "enabled", False):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{vdb_url.vectordb_url}/v1/health")
                if resp.status_code == 200:
                    vdb_status = resp.json()
        except Exception as exc:
            logger.debug("VDB health check failed: %s", exc)

    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    job_summary = tracker.summary() if tracker else {}

    pool_cfg = getattr(config, "pipeline", None)
    worker_config = {}
    if pool_cfg:
        worker_config = {
            "realtime_workers": pool_cfg.realtime_workers,
            "realtime_queue_size": pool_cfg.realtime_queue_size,
            "batch_workers": pool_cfg.batch_workers,
            "batch_queue_size": pool_cfg.batch_queue_size,
        }

    gateway_cfg = getattr(config, "gateway", None)
    gateway_info = {}
    if gateway_cfg:
        gateway_info = {
            "realtime_url": gateway_cfg.realtime_url,
            "batch_url": gateway_cfg.batch_url,
        }

    return JSONResponse(
        {
            "mode": config.mode,
            "backends": backends,
            "pool_stats": pool_stats,
            "vectordb": vdb_status,
            "job_summary": job_summary,
            "worker_config": worker_config,
            "gateway": gateway_info,
        }
    )


# ── Jobs SSE stream ─────────────────────────────────────────────────


@router.get("/api/jobs")
async def jobs_sse(request: Request) -> StreamingResponse:
    """SSE stream of job-tracker events with periodic summary heartbeats.

    Subscribes to the global event bus (no ``job_id`` filter) so the
    dashboard sees both per-document events and the J5 job lifecycle
    events for every job. The initial ``snapshot`` payload bundles both
    layers (``documents`` for back-compat with the legacy doc-grid view
    and ``jobs`` for the new job-aggregate view) so the SPA can render
    immediately without an extra REST hop.
    """
    from nemo_retriever.service.services.event_bus import get_event_bus
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    bus = get_event_bus()
    tracker = get_job_tracker()

    if bus is None:
        raise HTTPException(503, "Event bus not available")

    sub_id, queue = bus.subscribe()

    async def event_generator():
        try:
            if tracker:
                snapshot = {
                    "type": "snapshot",
                    "summary": tracker.summary(),
                    "jobs": [_serialize_job(j) for j in tracker.all_jobs()],
                    "documents": [rec.model_dump() for rec in tracker.all_documents()],
                }
                yield f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"

            last_heartbeat = asyncio.get_event_loop().time()

            while True:
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    # Job lifecycle events vs per-doc events are
                    # distinguished by the ``type`` field set by the
                    # tracker (``job_created`` etc. vs status strings).
                    evt_type = event.get("type", "")
                    sse_event = "job_lifecycle" if evt_type.startswith("job_") else "job_update"
                    yield f"event: {sse_event}\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    pass

                now = asyncio.get_event_loop().time()
                if now - last_heartbeat >= 5.0:
                    heartbeat = {"type": "heartbeat"}
                    if tracker:
                        heartbeat["summary"] = tracker.summary()
                    yield f"event: heartbeat\ndata: {json.dumps(heartbeat)}\n\n"
                    last_heartbeat = now

        finally:
            bus.unsubscribe(sub_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Jobs snapshot (REST fallback) ────────────────────────────────────


def _serialize_job(agg) -> dict:
    """Project a :class:`JobAggregate` to the wire shape the UI expects.

    Kept compact (no document records) so list views can paginate
    hundreds of jobs without bloating the payload.
    """
    return {
        "job_id": agg.job_id,
        "status": agg.status.value,
        "expected_documents": agg.expected_documents,
        "counts": dict(agg.counts),
        "created_at": agg.created_at,
        "started_at": agg.started_at,
        "finalized_at": agg.finalized_at,
        "elapsed_s": agg.elapsed_s,
        "label": agg.label,
        "trace_id": agg.trace_id,
        "document_ids": list(agg.document_ids),
    }


@router.get("/api/jobs/snapshot")
async def jobs_snapshot(request: Request) -> JSONResponse:
    """REST fallback for the SSE stream.

    Returns both ``jobs`` (the J2+ aggregate view) and ``documents``
    (per-doc rows for the legacy table). The SPA prefers ``jobs`` and
    falls back to ``documents`` for older builds.
    """
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    if tracker is None:
        return JSONResponse({"summary": {}, "jobs": [], "documents": []})

    return JSONResponse(
        {
            "summary": tracker.summary(),
            "jobs": [_serialize_job(j) for j in tracker.all_jobs()],
            "documents": [rec.model_dump() for rec in tracker.all_documents()],
        }
    )


# ── Jobs list / detail (J8 — paginated REST API for the UI) ──────────


@router.get("/api/jobs/list")
async def jobs_list(
    request: Request,
    status: str | None = None,
    offset: int = 0,
    limit: int = 50,
    sort: str = "created_desc",
) -> JSONResponse:
    """Paginated list of job aggregates, newest first by default.

    Parameters
    ----------
    status:
        Optional aggregate-status filter (``pending``, ``running``,
        ``completed``, ``failed``, ``partial_success``).
    offset:
        Zero-based page start (>= 0).
    limit:
        Page size, 1..500.
    sort:
        ``created_desc`` (default), ``created_asc``,
        ``finalized_desc``, ``finalized_asc``.

    Returns ``{jobs, total, total_filtered, offset, limit, sort}`` with
    a compact projection per job (see :func:`_serialize_job`).
    """
    from nemo_retriever.service.services.job_tracker import (
        JobAggregateStatus,
        get_job_tracker,
    )

    if offset < 0:
        raise HTTPException(400, "offset must be >= 0")
    if limit < 1 or limit > 500:
        raise HTTPException(400, "limit must be in [1, 500]")

    valid_sorts = {"created_desc", "created_asc", "finalized_desc", "finalized_asc"}
    if sort not in valid_sorts:
        raise HTTPException(400, f"sort must be one of {sorted(valid_sorts)}, got {sort!r}")

    tracker = get_job_tracker()
    if tracker is None:
        return JSONResponse(
            {
                "jobs": [],
                "total": 0,
                "total_filtered": 0,
                "offset": offset,
                "limit": limit,
                "sort": sort,
            }
        )

    jobs = list(tracker.all_jobs())

    if status is not None:
        valid_status = {s.value for s in JobAggregateStatus}
        if status not in valid_status:
            raise HTTPException(
                400,
                f"status must be one of {sorted(valid_status)}, got {status!r}",
            )
        filtered = [j for j in jobs if j.status.value == status]
    else:
        filtered = jobs

    # Sort newest/oldest by either creation or finalization timestamp.
    # ISO-8601 sorts lexicographically; ``None`` slots are pushed last
    # in descending order, first in ascending order so the UI always
    # surfaces *known* timestamps first.
    if sort.startswith("created"):
        key = lambda j: j.created_at  # noqa: E731
        reverse = sort == "created_desc"
    else:
        key = lambda j: (j.finalized_at or "")  # noqa: E731
        reverse = sort == "finalized_desc"
    filtered.sort(key=key, reverse=reverse)

    page = filtered[offset : offset + limit]
    return JSONResponse(
        {
            "jobs": [_serialize_job(j) for j in page],
            "total": len(jobs),
            "total_filtered": len(filtered),
            "offset": offset,
            "limit": limit,
            "sort": sort,
        }
    )


@router.get("/api/jobs/{job_id}")
async def jobs_detail(request: Request, job_id: str) -> JSONResponse:
    """Single-job aggregate view for the detail page.

    Returns the compact aggregate projection together with up to 500
    document records — enough to render the per-doc table immediately
    on page load. The full list is paginated separately at
    :func:`jobs_documents`.
    """
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(503, "Job tracker not available")

    agg = tracker.get_job(job_id)
    if agg is None:
        raise HTTPException(404, f"Job {job_id!r} not found")

    docs = tracker.job_documents(job_id)
    sample_cap = 500
    return JSONResponse(
        {
            **_serialize_job(agg),
            "documents": [d.model_dump() for d in docs[:sample_cap]],
            "documents_truncated": len(docs) > sample_cap,
        }
    )


@router.get("/api/jobs/{job_id}/documents")
async def jobs_documents(
    request: Request,
    job_id: str,
    status: str | None = None,
    offset: int = 0,
    limit: int = 100,
) -> JSONResponse:
    """Paginated documents for one job — backs the detail-page table."""
    from nemo_retriever.service.services.job_tracker import (
        DocumentStatus,
        get_job_tracker,
    )

    if offset < 0:
        raise HTTPException(400, "offset must be >= 0")
    if limit < 1 or limit > 1000:
        raise HTTPException(400, "limit must be in [1, 1000]")

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(503, "Job tracker not available")

    if tracker.get_job(job_id) is None:
        raise HTTPException(404, f"Job {job_id!r} not found")

    docs = tracker.job_documents(job_id)

    if status is not None:
        valid = {s.value for s in DocumentStatus}
        if status not in valid:
            raise HTTPException(
                400,
                f"status must be one of {sorted(valid)}, got {status!r}",
            )
        filtered = [d for d in docs if d.status.value == status]
    else:
        filtered = docs

    page = filtered[offset : offset + limit]
    return JSONResponse(
        {
            "job_id": job_id,
            "total": len(docs),
            "total_filtered": len(filtered),
            "offset": offset,
            "limit": limit,
            "items": [d.model_dump() for d in page],
        }
    )


# ── VDB tables ───────────────────────────────────────────────────────


@router.get("/api/vdb/tables")
async def vdb_tables(request: Request) -> JSONResponse:
    config = request.app.state.config
    vdb_cfg = getattr(config, "vectordb", None)

    if not vdb_cfg or not getattr(vdb_cfg, "enabled", False):
        return JSONResponse({"error": "VectorDB not enabled", "tables": []})

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{vdb_cfg.vectordb_url}/v1/health")
            resp.raise_for_status()
            health = resp.json()
            return JSONResponse(
                {
                    "tables": [
                        {
                            "name": health.get("table", ""),
                            "total_rows": health.get("total_rows", 0),
                            "exists": health.get("table_exists", False),
                        }
                    ],
                }
            )
    except Exception as exc:
        return JSONResponse({"error": str(exc), "tables": []})


# ── VDB query proxy ──────────────────────────────────────────────────


@router.post("/api/vdb/query")
async def vdb_query(req: VdbQueryRequest, request: Request) -> JSONResponse:
    config = request.app.state.config
    vdb_cfg = getattr(config, "vectordb", None)

    if not vdb_cfg or not getattr(vdb_cfg, "enabled", False):
        raise HTTPException(501, "VectorDB not enabled")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{vdb_cfg.vectordb_url}/v1/query",
                json={"query": req.query, "top_k": req.top_k},
            )
            resp.raise_for_status()
            return JSONResponse(resp.json())
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text[:500])
    except Exception as exc:
        raise HTTPException(502, f"VDB query failed: {exc}")
