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


@router.get("/api/overview")
async def overview(request: Request) -> JSONResponse:
    """Aggregate cluster status for the overview panel."""
    config = request.app.state.config

    backends = {}
    try:
        from nemo_retriever.service.services.proxy import get_proxy
        from nemo_retriever.service.services.pipeline_pool import PoolType

        proxy = get_proxy()
        if proxy is not None:
            backends["realtime"] = await proxy.check_backend(PoolType.REALTIME)
            backends["batch"] = await proxy.check_backend(PoolType.BATCH)
    except Exception as exc:
        logger.debug("Could not check backends: %s", exc)

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
            "vectordb": vdb_status,
            "job_summary": job_summary,
            "worker_config": worker_config,
            "gateway": gateway_info,
        }
    )


# ── Jobs SSE stream ─────────────────────────────────────────────────


@router.get("/api/jobs")
async def jobs_sse(request: Request) -> StreamingResponse:
    """SSE stream of job events with periodic summary heartbeats."""
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
                    "jobs": tracker.all_records(),
                }
                yield f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"

            last_heartbeat = asyncio.get_event_loop().time()

            while True:
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    yield f"event: job_update\ndata: {json.dumps(event)}\n\n"
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


@router.get("/api/jobs/snapshot")
async def jobs_snapshot(request: Request) -> JSONResponse:
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    if tracker is None:
        return JSONResponse({"summary": {}, "jobs": []})

    return JSONResponse(
        {
            "summary": tracker.summary(),
            "jobs": tracker.all_records(),
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
