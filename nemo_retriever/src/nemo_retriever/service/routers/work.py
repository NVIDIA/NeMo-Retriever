# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated, OpenAPI-hidden gateway work-broker endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from nemo_retriever.service.services.pipeline_pool import PoolType
from nemo_retriever.service.services.work_queue import StaleLease, get_work_broker

router = APIRouter(include_in_schema=False)


class ClaimRequest(BaseModel):
    pool: Literal["realtime", "batch"]
    worker_uid: str = Field(min_length=1, max_length=253)


class LeaseRequest(BaseModel):
    lease_id: str = Field(min_length=1)
    lease_generation: int = Field(ge=1)
    reason: Literal["release", "payload_fetch", "hash_mismatch"] = "release"


def _gateway_broker(request: Request):
    if request.app.state.config.mode != "gateway":
        raise HTTPException(status_code=404, detail="Work broker is available only on the gateway")
    broker = get_work_broker()
    if broker is None:
        raise HTTPException(status_code=503, detail="Work broker is not ready")
    return broker


@router.post("/internal/work/claim")
async def claim_work(request: Request, body: ClaimRequest) -> Response:
    broker = _gateway_broker(request)
    worker_ip = request.client.host if request.client is not None else ""
    record = await broker.claim(PoolType(body.pool), worker_uid=body.worker_uid, worker_ip=worker_ip)
    if record is None:
        return Response(status_code=204)
    return JSONResponse(broker.claim_payload(record, base_url=request.app.state.config.work_queue.gateway_url))


def _lease_from_headers(request: Request) -> tuple[str, int]:
    lease_id = request.headers.get("X-Work-Lease-Id", "")
    try:
        generation = int(request.headers.get("X-Work-Lease-Generation", ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid work lease generation") from exc
    if not lease_id:
        raise HTTPException(status_code=400, detail="Missing work lease identity")
    return lease_id, generation


@router.get("/internal/work/{work_id}/payload")
async def work_payload(request: Request, work_id: str) -> FileResponse:
    broker = _gateway_broker(request)
    lease_id, generation = _lease_from_headers(request)
    try:
        path = await broker.payload_path(work_id, lease_id, generation)
    except StaleLease as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return FileResponse(path, media_type="application/octet-stream")


@router.post("/internal/work/{work_id}/heartbeat")
async def heartbeat_work(request: Request, work_id: str, body: LeaseRequest) -> JSONResponse:
    broker = _gateway_broker(request)
    try:
        await broker.heartbeat(work_id, body.lease_id, body.lease_generation)
    except StaleLease as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse({"ok": True})


@router.post("/internal/work/{work_id}/release")
async def release_work(request: Request, work_id: str, body: LeaseRequest) -> JSONResponse:
    broker = _gateway_broker(request)
    try:
        await broker.release(work_id, body.lease_id, body.lease_generation, reason=body.reason)
    except StaleLease as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse({"ok": True})
