# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""System endpoints: liveness probe + capability discovery.

These are deliberately lightweight (no DB, no pool work) so liveness
probes never interfere with request processing and capability discovery
can be issued cheaply by every client at startup.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from nemo_retriever.service.config import NimEndpointsConfig, ServiceConfig
from nemo_retriever.service.models.responses import CapabilitiesResponse, CapabilityFlag, HealthResponse

router = APIRouter(tags=["system"])


_PROCESS_START = time.monotonic()


def _try_get_version() -> str:
    try:
        from nemo_retriever.version import get_build_version

        return get_build_version()
    except Exception:  # noqa: BLE001 — version is optional metadata
        return "unknown"


def _flag_for(url: str | None) -> CapabilityFlag:
    if not url:
        return CapabilityFlag(enabled=False, endpoints=[])
    endpoints = [u.strip() for u in url.split(",") if u.strip()]
    return CapabilityFlag(enabled=True, endpoints=endpoints)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe with pool capacity snapshot",
)
async def health(request: Request) -> HealthResponse:
    config: ServiceConfig = request.app.state.config
    pool = request.app.state.processing_pool
    return HealthResponse(
        status="ok",
        version=_try_get_version(),
        uptime_s=round(time.monotonic() - _PROCESS_START, 2),
        workers=config.processing.num_workers,
        in_flight_batches=pool.in_flight_batches() if hasattr(pool, "in_flight_batches") else 0,
        capacity=pool.capacity if hasattr(pool, "capacity") else 0,
        pool_size=pool.pool_size if hasattr(pool, "pool_size") else 0,
        draining=getattr(pool, "is_draining", False),
    )


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="Discover which NIM endpoints and optional features are wired up",
)
async def capabilities(request: Request) -> CapabilitiesResponse:
    config: ServiceConfig = request.app.state.config
    nim: NimEndpointsConfig = config.nim_endpoints

    auth_required = bool(getattr(getattr(config, "auth", None), "api_token", None))

    return CapabilitiesResponse(
        version=_try_get_version(),
        page_elements=_flag_for(nim.page_elements_invoke_url),
        ocr=_flag_for(nim.ocr_invoke_url),
        table_structure=_flag_for(nim.table_structure_invoke_url),
        graphic_elements=_flag_for(nim.graphic_elements_invoke_url),
        embed=_flag_for(nim.embed_invoke_url),
        rerank=_flag_for(nim.rerank_invoke_url),
        auth_required=auth_required,
        server_pdf_split=True,
        bulk_upload=True,
        resumable_sse=True,
        cancel=True,
    )
