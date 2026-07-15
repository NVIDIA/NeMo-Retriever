# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public collection/document lifecycle routes.

The gateway owns authentication and forwards only logical resource names and
the authenticated scope. LanceDB locations and physical table names never
cross this boundary.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

router = APIRouter(tags=["collections"])


async def _forward(request: Request, suffix: str) -> Response:
    config = request.app.state.config
    if not config.vectordb.enabled:
        raise HTTPException(404, "VectorDB is not enabled in the service configuration.")
    if config.mode in ("realtime", "batch"):
        raise HTTPException(404, "Collection management is available through the gateway.")

    target = f"{config.vectordb.vectordb_url.rstrip('/')}/v1/{suffix}"
    headers = {"Content-Type": request.headers.get("content-type", "application/json")}
    if scope := request.headers.get("X-NRL-Scope"):
        headers["X-NRL-Scope"] = scope
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.request(
                request.method, target, content=await request.body(), params=request.query_params, headers=headers,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Failed to reach VectorDB service: {type(exc).__name__}: {exc}") from exc
    return Response(
        content=response.content, status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


@router.api_route("/collections", methods=["GET", "POST"])
async def collections_root(request: Request) -> Response:
    return await _forward(request, "collections")


@router.api_route("/collections/{collection_name}", methods=["GET", "PATCH", "DELETE"])
async def collection_item(request: Request, collection_name: str) -> Response:
    return await _forward(request, f"collections/{collection_name}")


@router.get("/collections/{collection_name}/documents")
async def collection_documents(request: Request, collection_name: str) -> Response:
    return await _forward(request, f"collections/{collection_name}/documents")


@router.api_route(
    "/collections/{collection_name}/documents/{document_id}", methods=["GET", "DELETE"],
)
async def collection_document(request: Request, collection_name: str, document_id: str) -> Response:
    return await _forward(request, f"collections/{collection_name}/documents/{document_id}")
