# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest endpoints: general, per-page, and whole-document upload.

Each endpoint is mode-aware:

* **gateway** — record :class:`IngestMetrics` (the single authoritative store),
  then proxy the raw HTTP request to the correct backend worker pod.
* **standalone** — record :class:`IngestMetrics` *and* enqueue locally.
* **realtime / batch** — enqueue work to the local pipeline pool only.
  ``IngestMetrics`` is not initialised in worker modes, so ``get_metrics()``
  returns ``None`` and no per-item tracking occurs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from nemo_retriever.service.models.requests import IngestRequest
from nemo_retriever.service.models.responses import (
    DocumentIngestAccepted,
    IngestAccepted,
    JobStatusResponse,
    PageIngestAccepted,
)
from nemo_retriever.service.services.job_tracker import get_job_tracker
from nemo_retriever.service.services.metrics import get_metrics
from nemo_retriever.service.services.pipeline_pool import (
    PoolType,
    WorkItem,
    get_pipeline_pool,
)
from nemo_retriever.service.services.prometheus import (
    GATEWAY_FORWARD_DURATION,
    INGEST_BYTES_TOTAL,
    INGEST_DOCUMENTS_TOTAL,
    INGEST_PAGES_TOTAL,
    INGEST_REQUESTS_TOTAL,
)
from nemo_retriever.service.services.proxy import get_proxy
from nemo_retriever.service.utils.file_type import FileClassifier

_RETRY_AFTER_SECONDS = "5"
_DRY_RUN_HEADER = "X-Nemo-Dry-Run"

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mode(request: Request) -> str:
    return request.app.state.config.mode


def _is_dry_run(request: Request) -> bool:
    """Return ``True`` when the client sends the dry-run header.

    When present (any truthy value), worker pods skip pipeline enqueue
    and return an immediate 202.  The gateway forwards the header to the
    backend unchanged so the worker still sees it.
    """
    val = request.headers.get(_DRY_RUN_HEADER, "").strip().lower()
    return val not in ("", "0", "false", "no")


def _role(request: Request) -> str:
    return getattr(request.app.state, "prometheus_role", "standalone")


def _is_gateway(request: Request) -> bool:
    return _mode(request) == "gateway"


def _record_prometheus(
    request: Request,
    endpoint: str,
    status: str,
    *,
    file_size: int = 0,
    is_page: bool = False,
) -> None:
    role = _role(request)
    INGEST_REQUESTS_TOTAL.labels(role=role, endpoint=endpoint, status=status).inc()
    if file_size > 0:
        INGEST_BYTES_TOTAL.labels(role=role, endpoint=endpoint).inc(file_size)
    if is_page:
        INGEST_PAGES_TOTAL.labels(role=role).inc()
    else:
        INGEST_DOCUMENTS_TOTAL.labels(role=role).inc()


def _register_job(item_id: str) -> None:
    """Register a work item in the job tracker (best-effort, no-op if tracker absent)."""
    tracker = get_job_tracker()
    if tracker is not None:
        tracker.register(item_id)


async def _enqueue_or_reject(pool_type: PoolType, item: WorkItem) -> None:
    """Submit *item* to the pipeline pool, raising HTTP 429 if full."""
    pool = get_pipeline_pool()
    if pool is None:
        return
    if not await pool.submit(pool_type, item):
        raise HTTPException(
            status_code=429,
            detail=f"{pool_type.value} pipeline is at capacity — try again shortly",
            headers={"Retry-After": _RETRY_AFTER_SECONDS},
        )


async def _gateway_forward(request: Request, pool_type: PoolType) -> Response:
    """Proxy the entire HTTP request to the backend for *pool_type*."""
    import time

    proxy = get_proxy()
    if proxy is None:
        raise HTTPException(status_code=503, detail="Gateway proxy not initialised")
    t0 = time.monotonic()
    try:
        resp = await proxy.forward(request, pool_type)
    except Exception as exc:
        logger.exception(
            "Gateway forward to %s failed for %s %s",
            pool_type.value,
            request.method,
            request.url.path,
        )
        INGEST_REQUESTS_TOTAL.labels(
            role="gateway",
            endpoint=request.url.path,
            status="5xx",
        ).inc()
        raise HTTPException(
            status_code=502,
            detail=(f"Gateway failed to forward request to {pool_type.value} backend: " f"{type(exc).__name__}: {exc}"),
        )
    elapsed = time.monotonic() - t0
    GATEWAY_FORWARD_DURATION.labels(backend=pool_type.value).observe(elapsed)
    INGEST_REQUESTS_TOTAL.labels(
        role="gateway",
        endpoint=request.url.path,
        status=f"{resp.status_code // 100}xx",
    ).inc()
    return resp


def _file_size_from_upload(file: UploadFile, request: Request | None = None) -> int:
    """Best-effort file size without reading bytes.

    Checks ``UploadFile.size`` first, then falls back to the total cached
    body size stored by the gateway body-cache middleware.  The cached body
    includes multipart framing so it slightly overestimates, but it's good
    enough for throughput metrics.
    """
    if file.size is not None:
        return file.size
    if request is not None:
        cached = request.scope.get("_cached_body")
        if cached:
            return len(cached)
    return 0


def _parse_backend_json(resp: Response) -> dict:
    """Attempt to decode the backend response body as JSON."""
    try:
        return json.loads(resp.body)
    except Exception:
        return {}


# ------------------------------------------------------------------
# POST /v1/ingest  — general-purpose ingestion (unspecified mode)
# ------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestAccepted,
    status_code=202,
    summary="General-purpose ingestion endpoint",
)
async def ingest(
    request: Request,
    file: UploadFile = File(..., description="The file to ingest"),
    metadata: str = Form(default="{}", description="JSON-encoded IngestRequest metadata"),
) -> IngestAccepted | Response:
    try:
        meta = IngestRequest(**json.loads(metadata))
    except Exception:
        meta = IngestRequest()

    route = PoolType.REALTIME if meta.page_number is not None else PoolType.BATCH

    if _is_gateway(request):
        classification = FileClassifier.classify(file, filename_override=meta.filename or "")
        file_size = _file_size_from_upload(file, request)

        resp = await _gateway_forward(request, route)

        if resp.status_code in (200, 202):
            body = _parse_backend_json(resp)
            _record_prometheus(request, "/v1/ingest", "2xx", file_size=file_size)
            if (m := get_metrics()) is not None:
                m.record_request("/v1/ingest")
                if meta.job_id:
                    m.record_job_created(meta.job_id)
                m.record_document_accepted(
                    document_id=body.get("document_id", "unknown"),
                    job_id=meta.job_id,
                    filename=classification.filename,
                    file_category=classification.category.value,
                    content_type=classification.content_type,
                    file_size_bytes=file_size,
                    endpoint="/v1/ingest",
                )
        return resp

    # ── worker / standalone ──────────────────────────────────────
    dry_run = _is_dry_run(request)
    classification = FileClassifier.classify(file, filename_override=meta.filename or "")

    file_bytes = await file.read()
    content_sha256 = hashlib.sha256(file_bytes).hexdigest()
    document_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    if not dry_run:
        _register_job(document_id)
        await _enqueue_or_reject(
            route,
            WorkItem(id=document_id, payload=file_bytes, filename=file.filename),
        )

    _record_prometheus(request, "/v1/ingest", "2xx", file_size=len(file_bytes))

    if (m := get_metrics()) is not None:
        m.record_request("/v1/ingest")
        if meta.job_id:
            m.record_job_created(meta.job_id)
        m.record_document_accepted(
            document_id=document_id,
            job_id=meta.job_id,
            filename=classification.filename,
            file_category=classification.category.value,
            content_type=classification.content_type,
            file_size_bytes=len(file_bytes),
            endpoint="/v1/ingest",
        )

    return IngestAccepted(
        document_id=document_id,
        job_id=meta.job_id,
        content_sha256=content_sha256,
        status="accepted",
        created_at=now,
    )


# ------------------------------------------------------------------
# POST /v1/ingest/page  — single page from a pre-split document
# ------------------------------------------------------------------


@router.post(
    "/ingest/page",
    response_model=PageIngestAccepted,
    status_code=202,
    summary="Upload a single page belonging to a pre-split document",
)
async def ingest_page(
    request: Request,
    file: UploadFile = File(..., description="A single-page PDF or image"),
    document_id: str = Form(..., description="Client-assigned ID grouping pages from the same source document"),
    page_number: int = Form(..., description="1-based page number within the source document"),
    filename: str = Form(default="", description="Original source document filename"),
) -> PageIngestAccepted | Response:
    if _is_gateway(request):
        classification = FileClassifier.classify(file, filename_override=filename)
        file_size = _file_size_from_upload(file, request)

        resp = await _gateway_forward(request, PoolType.REALTIME)

        if resp.status_code in (200, 202):
            body = _parse_backend_json(resp)
            _record_prometheus(
                request,
                "/v1/ingest/page",
                "2xx",
                file_size=file_size,
                is_page=True,
            )
            if (m := get_metrics()) is not None:
                m.record_request("/v1/ingest/page")
                m.record_page_accepted(
                    page_id=body.get("page_id", "unknown"),
                    document_id=document_id,
                    endpoint="/v1/ingest/page",
                    page_number=page_number,
                    file_size_bytes=file_size,
                    file_category=classification.category.value,
                    content_type=classification.content_type,
                )
        return resp

    # ── worker / standalone ──────────────────────────────────────
    dry_run = _is_dry_run(request)
    classification = FileClassifier.classify(file, filename_override=filename)

    file_bytes = await file.read()
    content_sha256 = hashlib.sha256(file_bytes).hexdigest()
    page_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    if not dry_run:
        _register_job(page_id)
        await _enqueue_or_reject(
            PoolType.REALTIME,
            WorkItem(id=page_id, payload=file_bytes, filename=file.filename),
        )

    _record_prometheus(request, "/v1/ingest/page", "2xx", file_size=len(file_bytes), is_page=True)

    if (m := get_metrics()) is not None:
        m.record_request("/v1/ingest/page")
        m.record_page_accepted(
            page_id=page_id,
            document_id=document_id,
            endpoint="/v1/ingest/page",
            page_number=page_number,
            file_size_bytes=len(file_bytes),
            file_category=classification.category.value,
            content_type=classification.content_type,
        )

    return PageIngestAccepted(
        page_id=page_id,
        document_id=document_id,
        page_number=page_number,
        content_sha256=content_sha256,
        status="accepted",
        created_at=now,
    )


# ------------------------------------------------------------------
# POST /v1/ingest/document  — whole document (not individual pages)
# ------------------------------------------------------------------


@router.post(
    "/ingest/document",
    response_model=DocumentIngestAccepted,
    status_code=202,
    summary="Upload a complete document for ingestion (server handles page splitting)",
)
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="The full document to ingest"),
    metadata: str = Form(default="{}", description="JSON-encoded IngestRequest metadata"),
) -> DocumentIngestAccepted | Response:
    try:
        meta = IngestRequest(**json.loads(metadata))
    except Exception:
        meta = IngestRequest()

    if _is_gateway(request):
        classification = FileClassifier.classify(file, filename_override=meta.filename or "")
        file_size = _file_size_from_upload(file, request)

        resp = await _gateway_forward(request, PoolType.BATCH)

        if resp.status_code in (200, 202):
            body = _parse_backend_json(resp)
            _record_prometheus(request, "/v1/ingest/document", "2xx", file_size=file_size)
            if (m := get_metrics()) is not None:
                m.record_request("/v1/ingest/document")
                if meta.job_id:
                    m.record_job_created(meta.job_id)
                m.record_document_accepted(
                    document_id=body.get("document_id", "unknown"),
                    job_id=meta.job_id,
                    filename=classification.filename,
                    file_category=classification.category.value,
                    content_type=classification.content_type,
                    file_size_bytes=file_size,
                    endpoint="/v1/ingest/document",
                )
        return resp

    # ── worker / standalone ──────────────────────────────────────
    dry_run = _is_dry_run(request)
    classification = FileClassifier.classify(file, filename_override=meta.filename or "")

    file_bytes = await file.read()
    content_sha256 = hashlib.sha256(file_bytes).hexdigest()
    document_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    if not dry_run:
        _register_job(document_id)
        await _enqueue_or_reject(
            PoolType.BATCH,
            WorkItem(id=document_id, payload=file_bytes, filename=file.filename),
        )

    _record_prometheus(request, "/v1/ingest/document", "2xx", file_size=len(file_bytes))

    if (m := get_metrics()) is not None:
        m.record_request("/v1/ingest/document")
        if meta.job_id:
            m.record_job_created(meta.job_id)
        m.record_document_accepted(
            document_id=document_id,
            job_id=meta.job_id,
            filename=classification.filename,
            file_category=classification.category.value,
            content_type=classification.content_type,
            file_size_bytes=len(file_bytes),
            endpoint="/v1/ingest/document",
        )

    return DocumentIngestAccepted(
        document_id=document_id,
        filename=classification.filename,
        file_size_bytes=len(file_bytes),
        content_sha256=content_sha256,
        status="accepted",
        created_at=now,
    )


# ------------------------------------------------------------------
# GET /v1/ingest/status/{item_id}  — status for general ingest items
# GET /v1/ingest/page/status/{page_id}  — status for page items
# GET /v1/ingest/document/status/{document_id}  — status for document items
# ------------------------------------------------------------------


def _status_response(request: Request, item_id: str) -> JSONResponse:
    """Look up job status and return the appropriate HTTP code.

    Returns 200 for completed/failed, 202 for pending/processing, 404 if unknown.
    When returning a terminal (200) response, result_data is consumed from the
    tracker so memory is freed after the client has retrieved it.
    """
    from nemo_retriever.service.services.job_tracker import JobStatus

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(
            status_code=404,
            detail="Job tracker is not available on this pod. In split topology, query the worker pod directly.",
        )
    rec = tracker.get(item_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"No tracked job with id={item_id!r}")

    is_terminal = rec.status in (JobStatus.COMPLETED, JobStatus.FAILED)
    result_data = tracker.consume_result_data(item_id) if is_terminal else None

    body = JobStatusResponse(
        id=rec.id,
        status=rec.status.value,
        submitted_at=rec.submitted_at,
        started_at=rec.started_at,
        completed_at=rec.completed_at,
        elapsed_s=rec.elapsed_s,
        result_rows=rec.result_rows,
        result_data=result_data,
        error=rec.error,
    ).model_dump()

    if is_terminal:
        return JSONResponse(content=body, status_code=200)
    return JSONResponse(content=body, status_code=202)


@router.get(
    "/ingest/status/{item_id}",
    summary="Check processing status of a general ingest submission",
    responses={200: {"model": JobStatusResponse}, 202: {"model": JobStatusResponse}},
)
async def ingest_status(request: Request, item_id: str) -> JSONResponse:
    if _is_gateway(request):
        proxy = get_proxy()
        if proxy is None:
            raise HTTPException(status_code=503, detail="Gateway proxy not initialised")
        for pool_type in (PoolType.REALTIME, PoolType.BATCH):
            resp = await proxy.forward_get(request, pool_type, f"/v1/ingest/status/{item_id}")
            if resp.status_code != 404:
                return resp
        raise HTTPException(status_code=404, detail=f"No tracked job with id={item_id!r}")
    return _status_response(request, item_id)


@router.get(
    "/ingest/page/status/{page_id}",
    summary="Check processing status of a page ingest submission",
    responses={200: {"model": JobStatusResponse}, 202: {"model": JobStatusResponse}},
)
async def ingest_page_status(request: Request, page_id: str) -> JSONResponse:
    if _is_gateway(request):
        proxy = get_proxy()
        if proxy is None:
            raise HTTPException(status_code=503, detail="Gateway proxy not initialised")
        return await proxy.forward_get(request, PoolType.REALTIME, f"/v1/ingest/page/status/{page_id}")
    return _status_response(request, page_id)


@router.get(
    "/ingest/document/status/{document_id}",
    summary="Check processing status of a document ingest submission",
    responses={200: {"model": JobStatusResponse}, 202: {"model": JobStatusResponse}},
)
async def ingest_document_status(request: Request, document_id: str) -> JSONResponse:
    if _is_gateway(request):
        proxy = get_proxy()
        if proxy is None:
            raise HTTPException(status_code=503, detail="Gateway proxy not initialised")
        return await proxy.forward_get(request, PoolType.BATCH, f"/v1/ingest/document/status/{document_id}")
    return _status_response(request, document_id)


# ------------------------------------------------------------------
# GET /v1/ingest/pipeline-config  — introspect live pipeline setup
# ------------------------------------------------------------------


@router.get(
    "/ingest/pipeline-config",
    summary="Return the live pipeline configuration for this pod (or aggregated from backends via the gateway)",
)
async def pipeline_config(request: Request):
    """Return redacted pipeline configuration.

    * **worker / standalone** — returns the local pipeline configs directly.
    * **gateway** — fans out GET requests to one realtime and one batch
      backend pod, aggregates the responses, and returns them keyed by role.
    """
    mode = _mode(request)

    if _is_gateway(request):
        proxy = get_proxy()
        if proxy is None:
            raise HTTPException(status_code=503, detail="Gateway proxy not initialised")

        aggregated: dict[str, object] = {"source": "gateway", "mode": mode}
        for pool_type in (PoolType.REALTIME, PoolType.BATCH):
            label = pool_type.value
            try:
                resp = await proxy.forward_get(request, pool_type, "/v1/ingest/pipeline-config")
                if resp.status_code == 200:
                    aggregated[label] = json.loads(resp.body)
                else:
                    aggregated[label] = {
                        "error": f"HTTP {resp.status_code}",
                        "body": resp.body.decode(errors="replace")[:500],
                    }
            except Exception as exc:
                aggregated[label] = {"error": f"{type(exc).__name__}: {exc}"}

        return JSONResponse(content=aggregated)

    from nemo_retriever.service.services.pipeline_executor import get_pipeline_configs

    pool = get_pipeline_pool()
    pool_stats = pool.stats() if pool is not None else {}

    return JSONResponse(content={
        "source": mode,
        "mode": mode,
        "pipelines": get_pipeline_configs(),
        "pool_stats": pool_stats,
    })


# ------------------------------------------------------------------
# POST /v1/query  — vector search (proxied to vectordb pod)
# ------------------------------------------------------------------


@router.post(
    "/query",
    summary="Search ingested documents by semantic similarity",
)
async def query(request: Request) -> Response:
    """Proxy a query request to the VectorDB service.

    * **gateway / standalone** — forwards the JSON body to the vectordb pod.
    * **worker** — returns 404 (workers don't handle queries).
    """
    import httpx

    config = request.app.state.config

    if not config.vectordb.enabled:
        raise HTTPException(
            status_code=404,
            detail="VectorDB is not enabled in the service configuration.",
        )

    mode = _mode(request)
    if mode in ("realtime", "batch"):
        raise HTTPException(
            status_code=404,
            detail="Query endpoint is not available on worker pods. Use the gateway.",
        )

    vectordb_url = config.vectordb.vectordb_url.rstrip("/")
    target = f"{vectordb_url}/v1/query"

    body = await request.body()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                target,
                content=body,
                headers={"Content-Type": "application/json"},
            )
    except Exception as exc:
        logger.exception("Failed to proxy query to vectordb at %s", target)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach VectorDB service: {type(exc).__name__}: {exc}",
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type="application/json",
    )
