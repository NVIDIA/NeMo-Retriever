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

import asyncio
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.responses import StreamingResponse

from nemo_retriever.service.models.pipeline_spec import PipelineSpec
from nemo_retriever.service.models.requests import IngestRequest, JobCreateRequest
from nemo_retriever.service.models.responses import (
    DocumentIngestAccepted,
    DocumentStatusResponse,
    IngestAccepted,
    JobAggregateResponse,
    JobCreatedResponse,
    JobDocumentsPage,
    JobStatusResponse,
    PageIngestAccepted,
    SidecarUploadResponse,
)
from nemo_retriever.service.policy import PolicyError, validate_pipeline_spec
from nemo_retriever.service.services.event_bus import get_event_bus
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
_GATEWAY_DOC_ID_HEADER = "X-Gateway-Document-Id"
_GATEWAY_CALLBACK_HEADER = "X-Gateway-Callback-Url"
_GATEWAY_PIPELINE_SPEC_HEADER = "X-Gateway-Pipeline-Spec"
_GATEWAY_JOB_ID_HEADER = "X-Gateway-Job-Id"
_PAGE_THRESHOLD_FOR_BATCH = 5

# SSE keepalive cadence; tests monkey-patch this to a short value so
# they don't have to wait the production 30 s before the generator
# re-checks ``request.is_disconnected()`` and exits cleanly.
SSE_KEEPALIVE_TIMEOUT_S = 30.0

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


def _is_worker(request: Request) -> bool:
    """Return True for split-mode worker pods (``realtime`` or ``batch``).

    Workers don't own the ``JobTracker`` aggregate — the gateway does.
    When the gateway forwards an upload to a worker, the URL still
    contains the ``job_id``, but the worker must trust it (and not
    re-validate via ``_require_job``).
    """
    return _mode(request) in ("realtime", "batch")


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


def _register_document_under_job(
    *,
    document_id: str,
    job_id: str,
    filename: str | None = None,
) -> None:
    """Register a per-document tracker entry inside an existing job.

    Maps :class:`JobTrackerError` subclasses to HTTP responses so the
    upload routes can surface 404 (job missing) / 409 (full or
    finalized) without leaking the tracker's internal exception types.
    """
    from nemo_retriever.service.services.job_tracker import (
        JobFinalizedError,
        JobFullError,
        JobNotFoundError,
        JobTrackerError,
    )

    tracker = get_job_tracker()
    if tracker is None:
        return
    try:
        tracker.register_document(document_id, job_id=job_id, filename=filename)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except JobFullError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except JobFinalizedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except JobTrackerError as exc:
        raise HTTPException(status_code=getattr(exc, "status_code", 500), detail=str(exc)) from exc


def _require_job(job_id: str):
    """Look up an existing :class:`JobAggregate` or raise HTTP 404."""
    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker not available")
    agg = tracker.get_job(job_id)
    if agg is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return agg


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


def _build_callback_url(request: Request) -> str:
    """Build the internal callback URL pointing to THIS specific gateway pod.

    Uses ``POD_IP`` env (Kubernetes downward API) so the worker calls
    back to the exact gateway pod that accepted the upload, not the
    Service VIP which might route to a different replica.
    """
    pod_ip = os.environ.get("POD_IP")
    port = request.app.state.config.server.port
    if pod_ip:
        return f"http://{pod_ip}:{port}/v1/internal/job-callback"
    return f"http://localhost:{port}/v1/internal/job-callback"


async def _gateway_forward(
    request: Request,
    pool_type: PoolType,
    *,
    extra_headers: dict[str, str] | None = None,
) -> Response:
    """Proxy the entire HTTP request to the backend for *pool_type*."""
    import time

    proxy = get_proxy()
    if proxy is None:
        raise HTTPException(status_code=503, detail="Gateway proxy not initialised")
    t0 = time.monotonic()
    try:
        resp = await proxy.forward(request, pool_type, extra_headers=extra_headers)
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


def _check_upload_size(file: UploadFile, request: Request) -> None:
    """Reject uploads exceeding the configured size limit before buffering."""
    config = request.app.state.config
    limit = getattr(getattr(config, "resources", None), "max_upload_bytes", None)
    if limit is None:
        return
    size = file.size
    if size is not None and size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Upload size {size:,} bytes exceeds limit of {limit:,} bytes",
        )


def _count_pdf_pages(file_bytes: bytes) -> int:
    """Return the number of pages in a PDF, or 1 for non-PDF / errors."""
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(file_bytes)
        n = len(doc)
        doc.close()
        return n
    except Exception as exc:
        logger.warning("Could not determine PDF page count; defaulting to 1 page: %s", exc)
        return 1


def _route_by_page_count(file_bytes: bytes, meta: IngestRequest) -> PoolType:
    """Route to realtime for small docs (<threshold pages), batch for larger.

    When the client requested PDF page-chunking via
    :attr:`PipelineSpec.pdf_split`, we route to **batch** as soon as the
    document has more than one chunk's worth of pages — chunking is
    intrinsically a throughput-oriented operation.
    """
    if meta.page_number is not None:
        return PoolType.REALTIME
    pages = _count_pdf_pages(file_bytes)
    if meta.pipeline and meta.pipeline.pdf_split is not None:
        if pages > meta.pipeline.pdf_split.pages_per_chunk:
            return PoolType.BATCH
    return PoolType.REALTIME if pages < _PAGE_THRESHOLD_FOR_BATCH else PoolType.BATCH


def _build_policy(request: Request):  # -> PipelineOverridesPolicy
    """Build a :class:`PipelineOverridesPolicy` from the live app config.

    The ``caption_enabled`` flag is derived here so the trust boundary
    stays in one place — clients can only override caption settings when
    the operator has wired up ``nim_endpoints.caption_invoke_url``.
    """
    cfg = request.app.state.config
    caption_enabled = bool(getattr(cfg.nim_endpoints, "caption_invoke_url", None))
    return cfg.pipeline_overrides.to_policy(caption_enabled=caption_enabled)


def _resolve_pipeline_spec(request: Request, meta: IngestRequest) -> PipelineSpec | None:
    """Validate ``meta.pipeline`` against the service's override policy.

    Returns ``None`` when the spec is missing or empty so the worker
    short-circuits to the legacy startup-baked pipeline. Raises
    :class:`HTTPException` (the FastAPI-native error) for policy denials.
    """
    if meta.pipeline is None:
        return None
    policy = _build_policy(request)
    try:
        return validate_pipeline_spec(meta.pipeline, policy)
    except PolicyError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _spec_from_gateway_header(request: Request) -> PipelineSpec | None:
    """Recover and re-validate the spec forwarded by the gateway pod.

    The gateway has already validated against its own copy of the policy,
    but we re-validate on the worker as defense-in-depth: a misconfigured
    gateway or a pod with a different ``pipeline_overrides`` config will
    still see consistent enforcement.
    """
    raw = request.headers.get(_GATEWAY_PIPELINE_SPEC_HEADER)
    if not raw:
        return None
    try:
        spec = PipelineSpec.model_validate_json(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Malformed {_GATEWAY_PIPELINE_SPEC_HEADER!r} from gateway: {exc}",
        ) from exc
    policy = _build_policy(request)
    try:
        return validate_pipeline_spec(spec, policy)
    except PolicyError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _parse_backend_json(resp: Response) -> dict:
    """Attempt to decode the backend response body as JSON."""
    try:
        return json.loads(resp.body)
    except Exception:
        return {}


# ------------------------------------------------------------------
# POST /v1/ingest/job          — create a new job aggregate
# GET  /v1/ingest/job/{job_id} — fetch the aggregate's current state
# ------------------------------------------------------------------


def _aggregate_to_response(agg, *, documents: list[dict[str, Any]] | None = None) -> JobAggregateResponse:
    """Project a :class:`JobAggregate` to the wire response model."""
    return JobAggregateResponse(
        job_id=agg.job_id,
        expected_documents=agg.expected_documents,
        status=agg.status.value,
        created_at=agg.created_at,
        started_at=agg.started_at,
        finalized_at=agg.finalized_at,
        elapsed_s=agg.elapsed_s,
        label=agg.label,
        counts=dict(agg.counts),
        document_ids=list(agg.document_ids),
        documents=documents,
    )


@router.post(
    "/ingest/job",
    response_model=JobCreatedResponse,
    status_code=201,
    summary="Create a new ingestion job aggregate",
)
async def create_job(request: Request, body: JobCreateRequest) -> JobCreatedResponse:
    """Open a job that will receive ``expected_documents`` uploads.

    The server returns an opaque ``job_id`` the client uses for every
    subsequent ``POST /v1/ingest/job/{job_id}/document`` (or ``/page``)
    call. The job is in-memory only; gateway pod restarts erase it
    (this is intentional — see the J1 design notes).
    """
    from nemo_retriever.service.services.job_tracker import JobTrackerError

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker not available")
    job_id = uuid.uuid4().hex
    try:
        agg = tracker.register_job(
            job_id,
            expected_documents=body.expected_documents,
            label=body.label,
            metadata=body.metadata,
        )
    except JobTrackerError as exc:
        raise HTTPException(status_code=getattr(exc, "status_code", 500), detail=str(exc)) from exc

    if (m := get_metrics()) is not None:
        m.record_request("/v1/ingest/job")
        m.record_job_created(job_id)
    return JobCreatedResponse(
        job_id=agg.job_id,
        expected_documents=agg.expected_documents,
        status=agg.status.value,
        created_at=agg.created_at,
        label=agg.label,
    )


@router.get(
    "/ingest/job/{job_id}",
    response_model=JobAggregateResponse,
    summary="Fetch the current state of a job aggregate",
)
async def get_job(
    request: Request,
    job_id: str,
    include_documents: bool = False,
) -> JobAggregateResponse:
    """Return the live aggregate snapshot.

    Pass ``?include_documents=true`` to also return the per-document
    records (capped to the first 10k entries to keep payloads bounded).
    """
    agg = _require_job(job_id)
    documents: list[dict[str, Any]] | None = None
    if include_documents:
        tracker = get_job_tracker()
        docs = tracker.job_documents(job_id) if tracker is not None else []
        cap = 10_000
        documents = [d.model_dump() for d in docs[:cap]]
    return _aggregate_to_response(agg, documents=documents)


# ------------------------------------------------------------------
# GET /v1/ingest/job/{job_id}/documents
#   J6 — paginated per-document status under a job. Supports filtering
#   by ``status`` and slicing with ``offset`` / ``limit``. Preferred
#   over ``?include_documents=true`` for big jobs.
# GET /v1/ingest/job/{job_id}/document/{document_id}
#   J6 — single-document detail (replaces the legacy ``/ingest/status``
#   shape with a JSON body that exposes ``job_id`` + ``filename``).
# ------------------------------------------------------------------


def _document_to_response(rec, *, result_data=None) -> DocumentStatusResponse:
    """Project a :class:`DocumentRecord` to the wire response shape."""
    return DocumentStatusResponse(
        document_id=rec.id,
        job_id=rec.job_id,
        status=rec.status.value,
        submitted_at=rec.submitted_at,
        started_at=rec.started_at,
        completed_at=rec.completed_at,
        elapsed_s=rec.elapsed_s,
        filename=rec.filename,
        result_rows=rec.result_rows,
        result_data=result_data,
        error=rec.error,
    )


@router.get(
    "/ingest/job/{job_id}/documents",
    response_model=JobDocumentsPage,
    summary="Paginated per-document status under a job",
)
async def get_job_documents(
    request: Request,
    job_id: str,
    status: str | None = None,
    offset: int = 0,
    limit: int = 100,
) -> JobDocumentsPage:
    """Return per-document records for *job_id*, paginated.

    Parameters
    ----------
    status:
        Optional filter — one of ``pending``, ``processing``,
        ``completed``, ``failed``. The default returns every document.
    offset:
        Zero-based page start (must be ``>= 0``).
    limit:
        Page size; capped at 1000.

    The aggregate ``total`` is the absolute document count, and
    ``total_filtered`` is what the status filter selected; ``items``
    is the slice for the requested page.
    """
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be in [1, 1000]")

    agg = _require_job(job_id)
    tracker = get_job_tracker()
    docs = tracker.job_documents(job_id) if tracker is not None else []

    if status is not None:
        status_lower = status.lower()
        valid = {"pending", "processing", "completed", "failed"}
        if status_lower not in valid:
            raise HTTPException(
                status_code=400,
                detail=f"status must be one of {sorted(valid)}, got {status!r}",
            )
        filtered = [d for d in docs if d.status.value == status_lower]
    else:
        filtered = docs

    page = filtered[offset : offset + limit]
    return JobDocumentsPage(
        job_id=agg.job_id,
        total=len(docs),
        total_filtered=len(filtered),
        offset=offset,
        limit=limit,
        items=[_document_to_response(d) for d in page],
    )


@router.get(
    "/ingest/job/{job_id}/document/{document_id}",
    response_model=DocumentStatusResponse,
    summary="Fetch the current status of one document in a job",
    responses={
        200: {"model": DocumentStatusResponse},
        202: {"model": DocumentStatusResponse},
        404: {"description": "Job or document not found"},
    },
)
async def get_job_document(
    request: Request,
    job_id: str,
    document_id: str,
) -> Response:
    """Single-document detail nested under the owning job.

    Returns HTTP 200 when the document is terminal (consumes
    ``result_data`` so memory is freed for the caller) and HTTP 202
    while still pending/processing. A 404 is returned if either the
    job is unknown or the document does not belong to this job — the
    latter prevents leaking document existence across tenants.
    """
    from nemo_retriever.service.services.job_tracker import DocumentStatus

    _require_job(job_id)
    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker is not available.")
    rec = tracker.get_document(document_id)
    if rec is None or rec.job_id != job_id:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id!r} not found in job {job_id!r}",
        )
    is_terminal = rec.status in (DocumentStatus.COMPLETED, DocumentStatus.FAILED)
    result_data = tracker.consume_result_data(document_id) if is_terminal else None
    body = _document_to_response(rec, result_data=result_data).model_dump()
    return JSONResponse(content=body, status_code=200 if is_terminal else 202)


# ------------------------------------------------------------------
# POST /v1/ingest/job/{job_id}/document   — auto-routed by page count
# POST /v1/ingest/job/{job_id}/page       — explicit single page
# POST /v1/ingest/job/{job_id}/whole      — explicit whole document
# ------------------------------------------------------------------


@router.post(
    "/ingest/job/{job_id}/document",
    response_model=IngestAccepted,
    status_code=202,
    summary="Submit a document to an existing job (auto-routed by page count)",
)
async def submit_document_to_job(
    request: Request,
    job_id: str,
    file: UploadFile = File(..., description="The file to ingest"),
    metadata: str = Form(default="{}", description="JSON-encoded IngestRequest metadata"),
) -> IngestAccepted | Response:
    """General-purpose upload into a job.

    Routes to the realtime or batch pool based on PDF page count
    (matches the legacy ``POST /v1/ingest`` behaviour). The ``job_id``
    comes from the URL path; any ``job_id`` in the metadata body is
    ignored.
    """
    try:
        meta = IngestRequest(**json.loads(metadata))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")

    # Job lookup is gateway/standalone only — worker pods don't own the
    # JobTracker, so we must trust the gateway-forwarded URL.
    if not _is_worker(request):
        _require_job(job_id)
    _check_upload_size(file, request)
    validated_spec = _resolve_pipeline_spec(request, meta)

    if _is_gateway(request):
        classification = FileClassifier.classify(file, filename_override=meta.filename or "")
        file_size = _file_size_from_upload(file, request)

        file_bytes = await file.read()
        route = _route_by_page_count(file_bytes, meta)

        document_id = uuid.uuid4().hex
        content_sha256 = hashlib.sha256(file_bytes).hexdigest()
        now = datetime.now(timezone.utc).isoformat()

        _register_document_under_job(document_id=document_id, job_id=job_id, filename=file.filename)
        tracker = get_job_tracker()
        if tracker is not None:
            tracker.mark_processing(document_id)

        callback_url = _build_callback_url(request)
        extra_headers = {
            _GATEWAY_DOC_ID_HEADER: document_id,
            _GATEWAY_JOB_ID_HEADER: job_id,
            _GATEWAY_CALLBACK_HEADER: callback_url,
        }
        if validated_spec is not None:
            extra_headers[_GATEWAY_PIPELINE_SPEC_HEADER] = validated_spec.model_dump_json()
        resp = await _gateway_forward(request, route, extra_headers=extra_headers)

        if resp.status_code not in (200, 202):
            if tracker is not None:
                tracker.mark_failed(document_id, f"Worker returned HTTP {resp.status_code}")
            return resp

        _record_prometheus(request, "/v1/ingest/job/document", "2xx", file_size=file_size)
        if (m := get_metrics()) is not None:
            m.record_request("/v1/ingest/job/document")
            m.record_document_accepted(
                document_id=document_id,
                job_id=job_id,
                filename=classification.filename,
                file_category=classification.category.value,
                content_type=classification.content_type,
                file_size_bytes=file_size,
                endpoint="/v1/ingest/job/document",
            )

        return IngestAccepted(
            document_id=document_id,
            job_id=job_id,
            content_sha256=content_sha256,
            status="accepted",
            created_at=now,
        )

    # ── worker / standalone ──────────────────────────────────────
    classification = FileClassifier.classify(file, filename_override=meta.filename or "")

    file_bytes = await file.read()
    route = _route_by_page_count(file_bytes, meta)
    content_sha256 = hashlib.sha256(file_bytes).hexdigest()
    now = datetime.now(timezone.utc).isoformat()

    gw_doc_id = request.headers.get(_GATEWAY_DOC_ID_HEADER)
    gw_callback_url = request.headers.get(_GATEWAY_CALLBACK_HEADER)
    gw_job_id = request.headers.get(_GATEWAY_JOB_ID_HEADER) or job_id
    document_id = gw_doc_id or uuid.uuid4().hex

    worker_spec = _spec_from_gateway_header(request) if gw_doc_id else validated_spec

    if not gw_callback_url:
        _register_document_under_job(document_id=document_id, job_id=job_id, filename=file.filename)

    await _enqueue_or_reject(
        route,
        WorkItem(
            id=document_id,
            payload=file_bytes,
            filename=file.filename,
            callback_url=gw_callback_url,
            job_id=gw_job_id,
            pipeline_spec=worker_spec.model_dump(mode="json") if worker_spec is not None else None,
        ),
    )

    _record_prometheus(request, "/v1/ingest/job/document", "2xx", file_size=len(file_bytes))

    if (m := get_metrics()) is not None:
        m.record_request("/v1/ingest/job/document")
        m.record_document_accepted(
            document_id=document_id,
            job_id=gw_job_id,
            filename=classification.filename,
            file_category=classification.category.value,
            content_type=classification.content_type,
            file_size_bytes=len(file_bytes),
            endpoint="/v1/ingest/job/document",
        )

    return IngestAccepted(
        document_id=document_id,
        job_id=gw_job_id,
        content_sha256=content_sha256,
        status="accepted",
        created_at=now,
    )


@router.post(
    "/ingest/job/{job_id}/page",
    response_model=PageIngestAccepted,
    status_code=202,
    summary="Upload a single page belonging to a pre-split document inside a job",
)
async def submit_page_to_job(
    request: Request,
    job_id: str,
    file: UploadFile = File(..., description="A single-page PDF or image"),
    document_id: str = Form(..., description="Client-assigned ID grouping pages from the same source document"),
    page_number: int = Form(..., description="1-based page number within the source document"),
    filename: str = Form(default="", description="Original source document filename"),
) -> PageIngestAccepted | Response:
    # Job lookup is gateway/standalone only (workers don't own the
    # JobTracker — they trust the gateway-forwarded URL).
    if not _is_worker(request):
        _require_job(job_id)
    _check_upload_size(file, request)

    if _is_gateway(request):
        classification = FileClassifier.classify(file, filename_override=filename)
        file_size = _file_size_from_upload(file, request)

        page_id = uuid.uuid4().hex
        content_sha256 = hashlib.sha256((await file.read()) or b"").hexdigest()
        now = datetime.now(timezone.utc).isoformat()

        _register_document_under_job(document_id=page_id, job_id=job_id, filename=filename or file.filename)
        tracker = get_job_tracker()
        if tracker is not None:
            tracker.mark_processing(page_id)

        callback_url = _build_callback_url(request)
        resp = await _gateway_forward(
            request,
            PoolType.REALTIME,
            extra_headers={
                _GATEWAY_DOC_ID_HEADER: page_id,
                _GATEWAY_JOB_ID_HEADER: job_id,
                _GATEWAY_CALLBACK_HEADER: callback_url,
            },
        )

        if resp.status_code not in (200, 202):
            if tracker is not None:
                tracker.mark_failed(page_id, f"Worker returned HTTP {resp.status_code}")
            return resp

        _record_prometheus(
            request,
            "/v1/ingest/job/page",
            "2xx",
            file_size=file_size,
            is_page=True,
        )
        if (m := get_metrics()) is not None:
            m.record_request("/v1/ingest/job/page")
            m.record_page_accepted(
                page_id=page_id,
                document_id=document_id,
                endpoint="/v1/ingest/job/page",
                page_number=page_number,
                file_size_bytes=file_size,
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

    # ── worker / standalone ──────────────────────────────────────
    dry_run = _is_dry_run(request)
    classification = FileClassifier.classify(file, filename_override=filename)

    file_bytes = await file.read()
    content_sha256 = hashlib.sha256(file_bytes).hexdigest()
    now = datetime.now(timezone.utc).isoformat()

    gw_doc_id = request.headers.get(_GATEWAY_DOC_ID_HEADER)
    gw_callback_url = request.headers.get(_GATEWAY_CALLBACK_HEADER)
    gw_job_id = request.headers.get(_GATEWAY_JOB_ID_HEADER) or job_id
    page_id = gw_doc_id or uuid.uuid4().hex

    if not dry_run:
        if not gw_callback_url:
            _register_document_under_job(document_id=page_id, job_id=job_id, filename=filename or file.filename)
        await _enqueue_or_reject(
            PoolType.REALTIME,
            WorkItem(
                id=page_id,
                payload=file_bytes,
                filename=file.filename,
                callback_url=gw_callback_url,
                job_id=gw_job_id,
            ),
        )

    _record_prometheus(request, "/v1/ingest/job/page", "2xx", file_size=len(file_bytes), is_page=True)

    if (m := get_metrics()) is not None:
        m.record_request("/v1/ingest/job/page")
        m.record_page_accepted(
            page_id=page_id,
            document_id=document_id,
            endpoint="/v1/ingest/job/page",
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


@router.post(
    "/ingest/job/{job_id}/whole",
    response_model=DocumentIngestAccepted,
    status_code=202,
    summary="Upload a complete document into a job (server handles page splitting)",
)
async def submit_whole_document_to_job(
    request: Request,
    job_id: str,
    file: UploadFile = File(..., description="The full document to ingest"),
    metadata: str = Form(default="{}", description="JSON-encoded IngestRequest metadata"),
) -> DocumentIngestAccepted | Response:
    try:
        meta = IngestRequest(**json.loads(metadata))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")

    # Job lookup is gateway/standalone only (workers don't own the
    # JobTracker — they trust the gateway-forwarded URL).
    if not _is_worker(request):
        _require_job(job_id)
    _check_upload_size(file, request)
    validated_spec = _resolve_pipeline_spec(request, meta)

    if _is_gateway(request):
        classification = FileClassifier.classify(file, filename_override=meta.filename or "")
        file_size = _file_size_from_upload(file, request)

        document_id = uuid.uuid4().hex
        file_bytes = await file.read()
        content_sha256 = hashlib.sha256(file_bytes).hexdigest()
        now = datetime.now(timezone.utc).isoformat()

        _register_document_under_job(document_id=document_id, job_id=job_id, filename=file.filename)
        tracker = get_job_tracker()
        if tracker is not None:
            tracker.mark_processing(document_id)

        callback_url = _build_callback_url(request)
        extra_headers = {
            _GATEWAY_DOC_ID_HEADER: document_id,
            _GATEWAY_JOB_ID_HEADER: job_id,
            _GATEWAY_CALLBACK_HEADER: callback_url,
        }
        if validated_spec is not None:
            extra_headers[_GATEWAY_PIPELINE_SPEC_HEADER] = validated_spec.model_dump_json()
        resp = await _gateway_forward(request, PoolType.BATCH, extra_headers=extra_headers)

        if resp.status_code not in (200, 202):
            if tracker is not None:
                tracker.mark_failed(document_id, f"Worker returned HTTP {resp.status_code}")
            return resp

        _record_prometheus(request, "/v1/ingest/job/whole", "2xx", file_size=file_size)
        if (m := get_metrics()) is not None:
            m.record_request("/v1/ingest/job/whole")
            m.record_document_accepted(
                document_id=document_id,
                job_id=job_id,
                filename=classification.filename,
                file_category=classification.category.value,
                content_type=classification.content_type,
                file_size_bytes=file_size,
                endpoint="/v1/ingest/job/whole",
            )

        return DocumentIngestAccepted(
            document_id=document_id,
            filename=classification.filename,
            file_size_bytes=len(file_bytes),
            content_sha256=content_sha256,
            status="accepted",
            created_at=now,
        )

    # ── worker / standalone ──────────────────────────────────────
    dry_run = _is_dry_run(request)
    classification = FileClassifier.classify(file, filename_override=meta.filename or "")

    file_bytes = await file.read()
    content_sha256 = hashlib.sha256(file_bytes).hexdigest()
    now = datetime.now(timezone.utc).isoformat()

    gw_doc_id = request.headers.get(_GATEWAY_DOC_ID_HEADER)
    gw_callback_url = request.headers.get(_GATEWAY_CALLBACK_HEADER)
    gw_job_id = request.headers.get(_GATEWAY_JOB_ID_HEADER) or job_id
    document_id = gw_doc_id or uuid.uuid4().hex

    worker_spec = _spec_from_gateway_header(request) if gw_doc_id else validated_spec

    if not dry_run:
        if not gw_callback_url:
            _register_document_under_job(document_id=document_id, job_id=job_id, filename=file.filename)
        await _enqueue_or_reject(
            PoolType.BATCH,
            WorkItem(
                id=document_id,
                payload=file_bytes,
                filename=file.filename,
                callback_url=gw_callback_url,
                job_id=gw_job_id,
                pipeline_spec=worker_spec.model_dump(mode="json") if worker_spec is not None else None,
            ),
        )

    _record_prometheus(request, "/v1/ingest/job/whole", "2xx", file_size=len(file_bytes))

    if (m := get_metrics()) is not None:
        m.record_request("/v1/ingest/job/whole")
        m.record_document_accepted(
            document_id=document_id,
            job_id=gw_job_id,
            filename=classification.filename,
            file_category=classification.category.value,
            content_type=classification.content_type,
            file_size_bytes=len(file_bytes),
            endpoint="/v1/ingest/job/whole",
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
    """Look up document status and return the appropriate HTTP code.

    Returns 200 for completed/failed, 202 for pending/processing, 404 if unknown.
    When returning a terminal (200) response, result_data is consumed from the
    tracker so memory is freed after the client has retrieved it.
    """
    from nemo_retriever.service.services.job_tracker import DocumentStatus

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(
            status_code=503,
            detail="Job tracker is not available on this pod.",
        )
    rec = tracker.get_document(item_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"No tracked document with id={item_id!r}")

    is_terminal = rec.status in (DocumentStatus.COMPLETED, DocumentStatus.FAILED)
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


@router.post(
    "/ingest/sidecar",
    response_model=SidecarUploadResponse,
    status_code=201,
    summary="Upload a sidecar metadata file for use with vdb_upload",
)
async def ingest_sidecar(
    request: Request,
    file: UploadFile = File(..., description="Sidecar metadata payload (csv / json / parquet)."),
    ttl_s: float = Form(
        default=3600.0,
        description="Time-to-live in seconds; the sidecar auto-evicts after this window.",
    ),
    consume_on_read: bool = Form(
        default=True,
        description=(
            "When true (default) the worker removes the sidecar after its first read. "
            "Set to false to reuse the same metadata across multiple ingest batches."
        ),
    ),
) -> SidecarUploadResponse | Response:
    """Stash sidecar metadata in the service's in-memory store.

    Returns an opaque ``sidecar_id`` the caller passes through
    ``vdb_upload_params.meta_dataframe_id`` on subsequent ingest
    requests. Sidecars are scoped to the bearer token (when auth is
    enabled) and auto-evicted after ``ttl_s`` seconds.
    """
    from datetime import datetime, timezone
    from nemo_retriever.service.services.sidecar_store import get_sidecar_store

    _check_upload_size(file, request)

    # Forward to the gateway's backend so the realtime worker pool has
    # the sidecar available when the matching ingest call arrives. We
    # broadcast to both pools because the routing decision happens at
    # ingest time, not at sidecar-upload time.
    if _is_gateway(request):
        from nemo_retriever.service.services.proxy import get_proxy

        proxy = get_proxy()
        if proxy is None:
            raise HTTPException(status_code=503, detail="Gateway proxy not initialised")
        # Pick the realtime backend for the canonical response, then
        # mirror the upload to the batch backend so either worker pool
        # can resolve the id. If the mirror fails we still return 201
        # because the realtime store has the entry and most workloads
        # land there.
        realtime_resp = await proxy.forward(request, PoolType.REALTIME)
        try:
            await proxy.forward(request, PoolType.BATCH)
        except Exception as exc:
            logger.warning(
                "Sidecar mirror to batch backend failed (id from realtime still valid): %s",
                exc,
            )
        return realtime_resp

    store = get_sidecar_store()
    if store is None:
        raise HTTPException(status_code=503, detail="Sidecar store not initialised")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Sidecar upload is empty")

    # Owner-token scoping: use the bearer token when auth is enabled.
    auth_header = request.headers.get("Authorization", "")
    owner_token = auth_header.split(" ", 1)[1].strip() if auth_header.lower().startswith("bearer ") else None

    try:
        entry = store.put(
            filename=file.filename or "sidecar",
            content_type=file.content_type or "application/octet-stream",
            payload=payload,
            owner_token=owner_token,
            ttl_s=ttl_s,
            consume_on_read=consume_on_read,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    expires_iso = datetime.fromtimestamp(entry.expires_at, tz=timezone.utc).isoformat()
    return SidecarUploadResponse(
        sidecar_id=entry.sidecar_id,
        filename=entry.filename,
        content_type=entry.content_type,
        size_bytes=len(entry.payload),
        expires_at=expires_iso,
    )


@router.delete(
    "/ingest/sidecar/{sidecar_id}",
    status_code=204,
    summary="Delete a previously uploaded sidecar",
)
async def delete_sidecar(request: Request, sidecar_id: str) -> Response:
    """Explicit deletion lets callers free server memory before the TTL elapses."""
    from nemo_retriever.service.services.sidecar_store import get_sidecar_store

    if _is_gateway(request):
        from nemo_retriever.service.services.proxy import get_proxy

        proxy = get_proxy()
        if proxy is None:
            raise HTTPException(status_code=503, detail="Gateway proxy not initialised")
        # Mirror delete to both pools. We don't care which one had it.
        for pool in (PoolType.REALTIME, PoolType.BATCH):
            try:
                await proxy.forward(request, pool)
            except Exception as exc:
                logger.debug("Sidecar delete forward to %s failed: %s", pool.value, exc)
        return Response(status_code=204)

    store = get_sidecar_store()
    if store is None:
        raise HTTPException(status_code=503, detail="Sidecar store not initialised")
    store.delete(sidecar_id)
    return Response(status_code=204)


@router.get(
    "/ingest/status/{item_id}",
    summary="Check processing status of a general ingest submission",
    responses={200: {"model": JobStatusResponse}, 202: {"model": JobStatusResponse}},
)
async def ingest_status(request: Request, item_id: str) -> JSONResponse:
    return _status_response(request, item_id)


@router.get(
    "/ingest/page/status/{page_id}",
    summary="Check processing status of a page ingest submission",
    responses={200: {"model": JobStatusResponse}, 202: {"model": JobStatusResponse}},
)
async def ingest_page_status(request: Request, page_id: str) -> JSONResponse:
    return _status_response(request, page_id)


@router.get(
    "/ingest/document/status/{document_id}",
    summary="Check processing status of a document ingest submission",
    responses={200: {"model": JobStatusResponse}, 202: {"model": JobStatusResponse}},
)
async def ingest_document_status(request: Request, document_id: str) -> JSONResponse:
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

    policy = _build_policy(request)
    return JSONResponse(
        content={
            "source": mode,
            "mode": mode,
            "pipelines": get_pipeline_configs(),
            "pool_stats": pool_stats,
            "allowed_overrides": policy.describe(),
        }
    )


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


# ------------------------------------------------------------------
# POST /v1/internal/job-callback  — worker → gateway completion hook
# ------------------------------------------------------------------


@router.post(
    "/internal/job-callback",
    summary="Internal callback from worker pods to report job completion",
    include_in_schema=False,
)
async def job_callback(request: Request) -> JSONResponse:
    """Accept a completion notification from a worker pod.

    The gateway's ``JobTracker`` is updated and an SSE event is published
    so connected clients are notified instantly.
    """
    body = await request.json()
    item_id = body.get("id")
    if not item_id:
        raise HTTPException(status_code=400, detail="Missing 'id' field")

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker not available")

    status = body.get("status", "completed")
    if status == "failed":
        tracker.mark_failed(
            item_id,
            body.get("error", "unknown error"),
            elapsed_s=body.get("elapsed_s"),
        )
    else:
        tracker.mark_completed(
            item_id,
            result_rows=body.get("result_rows", 0),
            result_data=body.get("result_data"),
            elapsed_s=body.get("elapsed_s"),
        )

    bus = get_event_bus()
    sub_count = bus.subscriber_count if bus else 0
    logger.info(
        "Gateway callback: id=%s status=%s rows=%s subscribers=%d",
        item_id,
        status,
        body.get("result_rows", 0),
        sub_count,
    )
    return JSONResponse(content={"ok": True})


# ------------------------------------------------------------------
# GET /v1/ingest/job/{job_id}/events
#   SSE stream filtered to a single job aggregate.
#
# The legacy firehose ``GET /v1/ingest/events`` was removed in J4 to
# preserve job isolation between tenants — every public SSE consumer
# must declare which job it is observing. Dashboard internals (which
# are served from a separate router) still use a firehose subscription
# for the operator overview view, but that endpoint is privileged and
# lives under ``/dashboard``.
# ------------------------------------------------------------------


@router.get(
    "/ingest/job/{job_id}/events",
    summary="SSE stream of completion events for one job",
)
async def ingest_job_events(request: Request, job_id: str) -> StreamingResponse:
    """Stream Server-Sent Events for a single :class:`JobAggregate`.

    Emits:

    * a catch-up batch of already-terminal ``completed`` / ``failed``
      document events (so clients that connect after some documents
      already finished still see them),
    * live per-document ``processing`` / ``completed`` / ``failed``
      events,
    * lifecycle events (``job_progress``, ``job_finalized``,
      ``job_failed``) — these arrive once J5 wires them through the
      tracker; the endpoint already routes them.

    A keepalive comment is emitted every 30 seconds.
    """
    bus = get_event_bus()
    if bus is None:
        raise HTTPException(status_code=503, detail="Event bus not available")

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker not available")
    if tracker.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    async def event_generator():
        sub_id, queue = bus.subscribe(job_id=job_id)
        snapshot_count = 0
        live_count = 0
        logger.info("SSE subscriber %d connected (job_id=%s)", sub_id, job_id)
        try:
            snapshot = _snapshot_terminal_jobs(tracker, job_id=job_id)
            snapshot_count = len(snapshot)
            for rec in snapshot:
                yield f"event: {rec['status']}\ndata: {json.dumps(rec)}\n\n"
            logger.info(
                "SSE subscriber %d (job_id=%s): sent %d snapshot events",
                sub_id,
                job_id,
                snapshot_count,
            )

            while True:
                if await request.is_disconnected():
                    logger.info(
                        "SSE subscriber %d (job_id=%s): client disconnected",
                        sub_id,
                        job_id,
                    )
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_TIMEOUT_S)
                    live_count += 1
                    yield f"event: {event.get('type', 'status')}\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            bus.unsubscribe(sub_id)
            logger.info(
                "SSE subscriber %d (job_id=%s) closed (snapshot=%d live=%d)",
                sub_id,
                job_id,
                snapshot_count,
                live_count,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _snapshot_terminal_jobs(tracker: Any, *, job_id: str | None = None) -> list[dict[str, Any]]:
    """Return already-terminal documents so SSE clients get caught up.

    If *job_id* is supplied, only documents belonging to that job are
    returned; otherwise every terminal document across every job is
    yielded. The latter mode is only used by privileged dashboard
    endpoints — public per-job SSE always passes a *job_id*.
    """
    from nemo_retriever.service.services.job_tracker import DocumentStatus

    terminal = {DocumentStatus.COMPLETED, DocumentStatus.FAILED}
    if job_id is not None:
        recs = tracker.job_documents(job_id)
    else:
        recs = tracker.all_documents()
    return [
        {
            "type": rec.status.value,
            "id": rec.id,
            "job_id": rec.job_id,
            "status": rec.status.value,
            "result_rows": rec.result_rows,
            "elapsed_s": rec.elapsed_s,
            "error": rec.error,
        }
        for rec in recs
        if rec.status in terminal
    ]


# ------------------------------------------------------------------
# POST /v1/ingest/status/batch  — bulk status query
# ------------------------------------------------------------------


@router.post(
    "/ingest/status/batch",
    summary="Bulk status check for multiple items",
)
async def ingest_status_batch(request: Request) -> JSONResponse:
    """Return the status of many items in a single request.

    Accepts ``{"ids": ["id1", "id2", ...]}`` and returns a dict keyed by
    item id.  Works on the gateway's local tracker — no backend proxying.
    """
    _MAX_BATCH_IDS = 1000

    body = await request.json()
    ids = body.get("ids", [])
    if not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="'ids' must be a list")
    if len(ids) > _MAX_BATCH_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many IDs ({len(ids)}); maximum is {_MAX_BATCH_IDS}",
        )

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(status_code=503, detail="Job tracker not available")

    from nemo_retriever.service.services.job_tracker import DocumentStatus

    results: dict[str, dict[str, Any]] = {}
    for item_id in ids:
        rec = tracker.get_document(item_id)
        if rec is None:
            results[item_id] = {"status": "unknown"}
        else:
            results[item_id] = {
                "status": rec.status.value,
                "job_id": rec.job_id,
                "result_rows": rec.result_rows,
                "elapsed_s": rec.elapsed_s,
                "error": rec.error,
            }

    terminal_count = sum(
        1 for r in results.values() if r["status"] in (DocumentStatus.COMPLETED.value, DocumentStatus.FAILED.value)
    )
    return JSONResponse(
        content={
            "total": len(ids),
            "terminal": terminal_count,
            "pending": len(ids) - terminal_count,
            "items": results,
        }
    )
