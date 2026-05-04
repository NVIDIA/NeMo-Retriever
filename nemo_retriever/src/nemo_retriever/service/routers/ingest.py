# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest endpoints: POST/GET/PATCH/DELETE for jobs and documents."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_logger import record_event
from nemo_retriever.service.failure_types import EventCategory
from nemo_retriever.service.models.document import Document, ProcessingStatus
from nemo_retriever.service.models.event_log import EventOutcome, EventSeverity
from nemo_retriever.service.models.job import Job
from nemo_retriever.service.models.requests import IngestRequest
from nemo_retriever.service.models.responses import (
    BatchIngestAccepted,
    IngestAccepted,
    IngestComplete,
    IngestStatus,
    JobAccepted,
    JobCancelResponse,
    JobDeleteResponse,
    JobInputPage,
    JobListEntry,
    JobResults,
    JobsList,
    JobsPurgeResponse,
    JobsSummary,
    JobStatus,
    JobUpdateResponse,
    MetricSummary,
    PageSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])


@router.post(
    "/ingest",
    response_model=IngestAccepted,
    status_code=202,
    summary="Upload a single document (or page) for ingestion",
    responses={
        503: {
            "description": "Server is at capacity. Retry after a worker slot frees up.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Server busy — all worker slots are in use.",
                        "retry_after": 5,
                        "capacity": 0,
                        "pool_size": 32,
                    }
                }
            },
        }
    },
)
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="The document to ingest"),
    metadata: str = Form(default="{}", description="JSON-encoded IngestRequest metadata"),
) -> IngestAccepted | JSONResponse:
    pool = request.app.state.processing_pool
    req_id = getattr(request.state, "request_id", "")

    if pool.is_draining:
        repo_ref: Repository = request.app.state.repository
        record_event(
            repo_ref,
            category=EventCategory.INTERNAL.value,
            severity=EventSeverity.WARNING,
            outcome=EventOutcome.RECOVERED,
            summary="Ingest rejected: server is draining for shutdown",
            endpoint="/v1/ingest",
            request_id=req_id,
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "Server is draining for shutdown; not accepting new uploads."},
            headers={"Retry-After": "0"},
        )

    if not pool.has_capacity():
        logger.debug(
            "Ingest rejected — pool at capacity (%d/%d)",
            pool.pool_size,
            pool.pool_size,
        )
        repo_ref = request.app.state.repository
        record_event(
            repo_ref,
            category=EventCategory.INTERNAL.value,
            severity=EventSeverity.WARNING,
            outcome=EventOutcome.RECOVERED,
            summary="Ingest rejected: server busy (all worker slots in use)",
            endpoint="/v1/ingest",
            request_id=req_id,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Server busy — all worker slots are in use.",
                "retry_after": 10,
                "capacity": 0,
                "pool_size": pool.pool_size,
            },
            headers={"Retry-After": "10"},
        )

    try:
        meta = IngestRequest(**json.loads(metadata))
    except (json.JSONDecodeError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")

    file_bytes = await file.read()
    content_sha256 = hashlib.sha256(file_bytes).hexdigest()

    repo: Repository = request.app.state.repository

    existing = repo.get_document_by_sha(content_sha256)
    if existing is not None:
        logger.info("Duplicate document detected (sha=%s), returning existing record", content_sha256[:12])
        record_event(
            repo,
            category=EventCategory.DEDUP.value,
            severity=EventSeverity.INFO,
            outcome=EventOutcome.RECOVERED,
            summary=f"Duplicate document skipped (sha={content_sha256[:12]})",
            detail=f"SHA-256 collision with existing document {existing.id}",
            endpoint="/v1/ingest",
            document_id=existing.id,
            source_file=existing.filename,
            request_id=req_id,
        )

        job_id = meta.job_id
        if job_id is not None:
            existing_job = repo.get_job(job_id)
            if existing_job is None:
                filename = meta.filename or file.filename or "unknown"
                job = Job(
                    id=job_id,
                    filename=filename,
                    content_sha256="",
                    total_pages=meta.total_pages or 0,
                )
                repo.insert_job(job)
                repo.update_job_status(job_id, ProcessingStatus.PROCESSING)

            repo.increment_job_pages_submitted(job_id)
            pages_completed = repo.increment_job_pages_completed(job_id)
            logger.info(
                "[job %s] Page %s/%s deduped (sha=%s), counting as complete",
                job_id[:8],
                meta.page_number,
                meta.total_pages,
                content_sha256[:8],
            )

            event_bus = request.app.state.event_bus
            loop = request.app.state.processing_pool._event_loop

            job = repo.get_job(job_id)
            if job and pages_completed >= job.total_pages and job.total_pages > 0:
                repo.update_job_status(job_id, ProcessingStatus.COMPLETE)
                logger.info("[job %s] All %d pages of %s complete", job_id[:8], job.total_pages, job.filename)
                asyncio.run_coroutine_threadsafe(
                    event_bus.publish(
                        job_id,
                        {
                            "event": "job_complete",
                            "job_id": job_id,
                            "filename": job.filename,
                            "total_pages": job.total_pages,
                        },
                    ),
                    loop,
                )
            else:
                asyncio.run_coroutine_threadsafe(
                    event_bus.publish(
                        job_id,
                        {
                            "event": "page_complete",
                            "document_id": existing.id,
                            "job_id": job_id,
                            "pages_received": 0,
                            "total_pages": meta.total_pages or 0,
                        },
                    ),
                    loop,
                )

        return IngestAccepted(
            document_id=existing.id,
            job_id=job_id or existing.job_id,
            content_sha256=existing.content_sha256,
            status=existing.processing_status,
            created_at=existing.created_at,
        )

    filename = meta.filename or file.filename or "unknown"
    content_type = meta.content_type or file.content_type or "application/octet-stream"

    # Handle job creation / update when the client pre-splits pages
    job_id = meta.job_id
    if job_id is not None:
        existing_job = repo.get_job(job_id)
        if existing_job is None:
            job = Job(
                id=job_id,
                filename=filename,
                content_sha256="",
                total_pages=meta.total_pages or 0,
            )
            repo.insert_job(job)
            repo.update_job_status(job_id, ProcessingStatus.PROCESSING)

        repo.increment_job_pages_submitted(job_id)
        logger.info("[job %s] Received page %s/%s of %s", job_id[:8], meta.page_number, meta.total_pages, filename)

    new_doc = Document(
        job_id=job_id,
        filename=filename,
        content_type=content_type,
        content_sha256=content_sha256,
        file_size_bytes=len(file_bytes),
        page_number=meta.page_number,
        metadata_json=json.dumps(meta.metadata),
    )
    doc, lost_race = repo.insert_document_or_get(new_doc)
    if lost_race:
        # Another concurrent request inserted the same SHA between our
        # get_document_by_sha() check above and the INSERT here.  Treat
        # this submit as a dedup hit so we don't double-process the page.
        logger.info(
            "Document SHA-race detected: returning existing document %s for sha=%s",
            doc.id,
            content_sha256[:12],
        )
        return IngestAccepted(
            document_id=doc.id,
            job_id=job_id or doc.job_id,
            content_sha256=doc.content_sha256,
            status=doc.processing_status,
            created_at=doc.created_at,
        )
    logger.info("Document accepted: id=%s filename=%s sha=%s", doc.id, filename, content_sha256[:12])

    spool_path: str | None = None
    spool_store = getattr(pool, "_spool_store", None)
    if spool_store is not None:
        try:
            spool_path = spool_store.write(content_sha256, file_bytes)
            repo.update_document_spool_path(doc.id, spool_path)
        except OSError as exc:
            logger.exception("Spool write failed for doc %s", doc.id)
            record_event(
                repo,
                category=EventCategory.SPOOL.value,
                severity=EventSeverity.ERROR,
                outcome=EventOutcome.FAILED,
                summary=f"Spool write failed: {exc.__class__.__name__}: {exc}",
                detail=str(exc),
                stage="spool_write",
                endpoint="/v1/ingest",
                document_id=doc.id,
                source_file=filename,
                request_id=req_id,
            )
            return JSONResponse(
                status_code=503,
                content={
                    "detail": f"Failed to spool page bytes: {exc.__class__.__name__}: {exc}",
                    "document_id": doc.id,
                    "retry_after": 5,
                },
                headers={"Retry-After": "5"},
            )

    accepted = pool.try_submit(
        doc.id,
        content_sha256,
        file_bytes,
        filename,
        job_id=job_id,
        page_number=meta.page_number or 1,
        spool_path=spool_path,
    )
    if not accepted:
        repo.update_document_status(doc.id, ProcessingStatus.QUEUED)
        logger.warning("Document %s queued but buffer became full between check and submit", doc.id)
        record_event(
            repo,
            category=EventCategory.INTERNAL.value,
            severity=EventSeverity.WARNING,
            outcome=EventOutcome.FAILED,
            summary="Batch buffer full — page not submitted",
            detail="Buffer became full between capacity check and submit",
            stage="submit",
            endpoint="/v1/ingest",
            document_id=doc.id,
            source_file=filename,
            request_id=req_id,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Server busy — batch buffer is full.",
                "document_id": doc.id,
                "retry_after": 10,
                "capacity": 0,
                "pool_size": pool.pool_size,
            },
            headers={"Retry-After": "10"},
        )

    return IngestAccepted(
        document_id=doc.id,
        job_id=job_id,
        content_sha256=doc.content_sha256,
        status=doc.processing_status,
        created_at=doc.created_at,
    )


async def _build_status_response(repo: Repository, doc: Document) -> IngestStatus | IngestComplete:
    """Build the appropriate status or complete response for *doc*.

    Reads run in a worker thread so the asyncio loop stays responsive
    under load.
    """
    pages = await asyncio.to_thread(repo.get_page_results, doc.id)
    metrics_rows = await asyncio.to_thread(repo.get_metrics, doc.id)
    failure_log = (
        await asyncio.to_thread(repo.get_processing_log_for_document, doc.id)
        if doc.processing_status == "failed"
        else None
    )

    page_summaries = [PageSummary(page_number=p.page_number, content=json.loads(p.content_json)) for p in pages]
    metric_summaries = [
        MetricSummary(
            model_name=m.model_name,
            invocation_count=m.invocation_count,
            pages_processed=m.pages_processed,
            detections_count=m.detections_count,
            duration_ms=m.duration_ms,
        )
        for m in metrics_rows
    ]

    if doc.processing_status == ProcessingStatus.COMPLETE and doc.total_pages is not None:
        return IngestComplete(
            document_id=doc.id,
            filename=doc.filename,
            content_sha256=doc.content_sha256,
            status=doc.processing_status,
            total_pages=doc.total_pages,
            pages_received=doc.pages_received,
            metrics=metric_summaries,
            pages=page_summaries,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )

    return IngestStatus(
        document_id=doc.id,
        filename=doc.filename,
        content_sha256=doc.content_sha256,
        status=doc.processing_status,
        total_pages=doc.total_pages,
        pages_received=doc.pages_received,
        metrics=metric_summaries,
        pages=page_summaries,
        failure_type=failure_log.failure_type if failure_log else None,
        error_message=failure_log.error_message if failure_log else None,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get(
    "/ingest/status/{document_id}",
    response_model=IngestStatus | IngestComplete,
    summary="Get the processing status of a document",
)
async def get_ingest_status(
    request: Request,
    document_id: str,
) -> IngestStatus | IngestComplete:
    repo: Repository = request.app.state.repository
    doc = await asyncio.to_thread(repo.get_document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return await _build_status_response(repo, doc)


@router.get(
    "/ingest/jobs",
    response_model=JobsList,
    summary="List jobs with optional status / since-time filters and paging",
)
async def list_jobs(
    request: Request,
    status: str | None = Query(
        None,
        description="Filter by processing_status: queued | processing | complete | failed | cancelled",
    ),
    since: str | None = Query(
        None,
        description="ISO-8601 timestamp; only jobs whose updated_at >= since are returned.",
    ),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> JobsList:
    repo: Repository = request.app.state.repository
    rows = await asyncio.to_thread(
        repo.list_jobs,
        status=status,
        since=since,
        limit=limit,
        offset=offset,
    )
    entries = [
        JobListEntry(
            job_id=j.id,
            filename=j.filename,
            status=j.processing_status,
            total_pages=j.total_pages,
            pages_submitted=j.pages_submitted,
            pages_completed=j.pages_completed,
            created_at=j.created_at,
            updated_at=j.updated_at,
        )
        for j in rows
    ]
    return JobsList(jobs=entries, limit=limit, offset=offset, returned=len(entries))


@router.get(
    "/ingest/jobs/summary",
    response_model=JobsSummary,
    summary="Aggregate job counts grouped by processing status",
)
async def jobs_summary(request: Request) -> JobsSummary:
    repo: Repository = request.app.state.repository
    counts = await asyncio.to_thread(repo.count_jobs_by_status)
    payload = JobsSummary(
        queued=counts.get("queued", 0),
        processing=counts.get("processing", 0),
        complete=counts.get("complete", 0),
        failed=counts.get("failed", 0),
        cancelled=counts.get("cancelled", 0),
        draining=counts.get("draining", 0),
    )
    payload.total = sum(
        (
            payload.queued,
            payload.processing,
            payload.complete,
            payload.failed,
            payload.cancelled,
            payload.draining,
        )
    )
    return payload


def _aggregate_job_metrics(repo: Repository, job_id: str) -> list[MetricSummary]:
    """Aggregate per-document metrics rows up to a single per-model summary."""
    agg: dict[str, MetricSummary] = {}
    for m in repo.get_metrics_for_job(job_id):
        key = m.model_name
        if key not in agg:
            agg[key] = MetricSummary(
                model_name=m.model_name,
                invocation_count=0,
                pages_processed=0,
                detections_count=0,
                duration_ms=0.0,
            )
        agg[key].invocation_count += m.invocation_count
        agg[key].pages_processed += m.pages_processed
        agg[key].detections_count += m.detections_count
        agg[key].duration_ms += m.duration_ms
    return list(agg.values())


@router.get(
    "/ingest/job/{job_id}",
    response_model=JobStatus,
    summary="Get the aggregated processing status of a job",
)
async def get_job_status(
    request: Request,
    job_id: str,
) -> JobStatus:
    repo: Repository = request.app.state.repository
    job = await asyncio.to_thread(repo.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    documents = await asyncio.to_thread(repo.get_documents_for_job, job_id)
    metrics = await asyncio.to_thread(_aggregate_job_metrics, repo, job_id)

    return JobStatus(
        job_id=job.id,
        filename=job.filename,
        status=job.processing_status,
        total_pages=job.total_pages,
        pages_submitted=job.pages_submitted,
        pages_completed=job.pages_completed,
        metrics=metrics,
        document_ids=[d.id for d in documents],
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get(
    "/ingest/job/{job_id}/page/{page_number}",
    response_model=PageSummary,
    summary="Get the extracted content for a single input page of a job",
    responses={
        404: {"description": "Job, input page, or page result not found."},
        409: {"description": "Page is still processing (no results yet)."},
    },
)
async def get_job_page(
    request: Request,
    job_id: str,
    page_number: int,
) -> PageSummary:
    """Return the first output row for one input page of a job.

    The input ``page_number`` is the 1-based page as uploaded (matches the
    original PDF page).  If the pipeline expanded that page into multiple
    output rows, this endpoint returns the first one; use
    ``/v1/ingest/job/{job_id}/results`` to fetch all output rows for every
    input page in one call.
    """
    repo: Repository = request.app.state.repository
    job = await asyncio.to_thread(repo.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    doc = await asyncio.to_thread(repo.get_document_for_job_page, job_id, page_number)
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Page {page_number} not found for job {job_id}",
        )

    pages = await asyncio.to_thread(repo.get_page_results, doc.id)
    if not pages:
        raise HTTPException(
            status_code=409,
            detail=f"Page {page_number} of job {job_id} is still processing (status={doc.processing_status})",
        )

    first = pages[0]
    return PageSummary(page_number=first.page_number, content=json.loads(first.content_json))


@router.get(
    "/ingest/job/{job_id}/results",
    response_model=JobResults,
    summary="Get the reassembled per-page extracted content for an entire job",
)
async def get_job_results(
    request: Request,
    job_id: str,
) -> JobResults:
    """Return the per-input-page results for an entire job in input-page order.

    Each entry in ``pages`` corresponds to one input page (one POST to
    ``/v1/ingest`` with that ``page_number``) and contains the full list of
    output rows the pipeline produced for that input page.

    Pages whose pipeline run has not yet produced any output rows appear in
    the response with an empty ``pages`` list and the document's current
    ``status``, so callers can distinguish "in flight" from "produced no
    output".
    """
    repo: Repository = request.app.state.repository

    def _build() -> JobResults | None:
        job = repo.get_job(job_id)
        if job is None:
            return None
        documents = repo.get_documents_for_job(job_id)
        page_entries: list[JobInputPage] = []
        for doc in documents:
            page_results = repo.get_page_results(doc.id)
            page_entries.append(
                JobInputPage(
                    page_number=doc.page_number or 0,
                    document_id=doc.id,
                    status=doc.processing_status,
                    pages=[
                        PageSummary(page_number=p.page_number, content=json.loads(p.content_json)) for p in page_results
                    ],
                )
            )
        return JobResults(
            job_id=job.id,
            filename=job.filename,
            status=job.processing_status,
            total_pages=job.total_pages,
            pages_completed=job.pages_completed,
            metrics=_aggregate_job_metrics(repo, job_id),
            pages=page_entries,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    result = await asyncio.to_thread(_build)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return result


def _split_pdf_bytes(file_bytes: bytes) -> list[bytes]:
    """Split a PDF into single-page byte buffers using pypdfium2.

    Imported lazily so non-PDF code paths don't pay the import cost.
    """
    from nemo_retriever.pdf.split import _split_pdf_to_single_page_bytes

    return _split_pdf_to_single_page_bytes(file_bytes)


def _enqueue_one_page(
    *,
    repo: Repository,
    pool,
    job_id: str | None,
    filename: str,
    content_type: str,
    page_bytes: bytes,
    page_number: int,
    extra_metadata: dict | None = None,
) -> tuple[bool, IngestAccepted | dict]:
    """Insert one page document and submit it to the pool.

    Returns ``(accepted, payload)`` where ``payload`` is either an
    :class:`IngestAccepted` (when ``accepted`` is True) or a small
    diagnostic dict describing why the page was rejected.

    Handles SHA-based dedup the same way ``ingest_document`` does, but
    without the full HTTP-level branching (this is a helper for bulk
    paths that need consistent dedup behaviour).
    """
    content_sha256 = hashlib.sha256(page_bytes).hexdigest()
    existing = repo.get_document_by_sha(content_sha256)
    if existing is not None:
        if job_id is not None and existing.job_id != job_id:
            repo.increment_job_pages_submitted(job_id)
            repo.increment_job_pages_completed(job_id)
        return True, IngestAccepted(
            document_id=existing.id,
            job_id=job_id or existing.job_id,
            content_sha256=existing.content_sha256,
            status=existing.processing_status,
            created_at=existing.created_at,
        )

    new_doc = Document(
        job_id=job_id,
        filename=filename,
        content_type=content_type,
        content_sha256=content_sha256,
        file_size_bytes=len(page_bytes),
        page_number=page_number,
        metadata_json=json.dumps(extra_metadata or {}),
    )
    doc, lost_race = repo.insert_document_or_get(new_doc)

    if lost_race:
        if job_id is not None and doc.job_id != job_id:
            repo.increment_job_pages_submitted(job_id)
            repo.increment_job_pages_completed(job_id)
        return True, IngestAccepted(
            document_id=doc.id,
            job_id=job_id or doc.job_id,
            content_sha256=doc.content_sha256,
            status=doc.processing_status,
            created_at=doc.created_at,
        )

    if job_id is not None:
        repo.increment_job_pages_submitted(job_id)

    spool_path: str | None = None
    spool_store = getattr(pool, "_spool_store", None)
    if spool_store is not None:
        try:
            spool_path = spool_store.write(content_sha256, page_bytes)
            repo.update_document_spool_path(doc.id, spool_path)
        except OSError as exc:
            return False, {
                "document_id": doc.id,
                "filename": filename,
                "page_number": page_number,
                "reason": f"spool_write_failed: {exc.__class__.__name__}: {exc}",
            }

    accepted = pool.try_submit(
        doc.id,
        content_sha256,
        page_bytes,
        filename,
        job_id=job_id,
        page_number=page_number,
        spool_path=spool_path,
    )
    if not accepted:
        repo.update_document_status(doc.id, ProcessingStatus.QUEUED)
        return False, {
            "document_id": doc.id,
            "filename": filename,
            "page_number": page_number,
            "reason": "buffer_full_or_draining",
        }

    return True, IngestAccepted(
        document_id=doc.id,
        job_id=job_id,
        content_sha256=doc.content_sha256,
        status=doc.processing_status,
        created_at=doc.created_at,
    )


@router.post(
    "/ingest/job",
    response_model=JobAccepted,
    status_code=202,
    summary="Upload a whole document; the server splits PDFs into pages internally",
    responses={
        503: {"description": "Server is at capacity or draining."},
    },
)
async def ingest_whole_job(
    request: Request,
    file: UploadFile = File(..., description="The whole document to ingest (PDF or other)."),
    metadata: str = Form(default="{}", description="JSON-encoded IngestRequest metadata"),
) -> JobAccepted | JSONResponse:
    """Accept one whole document, split it server-side, and enqueue every page.

    For PDFs, splitting is performed with the same ``pypdfium2`` helper used
    by the CLI client, so callers no longer need to ship that dependency.
    For non-PDF inputs the file is treated as a single page.
    """
    pool = request.app.state.processing_pool
    repo: Repository = request.app.state.repository
    req_id = getattr(request.state, "request_id", "")

    if pool.is_draining:
        record_event(
            repo,
            category=EventCategory.INTERNAL.value,
            severity=EventSeverity.WARNING,
            outcome=EventOutcome.RECOVERED,
            summary="Ingest job rejected: server draining",
            stage="accept",
            endpoint="/v1/ingest/job",
            request_id=req_id,
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "Server is draining for shutdown; not accepting new uploads."},
            headers={"Retry-After": "0"},
        )

    if not pool.has_capacity():
        record_event(
            repo,
            category=EventCategory.INTERNAL.value,
            severity=EventSeverity.WARNING,
            outcome=EventOutcome.RECOVERED,
            summary="Ingest job rejected: server at capacity",
            stage="accept",
            endpoint="/v1/ingest/job",
            request_id=req_id,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Server busy — all worker slots are in use.",
                "retry_after": 10,
                "capacity": 0,
                "pool_size": pool.pool_size,
            },
            headers={"Retry-After": "10"},
        )

    try:
        meta = IngestRequest(**json.loads(metadata))
    except (json.JSONDecodeError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")

    file_bytes = await file.read()
    filename = meta.filename or file.filename or "unknown"
    content_type = meta.content_type or file.content_type or "application/octet-stream"

    is_pdf = filename.lower().endswith(".pdf") or content_type == "application/pdf"
    if is_pdf:
        try:
            pages = await asyncio.to_thread(_split_pdf_bytes, file_bytes)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Failed to split PDF: {exc}")
    else:
        pages = [file_bytes]

    total_pages = len(pages)
    job_id = meta.job_id or uuid.uuid4().hex

    job = Job(id=job_id, filename=filename, content_sha256="", total_pages=total_pages)
    try:
        repo.insert_job(job)
        repo.update_job_status(job_id, ProcessingStatus.PROCESSING)
    except Exception:  # noqa: BLE001 — job_id may already exist
        pass

    accepted_docs: list[str] = []
    rejections: list[dict] = []
    for idx, page_bytes in enumerate(pages, start=1):
        ok, payload = await asyncio.to_thread(
            _enqueue_one_page,
            repo=repo,
            pool=pool,
            job_id=job_id,
            filename=filename,
            content_type=content_type,
            page_bytes=page_bytes,
            page_number=idx,
            extra_metadata=meta.metadata,
        )
        if ok and isinstance(payload, IngestAccepted):
            accepted_docs.append(payload.document_id)
        elif not ok:
            rejections.append(payload)  # type: ignore[arg-type]

    return JobAccepted(
        job_id=job_id,
        filename=filename,
        total_pages=total_pages,
        document_ids=accepted_docs,
        status="processing",
        pages_accepted=len(accepted_docs),
        pages_rejected=len(rejections),
        pages_rejected_detail=rejections,
    )


@router.post(
    "/ingest/batch",
    response_model=BatchIngestAccepted,
    status_code=202,
    summary="Upload multiple pre-split pages in a single multipart request",
    responses={
        503: {"description": "Server is at capacity or draining."},
    },
)
async def ingest_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="One or more pre-split page files."),
    metadata: str = Form(
        default="{}",
        description=(
            "JSON-encoded {job_id?, filename?, content_type?, metadata?, page_numbers?: list[int]}. "
            "If page_numbers is provided it must align 1:1 with files."
        ),
    ),
) -> BatchIngestAccepted | JSONResponse:
    """Bulk equivalent of N x ``POST /v1/ingest`` calls in one request.

    Lower per-request overhead than uploading each page individually; useful
    when the client has already split a document into pages but wants to
    minimise HTTP round-trips.  All pages share a single ``job_id`` (one is
    generated if not supplied).
    """
    pool = request.app.state.processing_pool
    repo: Repository = request.app.state.repository
    req_id = getattr(request.state, "request_id", "")

    if pool.is_draining:
        record_event(
            repo,
            category=EventCategory.INTERNAL.value,
            severity=EventSeverity.WARNING,
            outcome=EventOutcome.RECOVERED,
            summary="Batch ingest rejected: server draining",
            stage="accept",
            endpoint="/v1/ingest/batch",
            request_id=req_id,
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "Server is draining for shutdown; not accepting new uploads."},
            headers={"Retry-After": "0"},
        )

    if not pool.has_capacity():
        record_event(
            repo,
            category=EventCategory.INTERNAL.value,
            severity=EventSeverity.WARNING,
            outcome=EventOutcome.RECOVERED,
            summary="Batch ingest rejected: server at capacity",
            stage="accept",
            endpoint="/v1/ingest/batch",
            request_id=req_id,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Server busy — all worker slots are in use.",
                "retry_after": 10,
                "capacity": 0,
                "pool_size": pool.pool_size,
            },
            headers={"Retry-After": "10"},
        )

    try:
        meta_dict = json.loads(metadata)
        meta = IngestRequest(**{k: v for k, v in meta_dict.items() if k != "page_numbers"})
        page_numbers = meta_dict.get("page_numbers")
        if page_numbers is not None and len(page_numbers) != len(files):
            raise ValueError("page_numbers length must match files length")
    except (json.JSONDecodeError, Exception) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")

    job_id = meta.job_id or uuid.uuid4().hex
    if not meta.job_id:
        job = Job(id=job_id, filename=meta.filename or "batch", content_sha256="", total_pages=len(files))
        try:
            await asyncio.to_thread(repo.insert_job, job)
            await asyncio.to_thread(repo.update_job_status, job_id, ProcessingStatus.PROCESSING)
        except Exception:  # noqa: BLE001
            pass

    accepted: list[IngestAccepted] = []
    rejected: list[dict] = []
    for idx, upload in enumerate(files):
        page_bytes = await upload.read()
        page_num = page_numbers[idx] if page_numbers else idx + 1
        ok, payload = await asyncio.to_thread(
            _enqueue_one_page,
            repo=repo,
            pool=pool,
            job_id=job_id,
            filename=meta.filename or upload.filename or f"page_{page_num}",
            content_type=meta.content_type or upload.content_type or "application/octet-stream",
            page_bytes=page_bytes,
            page_number=page_num,
            extra_metadata=meta.metadata,
        )
        if ok and isinstance(payload, IngestAccepted):
            accepted.append(payload)
        elif not ok:
            rejected.append(payload)  # type: ignore[arg-type]

    return BatchIngestAccepted(accepted=accepted, rejected=rejected)


@router.post(
    "/ingest/job/{job_id}/cancel",
    response_model=JobCancelResponse,
    summary="Cancel a job and any of its still-pending pages",
    responses={
        404: {"description": "Job not found."},
    },
)
async def cancel_job(
    request: Request,
    job_id: str,
) -> JobCancelResponse:
    """Best-effort cancellation.

    The job and any of its documents whose processing has not yet finished
    are marked ``cancelled``.  Pages already in flight (running inside a
    worker process) cannot be safely interrupted and will run to
    completion, but any subsequent batches that contain pages of the
    cancelled job are skipped before they're dispatched.
    """
    repo: Repository = request.app.state.repository
    pool = request.app.state.processing_pool

    info = await asyncio.to_thread(repo.cancel_job, job_id)
    if not info.get("found"):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    pool.cancel_job(job_id)

    job = await asyncio.to_thread(repo.get_job, job_id)
    return JobCancelResponse(
        job_id=job_id,
        status=job.processing_status if job else "cancelled",
        documents_cancelled=int(info.get("documents_cancelled", 0)),
        already_terminal=bool(info.get("already_terminal", 0)),
    )


@router.delete(
    "/ingest/job/{job_id}",
    response_model=JobDeleteResponse,
    summary="Delete a job and all of its documents, metrics, and results",
    responses={
        404: {"description": "Job not found."},
        409: {"description": "Job is still actively processing."},
    },
)
async def delete_job(
    request: Request,
    job_id: str,
    force: bool = Query(
        False,
        description="Force deletion even if the job is still queued or processing. "
        "Default: only terminal jobs (complete/failed/cancelled) can be deleted.",
    ),
) -> JobDeleteResponse:
    """Remove a job and all associated data from the database.

    By default only terminal jobs (complete, failed, cancelled) can be deleted.
    Pass ``?force=true`` to delete a queued or in-flight job (the pipeline will
    discard any remaining buffered pages silently).
    """
    repo: Repository = request.app.state.repository
    job = await asyncio.to_thread(repo.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if not force and job.processing_status in ("queued", "processing"):
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is still {job.processing_status}. "
            "Cancel it first or pass ?force=true to delete anyway.",
        )

    if job.processing_status in ("queued", "processing"):
        pool = request.app.state.processing_pool
        pool.cancel_job(job_id)

    counts = await asyncio.to_thread(repo.delete_job, job_id)
    return JobDeleteResponse(
        job_id=job_id,
        deleted=bool(counts.get("job_deleted")),
        documents_deleted=counts.get("documents_deleted", 0),
        metrics_deleted=counts.get("metrics_deleted", 0),
    )


@router.delete(
    "/ingest/jobs",
    response_model=JobsPurgeResponse,
    summary="Bulk-delete all terminal jobs (complete / failed / cancelled)",
)
async def purge_jobs(
    request: Request,
    status: list[str] | None = Query(
        None,
        description="Which terminal statuses to purge. "
        "Defaults to complete, failed, cancelled. "
        "Active statuses (queued, processing) are never purged.",
    ),
) -> JobsPurgeResponse:
    """Remove all finished jobs and their associated data to free up space.

    Only terminal statuses can be purged. Attempting to include ``queued``
    or ``processing`` in the status list is silently ignored.
    """
    allowed = {"complete", "failed", "cancelled"}
    if status:
        statuses = [s for s in status if s in allowed]
    else:
        statuses = sorted(allowed)

    if not statuses:
        return JobsPurgeResponse(jobs_deleted=0, documents_deleted=0, metrics_deleted=0, statuses_purged=[])

    repo: Repository = request.app.state.repository
    counts = await asyncio.to_thread(repo.purge_terminal_jobs, statuses)
    return JobsPurgeResponse(
        jobs_deleted=counts.get("jobs_deleted", 0),
        documents_deleted=counts.get("documents_deleted", 0),
        metrics_deleted=counts.get("metrics_deleted", 0),
        statuses_purged=statuses,
    )


@router.patch(
    "/ingest/job/{job_id}",
    response_model=JobUpdateResponse,
    summary="Update a job — currently supports re-queuing failed or cancelled jobs",
    responses={
        404: {"description": "Job not found."},
        409: {"description": "Job is not in a re-queueable state."},
    },
)
async def update_job(
    request: Request,
    job_id: str,
    action: str = Query(
        ...,
        description="Action to perform. Currently only ``requeue`` is supported.",
    ),
) -> JobUpdateResponse:
    """Modify job state.

    ``?action=requeue`` resets a failed or cancelled job to ``queued`` so
    its pages can be reprocessed on the next dispatch cycle. The
    ``pages_completed`` counter is reset to 0 and every document's status
    is set back to ``queued``.
    """
    if action != "requeue":
        raise HTTPException(status_code=400, detail=f"Unknown action '{action}'. Supported: requeue")

    repo: Repository = request.app.state.repository
    job = await asyncio.to_thread(repo.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.processing_status not in ("failed", "cancelled"):
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is '{job.processing_status}' — only failed or cancelled jobs can be re-queued.",
        )

    updated = await asyncio.to_thread(repo.requeue_job, job_id)
    if not updated:
        raise HTTPException(status_code=409, detail=f"Job {job_id} could not be re-queued.")

    refreshed = await asyncio.to_thread(repo.get_job, job_id)
    return JobUpdateResponse(
        job_id=job_id,
        status=refreshed.processing_status if refreshed else "queued",
        updated_at=refreshed.updated_at if refreshed else "",
    )
