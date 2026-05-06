# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MetricSummary(BaseModel):
    model_name: str
    invocation_count: int = 0
    pages_processed: int = 0
    detections_count: int = 0
    duration_ms: float = 0.0


class PageSummary(BaseModel):
    page_number: int
    content: dict[str, Any] = Field(default_factory=dict)


class IngestAccepted(BaseModel):
    document_id: str
    job_id: str | None = None
    content_sha256: str
    status: str
    created_at: str


class IngestStatus(BaseModel):
    document_id: str
    filename: str
    content_sha256: str
    status: str
    total_pages: int | None = None
    pages_received: int = 0
    metrics: list[MetricSummary] = Field(default_factory=list)
    pages: list[PageSummary] = Field(default_factory=list)
    failure_type: str | None = Field(
        default=None,
        description=(
            "When status='failed', a structured FailureType: pdf_parse | nim_timeout | "
            "nim_5xx | nim_4xx | oom | internal | cancelled | unknown."
        ),
    )
    error_message: str | None = None
    created_at: str
    updated_at: str


class IngestComplete(BaseModel):
    document_id: str
    filename: str
    content_sha256: str
    status: str
    total_pages: int
    pages_received: int = 0
    metrics: list[MetricSummary] = Field(default_factory=list)
    pages: list[PageSummary] = Field(default_factory=list)
    created_at: str
    updated_at: str


class JobStatus(BaseModel):
    job_id: str
    filename: str
    status: str
    total_pages: int
    pages_submitted: int = 0
    pages_completed: int = 0
    metrics: list[MetricSummary] = Field(default_factory=list)
    document_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of per-page document IDs created for this job. "
            "Each page upload produces one Document; entries are ordered by "
            "input page_number."
        ),
    )
    created_at: str
    updated_at: str


class JobInputPage(BaseModel):
    """One input page of a job, with the full extracted content."""

    page_number: int = Field(description="Input page number as uploaded (1-based for split PDFs).")
    document_id: str
    status: str
    pages: list[PageSummary] = Field(
        default_factory=list,
        description=(
            "Output rows produced by the pipeline for this input page. "
            "Multiple entries occur when content explosion expands one input "
            "page into N output rows (e.g. detected sub-elements)."
        ),
    )


class JobResults(BaseModel):
    """Reassembled per-page results for an entire job, in input-page order."""

    job_id: str
    filename: str
    status: str
    total_pages: int
    pages_completed: int
    metrics: list[MetricSummary] = Field(default_factory=list)
    pages: list[JobInputPage] = Field(default_factory=list)
    created_at: str
    updated_at: str


class JobListEntry(BaseModel):
    """One job row in the paginated job-listing response.

    A trimmed projection of :class:`JobStatus` (no metrics, no document_ids)
    suitable for browsing thousands of jobs cheaply.
    """

    job_id: str
    filename: str
    status: str
    total_pages: int
    pages_submitted: int = 0
    pages_completed: int = 0
    created_at: str
    updated_at: str


class JobsList(BaseModel):
    """Response for ``GET /v1/ingest/jobs``."""

    jobs: list[JobListEntry] = Field(default_factory=list)
    limit: int
    offset: int
    returned: int = Field(description="Number of jobs in this page (<= limit).")


class JobsSummary(BaseModel):
    """Aggregate job counts by status. Missing statuses default to 0."""

    queued: int = 0
    processing: int = 0
    complete: int = 0
    failed: int = 0
    cancelled: int = 0
    draining: int = 0
    total: int = 0


class JobAccepted(BaseModel):
    """Response for ``POST /v1/ingest/job`` (server-side split)."""

    job_id: str
    filename: str
    total_pages: int
    document_ids: list[str] = Field(default_factory=list)
    status: str = "processing"
    pages_accepted: int = Field(
        description="Number of pages accepted into the worker pool. Equals total_pages on success."
    )
    pages_rejected: int = Field(
        default=0,
        description=(
            "Number of pages that could not be queued (server busy / draining). "
            "Inspect pages_rejected_detail for reasons."
        ),
    )
    pages_rejected_detail: list[dict] = Field(default_factory=list)


class BatchIngestAccepted(BaseModel):
    """Response for ``POST /v1/ingest/batch``."""

    accepted: list[IngestAccepted] = Field(default_factory=list)
    rejected: list[dict] = Field(
        default_factory=list,
        description="Pages that could not be queued; each entry has filename, page_number, and reason.",
    )


class JobCancelResponse(BaseModel):
    """Response for ``POST /v1/ingest/job/{job_id}/cancel``."""

    job_id: str
    status: str = Field(description="The job's status AFTER the cancel call.")
    documents_cancelled: int = Field(
        description="Number of buffered/queued pages marked cancelled. Pages already in flight will still complete."
    )
    already_terminal: bool = Field(
        default=False,
        description="True if the job had already completed/failed/cancelled before this request.",
    )


class JobDeleteResponse(BaseModel):
    """Response for ``DELETE /v1/ingest/job/{job_id}``."""

    job_id: str
    deleted: bool = True
    documents_deleted: int = 0
    metrics_deleted: int = 0


class JobsPurgeResponse(BaseModel):
    """Response for ``DELETE /v1/ingest/jobs``."""

    jobs_deleted: int = 0
    documents_deleted: int = 0
    metrics_deleted: int = 0
    statuses_purged: list[str] = Field(default_factory=list)


class JobUpdateResponse(BaseModel):
    """Response for ``PATCH /v1/ingest/job/{job_id}``."""

    job_id: str
    status: str
    updated_at: str


class QueryHit(BaseModel):
    """A single search result from the vector store."""

    text: str = ""
    source: str = ""
    page_number: int = 0
    pdf_page: str = ""
    pdf_basename: str = ""
    source_id: str = ""
    path: str = ""
    stored_image_uri: str = ""
    content_type: str = ""
    bbox_xyxy_norm: str = ""
    score: float = Field(description="Distance score from the vector search (lower is closer for L2).")
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResultSet(BaseModel):
    """Results for one query string inside a batch response."""

    query: str
    hits: list[QueryHit] = Field(default_factory=list)
    total_hits: int = Field(description="Number of hits returned for this query.")


class QueryResponse(BaseModel):
    """Response for ``POST /v1/query``.

    Always returns one :class:`QueryResultSet` per input query, even when
    the request contained a single string (in that case ``results`` has
    exactly one entry).
    """

    results: list[QueryResultSet] = Field(default_factory=list)
    total_queries: int = Field(description="Number of queries processed.")
    lancedb_uri: str = Field(description="The LanceDB URI that was queried.")
    lancedb_table: str = Field(description="The LanceDB table that was queried.")
    embedding_model: str = Field(description="The embedding model used to vectorise the queries.")


class RerankHit(BaseModel):
    """A single reranked passage with its score and all original fields."""

    rerank_score: float = Field(description="Reranker logit score (higher is more relevant).")
    text: str = ""
    source: str = ""
    page_number: int = 0
    pdf_page: str = ""
    pdf_basename: str = ""
    source_id: str = ""
    path: str = ""
    stored_image_uri: str = ""
    content_type: str = ""
    bbox_xyxy_norm: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankResponse(BaseModel):
    """Response for ``POST /v1/rerank``."""

    query: str
    results: list[RerankHit] = Field(default_factory=list)
    total_results: int = Field(description="Number of reranked results returned.")
    model_name: str = Field(description="The reranker model that was used.")


class HealthResponse(BaseModel):
    """Response for ``GET /v1/health``."""

    status: str = "ok"
    version: str
    uptime_s: float
    workers: int
    in_flight_batches: int
    capacity: int
    pool_size: int
    draining: bool


class CapabilityFlag(BaseModel):
    enabled: bool
    endpoints: list[str] = Field(default_factory=list)


class CapabilitiesResponse(BaseModel):
    """Response for ``GET /v1/capabilities``.

    Lets clients introspect which NIM endpoints the server has wired up
    and which optional features are enabled, so they can validate their
    expectations before submitting work.
    """

    version: str
    page_elements: CapabilityFlag
    ocr: CapabilityFlag
    table_structure: CapabilityFlag
    graphic_elements: CapabilityFlag
    embed: CapabilityFlag
    rerank: CapabilityFlag
    auth_required: bool
    server_pdf_split: bool
    bulk_upload: bool
    resumable_sse: bool
    cancel: bool


# ------------------------------------------------------------------
# Event log (provenance) responses
# ------------------------------------------------------------------


class EventLogEntry(BaseModel):
    """A single provenance event surfaced via ``GET /v1/events``."""

    id: str
    timestamp: str
    job_id: str | None = None
    document_id: str | None = None
    source_file: str = ""
    page_number: int | None = None
    category: str
    severity: str
    outcome: str
    stage: str = ""
    summary: str = ""
    detail: str = ""
    stack_trace: str = ""
    endpoint: str = ""
    request_id: str = ""
    extra_json: str = "{}"
    created_at: str = ""


class EventLogList(BaseModel):
    """Paginated response for ``GET /v1/events``."""

    events: list[EventLogEntry] = Field(default_factory=list)
    limit: int
    offset: int
    returned: int = Field(description="Number of events in this page (<= limit).")


class EventCategorySummary(BaseModel):
    """One row of the per-category aggregation."""

    category: str
    count: int


class EventSummaryResponse(BaseModel):
    """Aggregated event counts for ``GET /v1/events/summary``."""

    categories: list[EventCategorySummary] = Field(default_factory=list)
    total: int = Field(description="Total events matching the filter.")


class EventUpdateRequest(BaseModel):
    """Body for ``PATCH /v1/events/{event_id}``."""

    outcome: str = Field(
        description="New outcome value (failed, recovered, in_progress).",
    )


class EventBulkAcknowledgeRequest(BaseModel):
    """Body for ``POST /v1/events/acknowledge``."""

    event_ids: list[str] = Field(
        ...,
        min_length=1,
        description="IDs of events to mark as recovered/acknowledged.",
    )
    outcome: str = Field(
        default="recovered",
        description="Outcome to set (default: recovered).",
    )


class EventBulkDeleteRequest(BaseModel):
    """Body for ``DELETE /v1/events`` (bulk delete)."""

    event_ids: list[str] | None = Field(
        default=None,
        description="Specific event IDs to delete. If omitted, filter-based deletion is used.",
    )
    category: str | None = Field(default=None, description="Delete all events with this category.")
    severity: str | None = Field(default=None, description="Delete all events with this severity.")
    outcome: str | None = Field(default=None, description="Delete all events with this outcome.")


class EventMutationResponse(BaseModel):
    """Response for event update/delete operations."""

    affected: int = Field(description="Number of events affected.")
    action: str = Field(description="The operation performed (updated, deleted).")
