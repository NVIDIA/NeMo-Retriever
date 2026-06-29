# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import Field

from nemo_retriever.common.schemas.base import RichModel


class IngestAccepted(RichModel):
    """Response for the general ``POST /v1/ingest`` endpoint."""

    document_id: str
    job_id: str | None = None
    content_sha256: str
    status: str
    created_at: str


class PageIngestAccepted(RichModel):
    """Response for ``POST /v1/ingest/page`` (single page from a split document)."""

    page_id: str
    document_id: str
    page_number: int
    content_sha256: str
    status: str
    created_at: str


class DocumentIngestAccepted(RichModel):
    """Response for ``POST /v1/ingest/document`` (whole document upload)."""

    document_id: str
    filename: str
    file_size_bytes: int
    content_sha256: str
    status: str
    created_at: str


class JobStatusResponse(RichModel):
    """Response for ``GET /v1/ingest/*/status/{id}`` status polling endpoints."""

    id: str
    status: str
    submitted_at: str
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_s: float | None = None
    result_rows: int | None = None
    result_data: list[dict[str, Any]] | None = None
    error: str | None = None


class SidecarUploadResponse(RichModel):
    """Response for ``POST /v1/ingest/sidecar`` — opaque id for a stored payload."""

    sidecar_id: str
    filename: str
    content_type: str
    size_bytes: int
    expires_at: str
    """ISO-8601 UTC timestamp after which the sidecar will be auto-evicted."""


class JobCreatedResponse(RichModel):
    """Response for ``POST /v1/ingest/job`` — newly minted job aggregate."""

    job_id: str
    expected_documents: int
    status: str
    created_at: str
    label: str | None = None
    trace_id: str | None = None


class JobAggregateResponse(RichModel):
    """Response for ``GET /v1/ingest/job/{job_id}`` — full aggregate snapshot.

    ``documents`` is included only when the request opts in via the
    ``include_documents=true`` query string (caps very-large payloads).
    The dedicated ``GET /v1/ingest/job/{job_id}/documents`` route is
    preferred for large jobs because it supports pagination + filtering.
    """

    job_id: str
    expected_documents: int
    status: str
    created_at: str
    started_at: str | None = None
    finalized_at: str | None = None
    elapsed_s: float | None = None
    label: str | None = None
    trace_id: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    document_ids: list[str] = Field(default_factory=list)
    documents: list[dict[str, Any]] | None = None


class DocumentStatusResponse(RichModel):
    """First-class per-document status (J6).

    Mirrors :class:`DocumentRecord` on the wire and adds ``job_id`` so
    callers can reverse-navigate to the parent aggregate without an
    extra lookup. The legacy :class:`JobStatusResponse` is preserved
    for the older ``GET /v1/ingest/status/{id}`` polling endpoints.
    """

    document_id: str
    job_id: str
    status: str
    submitted_at: str
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_s: float | None = None
    filename: str | None = None
    result_rows: int | None = None
    result_data: list[dict[str, Any]] | None = None
    error: str | None = None


class JobDocumentsPage(RichModel):
    """Paginated documents response for ``GET /v1/ingest/job/{job_id}/documents``.

    ``items`` is the slice for the requested page. ``total_filtered``
    reflects the result of any ``?status=...`` filter; ``total`` is
    the absolute document count on the job.
    """

    job_id: str
    total: int
    total_filtered: int
    offset: int
    limit: int
    items: list[DocumentStatusResponse] = Field(default_factory=list)
