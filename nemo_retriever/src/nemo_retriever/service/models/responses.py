# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_retriever.service.models.base import RichModel


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
