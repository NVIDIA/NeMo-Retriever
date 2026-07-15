# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from nemo_retriever.common.schemas.base import RichModel
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec


class IngestRequest(RichModel):
    """Metadata JSON sent alongside the uploaded file.

    ``job_id`` was the legacy free-form client tag; in J3+ it is the
    server-issued aggregate id and is supplied via the URL path rather
    than this body. The field is retained for back-compat with internal
    callers (Prometheus labelers) but the upload routes ignore it.
    """

    job_id: str | None = None
    filename: str | None = None
    content_type: str | None = None
    page_number: int | None = None
    total_pages: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Per-request pipeline overrides (see PipelineSpec). When None, the
    # server falls back to the static config baked at startup.
    pipeline: PipelineSpec | None = None


class JobCreateRequest(RichModel):
    """Body for ``POST /v1/ingest/job`` — open a new ingestion job.

    ``expected_documents`` is the count the client commits to uploading
    against the returned ``job_id``. The server rejects the 101st
    upload to a job created with ``expected_documents=100`` (J3).

    ``label`` is an optional human-readable tag surfaced in the
    dashboard so operators can identify the job in the history view.
    """

    expected_documents: int = Field(ge=1, description="Number of documents this job will receive")
    label: str | None = Field(default=None, description="Optional human-readable tag for the dashboard")
    metadata: dict[str, Any] = Field(default_factory=dict)
    retain_results: bool = Field(
        default=False,
        description=(
            "When false (default), completed documents keep only ``result_rows`` in the "
            "job tracker; row payloads are discarded after the pipeline finishes. Set true "
            "when the client will poll ``GET /v1/ingest/status/{id}`` to fetch "
            "``result_data``."
        ),
    )
    collection_name: str | None = Field(default=None, min_length=1, max_length=128)
    operation: str = Field(default="append", pattern=r"^(append|replace)$")
    target_document_id: str | None = None
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=256)
    document_manifest: list[dict[str, str]] = Field(default_factory=list)
    table_name: str | None = None
    lancedb_uri: str | None = None

    @model_validator(mode="after")
    def _reject_physical_collection_storage(self) -> "JobCreateRequest":
        if self.collection_name and (self.table_name or self.lancedb_uri):
            raise ValueError("collection-aware jobs cannot specify a table name or LanceDB URI")
        if self.document_manifest and len(self.document_manifest) != self.expected_documents:
            raise ValueError("document_manifest length must match expected_documents")
        return self
