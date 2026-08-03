# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from nemo_retriever.common.schemas.base import RichModel
from nemo_retriever.common.schemas.collections import DocumentId, IngestOperation
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


class DocumentManifestEntry(RichModel):
    """One immutable file identity in a resumable ingestion job."""

    manifest_entry_id: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    filename: str
    content_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


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
    operation: IngestOperation = IngestOperation.APPEND
    target_document_id: DocumentId | None = None
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=256)
    document_manifest: list[DocumentManifestEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_job_contract(self) -> "JobCreateRequest":
        if self.document_manifest and len(self.document_manifest) != self.expected_documents:
            raise ValueError("document_manifest length must match expected_documents")
        if len({entry.manifest_entry_id for entry in self.document_manifest}) != len(self.document_manifest):
            raise ValueError("document_manifest contains duplicate manifest_entry_id values")
        if self.operation is IngestOperation.REPLACE:
            if not self.collection_name:
                raise ValueError("replace requires collection_name")
            if self.expected_documents != 1:
                raise ValueError("replace requires exactly one document")
            if not self.target_document_id:
                raise ValueError("replace requires target_document_id")
        elif self.target_document_id is not None:
            raise ValueError("append does not accept target_document_id")
        return self
