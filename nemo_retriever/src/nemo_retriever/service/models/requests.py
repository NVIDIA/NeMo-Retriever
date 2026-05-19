# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import Field

from nemo_retriever.service.models.base import RichModel
from nemo_retriever.service.models.pipeline_spec import PipelineSpec


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
