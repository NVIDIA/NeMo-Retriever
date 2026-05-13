# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import Field

from nemo_retriever.service.models.base import RichModel
from nemo_retriever.service.models.pipeline_spec import PipelineSpec


class IngestRequest(RichModel):
    """Metadata JSON sent alongside the uploaded file."""

    job_id: str | None = None
    filename: str | None = None
    content_type: str | None = None
    page_number: int | None = None
    total_pages: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Per-request pipeline overrides (see PipelineSpec). When None, the
    # server falls back to the static config baked at startup.
    pipeline: PipelineSpec | None = None
