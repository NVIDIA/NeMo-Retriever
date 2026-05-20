# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from nemo_retriever.service.models.base import RichModel


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProcessingStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DRAINING = "draining"


class Document(RichModel):
    """Domain model representing a row in the ``documents`` table."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    job_id: str | None = None
    filename: str = "unknown"
    content_type: str = "application/octet-stream"
    content_sha256: str = ""
    file_size_bytes: int = 0
    page_number: int | None = None
    total_pages: int | None = None
    pages_received: int = 0
    processing_status: str = ProcessingStatus.QUEUED.value
    metadata_json: str = "{}"
    spool_path: str | None = None
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)

    @field_validator("processing_status", mode="before")
    @classmethod
    def _coerce_status(cls, v: Any) -> str:
        if isinstance(v, ProcessingStatus):
            return v.value
        return v

    def to_row(self) -> dict[str, Any]:
        return self.model_dump()
