# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any


class ProcessingStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DRAINING = "draining"


class Document:
    """Lightweight data object representing a row in the ``documents`` table."""

    def __init__(
        self,
        *,
        id: str | None = None,
        job_id: str | None = None,
        filename: str = "unknown",
        content_type: str = "application/octet-stream",
        content_sha256: str = "",
        file_size_bytes: int = 0,
        page_number: int | None = None,
        total_pages: int | None = None,
        pages_received: int = 0,
        processing_status: str | ProcessingStatus = ProcessingStatus.QUEUED,
        metadata_json: str = "{}",
        spool_path: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.id = id or uuid.uuid4().hex
        self.job_id = job_id
        self.filename = filename
        self.content_type = content_type
        self.content_sha256 = content_sha256
        self.file_size_bytes = file_size_bytes
        self.page_number = page_number
        self.total_pages = total_pages
        self.pages_received = pages_received
        self.processing_status = processing_status if isinstance(processing_status, str) else processing_status.value
        self.metadata_json = metadata_json
        self.spool_path = spool_path
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "content_sha256": self.content_sha256,
            "file_size_bytes": self.file_size_bytes,
            "page_number": self.page_number,
            "total_pages": self.total_pages,
            "pages_received": self.pages_received,
            "processing_status": self.processing_status,
            "metadata_json": self.metadata_json,
            "spool_path": self.spool_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
