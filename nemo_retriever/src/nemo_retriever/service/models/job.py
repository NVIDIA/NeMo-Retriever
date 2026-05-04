# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


class Job:
    """Lightweight data object representing a row in the ``jobs`` table."""

    def __init__(
        self,
        *,
        id: str | None = None,
        filename: str = "unknown",
        content_sha256: str = "",
        total_pages: int = 0,
        pages_submitted: int = 0,
        pages_completed: int = 0,
        processing_status: str = "queued",
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.id = id or uuid.uuid4().hex
        self.filename = filename
        self.content_sha256 = content_sha256
        self.total_pages = total_pages
        self.pages_submitted = pages_submitted
        self.pages_completed = pages_completed
        self.processing_status = processing_status
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "content_sha256": self.content_sha256,
            "total_pages": self.total_pages,
            "pages_submitted": self.pages_submitted,
            "pages_completed": self.pages_completed,
            "processing_status": self.processing_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
