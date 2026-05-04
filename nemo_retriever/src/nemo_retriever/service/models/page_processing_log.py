# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class PageProcessingLog:
    """Lightweight data object representing a row in ``page_processing_log``."""

    def __init__(
        self,
        *,
        id: str = "",
        document_id: str = "",
        job_id: str | None = None,
        source_file: str = "",
        page_number: int = 0,
        status: str = "complete",
        error_message: str | None = None,
        failure_type: str | None = None,
        detection_count: int = 0,
        processing_duration_ms: float = 0.0,
        started_at: str = "",
        completed_at: str = "",
        created_at: str | None = None,
    ) -> None:
        self.id = id
        self.document_id = document_id
        self.job_id = job_id
        self.source_file = source_file
        self.page_number = page_number
        self.status = status
        self.error_message = error_message
        self.failure_type = failure_type
        self.detection_count = detection_count
        self.processing_duration_ms = processing_duration_ms
        self.started_at = started_at
        self.completed_at = completed_at
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    @staticmethod
    def make_id(source_file: str, page_number: int) -> str:
        return f"{source_file}:{page_number}"

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "job_id": self.job_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "status": self.status,
            "error_message": self.error_message,
            "failure_type": self.failure_type,
            "detection_count": self.detection_count,
            "processing_duration_ms": self.processing_duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "created_at": self.created_at,
        }
