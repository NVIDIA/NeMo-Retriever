# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


class ProcessingMetric:
    """Lightweight data object representing a row in ``processing_metrics``."""

    def __init__(
        self,
        *,
        id: str | None = None,
        document_id: str = "",
        model_name: str = "",
        invocation_count: int = 0,
        pages_processed: int = 0,
        detections_count: int = 0,
        counts_by_label_json: str = "{}",
        duration_ms: float = 0.0,
        created_at: str | None = None,
    ) -> None:
        self.id = id or uuid.uuid4().hex
        self.document_id = document_id
        self.model_name = model_name
        self.invocation_count = invocation_count
        self.pages_processed = pages_processed
        self.detections_count = detections_count
        self.counts_by_label_json = counts_by_label_json
        self.duration_ms = duration_ms
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "model_name": self.model_name,
            "invocation_count": self.invocation_count,
            "pages_processed": self.pages_processed,
            "detections_count": self.detections_count,
            "counts_by_label_json": self.counts_by_label_json,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at,
        }
