# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import ConfigDict, Field

from nemo_retriever.service.models.base import RichModel


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Job(RichModel):
    """Domain model representing a row in the ``jobs`` table."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    filename: str = "unknown"
    content_sha256: str = ""
    total_pages: int = 0
    pages_submitted: int = 0
    pages_completed: int = 0
    processing_status: str = "queued"
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)

    def to_row(self) -> dict[str, Any]:
        return self.model_dump()
