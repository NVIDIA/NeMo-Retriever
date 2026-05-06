# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


class PageResult:
    """Lightweight data object representing a row in ``page_results``."""

    def __init__(
        self,
        *,
        id: str | None = None,
        document_id: str = "",
        page_number: int = 0,
        content_json: str = "{}",
        created_at: str | None = None,
    ) -> None:
        self.id = id or uuid.uuid4().hex
        self.document_id = document_id
        self.page_number = page_number
        self.content_json = content_json
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "content_json": self.content_json,
            "created_at": self.created_at,
        }
