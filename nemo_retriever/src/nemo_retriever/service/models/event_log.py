# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic model and enums for the ``event_log`` provenance table.

Every significant event that occurs during service-mode processing is
persisted as an :class:`EventRecord`.  The model mirrors the SQLite
``event_log`` table 1:1 so it can drive both the DB layer and the REST
API without translation.

Design notes
------------
* **Append-only** — rows are never updated or deleted.
* **Severity + outcome** together tell a UI whether the event is a
  permanent failure, a transient glitch that recovered, or a normal
  lifecycle marker.
* **extra_json** is the extensibility escape hatch for future
  provenance metadata (operator timings, retry counts, GPU memory
  snapshots, etc.) without schema migrations.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventSeverity(str, enum.Enum):
    """How serious the event is."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class EventOutcome(str, enum.Enum):
    """What happened after the event.

    * ``failed`` — the operation did not succeed and was not retried.
    * ``recovered`` — a transient error occurred but the system retried
      or fell back successfully.
    * ``in_progress`` — a normal lifecycle event (stage entered/exited)
      that is neither a failure nor a recovery.
    """

    FAILED = "failed"
    RECOVERED = "recovered"
    IN_PROGRESS = "in_progress"


class EventRecord(BaseModel):
    """A single provenance event, mirroring one row of ``event_log``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC instant when the event occurred.",
    )
    job_id: str | None = Field(default=None, description="FK -> jobs(id)")
    document_id: str | None = Field(default=None, description="FK -> documents(id)")
    source_file: str = Field(default="", description="Human-readable filename.")
    page_number: int | None = Field(default=None, description="1-based input page number.")
    category: str = Field(
        ...,
        description="Event category (from EventCategory enum).",
    )
    severity: EventSeverity = Field(default=EventSeverity.ERROR)
    outcome: EventOutcome = Field(default=EventOutcome.FAILED)
    stage: str = Field(
        default="",
        description="Pipeline stage name or endpoint step (e.g. 'page_elements', 'embed').",
    )
    summary: str = Field(
        default="",
        description="One-line human-readable summary.",
    )
    detail: str = Field(
        default="",
        description="Full error message or extended context.",
    )
    stack_trace: str = Field(
        default="",
        description="Python traceback (traceback.format_exc()) or empty.",
    )
    endpoint: str = Field(
        default="",
        description="Originating endpoint path (e.g. '/v1/query', 'pipeline').",
    )
    request_id: str = Field(
        default="",
        description="Correlates multiple events from one HTTP request or batch.",
    )
    extra_json: str = Field(
        default="{}",
        description="Extensible JSON blob for future provenance data.",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Row insertion timestamp.",
    )

    def to_row(self) -> dict[str, Any]:
        """Serialise to a flat dict suitable for SQLite ``INSERT``."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "job_id": self.job_id,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "category": self.category,
            "severity": self.severity.value if isinstance(self.severity, EventSeverity) else self.severity,
            "outcome": self.outcome.value if isinstance(self.outcome, EventOutcome) else self.outcome,
            "stage": self.stage,
            "summary": self.summary,
            "detail": self.detail,
            "stack_trace": self.stack_trace,
            "endpoint": self.endpoint,
            "request_id": self.request_id,
            "extra_json": self.extra_json,
            "created_at": self.created_at,
        }
