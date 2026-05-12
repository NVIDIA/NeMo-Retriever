# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory job status tracker for pipeline work items.

Tracks the lifecycle of each submitted work item from *pending* through
*processing* to *completed* or *failed*.  Status endpoints query this
tracker to report progress back to clients.

Singleton access follows the same optional pattern as the other service
singletons::

    if (tracker := get_job_tracker()) is not None:
        tracker.register(item_id)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from nemo_retriever.service.models.base import RichModel

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecord(RichModel):
    """Snapshot of a single tracked work item."""

    id: str
    status: JobStatus = JobStatus.PENDING
    submitted_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_s: float | None = None
    result_rows: int | None = None
    result_data: list[dict[str, Any]] | None = None
    error: str | None = None


class JobTracker:
    """Thread-safe in-memory store mapping item IDs to :class:`JobRecord`."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._started_mono: dict[str, float] = {}

    def register(self, job_id: str) -> None:
        """Register a newly submitted item as *pending*."""
        self._jobs[job_id] = JobRecord(
            id=job_id,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )

    def mark_processing(self, job_id: str) -> None:
        rec = self._jobs.get(job_id)
        if rec is None:
            return
        rec.status = JobStatus.PROCESSING
        rec.started_at = datetime.now(timezone.utc).isoformat()
        self._started_mono[job_id] = time.monotonic()

    def mark_completed(
        self,
        job_id: str,
        *,
        result_rows: int = 0,
        result_data: list[dict[str, Any]] | None = None,
    ) -> None:
        rec = self._jobs.get(job_id)
        if rec is None:
            return
        rec.status = JobStatus.COMPLETED
        rec.completed_at = datetime.now(timezone.utc).isoformat()
        rec.result_rows = result_rows
        rec.result_data = result_data
        t0 = self._started_mono.pop(job_id, None)
        rec.elapsed_s = round(time.monotonic() - t0, 4) if t0 is not None else None

    def mark_failed(self, job_id: str, error: str) -> None:
        rec = self._jobs.get(job_id)
        if rec is None:
            return
        rec.status = JobStatus.FAILED
        rec.completed_at = datetime.now(timezone.utc).isoformat()
        rec.error = error
        t0 = self._started_mono.pop(job_id, None)
        rec.elapsed_s = round(time.monotonic() - t0, 4) if t0 is not None else None

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def consume_result_data(self, job_id: str) -> list[dict[str, Any]] | None:
        """Return result_data for *job_id* and clear it from memory."""
        rec = self._jobs.get(job_id)
        if rec is None:
            return None
        data = rec.result_data
        rec.result_data = None
        return data

    def summary(self) -> dict[str, Any]:
        total = len(self._jobs)
        by_status = {s.value: 0 for s in JobStatus}
        for rec in self._jobs.values():
            by_status[rec.status.value] += 1
        return {"total_tracked": total, **by_status}


# ── Module-level singleton ───────────────────────────────────────────

_instance: JobTracker | None = None


def init_job_tracker() -> JobTracker:
    global _instance
    _instance = JobTracker()
    logger.info("Job tracker initialised")
    return _instance


def get_job_tracker() -> JobTracker | None:
    return _instance


def shutdown_job_tracker() -> None:
    global _instance
    if _instance is not None:
        summary = _instance.summary()
        logger.info("Job tracker shut down: %s", summary)
        _instance = None
