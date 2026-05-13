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


_TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})

DEFAULT_TTL_S: float = 4 * 3600  # 4 hours
DEFAULT_MAX_JOBS: int = 200_000
_EVICTION_INTERVAL: int = 500  # run eviction check every N registrations


class JobTracker:
    """Thread-safe in-memory store mapping item IDs to :class:`JobRecord`.

    Terminal records (completed/failed) are evicted after *ttl_s* seconds
    or when the total count exceeds *max_jobs*, whichever comes first.
    """

    def __init__(
        self,
        *,
        ttl_s: float = DEFAULT_TTL_S,
        max_jobs: int = DEFAULT_MAX_JOBS,
    ) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._started_mono: dict[str, float] = {}
        self._event_bus: Any = None
        self._ttl_s = ttl_s
        self._max_jobs = max_jobs
        self._reg_count = 0

    def set_event_bus(self, bus: Any) -> None:
        """Attach an :class:`EventBus` so state transitions publish SSE events."""
        self._event_bus = bus

    def register(self, job_id: str) -> None:
        """Register a newly submitted item as *pending*."""
        self._jobs[job_id] = JobRecord(
            id=job_id,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )
        self._reg_count += 1
        if self._reg_count % _EVICTION_INTERVAL == 0:
            self._evict()

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
        elapsed_s: float | None = None,
    ) -> None:
        rec = self._jobs.get(job_id)
        if rec is None:
            return
        rec.status = JobStatus.COMPLETED
        rec.completed_at = datetime.now(timezone.utc).isoformat()
        rec.result_rows = result_rows
        rec.result_data = result_data
        if elapsed_s is not None:
            rec.elapsed_s = elapsed_s
        else:
            t0 = self._started_mono.pop(job_id, None)
            rec.elapsed_s = round(time.monotonic() - t0, 4) if t0 is not None else None
        self._publish_event(rec)

    def mark_failed(self, job_id: str, error: str, *, elapsed_s: float | None = None) -> None:
        rec = self._jobs.get(job_id)
        if rec is None:
            return
        rec.status = JobStatus.FAILED
        rec.completed_at = datetime.now(timezone.utc).isoformat()
        rec.error = error
        if elapsed_s is not None:
            rec.elapsed_s = elapsed_s
        else:
            t0 = self._started_mono.pop(job_id, None)
            rec.elapsed_s = round(time.monotonic() - t0, 4) if t0 is not None else None
        self._publish_event(rec)

    def _evict(self) -> None:
        """Remove terminal records older than TTL, and oldest terminals if over max_jobs."""
        now = datetime.now(timezone.utc)
        expired: list[str] = []
        for jid, rec in self._jobs.items():
            if rec.status not in _TERMINAL_STATUSES:
                continue
            if rec.completed_at:
                try:
                    completed = datetime.fromisoformat(rec.completed_at)
                    if (now - completed).total_seconds() > self._ttl_s:
                        expired.append(jid)
                except (ValueError, TypeError):
                    pass

        for jid in expired:
            self._jobs.pop(jid, None)
            self._started_mono.pop(jid, None)

        if len(self._jobs) > self._max_jobs:
            terminal = [
                (jid, rec.completed_at or "") for jid, rec in self._jobs.items() if rec.status in _TERMINAL_STATUSES
            ]
            terminal.sort(key=lambda t: t[1])
            excess = len(self._jobs) - self._max_jobs
            for jid, _ in terminal[:excess]:
                self._jobs.pop(jid, None)
                self._started_mono.pop(jid, None)

        if expired or len(self._jobs) > self._max_jobs:
            logger.debug("Job tracker eviction: removed %d expired, %d total remaining", len(expired), len(self._jobs))

    def _publish_event(self, rec: JobRecord) -> None:
        if self._event_bus is None:
            return
        self._event_bus.publish_sync(
            {
                "type": rec.status.value,
                "id": rec.id,
                "status": rec.status.value,
                "result_rows": rec.result_rows,
                "elapsed_s": rec.elapsed_s,
                "error": rec.error,
            }
        )

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

    def all_records(self) -> list[dict[str, Any]]:
        """Export every job record as a plain dict (excluding bulky result_data)."""
        return [
            {
                "id": rec.id,
                "status": rec.status.value,
                "submitted_at": rec.submitted_at,
                "started_at": rec.started_at,
                "completed_at": rec.completed_at,
                "elapsed_s": rec.elapsed_s,
                "result_rows": rec.result_rows,
                "error": rec.error,
            }
            for rec in self._jobs.values()
        ]


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
