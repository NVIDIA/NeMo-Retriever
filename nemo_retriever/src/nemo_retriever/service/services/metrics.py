# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory ingest metrics singleton.

Everything lives in this one file for easy tracing and modification.
The singleton is **optional** — obtain it via ``get_metrics()`` which returns
``None`` when metrics have not been initialised (callers gate with a simple
``if`` check).

Three tracking granularities
-----------------------------
* **Job**      — a client submitting N documents/pages as a logical batch.
* **Document** — a single uploaded file.
* **Page**     — a single page (from ``/ingest/page`` or split server-side).

OpenTelemetry mapping
---------------------
Fields are chosen so they map cleanly onto OTel metric types:

* *Counters* — ``total_requests``, ``total_bytes_accepted``,
  ``total_documents_accepted``, ``total_pages_accepted``, ``total_errors``
* *Gauges*   — ``active_jobs``, ``active_documents``
* *Histograms* — ``processing_duration_s``, ``file_size_bytes`` on per-record
  models (ready for ``otel.metrics.Histogram.record()`` export).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pydantic import ConfigDict, Field

from nemo_retriever.service.models.base import RichModel

logger = logging.getLogger(__name__)

# ── Capacity knobs (tweak without touching the class) ────────────────

MAX_RECENT_PAGES: int = 2_000
MAX_DOCUMENTS: int = 10_000
MAX_JOBS: int = 1_000

# ── Pydantic models (all RichModel for pretty-printing) ─────────────


class PageMetric(RichModel):
    """Tracks a single ingested page."""

    page_id: str
    document_id: str
    endpoint: str
    page_number: int | None = None
    file_size_bytes: int = 0
    file_category: str = ""
    content_type: str = ""
    status: str = "accepted"
    accepted_at: str = ""
    completed_at: str | None = None
    processing_duration_s: float | None = None
    error: str | None = None


class DocumentMetric(RichModel):
    """Tracks a single ingested document."""

    document_id: str
    job_id: str | None = None
    filename: str = ""
    file_category: str = ""
    content_type: str = ""
    file_size_bytes: int = 0
    pages_submitted: int = 0
    pages_completed: int = 0
    pages_failed: int = 0
    status: str = "accepted"
    accepted_at: str = ""
    completed_at: str | None = None
    processing_duration_s: float | None = None
    error: str | None = None


class JobMetric(RichModel):
    """Tracks a logical batch/job of documents."""

    job_id: str
    documents_submitted: int = 0
    documents_completed: int = 0
    documents_failed: int = 0
    pages_total: int = 0
    pages_completed: int = 0
    pages_failed: int = 0
    bytes_total: int = 0
    status: str = "active"
    created_at: str = ""
    completed_at: str | None = None
    wall_duration_s: float | None = None


class IngestMetricsSnapshot(RichModel):
    """Top-level response returned by ``GET /v1/ingest/metrics``.

    Intentionally a flat snapshot so it serialises to a single JSON object
    that is easy to forward to OTel, Prometheus, or any time-series backend.
    """

    model_config = ConfigDict(ser_json_inf_nan="constants")

    # ── service-level (OTel counters) ────────────────────────────────
    service_started_at: str
    uptime_s: float

    total_requests: int = 0
    total_bytes_accepted: int = 0
    total_documents_accepted: int = 0
    total_pages_accepted: int = 0
    total_errors: int = 0

    # ── gauges ───────────────────────────────────────────────────────
    active_jobs: int = 0
    active_documents: int = 0

    # ── derived throughput ───────────────────────────────────────────
    throughput_pages_per_s: float = 0.0
    throughput_docs_per_s: float = 0.0
    throughput_bytes_per_s: float = 0.0

    # ── breakdowns (OTel counter attributes / labels) ────────────────
    requests_by_endpoint: dict[str, int] = Field(default_factory=dict)
    errors_by_type: dict[str, int] = Field(default_factory=dict)
    documents_by_category: dict[str, int] = Field(default_factory=dict)
    bytes_by_category: dict[str, int] = Field(default_factory=dict)

    # ── detailed records ─────────────────────────────────────────────
    jobs: list[JobMetric] = Field(default_factory=list)
    documents: list[DocumentMetric] = Field(default_factory=list)
    recent_pages: list[PageMetric] = Field(default_factory=list)


# ── Singleton service ────────────────────────────────────────────────


class IngestMetrics:
    """Thread-safe in-memory metrics store.

    All mutating helpers acquire ``_lock`` so they are safe to call from
    background tasks, OTel export threads, or the async event loop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)
        self._started_mono = time.monotonic()

        self._total_requests: int = 0
        self._total_bytes: int = 0
        self._total_documents: int = 0
        self._total_pages: int = 0
        self._total_errors: int = 0

        self._requests_by_endpoint: dict[str, int] = {}
        self._errors_by_type: dict[str, int] = {}
        self._documents_by_category: dict[str, int] = {}
        self._bytes_by_category: dict[str, int] = {}

        self._jobs: dict[str, JobMetric] = {}
        self._documents: dict[str, DocumentMetric] = {}
        self._pages: deque[PageMetric] = deque(maxlen=MAX_RECENT_PAGES)

    # ── recording helpers ────────────────────────────────────────────

    def record_request(self, endpoint: str) -> None:
        with self._lock:
            self._total_requests += 1
            self._requests_by_endpoint[endpoint] = self._requests_by_endpoint.get(endpoint, 0) + 1

    def record_error(self, error_type: str) -> None:
        with self._lock:
            self._total_errors += 1
            self._errors_by_type[error_type] = self._errors_by_type.get(error_type, 0) + 1

    def record_job_created(self, job_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if job_id not in self._jobs:
                if len(self._jobs) >= MAX_JOBS:
                    self._evict_oldest_job()
                self._jobs[job_id] = JobMetric(job_id=job_id, created_at=now)

    def record_document_accepted(
        self,
        *,
        document_id: str,
        job_id: str | None = None,
        filename: str = "",
        file_category: str = "",
        content_type: str = "",
        file_size_bytes: int = 0,
        endpoint: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._total_documents += 1
            self._total_bytes += file_size_bytes
            if file_category:
                self._documents_by_category[file_category] = self._documents_by_category.get(file_category, 0) + 1
                self._bytes_by_category[file_category] = self._bytes_by_category.get(file_category, 0) + file_size_bytes
            if len(self._documents) >= MAX_DOCUMENTS:
                self._evict_oldest_document()
            self._documents[document_id] = DocumentMetric(
                document_id=document_id,
                job_id=job_id,
                filename=filename,
                file_category=file_category,
                content_type=content_type,
                file_size_bytes=file_size_bytes,
                accepted_at=now,
            )
            if job_id and job_id in self._jobs:
                job = self._jobs[job_id]
                self._jobs[job_id] = job.model_copy(
                    update={
                        "documents_submitted": job.documents_submitted + 1,
                        "bytes_total": job.bytes_total + file_size_bytes,
                    }
                )

    def record_page_accepted(
        self,
        *,
        page_id: str,
        document_id: str,
        endpoint: str = "",
        page_number: int | None = None,
        file_size_bytes: int = 0,
        file_category: str = "",
        content_type: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._total_pages += 1
            self._total_bytes += file_size_bytes
            self._pages.append(
                PageMetric(
                    page_id=page_id,
                    document_id=document_id,
                    endpoint=endpoint,
                    page_number=page_number,
                    file_size_bytes=file_size_bytes,
                    file_category=file_category,
                    content_type=content_type,
                    accepted_at=now,
                )
            )
            if document_id in self._documents:
                doc = self._documents[document_id]
                self._documents[document_id] = doc.model_copy(
                    update={
                        "pages_submitted": doc.pages_submitted + 1,
                    }
                )
            job_id = self._documents[document_id].job_id if document_id in self._documents else None
            if job_id and job_id in self._jobs:
                job = self._jobs[job_id]
                self._jobs[job_id] = job.model_copy(
                    update={
                        "pages_total": job.pages_total + 1,
                    }
                )

    def record_document_completed(self, document_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if document_id not in self._documents:
                return
            doc = self._documents[document_id]
            self._documents[document_id] = doc.model_copy(
                update={
                    "status": "completed",
                    "completed_at": now,
                }
            )

    def record_page_completed(self, page_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            for i, page in enumerate(self._pages):
                if page.page_id == page_id:
                    self._pages[i] = page.model_copy(
                        update={
                            "status": "completed",
                            "completed_at": now,
                        }
                    )
                    break

    # ── single-record lookups ────────────────────────────────────────

    def get_job(self, job_id: str) -> JobMetric | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_document(self, document_id: str) -> DocumentMetric | None:
        with self._lock:
            return self._documents.get(document_id)

    def get_page(self, page_id: str) -> PageMetric | None:
        with self._lock:
            for page in self._pages:
                if page.page_id == page_id:
                    return page
            return None

    def get_job_documents(self, job_id: str) -> list[DocumentMetric]:
        with self._lock:
            return [d for d in self._documents.values() if d.job_id == job_id]

    def get_job_error_documents(self, job_id: str) -> list[DocumentMetric]:
        with self._lock:
            return [d for d in self._documents.values() if d.job_id == job_id and d.error is not None]

    # ── snapshot ─────────────────────────────────────────────────────

    def snapshot(self) -> IngestMetricsSnapshot:
        """Return a point-in-time immutable copy of all metrics."""
        uptime = time.monotonic() - self._started_mono
        with self._lock:
            return IngestMetricsSnapshot(
                service_started_at=self._started_at.isoformat(),
                uptime_s=round(uptime, 3),
                total_requests=self._total_requests,
                total_bytes_accepted=self._total_bytes,
                total_documents_accepted=self._total_documents,
                total_pages_accepted=self._total_pages,
                total_errors=self._total_errors,
                active_jobs=sum(1 for j in self._jobs.values() if j.status == "active"),
                active_documents=sum(1 for d in self._documents.values() if d.status == "accepted"),
                throughput_pages_per_s=(round(self._total_pages / uptime, 2) if uptime > 0 else 0.0),
                throughput_docs_per_s=(round(self._total_documents / uptime, 2) if uptime > 0 else 0.0),
                throughput_bytes_per_s=(round(self._total_bytes / uptime, 2) if uptime > 0 else 0.0),
                requests_by_endpoint=dict(self._requests_by_endpoint),
                errors_by_type=dict(self._errors_by_type),
                documents_by_category=dict(self._documents_by_category),
                bytes_by_category=dict(self._bytes_by_category),
                jobs=list(self._jobs.values()),
                documents=list(self._documents.values()),
                recent_pages=list(self._pages),
            )

    # ── internal eviction ────────────────────────────────────────────

    def _evict_oldest_job(self) -> None:
        completed = [jid for jid, j in self._jobs.items() if j.status != "active"]
        if completed:
            del self._jobs[completed[0]]
        elif self._jobs:
            del self._jobs[next(iter(self._jobs))]

    def _evict_oldest_document(self) -> None:
        completed = [did for did, d in self._documents.items() if d.status != "accepted"]
        if completed:
            del self._documents[completed[0]]
        elif self._documents:
            del self._documents[next(iter(self._documents))]


# ── Module-level singleton access ────────────────────────────────────

_instance: IngestMetrics | None = None


def init_metrics() -> IngestMetrics:
    """Create the global metrics instance (call once during app startup)."""
    global _instance
    _instance = IngestMetrics()
    logger.info("Ingest metrics service initialised")
    return _instance


def get_metrics() -> IngestMetrics | None:
    """Return the metrics singleton, or ``None`` if not initialised.

    This is the **only** function callers need. The optional pattern is::

        if (m := get_metrics()) is not None:
            m.record_document_accepted(...)
    """
    return _instance


def shutdown_metrics() -> None:
    """Tear down the singleton (call during app shutdown)."""
    global _instance
    if _instance is not None:
        logger.info("Ingest metrics service shut down")
        _instance = None
