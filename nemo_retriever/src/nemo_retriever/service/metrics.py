# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prometheus instrumentation for the nemo-retriever service.

Two layers of metrics are exposed at ``GET /metrics``:

1. **HTTP-level**, courtesy of ``prometheus-fastapi-instrumentator``:
   request totals, latency histograms, in-progress counts, etc., grouped
   by route template + status code family.

2. **Service-specific gauges** that surface the live state of the
   processing pool and the job database — so dashboards and alerts can
   answer questions like "how many pages are queued?", "is the pool
   draining?", "how many jobs failed today?".

The custom gauges are refreshed by a single asyncio background task
running in the main process; worker subprocesses contribute through the
already-aggregated database tables, so no IPC plumbing is required.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

if TYPE_CHECKING:
    from nemo_retriever.service.db.repository import Repository
    from nemo_retriever.service.processing.pool import ProcessingPool

logger = logging.getLogger(__name__)

DEFAULT_REFRESH_INTERVAL_S = 15.0

# Statuses we always emit (with value 0 when no jobs are present), so
# Prometheus dashboards don't have to special-case "missing series".
_KNOWN_JOB_STATUSES: tuple[str, ...] = (
    "queued",
    "processing",
    "complete",
    "failed",
    "cancelled",
    "draining",
)


class ServiceMetrics:
    """All custom Prometheus collectors for the nemo-retriever service.

    Lives on ``app.state.service_metrics`` so tests / introspection
    endpoints can poke at it.
    """

    def __init__(self) -> None:
        self.pool_workers = Gauge(
            "nemo_retriever_pool_workers",
            "Number of worker processes configured for the processing pool.",
        )
        self.pool_capacity = Gauge(
            "nemo_retriever_pool_capacity_pages",
            "Free pages the batch buffer can still accept before saturating.",
        )
        self.pool_buffered = Gauge(
            "nemo_retriever_pool_buffered_pages",
            "Pages currently sitting in the batch buffer waiting for dispatch.",
        )
        self.pool_in_flight = Gauge(
            "nemo_retriever_pool_in_flight_batches",
            "Batches submitted to a worker process but not yet completed.",
        )
        self.pool_draining = Gauge(
            "nemo_retriever_pool_draining",
            "1 when the pool is draining for shutdown, 0 otherwise.",
        )
        self.jobs_by_status = Gauge(
            "nemo_retriever_jobs",
            "Number of jobs currently tracked in the database, by status.",
            ["status"],
        )

    def refresh(self, repo: "Repository", pool: "ProcessingPool") -> None:
        """Re-read live state from the pool + DB and update every gauge.

        Designed to be cheap and exception-safe: a transient SQLite lock or
        an in-flight pool teardown should never crash the refresh loop.
        """
        try:
            self.pool_workers.set(pool.pool_size)
            self.pool_capacity.set(pool.capacity)
            buf = getattr(pool, "_buffer", None)
            self.pool_buffered.set(buf.buffered_count if buf is not None else 0)
            self.pool_in_flight.set(pool.in_flight_batches())
            self.pool_draining.set(1 if pool.is_draining else 0)
        except Exception:  # noqa: BLE001 — refresh must never raise
            logger.exception("Failed to refresh processing-pool gauges")

        try:
            counts = repo.count_jobs_by_status()
            seen: set[str] = set()
            for status, n in counts.items():
                self.jobs_by_status.labels(status=status).set(n)
                seen.add(status)
            for status in _KNOWN_JOB_STATUSES:
                if status not in seen:
                    self.jobs_by_status.labels(status=status).set(0)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to refresh DB-derived gauges")


async def _refresh_loop(
    metrics: ServiceMetrics,
    repo: "Repository",
    pool: "ProcessingPool",
    interval_s: float,
) -> None:
    """Background task: refresh gauges every ``interval_s`` seconds."""
    try:
        while True:
            metrics.refresh(repo, pool)
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        # Final refresh on shutdown so /metrics shows the latest snapshot
        # if anything scrapes during the drain window.
        metrics.refresh(repo, pool)
        raise


def setup_instrumentation(app: FastAPI) -> ServiceMetrics:
    """Install the Prometheus middleware and ``/metrics`` route.

    Must be called from :func:`create_app` *before* the application starts
    accepting requests — Starlette forbids ``add_middleware()`` once the
    lifespan has begun.

    The returned :class:`ServiceMetrics` instance is stashed on
    ``app.state.service_metrics`` so the lifespan can later refresh it
    against the live pool / repository.
    """
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/metrics", "/openapi.json", "/docs", "/redoc"],
    )
    instrumentator.instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,
        tags=["metrics"],
    )

    metrics = ServiceMetrics()
    app.state.service_metrics = metrics
    logger.info("Prometheus metrics endpoint registered at /metrics")
    return metrics


def start_refresh_loop(
    app: FastAPI,
    *,
    refresh_interval_s: float = DEFAULT_REFRESH_INTERVAL_S,
) -> asyncio.Task[None]:
    """Start the periodic gauge-refresh task.

    Call from inside the FastAPI lifespan **after** ``app.state.repository``
    and ``app.state.processing_pool`` are populated.  Returns the task so
    the caller can cancel it on shutdown.
    """
    metrics: ServiceMetrics = app.state.service_metrics
    repo: "Repository" = app.state.repository
    pool: "ProcessingPool" = app.state.processing_pool

    metrics.refresh(repo, pool)
    task = asyncio.create_task(
        _refresh_loop(metrics, repo, pool, refresh_interval_s),
        name="nemo-retriever-metrics-refresh",
    )
    logger.info(
        "Prometheus gauge refresh loop started (interval=%.1fs)",
        refresh_interval_s,
    )
    return task
