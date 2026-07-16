# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prometheus metrics for the retriever service.

Defines per-role counters, histograms, and gauges that each pod exports on
``GET /metrics``.  The metrics are created eagerly at import time so they can
be referenced from any module; the ``/metrics`` endpoint is wired up by
:func:`instrument_app`.

Metric naming follows the convention::

    nemo_retriever_<subsystem>_<metric>_<unit>

Label dimensions:

* ``role``     — ``standalone``, ``gateway``, ``realtime``, or ``batch``.
* ``endpoint`` — the HTTP path (e.g. ``/v1/ingest/page``).
* ``status``   — HTTP status code bucket (``2xx``, ``4xx``, ``5xx``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# ── Shared counters (all roles) ──────────────────────────────────────

INGEST_REQUESTS_TOTAL = Counter(
    "nemo_retriever_ingest_requests_total",
    "Total ingest requests received",
    ["role", "endpoint", "status"],
)

INGEST_BYTES_TOTAL = Counter(
    "nemo_retriever_ingest_bytes_total",
    "Total bytes accepted for ingestion",
    ["role", "endpoint"],
)

INGEST_DOCUMENTS_TOTAL = Counter(
    "nemo_retriever_ingest_documents_total",
    "Total documents accepted",
    ["role"],
)

INGEST_PAGES_TOTAL = Counter(
    "nemo_retriever_ingest_pages_total",
    "Total pages accepted",
    ["role"],
)

INGEST_ERRORS_TOTAL = Counter(
    "nemo_retriever_ingest_errors_total",
    "Total ingest errors",
    ["role", "error_type"],
)

# ── Gateway-specific ─────────────────────────────────────────────────

GATEWAY_FORWARD_DURATION = Histogram(
    "nemo_retriever_gateway_forward_duration_seconds",
    "Time spent forwarding a request to a backend pod",
    ["backend"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

WORK_QUEUE_ITEMS = Gauge("nemo_retriever_work_queue_items", "Gateway work items waiting to be claimed.", ["pool"])
WORK_QUEUE_BYTES = Gauge("nemo_retriever_work_queue_bytes", "Bytes waiting in the gateway work queue.", ["pool"])
WORK_QUEUE_ACTIVE_LEASES = Gauge(
    "nemo_retriever_work_queue_active_leases", "Current active gateway work leases.", ["pool"]
)
WORK_QUEUE_DEMAND = Gauge("nemo_retriever_work_queue_demand", "Queued work plus active gateway leases.", ["pool"])
WORK_QUEUE_MAX_ACTIVE_LEASES = Gauge(
    "nemo_retriever_work_queue_max_active_leases", "Configured gateway active-lease capacity.", ["pool"]
)
WORK_QUEUE_WAIT = Histogram(
    "nemo_retriever_work_queue_wait_seconds", "Time from gateway admission until claim.", ["pool"]
)
WORK_QUEUE_CLAIMS = Counter(
    "nemo_retriever_work_queue_claims_total",
    "Work claims issued by pool and worker pod.",
    ["pool", "worker_uid"],
)
WORK_QUEUE_EXPIRATIONS = Counter("nemo_retriever_work_queue_expirations_total", "Expired work leases.", ["pool"])
WORK_QUEUE_REQUEUES = Counter(
    "nemo_retriever_work_queue_requeues_total", "Work leases requeued after expiry or release.", ["pool", "reason"]
)
WORK_QUEUE_EXHAUSTED = Counter(
    "nemo_retriever_work_queue_exhausted_total", "Work items failed after exhausting delivery attempts.", ["pool"]
)
WORK_QUEUE_STALE_CALLBACKS = Counter(
    "nemo_retriever_work_queue_stale_callbacks_total",
    "Callbacks rejected because their lease was superseded.",
    ["pool"],
)

# ── Worker-specific ──────────────────────────────────────────────────

POOL_QUEUE_DEPTH = Gauge(
    "nemo_retriever_pool_queue_depth",
    "Current items waiting in the worker pool queue",
    ["pool"],
)

POOL_QUEUE_DEPTH_RATIO = Gauge(
    "nemo_retriever_pool_queue_depth_ratio",
    "Queue fill ratio (current depth / max queue size), in [0.0, 1.0]. "
    "This is the canonical signal for queue-depth-based HPA scaling.",
    ["pool"],
)

POOL_MAX_QUEUE_SIZE = Gauge(
    "nemo_retriever_pool_max_queue_size",
    "Configured maximum queue size for the worker pool. Set once at "
    "startup; published so prometheus-adapter (or any consumer) can "
    "compute the queue-depth ratio in PromQL without baking the "
    "denominator into the publisher.",
    ["pool"],
)

POOL_WORKERS = Gauge(
    "nemo_retriever_pool_workers",
    "Number of worker tasks configured for the pool.",
    ["pool"],
)

POOL_PROCESSED_TOTAL = Counter(
    "nemo_retriever_pool_processed_total",
    "Total work items processed by a pool, by terminal outcome.",
    ["pool", "outcome"],
)

POOL_PROCESSING_DURATION = Histogram(
    "nemo_retriever_pool_processing_duration_seconds",
    "Time spent processing a single work item",
    ["pool"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0),
)

POOL_DEFERRED_CALLBACKS = Gauge(
    "nemo_retriever_pool_deferred_callbacks",
    "Current gateway callback deliveries retrying in the background.",
    ["pool"],
)

POOL_CALLBACK_BACKPRESSURE_TOTAL = Counter(
    "nemo_retriever_pool_callback_backpressure_total",
    "Total completed work items that waited for bounded callback retry capacity.",
    ["pool"],
)

POOL_ACTIVE_SLOTS = Gauge(
    "nemo_retriever_pool_active_execution_slots", "Execution slots currently holding a claimed item.", ["pool"]
)
POOL_CLAIM_LATENCY = Histogram(
    "nemo_retriever_pool_claim_latency_seconds", "Worker long-poll claim request latency.", ["pool"]
)
POOL_HEARTBEAT_FAILURES = Counter(
    "nemo_retriever_pool_heartbeat_failures_total", "Failed worker lease heartbeat requests.", ["pool"]
)
POOL_COMPLETED_CLAIMS = Counter(
    "nemo_retriever_pool_completed_claims_total", "Claims whose callbacks were acknowledged.", ["pool"]
)


def instrument_app(app: "FastAPI", *, role: str) -> None:
    """Add a ``GET /metrics`` endpoint to *app* that exports Prometheus data.

    Also stores the *role* label on ``app.state`` so route handlers can
    record metrics with the correct role dimension.
    """
    from fastapi import Response as FastAPIResponse

    app.state.prometheus_role = role

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> FastAPIResponse:
        return FastAPIResponse(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    logger.info("Prometheus /metrics endpoint registered (role=%s)", role)
