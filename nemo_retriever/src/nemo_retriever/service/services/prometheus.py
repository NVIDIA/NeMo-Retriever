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

# ── Worker-specific ──────────────────────────────────────────────────

POOL_QUEUE_DEPTH = Gauge(
    "nemo_retriever_pool_queue_depth",
    "Current items waiting in the worker pool queue",
    ["pool"],
)

POOL_PROCESSING_DURATION = Histogram(
    "nemo_retriever_pool_processing_duration_seconds",
    "Time spent processing a single work item",
    ["pool"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0),
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
