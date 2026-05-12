# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracer, meter, and instrument accessors built on the OpenTelemetry API.

Depends only on the OpenTelemetry API package. When no provider is
registered the returned tracer and meter are no-ops with negligible
overhead, so this module is safe to import unconditionally from library
code.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Mapping

from opentelemetry import metrics, trace

from nemo_retriever.observability.attributes import INSTRUMENTATION_SCOPE

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Histogram, Meter
    from opentelemetry.trace import Tracer

try:
    _NR_VERSION: str | None = version("nemo-retriever")
except PackageNotFoundError:
    _NR_VERSION = None


@lru_cache(maxsize=1)
def get_tracer() -> "Tracer":
    """Return the NeMo Retriever tracer for the active global provider."""
    return trace.get_tracer(INSTRUMENTATION_SCOPE, _NR_VERSION)


@lru_cache(maxsize=1)
def get_meter() -> "Meter":
    """Return the NeMo Retriever meter for the active global provider."""
    return metrics.get_meter(INSTRUMENTATION_SCOPE, _NR_VERSION)


@lru_cache(maxsize=None)
def _counter(name: str, description: str, unit: str = "1") -> "Counter":
    return get_meter().create_counter(name=name, unit=unit, description=description)


@lru_cache(maxsize=None)
def _histogram(name: str, description: str, unit: str) -> "Histogram":
    return get_meter().create_histogram(name=name, unit=unit, description=description)


def reset_instrument_cache() -> None:
    """Clear cached tracer, meter, and instruments. Used by tests that swap providers."""
    get_tracer.cache_clear()
    get_meter.cache_clear()
    _counter.cache_clear()
    _histogram.cache_clear()


# Named instruments. Each is a thin alias so the cache key and description
# stay colocated with the call site.


def operator_duration_histogram() -> "Histogram":
    return _histogram(
        "nemo_retriever.operator.duration_ms",
        "Wall-clock duration of a single AbstractOperator.run() call.",
        "ms",
    )


def documents_processed_counter() -> "Counter":
    return _counter(
        "nemo_retriever.documents.processed",
        "Number of documents that flowed through an operator.",
    )


def documents_failed_counter() -> "Counter":
    return _counter(
        "nemo_retriever.documents.failed",
        "Number of operator invocations that raised an exception.",
    )


def jobs_submitted_counter() -> "Counter":
    return _counter(
        "nemo_retriever.jobs.submitted",
        "Number of ingest jobs accepted by the service (one increment per job_id).",
    )


def jobs_completed_counter() -> "Counter":
    return _counter(
        "nemo_retriever.jobs.completed",
        "Number of jobs that reached terminal status=complete.",
    )


def jobs_failed_counter() -> "Counter":
    return _counter(
        "nemo_retriever.jobs.failed",
        "Number of jobs that ended in status=failed or status=cancelled.",
    )


def pages_completed_counter() -> "Counter":
    return _counter(
        "nemo_retriever.pages.completed",
        "Number of pages whose pipeline run finished successfully.",
    )


def pages_failed_counter() -> "Counter":
    return _counter(
        "nemo_retriever.pages.failed",
        "Number of pages whose pipeline run terminally failed (any stage).",
    )


def safe_add(counter: "Counter", amount: int = 1, attributes: Mapping[str, Any] | None = None) -> None:
    """Add to a counter without ever raising. Use on pipeline hot paths.

    Telemetry must never break ingestion: a misconfigured exporter or
    metric instrument bug should not propagate as an unhandled
    exception. Use this from call sites where the increment is purely
    observational.
    """
    try:
        counter.add(amount, attributes=attributes)
    except Exception:  # noqa: BLE001 — telemetry must never raise
        pass
