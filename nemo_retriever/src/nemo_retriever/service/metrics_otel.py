# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry metrics for Retriever service ingestion.

The service also exposes native Prometheus metrics at ``/metrics``.  This
module is the complementary OTLP path used by the chart-owned Collector.
Telemetry setup and recording are deliberately best effort: an unavailable
collector must never prevent an ingestion request from completing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Mapping

from opentelemetry import metrics

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
except Exception:  # pragma: no cover - covered through configuration failure handling.
    OTLPMetricExporter = None  # type: ignore[assignment]
    MeterProvider = None  # type: ignore[assignment]
    PeriodicExportingMetricReader = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_DEFAULT_SERVICE_NAME = "nemo-retriever-service"
_METER_NAME = "nemo_retriever.service"
_CONFIGURED_PROVIDER: Any | None = None
_METRICS: dict[str, Any] = {}


def metrics_enabled_from_env(env: Mapping[str, str] | None = None) -> bool:
    """Return whether Helm-compatible environment enables OTLP metrics."""
    source = os.environ if env is None else env
    if source.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return False
    return bool(
        source.get("OTEL_METRICS_EXPORTER", "").strip().lower() == "otlp"
        and source.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    )


def configure_metrics(*, service_role: str, service_name: str | None = None) -> bool:
    """Configure process-wide OTLP metrics when enabled by environment."""
    global _CONFIGURED_PROVIDER

    if _CONFIGURED_PROVIDER is not None:
        return True
    if not metrics_enabled_from_env():
        return False

    provider: Any | None = None
    reader: Any | None = None
    exporter: Any | None = None
    try:
        if any(component is None for component in (OTLPMetricExporter, MeterProvider, PeriodicExportingMetricReader, Resource)):
            raise RuntimeError("OpenTelemetry metrics SDK/exporter packages are not importable")

        resolved_service_name = (service_name or os.environ.get("OTEL_SERVICE_NAME") or _DEFAULT_SERVICE_NAME).strip()
        resource = Resource.create({"service.name": resolved_service_name or _DEFAULT_SERVICE_NAME, "service.role": service_role})
        exporter = OTLPMetricExporter()
        reader = PeriodicExportingMetricReader(exporter)
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        if metrics.get_meter_provider() is not provider:
            raise RuntimeError("OpenTelemetry meter provider is already configured")

        _configure_instruments()
        _CONFIGURED_PROVIDER = provider
        logger.info("OpenTelemetry metrics configured: service=%s role=%s", resolved_service_name, service_role)
        return True
    except Exception as exc:
        _cleanup_partial_metrics_setup(provider=provider, reader=reader, exporter=exporter)
        logger.warning("OpenTelemetry metrics setup failed: %s", exc)
        return False


def instrument_app(app: "FastAPI", *, role: str) -> None:
    """Record request count, duration, and failures for ingest routes."""

    @app.middleware("http")
    async def otel_ingest_metrics(request: Any, call_next: Any) -> Any:
        if not request.url.path.startswith("/v1/ingest/"):
            return await call_next(request)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            record_ingest_request(role=role, endpoint=request.url.path, status="5xx", duration_s=time.perf_counter() - started)
            raise

        status = f"{response.status_code // 100}xx"
        record_ingest_request(role=role, endpoint=request.url.path, status=status, duration_s=time.perf_counter() - started)
        return response


def record_ingest_accepted(*, role: str, endpoint: str, file_size: int, is_page: bool) -> None:
    """Record accepted ingest payload cardinality and size."""
    attributes = {"role": role, "endpoint": endpoint}
    _add("bytes", file_size, attributes)
    _add("pages" if is_page else "documents", 1, {"role": role})


def record_ingest_request(*, role: str, endpoint: str, status: str, duration_s: float) -> None:
    """Record an ingest response and its duration."""
    attributes = {"role": role, "endpoint": endpoint, "status": status}
    _add("requests", 1, attributes)
    _record("request_duration", duration_s, attributes)
    if not status.startswith("2"):
        _add("errors", 1, attributes)


def force_flush(timeout_millis: int = 1000) -> None:
    """Best-effort flush of configured metrics."""
    if _CONFIGURED_PROVIDER is None:
        return
    try:
        _CONFIGURED_PROVIDER.force_flush(timeout_millis=timeout_millis)
    except Exception as exc:
        logger.warning("OpenTelemetry metrics flush failed: %s", exc)


def shutdown_metrics() -> None:
    """Best-effort shutdown of the configured metric provider."""
    global _CONFIGURED_PROVIDER
    provider = _CONFIGURED_PROVIDER
    _CONFIGURED_PROVIDER = None
    _METRICS.clear()
    if provider is None:
        return
    try:
        provider.shutdown()
    except Exception as exc:
        logger.warning("OpenTelemetry metrics shutdown failed: %s", exc)


def _configure_instruments() -> None:
    meter = metrics.get_meter(_METER_NAME)
    _METRICS.update(
        {
            "requests": meter.create_counter("nemo_retriever.ingest.requests", unit="{request}"),
            "errors": meter.create_counter("nemo_retriever.ingest.errors", unit="{error}"),
            "bytes": meter.create_counter("nemo_retriever.ingest.bytes", unit="By"),
            "documents": meter.create_counter("nemo_retriever.ingest.documents", unit="{document}"),
            "pages": meter.create_counter("nemo_retriever.ingest.pages", unit="{page}"),
            "request_duration": meter.create_histogram("nemo_retriever.ingest.request.duration", unit="s"),
        }
    )


def _add(name: str, value: int, attributes: Mapping[str, str]) -> None:
    instrument = _METRICS.get(name)
    if instrument is None or value <= 0:
        return
    try:
        instrument.add(value, attributes=dict(attributes))
    except Exception as exc:
        logger.warning("OpenTelemetry metric recording failed: %s", exc)


def _record(name: str, value: float, attributes: Mapping[str, str]) -> None:
    instrument = _METRICS.get(name)
    if instrument is None:
        return
    try:
        instrument.record(value, attributes=dict(attributes))
    except Exception as exc:
        logger.warning("OpenTelemetry metric recording failed: %s", exc)


def _cleanup_partial_metrics_setup(*, provider: Any | None, reader: Any | None, exporter: Any | None) -> None:
    for component in (provider, reader, exporter):
        if component is None:
            continue
        try:
            component.shutdown()
        except Exception:
            logger.debug("Ignoring OpenTelemetry metrics cleanup failure", exc_info=True)


def _reset_metrics_for_tests() -> None:
    """Reset metric globals and SDK state for isolated tests."""
    shutdown_metrics()
    try:
        metrics._METER_PROVIDER = None  # type: ignore[attr-defined]  # noqa: SLF001
        metrics._METER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # noqa: SLF001
    except AttributeError:
        logger.debug("OpenTelemetry metrics test reset skipped private provider state reset", exc_info=True)
