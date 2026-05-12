# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional one-shot OpenTelemetry SDK setup helper.

SDK + exporter packages live behind the ``[otel]`` extra and are imported
lazily so ``import nemo_retriever`` never pulls them in. Calling
:func:`configure` is opt-in — hosts that already configured OpenTelemetry
can skip it and the existing global providers are used as-is.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Callable, Literal, Sequence

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class OTELConfig:
    """User-facing OpenTelemetry SDK configuration.

    All fields are optional.  Standard ``OTEL_*`` environment variables
    take precedence inside the SDK itself, so values supplied here are
    fallbacks when the corresponding env var is unset.
    """

    enabled: bool = True
    service_name: str = "nemo-retriever"
    endpoint: str | None = None
    exporter: Literal["otlp", "console", "none"] = "otlp"
    sampling_ratio: float = 1.0
    auto_instrument: bool = True
    resource_attributes: dict[str, str] | None = None


def configure(config: OTELConfig | None = None) -> Callable[[], None]:
    """Install global tracer + meter providers and return a shutdown callback.

    When *config* is ``None``, defaults are used.  When ``enabled`` is
    false, the call is a no-op and the returned shutdown callback does
    nothing.  Callers should invoke the shutdown callback at process
    exit to flush in-flight spans/metrics through the exporter.
    """
    cfg = config or OTELConfig()
    if not cfg.enabled:
        return _noop_shutdown

    try:
        return _configure_sdk(cfg)
    except ImportError as exc:
        logger.warning(
            "OpenTelemetry SDK not installed (%s); pipeline will run "
            "without tracing.  Install with `pip install nemo-retriever[otel]`.",
            exc,
        )
        return _noop_shutdown


def _noop_shutdown() -> None:
    return None


_HF_HUB_URL_PATTERN = r"huggingface\.co"


def apply_otel_env_defaults(*, default_service_name: str) -> None:
    """Seed ``OTEL_*`` env vars with sensible defaults without clobbering
    values the caller has already set.

    Must run BEFORE :func:`configure` (the ``requests`` instrumentor reads
    ``OTEL_PYTHON_REQUESTS_EXCLUDED_URLS`` at instrument-time) and BEFORE
    :func:`collect_otel_env` (so Ray actors inherit the same values via
    runtime_env propagation).

    Currently sets:

    * ``OTEL_SERVICE_NAME`` — falls back to *default_service_name* when
      unset, preventing the SDK's ``"unknown_service"`` placeholder from
      surfacing in the collector.
    * ``OTEL_PYTHON_REQUESTS_EXCLUDED_URLS`` — appends a HuggingFace Hub
      regex so HEAD probes for optional config files (which legitimately
      return 404) don't show up as ``STATUS_CODE_ERROR`` spans.  Any
      caller-supplied patterns are preserved.
    """
    os.environ.setdefault("OTEL_SERVICE_NAME", default_service_name)

    existing = os.environ.get("OTEL_PYTHON_REQUESTS_EXCLUDED_URLS", "").strip()
    if _HF_HUB_URL_PATTERN not in existing:
        os.environ["OTEL_PYTHON_REQUESTS_EXCLUDED_URLS"] = (
            f"{existing},{_HF_HUB_URL_PATTERN}" if existing else _HF_HUB_URL_PATTERN
        )


def _configure_sdk(cfg: OTELConfig) -> Callable[[], None]:
    """Lazy SDK import + provider wiring; runs only when ``cfg.enabled``."""
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio

    resource_attrs: dict[str, str] = {"service.name": cfg.service_name}
    if cfg.resource_attributes:
        resource_attrs.update(cfg.resource_attributes)
    resource = Resource.create(resource_attrs)

    sampler = ParentBasedTraceIdRatio(max(0.0, min(cfg.sampling_ratio, 1.0)))
    tracer_provider = TracerProvider(resource=resource, sampler=sampler)

    span_exporter = _build_span_exporter(cfg)
    metric_reader = _build_metric_reader(cfg)

    if span_exporter is not None:
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    trace.set_tracer_provider(tracer_provider)

    if metric_reader is not None:
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    else:
        meter_provider = MeterProvider(resource=resource)
    metrics.set_meter_provider(meter_provider)

    from nemo_retriever.observability.tracer import reset_instrument_cache

    reset_instrument_cache()

    if cfg.auto_instrument:
        _install_auto_instrumentations()

    logger.info(
        "OpenTelemetry configured: service=%s exporter=%s endpoint=%s sampling=%.3f",
        cfg.service_name,
        cfg.exporter,
        cfg.endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "<env>"),
        cfg.sampling_ratio,
    )

    def _shutdown() -> None:
        try:
            tracer_provider.shutdown()
        except Exception:  # noqa: BLE001 — best effort
            logger.exception("Failed to shut down tracer provider cleanly")
        try:
            meter_provider.shutdown()
        except Exception:  # noqa: BLE001 — best effort
            logger.exception("Failed to shut down meter provider cleanly")

    return _shutdown


def _otlp_use_http(endpoint: str | None) -> bool:
    """Decide whether to use the OTLP/HTTP exporter or the gRPC one.

    OTLP supports both protocols and both use ``http://``-prefixed
    endpoints, so URL scheme alone is *not* a reliable signal.  Use:

    * ``OTEL_EXPORTER_OTLP_PROTOCOL`` env var when set
      (``grpc`` | ``http/protobuf`` | ``http/json``);
    * the canonical port as a fallback (``:4318`` → HTTP, anything else
      including ``:4317`` → gRPC);
    * gRPC otherwise (matches the SDK's own default).
    """
    proto = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "").strip().lower()
    if proto.startswith("http"):
        return True
    if proto == "grpc":
        return False
    if endpoint:
        if (
            ":4318" in endpoint
            or endpoint.rstrip("/").endswith("/v1/traces")
            or endpoint.rstrip("/").endswith("/v1/metrics")
        ):
            return True
    return False


def _build_span_exporter(cfg: OTELConfig):  # type: ignore[no-untyped-def]
    if cfg.exporter == "none":
        return None
    if cfg.exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()
    # Default: OTLP.  Endpoint resolves from cfg.endpoint, then env vars.
    endpoint = (
        cfg.endpoint
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    return _build_otlp_span_exporter(endpoint)


def _build_otlp_span_exporter(endpoint: str | None):  # type: ignore[no-untyped-def]
    if _otlp_use_http(endpoint):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPExporter

        return HTTPExporter(endpoint=endpoint) if endpoint else HTTPExporter()
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GRPCExporter

    return GRPCExporter(endpoint=endpoint) if endpoint else GRPCExporter()


def _build_metric_reader(cfg: OTELConfig):  # type: ignore[no-untyped-def]
    if cfg.exporter == "none":
        return None
    if cfg.exporter == "console":
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader

        return PeriodicExportingMetricReader(ConsoleMetricExporter())

    endpoint = (
        cfg.endpoint
        or os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )

    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    if _otlp_use_http(endpoint):
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as HTTPMetric

        return PeriodicExportingMetricReader(HTTPMetric(endpoint=endpoint) if endpoint else HTTPMetric())
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as GRPCMetric

    return PeriodicExportingMetricReader(GRPCMetric(endpoint=endpoint) if endpoint else GRPCMetric())


_AUTO_INSTRUMENTORS: Sequence[tuple[str, str]] = (
    ("opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
    ("opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
    ("opentelemetry.instrumentation.sqlite3", "SQLite3Instrumentor"),
    ("opentelemetry.instrumentation.logging", "LoggingInstrumentor"),
)


def _install_auto_instrumentations() -> None:
    """Install best-effort auto-instrumentations from the optional extra.

    Each instrumentor is independent: a missing module simply means that
    integration is skipped.  Already-installed instrumentors are no-ops.
    """
    for module_name, class_name in _AUTO_INSTRUMENTORS:
        try:
            module = __import__(module_name, fromlist=[class_name])
        except ImportError:
            continue
        instrumentor_cls = getattr(module, class_name, None)
        if instrumentor_cls is None:
            continue
        try:
            instrumentor_cls().instrument()
        except Exception:  # noqa: BLE001 — never break startup over a probe
            logger.exception("Failed to install %s.%s", module_name, class_name)
