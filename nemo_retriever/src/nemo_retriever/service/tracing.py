# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry tracing helpers for retriever service roles."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, MutableMapping

from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
except Exception:  # pragma: no cover - exercised through configure_tracing failure handling.
    OTLPSpanExporter = None  # type: ignore[assignment]

try:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:  # pragma: no cover - exercised through configure_tracing failure handling.
    BatchSpanProcessor = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]

TRACE_ID_HEADER = "x-trace-id"

logger = logging.getLogger(__name__)

_DEFAULT_SERVICE_NAME = "nemo-retriever-service"
_CONFIGURED_PROVIDER: Any | None = None
_TRACE_CONTEXT_PROPAGATOR = TraceContextTextMapPropagator()
_SENSITIVE_ATTRIBUTE_NAME_PARTS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "token",
        "password",
        "secret",
        "body",
        "payload",
        "file_bytes",
        "content",
    }
)


def tracing_enabled_from_env(env: Mapping[str, str] | None = None) -> bool:
    """Return whether Helm-compatible OpenTelemetry env enables tracing."""
    source = os.environ if env is None else env
    if source.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return False

    return source.get("OTEL_TRACES_EXPORTER", "").strip().lower() == "otlp"


def configure_tracing(*, service_role: str, service_name: str | None = None) -> bool:
    """Configure process-wide OTLP tracing when enabled by environment.

    Tracing is observability-only. Any setup failure is logged and reported as
    ``False`` without preventing service startup.
    """
    global _CONFIGURED_PROVIDER

    if _CONFIGURED_PROVIDER is not None:
        return True

    if not tracing_enabled_from_env():
        return False

    try:
        if OTLPSpanExporter is None or BatchSpanProcessor is None or Resource is None or TracerProvider is None:
            raise RuntimeError("OpenTelemetry SDK/exporter packages are not importable")

        resolved_service_name = (service_name or os.environ.get("OTEL_SERVICE_NAME") or _DEFAULT_SERVICE_NAME).strip()
        if not resolved_service_name:
            resolved_service_name = _DEFAULT_SERVICE_NAME

        resource = Resource.create({"service.name": resolved_service_name, "service.role": service_role})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter()
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        if trace.get_tracer_provider() is not provider:
            provider.shutdown()
            raise RuntimeError("OpenTelemetry tracer provider is already configured")

        _CONFIGURED_PROVIDER = provider
        logger.info("OpenTelemetry tracing configured: service=%s role=%s", resolved_service_name, service_role)
        return True
    except Exception as exc:
        logger.warning("OpenTelemetry tracing setup failed: %s", exc)
        return False


def get_tracer(name: str = "nemo_retriever.service") -> Any:
    """Return a tracer for service instrumentation."""
    return trace.get_tracer(name)


def start_span(
    name: str,
    *,
    kind: Any | None = None,
    context: Any | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Any:
    """Start a current span after removing sensitive attributes."""
    kwargs: dict[str, Any] = {}
    if kind is not None:
        kwargs["kind"] = kind
    if context is not None:
        kwargs["context"] = context
    sanitized_attributes = _sanitize_span_attributes(attributes)
    if sanitized_attributes is not None:
        kwargs["attributes"] = sanitized_attributes
    return get_tracer().start_as_current_span(name, **kwargs)


def current_trace_id_hex() -> str | None:
    """Return the current valid trace id as 32 lowercase hex chars."""
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return None
    return f"{context.trace_id:032x}"


def inject_trace_context(carrier: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Inject W3C trace context into a clean plain dict carrier."""
    output: MutableMapping[str, str] = carrier if carrier is not None else {}
    output.clear()
    _TRACE_CONTEXT_PROPAGATOR.inject(output)
    return dict(output)


def extract_trace_context(carrier: Mapping[str, str] | None) -> Any:
    """Extract W3C trace context from a carrier mapping."""
    return _TRACE_CONTEXT_PROPAGATOR.extract(dict(carrier or {}))


def force_flush(timeout_millis: int = 1000) -> None:
    """Best-effort flush of configured spans."""
    if _CONFIGURED_PROVIDER is None:
        return
    try:
        _CONFIGURED_PROVIDER.force_flush(timeout_millis=timeout_millis)
    except Exception as exc:
        logger.warning("OpenTelemetry tracing flush failed: %s", exc)


def _reset_tracing_for_tests() -> None:
    """Reset tracing globals so tests can configure providers repeatedly."""
    global _CONFIGURED_PROVIDER

    provider = _CONFIGURED_PROVIDER
    _CONFIGURED_PROVIDER = None
    if provider is not None:
        try:
            provider.shutdown()
        except Exception:
            logger.debug("Ignoring OpenTelemetry provider shutdown failure during test reset", exc_info=True)

    try:
        trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]  # noqa: SLF001
        trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]  # noqa: SLF001
    except AttributeError:
        logger.debug("OpenTelemetry test reset skipped private provider state reset", exc_info=True)


def span_attributes(attributes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return span attributes after filtering credentials and request bodies."""
    if attributes is None:
        return {}

    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        if _is_sensitive_attribute(key, value):
            continue
        sanitized[key] = value
    return sanitized


def span_attributes(attributes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return span attributes after filtering credentials and request bodies."""
    if attributes is None:
        return {}

    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        if _is_sensitive_attribute(key, value):
            continue
        sanitized[key] = value
    return sanitized


def _sanitize_span_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if attributes is None:
        return None
    return span_attributes(attributes)


def _is_sensitive_attribute(key: str, value: Any) -> bool:
    lowered = key.lower()
    normalized = lowered.replace("-", "_").replace(".", "_").replace(" ", "_")
    compact = normalized.replace("_", "")
    if any(part in lowered or part in normalized or part in compact for part in _SENSITIVE_ATTRIBUTE_NAME_PARTS):
        return True
    if isinstance(value, bytes):
        return True
    if isinstance(value, str):
        value_compact = value.lower().replace("-", "_").replace(".", "_").replace(" ", "_").replace("_", "")
        return "bearer" in value_compact or any(part in value_compact for part in _SENSITIVE_ATTRIBUTE_NAME_PARTS)
    return False
