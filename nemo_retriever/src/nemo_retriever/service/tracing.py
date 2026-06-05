# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry tracing helpers for retriever service roles."""

from __future__ import annotations

import logging
import os
import re
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
_W3C_TRACE_CONTEXT_HEADER_NAMES = frozenset({"traceparent", "tracestate"})

logger = logging.getLogger(__name__)

_DEFAULT_SERVICE_NAME = "nemo-retriever-service"
_CONFIGURED_PROVIDER: Any | None = None
_TRACE_CONTEXT_PROPAGATOR = TraceContextTextMapPropagator()
_SENSITIVE_ATTRIBUTE_TOKENS = frozenset({"authorization", "auth", "token", "password", "secret"})
_RAW_CONTENT_TOKENS = frozenset({"body", "payload", "content"})
_MEASUREMENT_TOKENS = frozenset({"length", "size", "type", "count"})


def tracing_enabled_from_env(env: Mapping[str, str] | None = None) -> bool:
    """Return whether Helm-compatible OpenTelemetry env enables tracing."""
    source = os.environ if env is None else env
    if source.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return False

    traces_exporter = source.get("OTEL_TRACES_EXPORTER", "").strip()
    endpoint = source.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    return bool(traces_exporter and traces_exporter.lower() != "none" and endpoint)


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

    if _global_tracer_provider_is_already_configured():
        return True

    provider: Any | None = None
    exporter: Any | None = None
    processor: Any | None = None
    processor_added = False

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
        processor_added = True
        trace.set_tracer_provider(provider)

        if trace.get_tracer_provider() is not provider:
            raise RuntimeError("OpenTelemetry tracer provider is already configured")

        _CONFIGURED_PROVIDER = provider
        logger.info("OpenTelemetry tracing configured: service=%s role=%s", resolved_service_name, service_role)
        return True
    except Exception as exc:
        _cleanup_partial_tracing_setup(provider=provider, processor=processor, exporter=exporter, processor_added=processor_added)
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


def inject_trace_context(carrier: MutableMapping[str, str] | None = None) -> MutableMapping[str, str]:
    """Inject W3C trace context into a clean or provided mutable carrier."""
    output: MutableMapping[str, str] = {} if carrier is None else carrier
    for key in list(output):
        if key.lower() in _W3C_TRACE_CONTEXT_HEADER_NAMES:
            del output[key]
    _TRACE_CONTEXT_PROPAGATOR.inject(output)
    return output


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


def _sanitize_span_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if attributes is None:
        return None

    return {key: value for key, value in attributes.items() if not _is_sensitive_span_attribute_name(key)}


def _global_tracer_provider_is_already_configured() -> bool:
    provider = trace.get_tracer_provider()
    proxy_provider_type = getattr(trace, "ProxyTracerProvider", None)
    if proxy_provider_type is not None:
        return not isinstance(provider, proxy_provider_type)
    return TracerProvider is not None and isinstance(provider, TracerProvider)


def _cleanup_partial_tracing_setup(*, provider: Any, processor: Any, exporter: Any, processor_added: bool) -> None:
    if not processor_added:
        _shutdown_quietly(processor)
        _shutdown_quietly(exporter)
    _shutdown_quietly(provider)


def _shutdown_quietly(resource: Any | None) -> None:
    if resource is None:
        return
    shutdown = getattr(resource, "shutdown", None)
    if shutdown is None:
        return
    try:
        shutdown()
    except Exception:
        logger.debug("Ignoring OpenTelemetry cleanup failure", exc_info=True)


def _is_sensitive_span_attribute_name(key: str) -> bool:
    tokens = _attribute_name_tokens(key)
    token_set = set(tokens)
    compact = "".join(tokens)

    if token_set & _SENSITIVE_ATTRIBUTE_TOKENS:
        return True
    if "api" in token_set and "key" in token_set:
        return True
    if "apikey" in compact:
        return True
    if compact in {"body", "payload", "content", "filebytes"}:
        return True
    if "file" in token_set and "bytes" in token_set:
        return True
    if "request" in token_set and token_set & _RAW_CONTENT_TOKENS:
        return True
    if "raw" in token_set and (token_set & _RAW_CONTENT_TOKENS or "bytes" in token_set):
        return True
    if token_set & _RAW_CONTENT_TOKENS and not token_set & _MEASUREMENT_TOKENS:
        return True
    return False


def _attribute_name_tokens(key: str) -> list[str]:
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return [token for token in re.split(r"[^A-Za-z0-9]+", camel_split.lower()) if token]
