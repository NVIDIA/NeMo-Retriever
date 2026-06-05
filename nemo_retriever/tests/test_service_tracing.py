# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for service-mode OpenTelemetry tracing helpers."""

from __future__ import annotations

import re
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import ServiceConfig
from nemo_retriever.service.tracing import (
    _reset_tracing_for_tests,
    configure_tracing,
    current_trace_id_hex,
    extract_trace_context,
    inject_trace_context,
    start_span,
    tracing_enabled_from_env,
)


class _CollectingExporter:
    def __init__(self, exported: list[Any]) -> None:
        self._exported = exported

    def export(self, spans: Any) -> SpanExportResult:
        self._exported.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


@pytest.fixture(autouse=True)
def reset_tracing() -> None:
    _reset_tracing_for_tests()
    yield
    _reset_tracing_for_tests()


@pytest.fixture
def exported_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    exported: list[Any] = []
    monkeypatch.setattr(
        "nemo_retriever.service.tracing.OTLPSpanExporter",
        lambda *args, **kwargs: _CollectingExporter(exported),
    )
    monkeypatch.setattr("nemo_retriever.service.tracing.BatchSpanProcessor", SimpleSpanProcessor)
    return exported


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"OTEL_TRACES_EXPORTER": "otlp"}, False),
        ({"OTEL_TRACES_EXPORTER": "OTLP"}, False),
        ({"OTEL_TRACES_EXPORTER": "", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"}, False),
        ({"OTEL_TRACES_EXPORTER": "none", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"}, False),
        ({"OTEL_TRACES_EXPORTER": "otlp", "OTEL_EXPORTER_OTLP_ENDPOINT": ""}, False),
        ({"OTEL_TRACES_EXPORTER": "otlp", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317", "OTEL_SDK_DISABLED": "true"}, False),
        ({"OTEL_TRACES_EXPORTER": "otlp", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"}, True),
        ({"OTEL_TRACES_EXPORTER": "OTLP", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"}, True),
        ({"OTEL_TRACES_EXPORTER": "jaeger", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"}, False),
        ({"OTEL_TRACES_EXPORTER": "zipkin", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"}, False),
        ({"OTEL_TRACES_EXPORTER": "console", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"}, False),
        ({"OTEL_TRACES_EXPORTER": "custom", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"}, False),
    ],
)
def test_tracing_enabled_from_env_requires_exporter_and_endpoint(env: dict[str, str], expected: bool) -> None:
    assert tracing_enabled_from_env(env) is expected


def test_tracing_helper_keeps_span_attribute_sanitizer_private() -> None:
    import nemo_retriever.service.tracing as tracing_module

    assert not hasattr(tracing_module, "span_attributes")


def test_start_span_is_noop_when_tracer_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("tracer unavailable")

    monkeypatch.setattr("nemo_retriever.service.tracing.get_tracer", _raise)

    with start_span("service.unavailable"):
        assert current_trace_id_hex() is None


def test_start_span_is_noop_when_context_enter_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingSpanContext:
        def __enter__(self) -> Any:
            raise RuntimeError("span enter failed")

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
            raise AssertionError("span context should not be entered")

    class _Tracer:
        def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
            return _FailingSpanContext()

    monkeypatch.setattr("nemo_retriever.service.tracing.get_tracer", lambda: _Tracer())

    with start_span("service.enter-failure"):
        assert current_trace_id_hex() is None


def test_start_span_does_not_swallow_user_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SpanContext:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
            return False

    class _Tracer:
        def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
            return _SpanContext()

    monkeypatch.setattr("nemo_retriever.service.tracing.get_tracer", lambda: _Tracer())

    with pytest.raises(RuntimeError, match="application failure"):
        with start_span("service.user-failure"):
            raise RuntimeError("application failure")


def test_configure_tracing_creates_spans_with_hex_trace_id(
    monkeypatch: pytest.MonkeyPatch, exported_spans: list[Any]
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")

    assert configure_tracing(service_role="standalone") is True

    with start_span("service.unit"):
        trace_id = current_trace_id_hex()

    assert trace_id is not None
    assert re.fullmatch(r"[0-9a-f]{32}", trace_id)
    assert exported_spans


def test_configure_tracing_treats_existing_global_provider_as_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    external_provider = TracerProvider()
    trace.set_tracer_provider(external_provider)
    monkeypatch.setattr(
        "nemo_retriever.service.tracing.OTLPSpanExporter",
        lambda *args, **kwargs: pytest.fail("configure_tracing should not construct an exporter"),
    )

    try:
        assert configure_tracing(service_role="standalone") is True
        assert configure_tracing(service_role="standalone") is True
        assert trace.get_tracer_provider() is external_provider
    finally:
        external_provider.shutdown()


def test_configure_tracing_cleans_up_partial_setup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    cleanup_calls: list[str] = []

    class _FailingProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

        def add_span_processor(self, processor: Any) -> None:
            raise RuntimeError("processor install failed")

        def shutdown(self) -> None:
            cleanup_calls.append("provider")

    class _FakeExporter:
        def shutdown(self) -> None:
            cleanup_calls.append("exporter")

    class _FakeProcessor:
        def __init__(self, exporter: Any) -> None:
            self.exporter = exporter

        def shutdown(self) -> None:
            cleanup_calls.append("processor")

    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    monkeypatch.setattr("nemo_retriever.service.tracing.TracerProvider", _FailingProvider)
    monkeypatch.setattr("nemo_retriever.service.tracing.OTLPSpanExporter", _FakeExporter)
    monkeypatch.setattr("nemo_retriever.service.tracing.BatchSpanProcessor", _FakeProcessor)

    assert configure_tracing(service_role="standalone") is False
    assert cleanup_calls == ["processor", "exporter", "provider"]


def test_trace_context_inject_extract_round_trips_traceparent(
    monkeypatch: pytest.MonkeyPatch, exported_spans: list[Any]
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    assert configure_tracing(service_role="standalone") is True

    with start_span("service.parent"):
        parent_trace_id = current_trace_id_hex()
        carrier = inject_trace_context()

    assert "traceparent" in carrier
    assert set(carrier).issubset({"traceparent", "tracestate"})

    extracted = extract_trace_context(carrier)
    with start_span("service.child", context=extracted):
        assert current_trace_id_hex() == parent_trace_id


def test_trace_context_inject_mutates_existing_carrier_preserving_unrelated_headers(
    monkeypatch: pytest.MonkeyPatch, exported_spans: list[Any]
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    assert configure_tracing(service_role="standalone") is True

    existing_carrier = {
        "traceparent": "00-00000000000000000000000000000001-0000000000000001-01",
        "x-required": "keep",
    }
    with start_span("service.parent"):
        carrier = inject_trace_context(existing_carrier)

    assert carrier is existing_carrier
    assert carrier["x-required"] == "keep"
    assert "traceparent" in carrier
    assert carrier["traceparent"] != "00-00000000000000000000000000000001-0000000000000001-01"


def test_trace_context_inject_removes_mixed_case_w3c_keys(
    monkeypatch: pytest.MonkeyPatch, exported_spans: list[Any]
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    assert configure_tracing(service_role="standalone") is True

    existing_carrier = {
        "Traceparent": "00-00000000000000000000000000000001-0000000000000001-01",
        "TRACEPARENT": "00-00000000000000000000000000000002-0000000000000002-01",
        "Tracestate": "vendor=old",
        "x-required": "keep",
    }
    with start_span("service.parent"):
        carrier = inject_trace_context(existing_carrier)

    assert carrier is existing_carrier
    assert carrier["x-required"] == "keep"
    assert "traceparent" in carrier
    assert "Traceparent" not in carrier
    assert "TRACEPARENT" not in carrier
    assert "Tracestate" not in carrier
    assert {key for key in carrier if key.lower() in {"traceparent", "tracestate"}} <= {"traceparent", "tracestate"}


def test_start_span_drops_sensitive_keys_and_keeps_benign_metadata(
    monkeypatch: pytest.MonkeyPatch, exported_spans: list[Any]
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    assert configure_tracing(service_role="standalone") is True

    raw_attributes = {
        "Authorization": "Bearer abc",
        "auth": "abc",
        "auth.header": "Bearer abc",
        "credential": "abc",
        "credentials": {"token": "abc"},
        "user_credentials": "abc",
        "x-api-key": "abc",
        "API key": "abc",
        "apiKey": "abc",
        "access_token": "abc",
        "password": "abc",
        "client_secret": "abc",
        "request_body": "{}",
        "body": "{}",
        "payload": b"raw",
        "response_payload": b"raw",
        "payload_text": "sensitive",
        "file_bytes": b"raw",
        "content": "sensitive",
        "document_content": "sensitive",
        "body_text": "sensitive",
        "raw_content": "sensitive",
        "content_type": "application/json",
        "content_encoding": "gzip",
        "content_language": "en",
        "content_length": 123,
        "payload_size": 456,
        "body_length": 789,
        "safe.status": "ok",
        "document_count": 2,
    }

    with start_span("service.sanitize", attributes=raw_attributes):
        pass

    attrs = dict(exported_spans[-1].attributes)
    assert attrs == {
        "content_type": "application/json",
        "content_encoding": "gzip",
        "content_language": "en",
        "content_length": 123,
        "payload_size": 456,
        "body_length": 789,
        "safe.status": "ok",
        "document_count": 2,
    }


def test_create_app_configures_tracing_for_service_role(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def _configure_tracing(*, service_role: str, service_name: str | None = None) -> bool:
        calls.append((service_role, service_name))
        return True

    monkeypatch.setattr("nemo_retriever.service.tracing.configure_tracing", _configure_tracing)
    monkeypatch.setattr("nemo_retriever.service.app._configure_logging", lambda config: None)
    monkeypatch.setattr("nemo_retriever.service.app._apply_resource_limits", lambda config: None)

    app = create_app(ServiceConfig(mode="gateway"))

    assert app.state.config.mode == "gateway"
    assert calls == [("gateway", None)]
