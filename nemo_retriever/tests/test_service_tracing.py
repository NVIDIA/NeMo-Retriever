# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for service-mode OpenTelemetry tracing helpers."""

from __future__ import annotations

import re
from typing import Any

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import ServiceConfig
from nemo_retriever.service.tracing import (
    _reset_tracing_for_tests,
    configure_tracing,
    current_trace_id_hex,
    extract_trace_context,
    inject_trace_context,
    span_attributes,
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
        ({"OTEL_TRACES_EXPORTER": "otlp"}, True),
        ({"OTEL_TRACES_EXPORTER": "", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"}, False),
        ({"OTEL_TRACES_EXPORTER": "none", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"}, False),
        ({"OTEL_TRACES_EXPORTER": "otlp", "OTEL_EXPORTER_OTLP_ENDPOINT": ""}, True),
        (
            {
                "OTEL_TRACES_EXPORTER": "otlp",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317",
                "OTEL_SDK_DISABLED": "true",
            },
            False,
        ),
        ({"OTEL_TRACES_EXPORTER": "otlp", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"}, True),
    ],
)
def test_tracing_enabled_from_env_requires_otlp_exporter(env: dict[str, str], expected: bool) -> None:
    assert tracing_enabled_from_env(env) is expected


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


def test_trace_context_inject_extract_round_trips_traceparent(
    monkeypatch: pytest.MonkeyPatch, exported_spans: list[Any]
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    assert configure_tracing(service_role="standalone") is True

    with start_span("service.parent"):
        parent_trace_id = current_trace_id_hex()
        carrier = inject_trace_context({"x-unrelated": "keep-out"})

    assert set(carrier) == {"traceparent"}

    extracted = extract_trace_context(carrier)
    with start_span("service.child", context=extracted):
        assert current_trace_id_hex() == parent_trace_id


def test_span_attributes_drop_sensitive_keys(monkeypatch: pytest.MonkeyPatch, exported_spans: list[Any]) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    assert configure_tracing(service_role="standalone") is True

    raw_attributes = {
        "Authorization": "Bearer abc",
        "x_api_key": "abc",
        "apiKey": "abc",
        "access_token": "abc",
        "password": "abc",
        "client_secret": "abc",
        "request_body": "{}",
        "payload": b"raw",
        "file_bytes": b"raw",
        "content": "sensitive",
        "safe.status": "ok",
        "document_count": 2,
    }
    assert span_attributes(raw_attributes) == {"safe.status": "ok", "document_count": 2}

    with start_span("service.sanitize", attributes=raw_attributes):
        pass

    attrs = dict(exported_spans[-1].attributes)
    assert attrs == {"safe.status": "ok", "document_count": 2}


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
