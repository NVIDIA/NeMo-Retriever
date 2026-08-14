# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for service-mode OpenTelemetry metric instrumentation."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from nemo_retriever.service import metrics_otel


class _Exporter:
    def shutdown(self) -> None:
        return None


@pytest.fixture(autouse=True)
def reset_metrics() -> None:
    metrics_otel._reset_metrics_for_tests()
    yield
    metrics_otel._reset_metrics_for_tests()


def _metric_points(reader: InMemoryMetricReader) -> dict[str, list[Any]]:
    data = reader.get_metrics_data()
    assert data is not None
    return {
        metric.name: list(metric.data.data_points)
        for resource_metric in data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
    }


def test_metrics_enabled_from_env_requires_otlp_exporter_and_endpoint() -> None:
    assert not metrics_otel.metrics_enabled_from_env({})
    assert not metrics_otel.metrics_enabled_from_env({"OTEL_METRICS_EXPORTER": "otlp"})
    assert not metrics_otel.metrics_enabled_from_env({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"})
    assert metrics_otel.metrics_enabled_from_env(
        {"OTEL_METRICS_EXPORTER": "OTLP", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel:4317"}
    )


def test_configured_metrics_export_ingest_core_instruments(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = InMemoryMetricReader()
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setattr(metrics_otel, "OTLPMetricExporter", _Exporter)
    monkeypatch.setattr(metrics_otel, "PeriodicExportingMetricReader", lambda exporter: reader)

    assert metrics_otel.configure_metrics(service_role="standalone", service_name="metrics-test")
    metrics_otel.record_ingest_request(
        role="standalone", endpoint="/v1/ingest/job/document", status="2xx", duration_s=0.25
    )
    metrics_otel.record_ingest_request(
        role="standalone", endpoint="/v1/ingest/job/document", status="4xx", duration_s=0.1
    )
    metrics_otel.record_ingest_accepted(
        role="standalone", endpoint="/v1/ingest/job/document", file_size=42, is_page=False
    )
    metrics_otel.record_ingest_accepted(
        role="standalone", endpoint="/v1/ingest/job/page", file_size=7, is_page=True
    )

    points = _metric_points(reader)
    assert sum(point.value for point in points["nemo_retriever.ingest.requests"]) == 2
    assert sum(point.value for point in points["nemo_retriever.ingest.errors"]) == 1
    assert sum(point.value for point in points["nemo_retriever.ingest.bytes"]) == 49
    assert points["nemo_retriever.ingest.documents"][0].value == 1
    assert points["nemo_retriever.ingest.pages"][0].value == 1
    assert sum(point.count for point in points["nemo_retriever.ingest.request.duration"]) == 2


def test_disabled_metrics_do_not_create_instruments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")

    assert not metrics_otel.configure_metrics(service_role="standalone")
    metrics_otel.record_ingest_accepted(role="standalone", endpoint="/v1/ingest/job/document", file_size=1, is_page=False)
    assert not metrics_otel._METRICS
