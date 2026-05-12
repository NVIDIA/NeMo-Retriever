# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal essential tests for the OpenTelemetry observability module.

Each kept test guards a contract that has no other coverage:

* ``test_operator_span_success`` / ``test_operator_span_failure`` —
  happy path + exception path span/metric emission, including the
  ``nim.kind`` derivation and ``error.class`` label.
* ``test_run_pipeline_batch_inherits_per_page_context`` — service mode
  worker span parents under the inbound HTTP span via the W3C carrier
  captured at ``try_submit`` time.
* ``test_ray_operator_span_wrapper_links_to_driver`` — Ray batch mode
  actor span inherits the driver's ``pipeline_span``.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd
import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.observability import (
    attributes as A,
    inject_current_context,
    operator_span,
    pipeline_span,
    reset_instrument_cache,
)


# OTEL Python allows ``set_*_provider`` only once per process — install
# SDK providers a single time and reset the in-memory exporter/reader
# state between tests.
_SPAN_EXPORTER = InMemorySpanExporter()
_METRIC_READER: InMemoryMetricReader | None = None
_PROVIDERS_INSTALLED = False


def _install_session_providers() -> InMemoryMetricReader:
    global _METRIC_READER, _PROVIDERS_INSTALLED
    if _PROVIDERS_INSTALLED:
        assert _METRIC_READER is not None
        return _METRIC_READER

    tp = trace.get_tracer_provider()
    provider = tp if isinstance(tp, TracerProvider) else TracerProvider()
    if not isinstance(tp, TracerProvider):
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(_SPAN_EXPORTER))

    _METRIC_READER = InMemoryMetricReader()
    existing = metrics.get_meter_provider()
    if not isinstance(existing, MeterProvider):
        metrics.set_meter_provider(MeterProvider(metric_readers=[_METRIC_READER]))
    else:
        raise RuntimeError("A non-test MeterProvider was already installed; in-memory reader cannot attach.")
    _PROVIDERS_INSTALLED = True
    return _METRIC_READER


@pytest.fixture
def otel_capture() -> Iterator[tuple[InMemorySpanExporter, InMemoryMetricReader]]:
    reader = _install_session_providers()
    reset_instrument_cache()
    _SPAN_EXPORTER.clear()
    yield _SPAN_EXPORTER, reader


def _finished_spans(exporter: InMemorySpanExporter):
    return list(exporter.get_finished_spans())


def _metric_names(metric_reader: InMemoryMetricReader) -> set[str]:
    names: set[str] = set()
    data = metric_reader.get_metrics_data()
    if data is None:
        return names
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                names.add(metric.name)
    return names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_operator_span_success(otel_capture) -> None:
    """Happy path: operator_span emits one span + records duration histogram + processed counter."""
    span_exporter, metric_reader = otel_capture

    with operator_span(
        "FooOp",
        run_mode="inprocess",
        operator_class="OCRV2Actor",  # implies nim.kind=ocr
        operator_index=0,
        batch_size=12,
    ):
        pass

    spans = _finished_spans(span_exporter)
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "operator.FooOp"
    assert span.attributes[A.OPERATOR_NAME] == "FooOp"
    assert span.attributes[A.RUN_MODE] == "inprocess"
    assert span.attributes[A.BATCH_SIZE] == 12
    # nim.kind derived from operator_class via the substring lookup —
    # protects the canonical role mapping used by Zipkin filters.
    assert span.attributes[A.NIM_KIND] == "ocr"

    names = _metric_names(metric_reader)
    assert "nemo_retriever.operator.duration_ms" in names
    assert "nemo_retriever.documents.processed" in names


def test_operator_span_failure(otel_capture) -> None:
    """Exception path: span records the exception + ``documents.failed``
    counter fires with ``error.class`` attribute equal to the exception
    type's ``__name__``."""
    span_exporter, metric_reader = otel_capture

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        with operator_span("BarOp", run_mode="service", batch_size=4):
            raise _Boom("boom")

    spans = _finished_spans(span_exporter)
    assert len(spans) == 1
    assert any(event.name == "exception" for event in spans[0].events)

    data = metric_reader.get_metrics_data()
    saw_failed_with_error_class = False
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != "nemo_retriever.documents.failed":
                    continue
                for point in metric.data.data_points:
                    if point.attributes.get(A.ERROR_CLASS) == "_Boom":
                        saw_failed_with_error_class = True
    assert saw_failed_with_error_class


def test_run_pipeline_batch_inherits_per_page_context(otel_capture, monkeypatch) -> None:
    """Service-mode worker batch span must parent under the inbound HTTP
    span via per-page ``trace_context`` carriers captured at submit time.
    Without this the trace tree splits across process boundaries.
    """
    span_exporter, _ = otel_capture

    from nemo_retriever.service.processing import pool as pool_mod

    class _FakeRepo:
        def update_document_status(self, *_a, **_kw):
            return None

        def get_document(self, *_a, **_kw):
            return None

        def get_job(self, *_a, **_kw):
            return None

    monkeypatch.setattr(pool_mod, "DatabaseEngine", lambda *_a, **_kw: object())
    monkeypatch.setattr(pool_mod, "Repository", lambda _e: _FakeRepo())
    monkeypatch.setattr(pool_mod, "_worker_chain", [])

    with pipeline_span("service") as parent:
        parent_trace_id = parent.get_span_context().trace_id
        carrier = inject_current_context()

    descriptors = [
        {
            "document_id": f"doc-{i}",
            "content_sha256": "h",
            "file_bytes": b"x",
            "filename": "x.pdf",
            "job_id": "job-1",
            "page_number": i,
            "spool_path": None,
            "trace_context": carrier,
        }
        for i in range(3)
    ]
    pool_mod._run_pipeline_batch(descriptors, db_path="/tmp/unused.db")

    worker_spans = [s for s in _finished_spans(span_exporter) if s.name == "pool.run_pipeline_batch"]
    assert len(worker_spans) == 1
    span = worker_spans[0]
    assert span.context.trace_id == parent_trace_id
    # Uniform batch → singular correlation attrs populated for Zipkin filters.
    assert span.attributes[A.JOB_ID] == "job-1"
    assert span.attributes[A.BATCH_DOCUMENT_COUNT] == 3


def test_ray_operator_span_wrapper_links_to_driver(otel_capture) -> None:
    """Ray-actor-side wrapper must produce an ``operator.<name>`` span
    parented to the driver's ``pipeline_span`` via the W3C carrier — the
    whole point of the Ray batch instrumentation."""
    span_exporter, _ = otel_capture

    from nemo_retriever.observability import RayOperatorSpanWrapper

    class _AddOneOp(AbstractOperator):
        def preprocess(self, d, **kw):
            return d

        def process(self, d, **kw):
            d = d.copy()
            d["v"] = d["v"] + 1
            return d

        def postprocess(self, d, **kw):
            return d

    with pipeline_span("batch") as parent:
        parent_trace_id = parent.get_span_context().trace_id
        carrier = inject_current_context()

    wrapper = RayOperatorSpanWrapper(
        _otel_operator_class=_AddOneOp,
        _otel_node_name="AddOne",
        _otel_node_index=2,
        _otel_parent_carrier=carrier,
    )
    result = wrapper(pd.DataFrame({"v": [10, 20, 30]}))
    assert list(result["v"]) == [11, 21, 31]

    op_spans = [s for s in _finished_spans(span_exporter) if s.name == "operator.AddOne"]
    assert len(op_spans) == 1
    span = op_spans[0]
    assert span.context.trace_id == parent_trace_id
    assert span.attributes[A.RUN_MODE] == "batch"
    assert span.attributes[A.BATCH_SIZE] == 3
