# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracing coverage for process-isolated pipeline execution."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from nemo_retriever.service import tracing
from nemo_retriever.service.services import pipeline_executor


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


class _FakeIngestor:
    def ingest(self) -> pd.DataFrame:
        return pd.DataFrame([{"document_id": "doc-1", "text": "chunk"}])


@pytest.fixture
def exported_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    exported: list[Any] = []
    tracing._reset_tracing_for_tests()
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setattr(
        "nemo_retriever.service.tracing.OTLPSpanExporter",
        lambda *args, **kwargs: _CollectingExporter(exported),
    )
    monkeypatch.setattr("nemo_retriever.service.tracing.BatchSpanProcessor", SimpleSpanProcessor)
    try:
        yield exported
    finally:
        tracing._reset_tracing_for_tests()


def test_run_pipeline_in_process_links_child_span_to_parent_trace(
    monkeypatch: pytest.MonkeyPatch,
    exported_spans: list[Any],
) -> None:
    def _fake_build_graph_ingestor_from_spec(*args: Any, **kwargs: Any) -> tuple[_FakeIngestor, str, bool]:
        return _FakeIngestor(), "pdf", False

    monkeypatch.setattr(
        pipeline_executor,
        "_build_graph_ingestor_from_spec",
        _fake_build_graph_ingestor_from_spec,
    )

    tracing.configure_tracing(service_role="parent-test")
    with tracing.start_span("parent.request"):
        parent_trace_id = tracing.current_trace_id_hex()
        carrier = dict(tracing.inject_trace_context())

        row_count, result_data, _elapsed = pipeline_executor._run_pipeline_in_process(
            "contract.pdf",
            b"%PDF-1.4\n",
            {},
            None,
            trace_context=carrier,
            pool_label="Realtime",
            service_role="standalone",
        )

    assert parent_trace_id is not None
    assert row_count == 1
    assert result_data

    pipeline_span = next(span for span in exported_spans if span.name == "pipeline.ingest")
    assert f"{pipeline_span.context.trace_id:032x}" == parent_trace_id
    assert pipeline_span.attributes["pool"] == "realtime"
    assert pipeline_span.attributes["document.filename"] == "contract.pdf"


def test_run_pipeline_in_process_continues_when_trace_extraction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_build_graph_ingestor_from_spec(*args: Any, **kwargs: Any) -> tuple[_FakeIngestor, str, bool]:
        return _FakeIngestor(), "pdf", False

    def _raise_extract(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("bad carrier")

    monkeypatch.setattr(
        pipeline_executor,
        "_build_graph_ingestor_from_spec",
        _fake_build_graph_ingestor_from_spec,
    )
    monkeypatch.setattr(tracing, "extract_trace_context", _raise_extract)

    row_count, result_data, _elapsed = pipeline_executor._run_pipeline_in_process(
        "contract.pdf",
        b"%PDF-1.4\n",
        {},
        None,
        trace_context={"traceparent": "invalid"},
        pool_label="Realtime",
        service_role="standalone",
    )

    assert row_count == 1
    assert result_data == [{"document_id": "doc-1", "text": "chunk"}]


def test_make_work_fn_continues_when_trace_capture_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Params:
        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {}

    class _WorkItem:
        id = "item-1"
        filename = "contract.pdf"
        payload = b"%PDF-1.4\n"
        pipeline_spec = None

    def _fake_process_pool_executor(*args: Any, **kwargs: Any) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=1)

    def _raise_inject(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("inject failed")

    def _fake_run_pipeline_in_process(*args: Any, **kwargs: Any) -> tuple[int, list[dict[str, Any]], float]:
        return 1, [{"document_id": "doc-1", "text": "chunk"}], 0.01

    monkeypatch.setattr(pipeline_executor, "build_extract_params", lambda nim: _Params())
    monkeypatch.setattr(pipeline_executor, "build_embed_params", lambda nim: None)
    monkeypatch.setattr(pipeline_executor, "build_caption_params", lambda nim: None)
    monkeypatch.setattr(pipeline_executor, "build_asr_params", lambda nim: None)
    monkeypatch.setattr(pipeline_executor, "ProcessPoolExecutor", _fake_process_pool_executor)
    monkeypatch.setattr(pipeline_executor, "_run_pipeline_in_process", _fake_run_pipeline_in_process)
    monkeypatch.setattr(tracing, "inject_trace_context", _raise_inject)

    config = SimpleNamespace(
        nim_endpoints=SimpleNamespace(model_dump=lambda *args, **kwargs: {}),
        vectordb=SimpleNamespace(enabled=False, vectordb_url=None),
        pipeline=SimpleNamespace(
            realtime_workers=1,
            batch_workers=1,
            realtime_queue_size=1,
            batch_queue_size=1,
        ),
        mode="standalone",
    )

    work = pipeline_executor._make_work_fn(config, label="Realtime")
    try:
        row_count, result_data = asyncio.run(work(_WorkItem()))
    finally:
        pipeline_executor.shutdown_process_executors()

    assert row_count == 1
    assert result_data == [{"document_id": "doc-1", "text": "chunk"}]
