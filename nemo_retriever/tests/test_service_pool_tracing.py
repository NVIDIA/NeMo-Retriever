# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenTelemetry context propagation through pipeline pools."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from nemo_retriever.service import tracing
from nemo_retriever.service.services.pipeline_pool import WorkItem, _Pool


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


def _run(coro: Any) -> Any:
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def reset_tracing() -> None:
    tracing._reset_tracing_for_tests()
    yield
    tracing._reset_tracing_for_tests()


@pytest.fixture
def exported_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    exported: list[Any] = []
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    monkeypatch.setattr(
        "nemo_retriever.service.tracing.OTLPSpanExporter",
        lambda *args, **kwargs: _CollectingExporter(exported),
    )
    monkeypatch.setattr("nemo_retriever.service.tracing.BatchSpanProcessor", SimpleSpanProcessor)
    assert tracing.configure_tracing(service_role="standalone") is True
    return exported


def test_pool_process_span_uses_work_item_trace_context(exported_spans: list[Any]) -> None:
    async def body() -> None:
        done = asyncio.Event()

        async def work(_item: WorkItem) -> int:
            done.set()
            return 1

        with tracing.start_span("test.parent"):
            parent_trace_id = tracing.current_trace_id_hex()
            carrier = dict(tracing.inject_trace_context())

        assert parent_trace_id is not None

        pool = _Pool(name="rt-trace", num_workers=1, max_queue_size=4, work_fn=work)
        pool.start()
        try:
            assert await pool.submit(
                WorkItem(
                    id="doc-1",
                    job_id="job-1",
                    trace_context=carrier,
                )
            )
            await asyncio.wait_for(done.wait(), timeout=2.0)
            await asyncio.sleep(0.05)
        finally:
            await pool.shutdown()

        spans_by_name = {span.name: span for span in exported_spans}
        assert "pool.rt-trace.process" in spans_by_name
        pool_span = spans_by_name["pool.rt-trace.process"]
        assert f"{pool_span.context.trace_id:032x}" == parent_trace_id

        attrs = dict(pool_span.attributes)
        assert attrs["pool"] == "rt-trace"
        assert attrs["document.id"] == "doc-1"
        assert attrs["job.id"] == "job-1"
        assert attrs["queue.wait_ms"] >= 0.0

    _run(body())


def test_pool_processes_item_when_trace_context_extraction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_carrier: dict[str, str]) -> Any:
        raise RuntimeError("propagator unavailable")

    monkeypatch.setattr("nemo_retriever.service.tracing.extract_trace_context", _raise)

    async def body() -> None:
        done = asyncio.Event()

        async def work(_item: WorkItem) -> int:
            done.set()
            return 1

        pool = _Pool(name="rt-trace-fallback", num_workers=1, max_queue_size=4, work_fn=work)
        pool.start()
        try:
            assert await pool.submit(
                WorkItem(
                    id="doc-1",
                    job_id="job-1",
                    trace_context={"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"},
                )
            )
            await asyncio.wait_for(done.wait(), timeout=2.0)
        finally:
            await pool.shutdown()

    _run(body())
