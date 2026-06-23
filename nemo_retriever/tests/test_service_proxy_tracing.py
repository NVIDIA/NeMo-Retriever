# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for gateway proxy W3C trace-context propagation."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi import Request
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from nemo_retriever.service.config import GatewayConfig
from nemo_retriever.service.services.pipeline_pool import PoolType
from nemo_retriever.service.services import proxy as proxy_module
from nemo_retriever.service.services.proxy import GatewayProxy
from nemo_retriever.service.tracing import (
    _reset_tracing_for_tests,
    configure_tracing,
    current_trace_id_hex,
    start_span,
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


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def request(self, *, method: str, url: str, content: bytes, headers: dict[str, str]) -> httpx.Response:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "content": content,
                "headers": dict(headers),
            }
        )
        return httpx.Response(202, content=b'{"ok": true}', headers={"content-type": "application/json"})

    async def get(self, path: str, *, headers: dict[str, str]) -> httpx.Response:
        self.calls.append(
            {
                "method": "GET",
                "url": path,
                "headers": dict(headers),
            }
        )
        return httpx.Response(200, content=b'{"config": true}', headers={"content-type": "application/json"})


@pytest.fixture(autouse=True)
def reset_tracing() -> None:
    _reset_tracing_for_tests()
    yield
    _reset_tracing_for_tests()


@pytest.fixture
def configured_tracing(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    exported: list[Any] = []
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setattr(
        "nemo_retriever.service.tracing.OTLPSpanExporter",
        lambda *args, **kwargs: _CollectingExporter(exported),
    )
    monkeypatch.setattr("nemo_retriever.service.tracing.BatchSpanProcessor", SimpleSpanProcessor)

    assert configure_tracing(service_role="gateway") is True
    return exported


def test_forward_injects_traceparent_and_preserves_forwarding_inputs(configured_tracing: list[Any]) -> None:
    backend = _RecordingClient()
    proxy = _make_proxy(batch=backend)
    body = b'{"document": "raw payload"}'
    request = _make_request(
        method="POST",
        path="/v1/ingest/document",
        headers={
            "host": "gateway.local",
            "transfer-encoding": "chunked",
            "content-type": "application/json",
            "traceparent": "00-00000000000000000000000000000001-0000000000000001-01",
            "x-request-id": "keep-me",
        },
        body=body,
    )

    async def _forward_under_span() -> str:
        with start_span("proxy.forward.parent"):
            trace_id = current_trace_id_hex()
            assert trace_id is not None
            response = await proxy.forward(
                request,
                PoolType.BATCH,
                extra_headers={"x-extra": "extra-value"},
            )
            assert response.status_code == 202
            return trace_id

    trace_id = asyncio.run(_forward_under_span())

    assert len(backend.calls) == 1
    call = backend.calls[0]
    headers = call["headers"]
    assert call["method"] == "POST"
    assert call["url"] == "/v1/ingest/document"
    assert call["content"] == body
    assert headers["content-type"] == "application/json"
    assert headers["x-request-id"] == "keep-me"
    assert headers["x-extra"] == "extra-value"
    assert "host" not in {key.lower() for key in headers}
    assert "transfer-encoding" not in {key.lower() for key in headers}
    assert headers["traceparent"].split("-")[1] == trace_id
    assert headers["traceparent"] != "00-00000000000000000000000000000001-0000000000000001-01"


def test_forward_get_injects_traceparent_before_backend_get(configured_tracing: list[Any]) -> None:
    backend = _RecordingClient()
    proxy = _make_proxy(realtime=backend)
    request = _make_request(
        method="GET",
        path="/v1/ingest/pipeline-config",
        headers={
            "host": "gateway.local",
            "Traceparent": "00-00000000000000000000000000000002-0000000000000002-01",
            "x-request-id": "keep-me",
        },
    )

    async def _forward_under_span() -> str:
        with start_span("proxy.forward_get.parent"):
            trace_id = current_trace_id_hex()
            assert trace_id is not None
            response = await proxy.forward_get(request, PoolType.REALTIME, "/v1/ingest/pipeline-config")
            assert response.status_code == 200
            return trace_id

    trace_id = asyncio.run(_forward_under_span())

    assert len(backend.calls) == 1
    call = backend.calls[0]
    headers = call["headers"]
    assert call["method"] == "GET"
    assert call["url"] == "/v1/ingest/pipeline-config"
    assert headers["x-request-id"] == "keep-me"
    assert "host" not in {key.lower() for key in headers}
    assert "Traceparent" not in headers
    assert headers["traceparent"].split("-")[1] == trace_id


def test_forward_sends_backend_request_when_trace_injection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _RecordingClient()
    proxy = _make_proxy(batch=backend)
    body = b'{"document": "raw payload"}'
    request = _make_request(
        method="POST",
        path="/v1/ingest/document",
        headers={
            "host": "gateway.local",
            "transfer-encoding": "chunked",
            "content-type": "application/json",
            "x-request-id": "keep-me",
        },
        body=body,
    )

    def _raise_injection_error(headers: dict[str, str]) -> None:
        raise RuntimeError("propagator unavailable")

    monkeypatch.setattr(proxy_module.tracing_module, "inject_trace_context", _raise_injection_error)

    async def _forward() -> Any:
        return await proxy.forward(
            request,
            PoolType.BATCH,
            extra_headers={"x-extra": "extra-value"},
        )

    response = asyncio.run(_forward())

    assert response.status_code == 202
    assert response.body == b'{"ok": true}'
    assert len(backend.calls) == 1
    call = backend.calls[0]
    headers = call["headers"]
    assert call["method"] == "POST"
    assert call["url"] == "/v1/ingest/document"
    assert call["content"] == body
    assert headers["content-type"] == "application/json"
    assert headers["x-request-id"] == "keep-me"
    assert headers["x-extra"] == "extra-value"
    assert "host" not in {key.lower() for key in headers}
    assert "transfer-encoding" not in {key.lower() for key in headers}


def test_forward_get_sends_backend_request_when_trace_injection_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _RecordingClient()
    proxy = _make_proxy(realtime=backend)
    request = _make_request(
        method="GET",
        path="/v1/ingest/pipeline-config",
        headers={
            "host": "gateway.local",
            "x-request-id": "keep-me",
        },
    )

    def _raise_injection_error(headers: dict[str, str]) -> None:
        raise RuntimeError("propagator unavailable")

    monkeypatch.setattr(proxy_module.tracing_module, "inject_trace_context", _raise_injection_error)

    async def _forward_get() -> Any:
        return await proxy.forward_get(request, PoolType.REALTIME, "/v1/ingest/pipeline-config")

    response = asyncio.run(_forward_get())

    assert response.status_code == 200
    assert response.body == b'{"config": true}'
    assert len(backend.calls) == 1
    call = backend.calls[0]
    headers = call["headers"]
    assert call["method"] == "GET"
    assert call["url"] == "/v1/ingest/pipeline-config"
    assert headers["x-request-id"] == "keep-me"
    assert "host" not in {key.lower() for key in headers}


def _make_proxy(
    *,
    realtime: _RecordingClient | None = None,
    batch: _RecordingClient | None = None,
) -> GatewayProxy:
    proxy = GatewayProxy.__new__(GatewayProxy)
    proxy._config = GatewayConfig(  # noqa: SLF001
        realtime_url="http://realtime.test",
        batch_url="http://batch.test",
    )
    proxy._realtime = realtime or _RecordingClient()  # noqa: SLF001
    proxy._batch = batch or _RecordingClient()  # noqa: SLF001
    return proxy


def _make_request(
    *,
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes = b"",
) -> Request:
    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(key.encode("latin-1"), value.encode("latin-1")) for key, value in headers.items()],
        "scheme": "http",
        "server": ("gateway.local", 80),
        "client": ("testclient", 123),
    }
    return Request(scope, _receive)
