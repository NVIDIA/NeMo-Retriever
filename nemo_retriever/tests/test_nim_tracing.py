# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenTelemetry propagation in NIM HTTP clients."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import requests
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.trace import StatusCode

from nemo_retriever.models.nim.primitives.nim_client import NimClient as InternalNimClient
from nemo_retriever.models.nim.nim import NIMClient as HttpNIMClient
from nemo_retriever.models.nim.nim import _post_with_retries
from nemo_retriever.service import tracing


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


def test_post_with_retries_injects_trace_context_and_emits_safe_span(
    monkeypatch: pytest.MonkeyPatch,
    exported_spans: list[Any],
) -> None:
    captured_headers: list[dict[str, str]] = []

    class _Response:
        status_code = 200
        text = '{"ok": true}'

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, bool]:
            return {"ok": True}

    def _post(*args: Any, headers: dict[str, str], **kwargs: Any) -> _Response:
        captured_headers.append(dict(headers))
        return _Response()

    monkeypatch.setattr("nemo_retriever.models.nim.nim.requests.post", _post)
    original_headers = {
        "Accept": "application/json",
        "Authorization": "Bearer secret-token",
    }
    payload = {"input": [{"secret": "request-body"}]}

    with tracing.start_span("test.parent"):
        parent_trace_id = tracing.current_trace_id_hex()
        result = _post_with_retries(
            invoke_url="http://nim.example/v1/infer",
            payload=payload,
            headers=original_headers,
            timeout_s=10,
            max_retries=1,
            max_429_retries=1,
        )

    assert result == {"ok": True}
    assert parent_trace_id is not None
    assert captured_headers
    assert "traceparent" in captured_headers[0]
    assert captured_headers[0]["Authorization"] == "Bearer secret-token"
    assert "traceparent" not in original_headers

    spans_by_name = {span.name: span for span in exported_spans}
    assert "nim.http.post" in spans_by_name
    span = spans_by_name["nim.http.post"]
    assert f"{span.context.trace_id:032x}" == parent_trace_id

    attrs = dict(span.attributes)
    assert attrs["http.method"] == "POST"
    assert attrs["nim.endpoint"] == "http://nim.example/v1/infer"
    assert attrs["retry.attempt"] == 0
    assert attrs["http.status_code"] == 200
    assert "Authorization" not in attrs
    assert "request.body" not in attrs
    assert "payload" not in attrs


class _PrimitiveModelInterface:
    def name(self) -> str:
        return "page-elements"

    def prepare_data_for_inference(self, data: Any) -> Any:
        return data

    def format_input(self, data: Any, **kwargs: Any) -> tuple[list[Any], list[dict[str, Any]]]:
        return [data], [{"original_image_shapes": None}]

    def parse_output(self, response: Any, **kwargs: Any) -> Any:
        return response

    def process_inference_results(self, parsed_output: Any, **kwargs: Any) -> Any:
        return parsed_output


def _span_by_name(exported_spans: list[Any], name: str) -> Any:
    return next(span for span in exported_spans if span.name == name)


def test_internal_nim_client_http_injects_trace_context_and_emits_infer_span(
    monkeypatch: pytest.MonkeyPatch,
    exported_spans: list[Any],
) -> None:
    captured_headers: list[dict[str, str]] = []

    class _Response:
        status_code = 200
        reason = "OK"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, bool]:
            return {"ok": True}

    def _post(*args: Any, headers: dict[str, str], **kwargs: Any) -> _Response:
        captured_headers.append(dict(headers))
        return _Response()

    monkeypatch.setattr("nemo_retriever.models.nim.primitives.nim_client.requests.post", _post)

    client = InternalNimClient(
        model_interface=_PrimitiveModelInterface(),
        protocol="http",
        endpoints=("", "http://nim.example/v1/infer"),
        auth_token="secret-token",
        max_retries=1,
        max_429_retries=1,
    )

    with tracing.start_span("test.parent"):
        parent_trace_id = tracing.current_trace_id_hex()
        parsed_output, batch_data = client._process_batch(
            {"request": "body"},
            batch_data={"batch": 1},
            model_name="detector",
        )

    assert parsed_output == {"ok": True}
    assert batch_data == {"batch": 1}
    assert parent_trace_id is not None
    assert captured_headers
    assert "traceparent" in captured_headers[0]
    assert captured_headers[0]["Authorization"] == "Bearer secret-token"
    assert "traceparent" not in client.headers

    span = _span_by_name(exported_spans, "nim.infer")
    assert f"{span.context.trace_id:032x}" == parent_trace_id
    assert span.attributes["nim.model"] == "detector"
    assert span.attributes["nim.protocol"] == "http"
    assert span.attributes["nim.service"] == "page-elements"
    assert "Authorization" not in span.attributes
    assert "request.body" not in span.attributes


def test_internal_nim_client_grpc_passes_trace_context_headers_when_supported(
    monkeypatch: pytest.MonkeyPatch,
    exported_spans: list[Any],
) -> None:
    captured_headers: list[dict[str, str]] = []

    class _InferInput:
        def __init__(self, name: str, shape: Any, datatype: str) -> None:
            self.name = name
            self.shape = shape
            self.datatype = datatype

        def set_data_from_numpy(self, value: Any) -> None:
            self.value = value

    class _InferRequestedOutput:
        def __init__(self, name: str) -> None:
            self._name = name

        def name(self) -> str:
            return self._name

    class _Response:
        def as_numpy(self, name: str) -> np.ndarray:
            return np.array([[1.0]], dtype=np.float32)

    class _InferenceServerException(Exception):
        def status(self) -> str:
            return "StatusCode.UNKNOWN"

        def message(self) -> str:
            return str(self)

    class _FakeGrpcClient:
        def infer(self, *, headers: dict[str, str] | None = None, **kwargs: Any) -> _Response:
            captured_headers.append(dict(headers or {}))
            return _Response()

        def close(self) -> None:
            return None

    class _GrpcModule:
        InferenceServerException = _InferenceServerException
        InferInput = _InferInput
        InferRequestedOutput = _InferRequestedOutput

        class InferenceServerClient:
            def __init__(self, url: str) -> None:
                self._client = _FakeGrpcClient()

            def infer(self, **kwargs: Any) -> _Response:
                return self._client.infer(**kwargs)

            def close(self) -> None:
                return None

    monkeypatch.setattr("nemo_retriever.models.nim.primitives.nim_client._triton_grpc", lambda: _GrpcModule)

    client = InternalNimClient(
        model_interface=_PrimitiveModelInterface(),
        protocol="grpc",
        endpoints=("nim-grpc:8001", ""),
        max_retries=1,
        max_429_retries=1,
    )

    with tracing.start_span("test.parent"):
        parent_trace_id = tracing.current_trace_id_hex()
        parsed_output, batch_data = client._process_batch(
            np.array([[1.0]], dtype=np.float32),
            batch_data={"batch": 2},
            model_name="detector-grpc",
        )

    assert isinstance(parsed_output, np.ndarray)
    assert batch_data == {"batch": 2}
    assert parent_trace_id is not None
    assert captured_headers
    assert "traceparent" in captured_headers[0]

    span = _span_by_name(exported_spans, "nim.infer")
    assert f"{span.context.trace_id:032x}" == parent_trace_id
    assert span.attributes["nim.model"] == "detector-grpc"
    assert span.attributes["nim.protocol"] == "grpc"
    assert span.attributes["nim.service"] == "page-elements"


def test_public_nim_client_executor_path_preserves_parent_trace_context(
    monkeypatch: pytest.MonkeyPatch,
    exported_spans: list[Any],
) -> None:
    captured_headers: list[dict[str, str]] = []

    class _Response:
        status_code = 200
        text = "{}"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[dict[str, bool]]]:
            return {"data": [{"ok": True}]}

    def _post(*args: Any, headers: dict[str, str], **kwargs: Any) -> _Response:
        captured_headers.append(dict(headers))
        return _Response()

    monkeypatch.setattr("nemo_retriever.models.nim.nim.requests.post", _post)
    client = HttpNIMClient(max_pool_workers=1)
    try:
        with tracing.start_span("test.parent"):
            parent_trace_id = tracing.current_trace_id_hex()
            result = client.invoke_image_inference_batches(
                invoke_url="http://nim.example/v1/infer",
                image_b64_list=["iVBORw0KGgo="],
                max_batch_size=1,
                max_retries=1,
                max_429_retries=1,
            )
    finally:
        client.shutdown()

    assert result == [{"ok": True}]
    assert parent_trace_id is not None
    assert captured_headers
    assert "traceparent" in captured_headers[0]

    span = _span_by_name(exported_spans, "nim.http.post")
    assert f"{span.context.trace_id:032x}" == parent_trace_id


def test_internal_nim_client_infer_executor_path_preserves_parent_trace_context(
    monkeypatch: pytest.MonkeyPatch,
    exported_spans: list[Any],
) -> None:
    captured_headers: list[dict[str, str]] = []

    class _Response:
        status_code = 200
        reason = "OK"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, bool]:
            return {"ok": True}

    def _post(*args: Any, headers: dict[str, str], **kwargs: Any) -> _Response:
        captured_headers.append(dict(headers))
        return _Response()

    monkeypatch.setattr("nemo_retriever.models.nim.primitives.nim_client.requests.post", _post)
    client = InternalNimClient(
        model_interface=_PrimitiveModelInterface(),
        protocol="http",
        endpoints=("", "http://nim.example/v1/infer"),
        max_retries=1,
        max_429_retries=1,
    )

    with tracing.start_span("test.parent"):
        parent_trace_id = tracing.current_trace_id_hex()
        result = client.infer({"request": "body"}, model_name="detector", max_pool_workers=1)

    assert result == [{"ok": True}]
    assert parent_trace_id is not None
    assert captured_headers
    assert "traceparent" in captured_headers[0]

    span = _span_by_name(exported_spans, "nim.infer")
    assert f"{span.context.trace_id:032x}" == parent_trace_id


def test_public_nim_client_chat_executor_path_preserves_parent_trace_context(
    monkeypatch: pytest.MonkeyPatch,
    exported_spans: list[Any],
) -> None:
    captured_headers: list[dict[str, str]] = []

    class _Response:
        status_code = 200
        text = "{}"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[dict[str, dict[str, str]]]]:
            return {"choices": [{"message": {"content": "hello"}}]}

    def _post(*args: Any, headers: dict[str, str], **kwargs: Any) -> _Response:
        captured_headers.append(dict(headers))
        return _Response()

    monkeypatch.setattr("nemo_retriever.models.nim.nim.requests.post", _post)
    client = HttpNIMClient(max_pool_workers=1)
    try:
        with tracing.start_span("test.parent"):
            parent_trace_id = tracing.current_trace_id_hex()
            result = client.invoke_chat_completions(
                invoke_url="http://nim.example/v1/chat/completions",
                messages_list=[[{"role": "user", "content": "hi"}]],
                max_retries=1,
                max_429_retries=1,
            )
    finally:
        client.shutdown()

    assert result == ["hello"]
    assert parent_trace_id is not None
    assert captured_headers
    assert "traceparent" in captured_headers[0]

    span = _span_by_name(exported_spans, "nim.http.post")
    assert f"{span.context.trace_id:032x}" == parent_trace_id


def test_internal_dynamic_batching_keeps_different_trace_contexts_in_separate_batches(
    exported_spans: list[Any],
) -> None:
    del exported_spans
    client = InternalNimClient(
        model_interface=_PrimitiveModelInterface(),
        protocol="http",
        endpoints=("", "http://nim.example/v1/infer"),
        enable_dynamic_batching=True,
        dynamic_batch_timeout=0.01,
    )
    client._batch_size = 2
    captured_batches: list[list[Any]] = []

    def _capture_batch(requests: list[Any]) -> None:
        captured_batches.append(list(requests))
        client._stop_event.set()

    client._process_dynamic_batch = _capture_batch  # type: ignore[method-assign]

    with tracing.start_span("test.parent.one"):
        first_trace_id = tracing.current_trace_id_hex()
        first_future = client.submit("one", "detector", (1, 1))
    with tracing.start_span("test.parent.two"):
        second_trace_id = tracing.current_trace_id_hex()
        second_future = client.submit("two", "detector", (1, 1))

    assert first_trace_id is not None and second_trace_id is not None
    assert first_trace_id != second_trace_id

    client._batcher_loop()

    assert len(captured_batches) == 1
    assert [request.data for request in captured_batches[0]] == ["one"]
    assert client._request_queue.qsize() == 1
    assert first_future.done() is False
    assert second_future.done() is False


def test_post_with_retries_marks_final_500_span_as_error_without_secrets(
    monkeypatch: pytest.MonkeyPatch,
    exported_spans: list[Any],
) -> None:
    class _Response:
        status_code = 500
        text = "server body with secret-token"

        def raise_for_status(self) -> None:
            raise requests.HTTPError("HTTP 500 from upstream", response=self)

        def json(self) -> dict[str, bool]:
            return {"ok": False}

    def _post(*args: Any, **kwargs: Any) -> _Response:
        return _Response()

    monkeypatch.setattr("nemo_retriever.models.nim.nim.requests.post", _post)

    with pytest.raises(requests.HTTPError):
        _post_with_retries(
            invoke_url="http://nim.example/v1/infer?api_key=secret-token",
            payload={
                "input": [
                    {"Authorization": "Bearer secret-token", "secret": "payload"},
                ],
            },
            headers={"Authorization": "Bearer secret-token"},
            timeout_s=10,
            max_retries=1,
            max_429_retries=1,
        )

    span = _span_by_name(exported_spans, "nim.http.post")
    attrs = dict(span.attributes)
    assert attrs["http.status_code"] == 500
    assert span.status.status_code == StatusCode.ERROR or attrs.get("error.type") == "HTTP 500"
    attr_text = repr(attrs)
    assert "secret-token" not in attr_text
    assert "Bearer" not in attr_text
    assert "payload" not in attr_text
    assert "server body" not in attr_text
