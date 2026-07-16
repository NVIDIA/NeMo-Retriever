# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

from starlette.requests import Request

from nemo_retriever.service.app import _GatewayBodyCacheMiddleware


def _scope(headers: list[tuple[bytes, bytes]]) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/ingest/job/job-id/document",
        "raw_path": b"/v1/ingest/job/job-id/document",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


async def _call_asgi(
    app: Any,
    *,
    headers: list[tuple[bytes, bytes]],
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    sent: list[dict[str, Any]] = []
    receive_calls = 0
    pending = list(messages)

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(_scope(headers), receive, send)
    return sent, receive_calls


def _status(sent: list[dict[str, Any]]) -> int:
    return next(message["status"] for message in sent if message["type"] == "http.response.start")


def _body(sent: list[dict[str, Any]]) -> bytes:
    return b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")


class _NeverCalledApp:
    called = False

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        self.called = True
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


class _BodyReadingApp:
    called = False
    cached_body: bytes | None = None
    body: bytes | None = None

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        self.called = True
        self.cached_body = scope.get("_cached_body")
        self.body = await Request(scope, receive).body()

        await send({"type": "http.response.start", "status": 202, "headers": []})
        await send({"type": "http.response.body", "body": b"accepted"})


class _MultipartParsingApp:
    called = False
    cached_body: bytes | None = None
    filename: str | None = None
    metadata: str | None = None
    file_content: bytes | None = None

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        self.called = True
        self.cached_body = scope.get("_cached_body")

        form = await Request(scope, receive).form()
        upload = form["file"]
        self.filename = upload.filename
        self.metadata = str(form["metadata"])
        self.file_content = await upload.read()

        await send({"type": "http.response.start", "status": 202, "headers": []})
        await send({"type": "http.response.body", "body": b"accepted"})


def test_gateway_body_cache_no_limit_passes_large_body_through() -> None:
    downstream = _BodyReadingApp()
    app = _GatewayBodyCacheMiddleware(downstream, max_body_bytes=None)
    payload = b"x" * 1_000_000

    sent, receive_calls = asyncio.run(
        _call_asgi(
            app,
            headers=[(b"content-length", str(len(payload)).encode())],
            messages=[{"type": "http.request", "body": payload, "more_body": False}],
        )
    )

    assert _status(sent) == 202
    assert _body(sent) == b"accepted"
    assert receive_calls == 1
    assert downstream.called is True
    assert downstream.cached_body == payload
    assert downstream.body == payload


def test_gateway_body_cache_rejects_oversized_content_length_without_reading_body() -> None:
    downstream = _NeverCalledApp()
    app = _GatewayBodyCacheMiddleware(downstream, max_body_bytes=4)

    sent, receive_calls = asyncio.run(
        _call_asgi(
            app,
            headers=[(b"content-length", b"5")],
            messages=[{"type": "http.request", "body": b"12345", "more_body": False}],
        )
    )

    assert _status(sent) == 413
    assert b"exceeds limit" in _body(sent)
    assert receive_calls == 0
    assert downstream.called is False


def test_gateway_body_cache_accepts_content_length_at_limit() -> None:
    downstream = _BodyReadingApp()
    payload = b"12345"
    app = _GatewayBodyCacheMiddleware(downstream, max_body_bytes=len(payload))

    sent, receive_calls = asyncio.run(
        _call_asgi(
            app,
            headers=[(b"content-length", str(len(payload)).encode())],
            messages=[{"type": "http.request", "body": payload, "more_body": False}],
        )
    )

    assert _status(sent) == 202
    assert _body(sent) == b"accepted"
    assert receive_calls == 1
    assert downstream.called is True
    assert downstream.cached_body == payload
    assert downstream.body == payload


def test_gateway_body_cache_rejects_chunked_upload_after_limit_is_exceeded() -> None:
    downstream = _NeverCalledApp()
    app = _GatewayBodyCacheMiddleware(downstream, max_body_bytes=5)

    sent, receive_calls = asyncio.run(
        _call_asgi(
            app,
            headers=[(b"transfer-encoding", b"chunked")],
            messages=[
                {"type": "http.request", "body": b"abc", "more_body": True},
                {"type": "http.request", "body": b"def", "more_body": True},
                {"type": "http.request", "body": b"ghi", "more_body": False},
            ],
        )
    )

    assert _status(sent) == 413
    assert b"6 bytes exceeds limit of 5 bytes" in _body(sent)
    assert receive_calls == 2
    assert downstream.called is False


def test_gateway_body_cache_accepts_chunked_upload_at_limit() -> None:
    downstream = _BodyReadingApp()
    payload = b"abcdef"
    app = _GatewayBodyCacheMiddleware(downstream, max_body_bytes=len(payload))

    sent, receive_calls = asyncio.run(
        _call_asgi(
            app,
            headers=[],
            messages=[
                {"type": "http.request", "body": payload[:3], "more_body": True},
                {"type": "http.request", "body": payload[3:], "more_body": False},
            ],
        )
    )

    assert _status(sent) == 202
    assert _body(sent) == b"accepted"
    assert receive_calls == 2
    assert downstream.called is True
    assert downstream.cached_body == payload
    assert downstream.body == payload


def test_gateway_body_cache_replays_valid_multipart_upload_under_limit() -> None:
    downstream = _MultipartParsingApp()
    app = _GatewayBodyCacheMiddleware(downstream, max_body_bytes=1024)

    boundary = b"----nemo-retriever-test-boundary"
    body = (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n{}\r\n'
        + b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="file"; filename="doc.txt"\r\n'
        + b"Content-Type: text/plain\r\n\r\nhello gateway\r\n"
        + b"--"
        + boundary
        + b"--\r\n"
    )

    sent, receive_calls = asyncio.run(
        _call_asgi(
            app,
            headers=[
                (b"content-type", b"multipart/form-data; boundary=" + boundary),
                (b"content-length", str(len(body)).encode()),
            ],
            messages=[
                {"type": "http.request", "body": body[:40], "more_body": True},
                {"type": "http.request", "body": body[40:], "more_body": False},
            ],
        )
    )

    assert _status(sent) == 202
    assert _body(sent) == b"accepted"
    assert receive_calls == 2
    assert downstream.called is True
    assert downstream.cached_body == body
    assert downstream.filename == "doc.txt"
    assert downstream.metadata == "{}"
    assert downstream.file_content == b"hello gateway"
