# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end regression test for the 26.05-RC2 version-mismatch bug.

The customer-reported failure mode:

    The published 26.05-RC2 Python SDK calls legacy ``/v1/ingest`` /
    ``/v1/ingest/events`` routes against an nrl-service image that
    expects the newer job-scoped API. The documented service-mode flow
    appears to run but returns an empty result with no successful
    ``document_complete`` event.

The fix has two halves:

1. The server returns ``410 Gone`` with a migration body for the
   removed legacy routes (covered in :mod:`test_service_sse`).
2. The client translates ``404`` / ``410`` from
   ``POST /v1/ingest/job`` into
   :class:`RetrieverServiceCompatibilityError` (unit-tested in
   :mod:`test_service_client_compat`).

What's missing: an end-to-end test that follows the *customer's*
entry point — :meth:`ServiceIngestor.ingest` — through every layer
(sync method → ``ingest_stream`` → async generator → background
thread → ``_AsyncToSyncBridge`` → ``RetrieverServiceClient``) and
confirms the compat error reaches the caller intact instead of being
swallowed.  Without this test a future refactor that catches
``Exception`` in the bridge would silently re-introduce the original
empty-result regression.
"""

from __future__ import annotations

import httpx
import pytest

import nemo_retriever.service.client as _client_module
from nemo_retriever import RetrieverServiceCompatibilityError
from nemo_retriever.service.service_ingestor import ServiceIngestor


# ----------------------------------------------------------------------
# Mock transport plumbing: route every HTTP call into a handler dict
# ----------------------------------------------------------------------


def _install_mock_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Replace ``httpx.AsyncClient`` inside ``service.client`` with a mock.

    We patch the symbol on the imported module rather than on the
    ``httpx`` package globally so other tests running in the same
    process keep their real transport.  The factory wraps the real
    ``AsyncClient`` constructor and injects a ``MockTransport`` whose
    handler is the one the test passes in.
    """
    original = _client_module.httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return original(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(_client_module.httpx, "AsyncClient", _factory)


# ----------------------------------------------------------------------
# Test: 404 on POST /v1/ingest/job ⇒ ingest() raises the compat error
# ----------------------------------------------------------------------


def test_service_ingestor_ingest_surfaces_compat_error_on_404(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Customer entry point: ``ServiceIngestor.ingest()`` must raise, not return empty.

    This pins the exact end-to-end behavior that 26.05-RC2 customers
    expected.  Before the fix the documented flow produced an empty
    :class:`ServiceIngestResult` with ``len(result) == 0`` and no
    actionable error.  After the fix:

      * the error propagates through the async-to-sync bridge,
      * the message names the URL that failed,
      * the message names the replacement route and remediation,
      * the exception type is the dedicated compatibility error so
        callers can catch it specifically.
    """
    calls: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        # Simulate an old service: every job-scoped path returns 404.
        return httpx.Response(404, text="Not Found")

    _install_mock_transport(monkeypatch, _handler)

    pdf = tmp_path / "tiny.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy content")

    ing = ServiceIngestor(base_url="http://nrl:7670", documents=[str(pdf)])

    with pytest.raises(RetrieverServiceCompatibilityError) as ei:
        ing.ingest()

    msg = str(ei.value)
    assert "http://nrl:7670/v1/ingest/job" in msg
    assert "HTTP 404" in msg
    assert "POST /v1/ingest/job" in msg
    # Pin that the SDK gave up on the very first request — we must not
    # silently retry against the legacy paths.
    assert calls == ["POST /v1/ingest/job"], calls


def test_service_ingestor_ingest_surfaces_compat_error_on_410(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """An explicit ``410 Gone`` from a new service raises the same error.

    The mirror-image case: the SDK is the older 26.05-RC2 build, but
    the deployed service ships the 410 stub that explains the legacy
    route is gone.  Our new SDK targets the new route, so the 410 is
    delivered to the new client too — and it must still produce the
    same :class:`RetrieverServiceCompatibilityError` so the customer
    sees one consistent failure type regardless of which side is stale.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            410,
            json={
                "detail": ("POST /v1/ingest was removed in retriever-service 26.05"),
            },
        )

    _install_mock_transport(monkeypatch, _handler)

    pdf = tmp_path / "tiny.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    ing = ServiceIngestor(base_url="http://nrl:7670", documents=[str(pdf)])

    with pytest.raises(RetrieverServiceCompatibilityError) as ei:
        ing.ingest()

    msg = str(ei.value)
    assert "HTTP 410" in msg
    assert "POST /v1/ingest/job" in msg


def test_service_ingestor_ingest_stream_also_raises_for_404(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Streaming entry point must surface the same error before yielding events.

    Customers that subscribe to ``ingest_stream()`` would otherwise
    silently get an empty iterator — same bug class, different surface.
    The exception must be raised on the first ``next()`` call (i.e.
    *before* any document event is yielded), so a ``for evt in
    ingest_stream(): ...`` loop terminates immediately with a clear
    error rather than entering a no-op body.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    _install_mock_transport(monkeypatch, _handler)

    pdf = tmp_path / "tiny.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    ing = ServiceIngestor(base_url="http://nrl:7670", documents=[str(pdf)])

    stream = ing.ingest_stream()
    with pytest.raises(RetrieverServiceCompatibilityError):
        next(stream)


# ----------------------------------------------------------------------
# Test: ingest_async() inherits the same propagation contract
# ----------------------------------------------------------------------


def test_service_ingestor_ingest_async_future_reraises_compat_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """``ingest_async().result()`` must re-raise the compat error.

    The asynchronous wrapper runs :meth:`ingest` on a background
    thread; without explicit handling it could capture the exception
    into the future and force callers to call ``.exception()``
    explicitly.  We pin that ``.result()`` re-raises so the synchronous
    and asynchronous surfaces share one error contract.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    _install_mock_transport(monkeypatch, _handler)

    pdf = tmp_path / "tiny.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    ing = ServiceIngestor(base_url="http://nrl:7670", documents=[str(pdf)])

    future = ing.ingest_async()
    with pytest.raises(RetrieverServiceCompatibilityError):
        future.result(timeout=30)


# ----------------------------------------------------------------------
# Test: connection-level error stays a different error type
# ----------------------------------------------------------------------


def test_connection_error_is_not_misreported_as_version_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A real network failure must not be coded as a version mismatch.

    If the service is down (or the URL is wrong) the SDK should fail
    with a transport-level error, not :class:`RetrieverServiceCompatibilityError`.
    Mis-classifying transport failures as version mismatches would
    send operators chasing the wrong root cause.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    _install_mock_transport(monkeypatch, _handler)

    pdf = tmp_path / "tiny.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    ing = ServiceIngestor(base_url="http://nrl:7670", documents=[str(pdf)])

    with pytest.raises(Exception) as ei:
        ing.ingest()
    # Whatever the exact transport error, it must NOT be the compat
    # error — that's the only assertion that matters here.
    assert not isinstance(ei.value, RetrieverServiceCompatibilityError)
