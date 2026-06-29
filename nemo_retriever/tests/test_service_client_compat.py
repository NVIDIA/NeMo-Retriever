# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for SDK/service version-mismatch handling.

These tests pin the customer-facing failure mode reported in 26.05-RC2:
a Python SDK wheel whose ``RetrieverServiceClient`` targets the new
job-scoped ingest API (``POST /v1/ingest/job`` +
``POST /v1/ingest/job/{job_id}/document`` +
``GET /v1/ingest/job/{job_id}/events``) is paired with a service image
that does not advertise that route, *or* an older SDK is paired with a
new service image that returns ``410 Gone`` for the removed legacy
routes.

The contract we pin here:

* On any first call to ``POST /v1/ingest/job`` that returns ``404`` or
  ``410``, :class:`RetrieverServiceCompatibilityError` is raised with a
  message that:
    - names the URL that failed,
    - reports the HTTP status,
    - explains the SDK/service version-mismatch root cause,
    - names the replacement routes,
    - tells the operator how to fix it (upgrade chart, or downgrade SDK).
* On a 404/410 from ``POST /v1/ingest/job/{job_id}/document`` (a mid-
  rollout case where the job was created on a new pod but the upload
  was routed to a stale one), the same error type is raised.
* The error is re-exported from the top-level :mod:`nemo_retriever`
  package so customers can ``except RetrieverServiceCompatibilityError``
  without depending on internal modules.

We drive the client with :class:`httpx.MockTransport` rather than
running a real FastAPI server — the goal is to pin the SDK-side error
translation, not the server route shape (that is covered separately in
:mod:`test_service_sse`).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from nemo_retriever.service.client import (
    RetrieverServiceClient,
    RetrieverServiceCompatibilityError,
    _compat_error_message,
    _is_api_mismatch_status,
)


# ----------------------------------------------------------------------
# Helpers: drive _create_job and _upload_one directly against MockTransport
# ----------------------------------------------------------------------


def _run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def _make_transport(handler) -> httpx.MockTransport:
    """Wrap a sync handler as a MockTransport with the standard signature."""
    return httpx.MockTransport(handler)


# ----------------------------------------------------------------------
# Helper-function unit tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "status, is_mismatch",
    [
        (200, False),
        (201, False),
        (400, False),
        (403, False),
        (404, True),
        (410, True),
        (500, False),
        (503, False),
    ],
)
def test_is_api_mismatch_status_pins_the_404_410_pair(status: int, is_mismatch: bool) -> None:
    """Only 404 and 410 must trigger the compat-error translation.

    400/403/500/503 are real client/server errors, not route-mismatches,
    and must continue to surface through the generic ``HTTPStatusError``
    path so callers can distinguish "bad payload" or "service overloaded"
    from "service is the wrong version".
    """
    assert _is_api_mismatch_status(status) is is_mismatch


def test_compat_error_message_names_url_status_and_replacement_routes() -> None:
    """The customer-facing message must contain every actionable detail.

    Operators reading the traceback should see:
      * the URL that returned 404/410 (so they know which pod is wrong),
      * the HTTP status (so they don't confuse it with a network error),
      * the new job-scoped route names (so they can grep the SDK),
      * the recommended remediation (upgrade chart or downgrade SDK).
    """
    msg = _compat_error_message(
        url="http://example:7670/v1/ingest/job",
        status=404,
        body="Not Found",
    )
    assert "http://example:7670/v1/ingest/job" in msg
    assert "HTTP 404" in msg
    assert "POST /v1/ingest/job" in msg
    assert "POST /v1/ingest/job/{job_id}/document" in msg
    assert "GET /v1/ingest/job/{job_id}/events" in msg
    # Remediation must mention both directions of the alignment.
    assert "Upgrade" in msg or "upgrade" in msg
    assert "downgrade" in msg or "Downgrade" in msg


def test_compat_error_message_clips_long_body() -> None:
    """A pathologically long server response must not flood the traceback."""
    huge = "x" * 10_000
    msg = _compat_error_message(url="http://x/v1/ingest/job", status=410, body=huge)
    # We accept the body section being clipped to 500 chars; everything
    # past that must be elided so the message stays readable.
    assert msg.count("x") <= 600


# ----------------------------------------------------------------------
# _create_job — the primary entry point most customers will trip over
# ----------------------------------------------------------------------


@pytest.mark.parametrize("status", [404, 410])
def test_create_job_raises_compat_error_for_404_and_410(status: int) -> None:
    """The very first SDK call ⇒ clear compat error, not silent empty result.

    Reproduces the 26.05-RC2 customer scenario: ``POST /v1/ingest/job``
    on a service image that does not advertise that route. Before this
    fix the client surfaced a generic ``httpx.HTTPStatusError`` (which
    callers often catch+log+continue), so the documented service-mode
    flow returned an empty :class:`ServiceIngestResult` with no clue
    that the SDK was the wrong version.
    """
    request_paths: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        request_paths.append(request.url.path)
        if status == 410:
            return httpx.Response(
                410,
                json={"detail": "POST /v1/ingest was removed in 26.05"},
            )
        return httpx.Response(404)

    rc = RetrieverServiceClient(base_url="http://nrl:7670")

    async def _call() -> None:
        async with httpx.AsyncClient(transport=_make_transport(_handler)) as client:
            await rc._create_job(client, expected_documents=1)

    with pytest.raises(RetrieverServiceCompatibilityError) as ei:
        _run_async(_call())

    msg = str(ei.value)
    assert "http://nrl:7670/v1/ingest/job" in msg
    assert f"HTTP {status}" in msg
    assert "POST /v1/ingest/job" in msg
    # Pin that exactly one request landed on /v1/ingest/job.
    assert request_paths == ["/v1/ingest/job"], request_paths


def test_create_job_500_still_raises_generic_http_status_error() -> None:
    """A real server error must NOT be misreported as a version mismatch.

    If the deployed service is the right version but transiently broken
    (500/503/etc.) the SDK should surface the existing
    :class:`httpx.HTTPStatusError` so retry/alerting logic in the
    caller still triggers as before. Mis-coding this as a
    ``RetrieverServiceCompatibilityError`` would hide a real outage.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    rc = RetrieverServiceClient(base_url="http://nrl:7670")

    async def _call() -> None:
        async with httpx.AsyncClient(transport=_make_transport(_handler)) as client:
            await rc._create_job(client, expected_documents=1)

    with pytest.raises(httpx.HTTPStatusError) as ei:
        _run_async(_call())
    assert "HTTP 500" in str(ei.value)


def test_create_job_success_returns_job_id() -> None:
    """Smoke test: a healthy 201 response is unaffected by the compat logic."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"job_id": "JOB-1", "trace_id": "trace-123"})

    rc = RetrieverServiceClient(base_url="http://nrl:7670")

    async def _call():
        async with httpx.AsyncClient(transport=_make_transport(_handler)) as client:
            return await rc._create_job(client, expected_documents=1)

    created = _run_async(_call())
    assert created.job_id == "JOB-1"
    assert created.trace_id == "trace-123"


def test_stream_job_created_event_includes_trace_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public streaming path preserves the trace id returned by job creation."""

    rc = RetrieverServiceClient(base_url="http://nrl:7670")

    async def _create_job(*args, **kwargs):
        return SimpleNamespace(job_id="JOB-1", trace_id="trace-123")

    async def _consume_sse(_client, _pending, uploads_done, _tracker, **_kwargs):
        uploads_done.set()

    monkeypatch.setattr(rc, "_create_job", _create_job)
    monkeypatch.setattr(rc, "_consume_sse", _consume_sse)

    async def _first_event() -> dict[str, object]:
        async for event in rc.aingest_documents_stream([]):
            return event
        raise AssertionError("stream produced no events")

    assert _run_async(_first_event()) == {
        "event": "job_created",
        "job_id": "JOB-1",
        "expected_documents": 0,
        "trace_id": "trace-123",
    }


# ----------------------------------------------------------------------
# _upload_one — guards the mid-rollout case (new gateway, stale worker)
# ----------------------------------------------------------------------


def test_upload_one_raises_compat_error_on_404(tmp_path: Path) -> None:
    """Uploads to a stale pod (404 on the document path) ⇒ compat error.

    This guards the rolling-upgrade case: ``POST /v1/ingest/job`` lands
    on a new pod and succeeds, but the per-document upload is routed by
    the load balancer to a stale pod that does not implement
    ``POST /v1/ingest/job/{job_id}/document``.  Without this guard the
    upload would surface as a generic ``HTTPStatusError`` and get
    captured into ``upload_failures`` — i.e. ANOTHER silent empty
    result mode.
    """
    fpath = tmp_path / "doc.pdf"
    fpath.write_bytes(b"%PDF-1.4 dummy")

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    rc = RetrieverServiceClient(base_url="http://nrl:7670")

    async def _call() -> None:
        async with httpx.AsyncClient(transport=_make_transport(_handler)) as client:
            await rc._upload_one(client, fpath, job_id="JOB-1")

    with pytest.raises(RetrieverServiceCompatibilityError) as ei:
        _run_async(_call())
    assert "/v1/ingest/job/JOB-1/document" in str(ei.value)


def test_upload_one_410_surfaces_replacement_routes(tmp_path: Path) -> None:
    """A 410 with a server-side migration body still surfaces the new routes."""
    fpath = tmp_path / "doc.pdf"
    fpath.write_bytes(b"%PDF-1.4 dummy")

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            410,
            json={"detail": "POST /v1/ingest was removed in 26.05"},
        )

    rc = RetrieverServiceClient(base_url="http://nrl:7670")

    async def _call() -> None:
        async with httpx.AsyncClient(transport=_make_transport(_handler)) as client:
            await rc._upload_one(client, fpath, job_id="JOB-1")

    with pytest.raises(RetrieverServiceCompatibilityError) as ei:
        _run_async(_call())
    msg = str(ei.value)
    assert "HTTP 410" in msg
    assert "POST /v1/ingest/job/{job_id}/document" in msg


# ----------------------------------------------------------------------
# Top-level re-export — callers should not need to import a private path
# ----------------------------------------------------------------------


def test_error_is_reexported_from_nemo_retriever_package() -> None:
    """Customers must be able to ``from nemo_retriever import …`` the error.

    The compat error becomes the canonical signal customers will write
    ``except`` blocks against. Forcing them to import from
    ``nemo_retriever.service.client`` would bind their code to an
    internal module path; the top-level re-export is part of the public
    contract.
    """
    import nemo_retriever

    assert nemo_retriever.RetrieverServiceCompatibilityError is RetrieverServiceCompatibilityError
    assert "RetrieverServiceCompatibilityError" in nemo_retriever.__all__


# ----------------------------------------------------------------------
# Misc: avoid pytest collection touching the asyncio loop helper
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_event_loop() -> Iterator[None]:
    """Ensure each test uses a fresh asyncio loop.

    ``asyncio.run`` creates and tears down a loop per call, so this
    fixture is largely defensive — it just makes the intent explicit
    and protects against future tests that monkey-patch ``get_event_loop``.
    """
    yield
