# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async client for submitting documents to the retriever service.

Uploads whole documents via ``POST /v1/ingest/job/{job_id}/document``
(after opening a job aggregate with ``POST /v1/ingest/job``), tracks
completion via the per-job ``GET /v1/ingest/job/{job_id}/events`` SSE
stream (with ``POST /v1/ingest/status/batch`` bulk-poll fallback), and
surfaces results through both materialized and streaming interfaces.

The SSE connection is opened **before** uploads begin so that completion
events for fast-finishing documents are never missed.  A ``seen_terminal``
buffer reconciles events that arrive before the client registers the
corresponding ``document_id`` from the upload response.

API compatibility
-----------------
The Retriever Service v2 refactor (multi-pod) removed the legacy
single-shot ``POST /v1/ingest`` and the firehose
``GET /v1/ingest/events`` routes in favor of the job-scoped API used
here.  Older SDK builds may still call the legacy routes; the server
now returns ``410 Gone`` with a migration body for those.  This client
detects the matching failure mode on its own side — a ``404`` or
``410`` from the very first call to ``POST /v1/ingest/job`` — and
raises :class:`RetrieverServiceCompatibilityError` so callers see a
single, actionable "SDK and service versions are out of sync" message
instead of an empty/no-completion result.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import httpx
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)

_BULK_POLL_INTERVAL_S = 5.0
_BULK_POLL_TIMEOUT_S = 1800.0
_MAX_UPLOAD_RETRIES = 10
_DEFAULT_RETRY_AFTER = 2.0

_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
)


# ------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------


class RetrieverServiceCompatibilityError(RuntimeError):
    """Raised when the SDK and the retriever service disagree on the API.

    The Retriever Service v2 refactor removed the legacy
    ``POST /v1/ingest`` / ``GET /v1/ingest/events`` routes in favor of
    job-scoped routes (``POST /v1/ingest/job`` +
    ``POST /v1/ingest/job/{job_id}/document`` +
    ``GET /v1/ingest/job/{job_id}/events``).  Whenever the very first
    call from this client — opening a job aggregate via
    ``POST /v1/ingest/job`` — returns ``404`` (route missing) or
    ``410`` (route removed, with migration body), the deployed
    nrl-service is older than this SDK build.  Raising a dedicated
    error type lets callers surface a single, actionable message
    instead of the previous silent "no document_complete event"
    failure mode that 26.05-RC2 customers reported.
    """


def _is_api_mismatch_status(status: int) -> bool:
    """Return ``True`` for HTTP status codes that signal a route mismatch.

    The new client points at ``POST /v1/ingest/job`` (and friends).
    Servers that predate the multi-pod refactor return ``404`` for that
    path; servers carrying the explicit legacy stubs added alongside
    this client return ``410 Gone`` with a migration body.  Either is a
    deterministic "wrong service version" signal.
    """
    return status in (404, 410)


def _compat_error_message(
    *,
    url: str,
    status: int,
    body: str,
) -> str:
    """Build the customer-facing message attached to compatibility errors."""
    body_clip = (body or "(empty)").strip()[:500]
    return (
        f"Retriever service rejected {url} with HTTP {status}. "
        "This signals an SDK/service version mismatch: this Python "
        "SDK targets the job-scoped ingest API "
        "(POST /v1/ingest/job + POST /v1/ingest/job/{job_id}/document "
        "+ GET /v1/ingest/job/{job_id}/events) introduced in 26.05, "
        "but the deployed nrl-service does not advertise that route. "
        "Upgrade the chart/image to a 26.05+ build, or downgrade the "
        "Python SDK to match the deployed service version. Server "
        f"response body: {body_clip}"
    )


# ------------------------------------------------------------------
# Progress bar helper
# ------------------------------------------------------------------


def _make_progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


# ------------------------------------------------------------------
# Document-level tracker
# ------------------------------------------------------------------


class DocumentTracker:
    """Lightweight tracker for document-level completion."""

    def __init__(self) -> None:
        self.pending: set[str] = set()
        self.completed: int = 0
        self.failed: int = 0
        self.errors: list[tuple[str, str]] = []
        self.results: list[dict[str, Any]] = []

    @property
    def done_count(self) -> int:
        return self.completed + self.failed

    def mark_completed(self, doc_id: str, event: dict[str, Any]) -> None:
        self.pending.discard(doc_id)
        self.completed += 1
        self.results.append(event)

    def mark_failed(self, doc_id: str, error: str | None, event: dict[str, Any]) -> None:
        self.pending.discard(doc_id)
        self.failed += 1
        if error:
            self.errors.append((doc_id, error))
        self.results.append(event)


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------


class RetrieverServiceClient:
    """Submits documents to a running retriever service and tracks results.

    Opens a job aggregate with ``POST /v1/ingest/job`` (sized to the
    number of files), then uses ``POST /v1/ingest/job/{job_id}/document``
    for each upload. Completion is tracked via the per-job
    ``GET /v1/ingest/job/{job_id}/events`` SSE stream with
    ``POST /v1/ingest/status/batch`` as a bulk-poll fallback.

    The first request issued by every entry point is ``POST /v1/ingest/job``;
    if that returns ``404`` or ``410`` the client raises
    :class:`RetrieverServiceCompatibilityError` to surface a clear
    SDK/service version-mismatch message rather than silently producing
    an empty result list.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7670",
        max_concurrency: int = 8,
        *,
        api_token: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_concurrency = max_concurrency
        self._api_token = (api_token or "").strip() or None

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_token}"} if self._api_token else {}

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    async def _create_job(
        self,
        client: httpx.AsyncClient,
        *,
        expected_documents: int,
        label: str | None = None,
        retain_results: bool = False,
    ) -> str:
        """Open a server-side job aggregate and return the assigned ``job_id``.

        Every upload made through this client must reference a job (J3+).
        We open one job per ``ingest_documents`` / ``aingest_documents_stream``
        call sized to the number of files supplied.
        """
        url = f"{self._base_url}/v1/ingest/job"
        payload: dict[str, Any] = {
            "expected_documents": expected_documents,
            "retain_results": retain_results,
        }
        if label is not None:
            payload["label"] = label
        resp = await client.post(url, json=payload)
        # A 404/410 here means the deployed service does not advertise
        # the job-scoped ingest API.  Surface a dedicated compatibility
        # error instead of a generic HTTPStatusError so callers see one
        # actionable message — see the 26.05-RC2 release-integration
        # regression report.
        if _is_api_mismatch_status(resp.status_code):
            raise RetrieverServiceCompatibilityError(
                _compat_error_message(
                    url=url,
                    status=resp.status_code,
                    body=resp.text if resp.text else "",
                )
            )
        if resp.status_code >= 400:
            detail = resp.text[:500] if resp.text else "(empty)"
            raise httpx.HTTPStatusError(
                f"Job creation failed: HTTP {resp.status_code}: {detail}",
                request=resp.request,
                response=resp,
            )
        body = resp.json()
        job_id = body.get("job_id")
        if not job_id:
            raise RuntimeError(f"Job creation returned no job_id: {body!r}")
        return job_id

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    async def _upload_one(
        self,
        client: httpx.AsyncClient,
        file_path: Path,
        *,
        job_id: str,
        metadata: dict[str, Any] | None = None,
        pipeline_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Upload a file under an existing job, with retry on 429 + transient errors.

        Posts to ``POST /v1/ingest/job/{job_id}/document``; ``pipeline_spec``
        (when provided) is attached to ``metadata`` under the ``pipeline``
        key so the server can validate and apply it. Returns the parsed
        JSON response (contains ``document_id`` and ``job_id``).
        """
        file_bytes = file_path.read_bytes()
        filename = file_path.name
        meta_payload: dict[str, Any] = dict(metadata or {})
        if pipeline_spec is not None:
            meta_payload["pipeline"] = pipeline_spec
        meta_json = json.dumps(meta_payload)
        transport_attempts = 0
        url = f"{self._base_url}/v1/ingest/job/{job_id}/document"

        for attempt in range(1, _MAX_UPLOAD_RETRIES + 1):
            try:
                resp = await client.post(
                    url,
                    files={"file": (filename, file_bytes, "application/octet-stream")},
                    data={"metadata": meta_json},
                )
            except _TRANSIENT_ERRORS as exc:
                transport_attempts += 1
                if transport_attempts > 5:
                    raise
                delay = min(_DEFAULT_RETRY_AFTER * (2 ** (transport_attempts - 1)), 60.0)
                logger.debug("Transient %s uploading %s, retry in %.1fs", type(exc).__name__, filename, delay)
                await asyncio.sleep(delay)
                continue

            if resp.status_code == 429:
                delay = float(resp.headers.get("retry-after", _DEFAULT_RETRY_AFTER))
                logger.debug("429 for %s, retry in %.1fs (attempt %d)", filename, delay, attempt)
                await asyncio.sleep(delay)
                continue

            # Same compatibility-mismatch translation as `_create_job`.
            # If the service did not have the job-scoped upload route at
            # job-create time it would have already failed there; a
            # 404/410 here usually means a rolling upgrade pointed the
            # client at a stale pod after the job was created on a
            # newer one.  Either way, the actionable advice is the same.
            if _is_api_mismatch_status(resp.status_code):
                raise RetrieverServiceCompatibilityError(
                    _compat_error_message(
                        url=url,
                        status=resp.status_code,
                        body=resp.text if resp.text else "",
                    )
                )

            if resp.status_code >= 400:
                detail = resp.text[:500] if resp.text else "(empty)"
                raise httpx.HTTPStatusError(
                    f"Upload of {filename} returned HTTP {resp.status_code}: {detail}",
                    request=resp.request,
                    response=resp,
                )

            return resp.json()

        raise RuntimeError(f"Upload of {filename} failed after {_MAX_UPLOAD_RETRIES} retries")

    # ------------------------------------------------------------------
    # SSE consumer
    # ------------------------------------------------------------------

    async def _consume_sse(
        self,
        client: httpx.AsyncClient,
        pending: set[str],
        uploads_done: asyncio.Event,
        tracker: DocumentTracker,
        *,
        job_id: str,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        """Consume the per-job SSE stream until all pending items resolve.

        Subscribes to ``GET /v1/ingest/job/{job_id}/events`` (J4+). The
        firehose endpoint was removed deliberately — every public SSE
        consumer must declare the job it is observing. ``pending`` is
        the set of document ids we expect to terminate; reconciliation
        handles the race where an event arrives before the upload
        response added the id to ``pending``.
        """
        url = f"{self._base_url}/v1/ingest/job/{job_id}/events"
        seen_terminal: set[str] = set()
        seen_events: dict[str, dict[str, Any]] = {}

        def _reconcile() -> None:
            late = pending & seen_terminal
            if not late:
                return
            for doc_id in late:
                evt = seen_events.get(doc_id, {})
                status = evt.get("status", "completed")
                if status == "failed":
                    tracker.mark_failed(doc_id, evt.get("error"), evt)
                else:
                    tracker.mark_completed(doc_id, evt)
                if on_event:
                    on_event(evt)
            pending.difference_update(late)
            logger.debug("Reconciled %d late items", len(late))

        def _is_done() -> bool:
            if not uploads_done.is_set():
                return False
            _reconcile()
            return not pending

        _JOB_LIFECYCLE_EVENTS = {
            "job_created",
            "job_started",
            "job_progress",
            "job_finalized",
            "job_partial",
            "job_failed",
        }

        try:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"SSE endpoint returned HTTP {response.status_code}")

                event_type = ""
                data_buf = ""

                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_buf = line[5:].strip()
                    elif line == "" and data_buf:
                        try:
                            event = json.loads(data_buf)
                        except json.JSONDecodeError:
                            data_buf = ""
                            event_type = ""
                            continue

                        # Capture event name (SSE 'event:' field) inline so
                        # downstream consumers can demux without re-parsing
                        # the wire format.
                        if event_type and "event" not in event:
                            event["event"] = event_type
                        evt_name = event.get("event", event_type)
                        data_buf = ""
                        event_type = ""

                        if evt_name in _JOB_LIFECYCLE_EVENTS:
                            if on_event:
                                on_event(event)
                            if _is_done():
                                break
                            continue

                        item_id = event.get("id", "")
                        status = event.get("status", evt_name)

                        seen_terminal.add(item_id)
                        seen_events[item_id] = event

                        if item_id not in pending:
                            if _is_done():
                                break
                            continue

                        pending.discard(item_id)
                        error_msg = event.get("error")
                        if status == "failed":
                            tracker.mark_failed(item_id, error_msg, event)
                        else:
                            tracker.mark_completed(item_id, event)

                        if on_event:
                            on_event(event)

                        if _is_done():
                            break
                    elif line.startswith(":"):
                        if _is_done():
                            break

        except Exception as exc:
            logger.warning("SSE stream error: %s: %s", type(exc).__name__, exc)

        if pending:
            logger.info("SSE closed with %d items pending — falling back to bulk poll", len(pending))
            await self._bulk_poll_fallback(client, pending, tracker, on_event)

    # ------------------------------------------------------------------
    # Bulk-poll fallback
    # ------------------------------------------------------------------

    async def _bulk_poll_fallback(
        self,
        client: httpx.AsyncClient,
        pending: set[str],
        tracker: DocumentTracker,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        """Poll ``POST /v1/ingest/status/batch`` until all pending items resolve."""
        url = f"{self._base_url}/v1/ingest/status/batch"
        t0 = time.perf_counter()
        consecutive_errors = 0

        while pending and (time.perf_counter() - t0) < _BULK_POLL_TIMEOUT_S:
            try:
                resp = await client.post(url, json={"ids": list(pending)})
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    logger.error("Bulk poll: too many errors, giving up")
                    break
                logger.warning("Bulk poll error (%s), attempt %d", exc, consecutive_errors)
                await asyncio.sleep(_BULK_POLL_INTERVAL_S * min(consecutive_errors, 6))
                continue

            if resp.status_code != 200:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    break
                await asyncio.sleep(_BULK_POLL_INTERVAL_S)
                continue

            consecutive_errors = 0
            body = resp.json()
            items = body.get("items", {})
            for doc_id, info in items.items():
                if doc_id not in pending:
                    continue
                status = info.get("status", "")
                if status in ("completed", "failed"):
                    pending.discard(doc_id)
                    event = {"id": doc_id, "status": status, "result_rows": info.get("result_rows", 0)}
                    error_msg = info.get("error")
                    if error_msg:
                        event["error"] = error_msg
                    if status == "failed":
                        tracker.mark_failed(doc_id, error_msg, event)
                    else:
                        tracker.mark_completed(doc_id, event)
                    if on_event:
                        on_event(event)

            if pending:
                await asyncio.sleep(_BULK_POLL_INTERVAL_S)

    # ------------------------------------------------------------------
    # Public: materialized ingest (upload all, wait for all, return list)
    # ------------------------------------------------------------------

    async def ingest_documents(
        self,
        files: list[Path],
        *,
        on_file_submitted: Callable[[str, str], Any] | None = None,
        show_progress: bool = True,
        pipeline_spec: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Upload documents and wait for all to complete.

        Opens the SSE stream first, then uploads concurrently.  Returns
        one result dict per document containing ``id``, ``status``,
        ``result_rows``, ``elapsed_s``, and optionally ``error``.

        Parameters
        ----------
        on_file_submitted
            Called with ``(filename, document_id)`` after each upload.
        show_progress
            Show Rich progress bars during upload and SSE tracking.
        pipeline_spec
            Optional :class:`PipelineSpec` dict attached to every upload's
            ``metadata`` form blob (see
            :meth:`aingest_documents_stream` for details).
        """
        tracker = DocumentTracker()
        pending = tracker.pending
        uploads_done = asyncio.Event()
        document_ids: list[str] = []

        pool_limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        timeout = httpx.Timeout(timeout=None, connect=30.0)

        t0 = time.monotonic()

        async with httpx.AsyncClient(
            timeout=timeout,
            limits=pool_limits,
            headers=self._auth_headers,
        ) as client:
            job_id = await self._create_job(client, expected_documents=len(files))
            upload_sem = asyncio.Semaphore(self._max_concurrency)
            upload_failures: list[tuple[str, str]] = []

            async def _upload_one_file(fpath: Path) -> None:
                async with upload_sem:
                    try:
                        resp_json = await self._upload_one(client, fpath, job_id=job_id, pipeline_spec=pipeline_spec)
                        doc_id = resp_json.get("document_id", "")
                        if doc_id:
                            pending.add(doc_id)
                            document_ids.append(doc_id)
                            if on_file_submitted:
                                on_file_submitted(fpath.name, doc_id)
                    except Exception as exc:
                        upload_failures.append((fpath.name, str(exc)))
                        logger.error("Upload failed for %s: %s", fpath.name, exc)

            progress_ctx = _make_progress() if show_progress else None

            async def _upload_all() -> None:
                tasks = [asyncio.create_task(_upload_one_file(f)) for f in files]
                if progress_ctx:
                    task_id = progress_ctx.add_task("[blue]Uploading", total=len(files))
                    done = 0
                    for coro in asyncio.as_completed(tasks):
                        await coro
                        done += 1
                        progress_ctx.update(task_id, completed=done)
                else:
                    await asyncio.gather(*tasks)
                uploads_done.set()

            if progress_ctx:
                with progress_ctx:
                    sse_task = asyncio.create_task(
                        self._consume_sse(client, pending, uploads_done, tracker, job_id=job_id)
                    )
                    await asyncio.sleep(0.3)
                    await _upload_all()
                    await sse_task
            else:
                sse_task = asyncio.create_task(self._consume_sse(client, pending, uploads_done, tracker, job_id=job_id))
                await asyncio.sleep(0.3)
                await _upload_all()
                await sse_task

        elapsed = time.monotonic() - t0

        for fname, err in upload_failures:
            tracker.errors.append((fname, f"upload failed: {err}"))

        if show_progress:
            mins, secs = divmod(elapsed, 60)
            pps_str = f"{len(files) / elapsed:.1f}" if elapsed > 0 else "N/A"
            print(
                f"\n  Completed: {tracker.completed}  Failed: {tracker.failed}  "
                f"Upload errors: {len(upload_failures)}"
            )
            print(f"  Wall time: {int(mins)}m {secs:.1f}s  ({pps_str} docs/sec)")
            if tracker.errors:
                print(f"\n  Errors ({len(tracker.errors)}):")
                for eid, emsg in tracker.errors[:20]:
                    print(f"    {eid}: {emsg[:100]}")

        return tracker.results

    # ------------------------------------------------------------------
    # Public: streaming ingest (yields events as they arrive)
    # ------------------------------------------------------------------

    async def aingest_documents_stream(
        self,
        files: list[Path],
        *,
        pipeline_spec: dict[str, Any] | None = None,
        retain_results: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async generator: upload files, yield events as documents complete.

        Opens SSE before uploading.  Yields dicts with:

        * ``{"event": "upload_complete", "filename": ..., "document_id": ...}``
        * ``{"event": "document_complete", "document_id": ..., "status": ...,
              "result_rows": ..., "elapsed_s": ..., "error": ...}``

        When *pipeline_spec* is provided it is attached to each upload's
        ``metadata`` form blob under the ``pipeline`` key so the server can
        validate and apply per-request pipeline overrides.
        """
        tracker = DocumentTracker()
        pending = tracker.pending
        uploads_done = asyncio.Event()
        event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        pool_limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        timeout = httpx.Timeout(timeout=None, connect=30.0)

        async with httpx.AsyncClient(
            timeout=timeout,
            limits=pool_limits,
            headers=self._auth_headers,
        ) as client:
            job_id = await self._create_job(
                client,
                expected_documents=len(files),
                retain_results=retain_results,
            )
            yield {
                "event": "job_created",
                "job_id": job_id,
                "expected_documents": len(files),
            }
            upload_sem = asyncio.Semaphore(self._max_concurrency)

            async def _upload_one_file(fpath: Path) -> None:
                async with upload_sem:
                    try:
                        resp_json = await self._upload_one(client, fpath, job_id=job_id, pipeline_spec=pipeline_spec)
                        doc_id = resp_json.get("document_id", "")
                        if doc_id:
                            pending.add(doc_id)
                            await event_queue.put(
                                {
                                    "event": "upload_complete",
                                    "filename": fpath.name,
                                    "document_id": doc_id,
                                    "job_id": job_id,
                                }
                            )
                    except Exception as exc:
                        logger.error("Upload failed for %s: %s", fpath.name, exc)
                        await event_queue.put(
                            {
                                "event": "upload_failed",
                                "filename": fpath.name,
                                "error": str(exc),
                                "job_id": job_id,
                            }
                        )

            async def _upload_all() -> None:
                tasks = [asyncio.create_task(_upload_one_file(f)) for f in files]
                await asyncio.gather(*tasks, return_exceptions=True)
                uploads_done.set()

            def _on_sse_event(event: dict[str, Any]) -> None:
                event_name = event.get("event")
                if event_name in {
                    "job_created",
                    "job_started",
                    "job_progress",
                    "job_finalized",
                    "job_partial",
                    "job_failed",
                }:
                    payload = dict(event)
                    payload.setdefault("job_id", job_id)
                    event_queue.put_nowait(payload)
                    return
                doc_id = event.get("id", "")
                status = event.get("status", "completed")
                if status not in ("completed", "failed"):
                    return
                event_queue.put_nowait(
                    {
                        "event": "document_complete",
                        "document_id": doc_id,
                        "status": status,
                        "result_rows": event.get("result_rows", 0),
                        "elapsed_s": event.get("elapsed_s"),
                        "error": event.get("error"),
                        "job_id": job_id,
                    }
                )

            async def _sse_then_signal() -> None:
                await self._consume_sse(
                    client,
                    pending,
                    uploads_done,
                    tracker,
                    job_id=job_id,
                    on_event=_on_sse_event,
                )
                await event_queue.put(None)

            sse_task = asyncio.create_task(_sse_then_signal())
            await asyncio.sleep(0.3)
            upload_task = asyncio.create_task(_upload_all())

            while True:
                event = await event_queue.get()
                if event is None:
                    break
                yield event

            await upload_task
            await sse_task
