# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async client for submitting documents to the retriever service.

Uploads whole documents via ``POST /v1/ingest``, tracks completion via
the ``GET /v1/ingest/events`` SSE stream (with ``POST /v1/ingest/status/batch``
bulk-poll fallback), and surfaces results through both materialized and
streaming interfaces.

The SSE connection is opened **before** uploads begin so that completion
events for fast-finishing documents are never missed.  A ``seen_terminal``
buffer reconciles events that arrive before the client registers the
corresponding ``document_id`` from the upload response.
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

    Uses ``POST /v1/ingest`` for uploads, ``GET /v1/ingest/events`` SSE
    for real-time completion tracking, and ``POST /v1/ingest/status/batch``
    as a bulk-poll fallback.
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
    # Upload
    # ------------------------------------------------------------------

    async def _upload_one(
        self,
        client: httpx.AsyncClient,
        file_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Upload a file to ``POST /v1/ingest`` with retry on 429 and transient errors.

        Returns the parsed JSON response (contains ``document_id``).
        """
        file_bytes = file_path.read_bytes()
        filename = file_path.name
        meta_json = json.dumps(metadata or {})
        transport_attempts = 0

        for attempt in range(1, _MAX_UPLOAD_RETRIES + 1):
            try:
                resp = await client.post(
                    f"{self._base_url}/v1/ingest",
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
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        """Consume ``GET /v1/ingest/events`` SSE stream until all pending items resolve.

        Uses ``seen_terminal`` reconciliation to handle events that arrive
        before the upload response adds the ``document_id`` to ``pending``.
        """
        url = f"{self._base_url}/v1/ingest/events"
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

                        item_id = event.get("id", "")
                        status = event.get("status", event_type)
                        data_buf = ""
                        event_type = ""

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
            upload_sem = asyncio.Semaphore(self._max_concurrency)
            upload_failures: list[tuple[str, str]] = []

            async def _upload_one_file(fpath: Path) -> None:
                async with upload_sem:
                    try:
                        resp_json = await self._upload_one(client, fpath)
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
                    sse_task = asyncio.create_task(self._consume_sse(client, pending, uploads_done, tracker))
                    await asyncio.sleep(0.3)
                    await _upload_all()
                    await sse_task
            else:
                sse_task = asyncio.create_task(self._consume_sse(client, pending, uploads_done, tracker))
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
    ) -> AsyncIterator[dict[str, Any]]:
        """Async generator: upload files, yield events as documents complete.

        Opens SSE before uploading.  Yields dicts with:

        * ``{"event": "upload_complete", "filename": ..., "document_id": ...}``
        * ``{"event": "document_complete", "document_id": ..., "status": ...,
              "result_rows": ..., "elapsed_s": ..., "error": ...}``
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
            upload_sem = asyncio.Semaphore(self._max_concurrency)

            async def _upload_one_file(fpath: Path) -> None:
                async with upload_sem:
                    try:
                        resp_json = await self._upload_one(client, fpath)
                        doc_id = resp_json.get("document_id", "")
                        if doc_id:
                            pending.add(doc_id)
                            await event_queue.put(
                                {
                                    "event": "upload_complete",
                                    "filename": fpath.name,
                                    "document_id": doc_id,
                                }
                            )
                    except Exception as exc:
                        logger.error("Upload failed for %s: %s", fpath.name, exc)
                        await event_queue.put(
                            {
                                "event": "upload_failed",
                                "filename": fpath.name,
                                "error": str(exc),
                            }
                        )

            async def _upload_all() -> None:
                tasks = [asyncio.create_task(_upload_one_file(f)) for f in files]
                await asyncio.gather(*tasks, return_exceptions=True)
                uploads_done.set()

            def _on_sse_event(event: dict[str, Any]) -> None:
                doc_id = event.get("id", "")
                status = event.get("status", "completed")
                event_queue.put_nowait(
                    {
                        "event": "document_complete",
                        "document_id": doc_id,
                        "status": status,
                        "result_rows": event.get("result_rows", 0),
                        "elapsed_s": event.get("elapsed_s"),
                        "error": event.get("error"),
                    }
                )

            async def _sse_then_signal() -> None:
                await self._consume_sse(client, pending, uploads_done, tracker, on_event=_on_sse_event)
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
