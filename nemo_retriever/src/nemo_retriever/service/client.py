# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async CLI client for submitting documents to the retriever service.

Key features:
- Whole-document upload (server-side PDF splitting via POST /v1/ingest/job)
- Job-level tracking (one job per input file)
- SSE subscription for job_complete events, with polling fallback
- On job_complete, fetches full results from GET /v1/ingest/job/{id}/results
- Rich live progress bars and metrics table
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from contextlib import nullcontext
from typing import Any, AsyncIterator, Callable

import httpx
from rich.bar import Bar
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)

_DEFAULT_RETRY_AFTER = 10.0
_MAX_BACKOFF = 120.0
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0

# Transport-level errors we retry on.  These show up when the connection
# is reset mid-flight (typical of Kubernetes NodePort / kube-proxy under
# bursty load): peer closes TCP before sending response headers.
_TRANSIENT_HTTPX_ERRORS: tuple[type[Exception], ...] = (
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
)
_UPLOAD_TRANSIENT_RETRIES = 5
_SSE_TRANSIENT_RETRIES = 10

console = Console()




# ------------------------------------------------------------------
# Failure record
# ------------------------------------------------------------------


class _FailedPage:
    """Captures a single page failure for end-of-run reporting."""

    __slots__ = ("source_file", "page_number", "error")

    def __init__(self, source_file: str, page_number: int, error: str) -> None:
        self.source_file = source_file
        self.page_number = page_number
        self.error = error

    @property
    def page_id(self) -> str:
        return f"{self.source_file}:{self.page_number}"


# ------------------------------------------------------------------
# Job tracker
# ------------------------------------------------------------------


class _JobTracker:
    """Tracks per-job progress, aggregated metrics, and failures."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.metrics: dict[str, dict[str, int]] = {}
        self.failed_pages: list[_FailedPage] = []

    def register_job(self, job_id: str, filename: str, total_pages: int) -> None:
        self.jobs[job_id] = {
            "filename": filename,
            "total_pages": total_pages,
            "pages_completed": 0,
            "pages_failed": 0,
            "status": "uploading",
            "busy_pages": 0,
        }

    def mark_busy(self, job_id: str) -> None:
        """A page for this job got a 503; show the job as server-busy."""
        if job_id in self.jobs:
            j = self.jobs[job_id]
            j["busy_pages"] = j.get("busy_pages", 0) + 1
            if j["status"] in ("uploading",):
                j["status"] = "server_busy"

    def mark_not_busy(self, job_id: str) -> None:
        """A previously-503'd page was accepted; clear busy state if no others waiting."""
        if job_id in self.jobs:
            j = self.jobs[job_id]
            j["busy_pages"] = max(j.get("busy_pages", 1) - 1, 0)
            if j["busy_pages"] == 0 and j["status"] == "server_busy":
                j["status"] = "uploading"

    def page_completed(self, job_id: str) -> None:
        """Increment by one; capped at total_pages to avoid over-count."""
        if job_id in self.jobs:
            j = self.jobs[job_id]
            j["pages_completed"] = min(j["pages_completed"] + 1, j["total_pages"])
            if j["status"] in ("uploading", "server_busy"):
                j["status"] = "processing"

    def set_pages_completed(self, job_id: str, count: int) -> None:
        """Set completed count (from polling); only moves forward."""
        if job_id in self.jobs:
            j = self.jobs[job_id]
            j["pages_completed"] = min(max(j["pages_completed"], count), j["total_pages"])
            if count > 0 and j["status"] in ("uploading", "server_busy"):
                j["status"] = "processing"

    def job_done(self, job_id: str, status: str = "complete") -> None:
        if job_id in self.jobs:
            j = self.jobs[job_id]
            j["status"] = status
            j["pages_completed"] = j["total_pages"]


    @property
    def total_pages(self) -> int:
        return sum(j["total_pages"] for j in self.jobs.values())

    @property
    def total_completed(self) -> int:
        return sum(j["pages_completed"] for j in self.jobs.values())

    @property
    def total_failed(self) -> int:
        return len(self.failed_pages)

    @property
    def all_done(self) -> bool:
        return all(j["status"] in ("complete", "failed") for j in self.jobs.values())


# ------------------------------------------------------------------
# Rich live display
# ------------------------------------------------------------------


class _LiveDisplay:
    """Manages a compact Rich live display that scales to hundreds of files.

    Only the overall progress bar and *actively-processing* files are shown.
    Completed and queued jobs are counted but not listed individually.
    """

    def __init__(self, tracker: _JobTracker) -> None:
        self._tracker = tracker
        self._overall = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=50),
            MofNCompleteColumn(),
            TextColumn("pages"),
            TimeElapsedColumn(),
        )
        self._overall_task = self._overall.add_task("Overall", total=0)
        self._t0 = time.monotonic()

    def add_job(self, job_id: str, filename: str, total_pages: int) -> None:
        self._overall.update(self._overall_task, total=self._tracker.total_pages)

    def _mini_bar(self, completed: int, total: int, width: int = 20) -> Bar:
        """Build a small inline progress bar."""
        return Bar(size=total, begin=0, end=completed, width=width)

    def build_renderable(self) -> Group:
        t = self._tracker
        self._overall.update(self._overall_task, completed=t.total_completed)

        parts: list[Any] = [self._overall]

        # --- Status summary line ---
        n_complete = sum(1 for j in t.jobs.values() if j["status"] in ("complete", "failed") and j["pages_failed"] == 0)
        n_failed_jobs = sum(1 for j in t.jobs.values() if j.get("pages_failed", 0) > 0)
        n_busy = sum(1 for j in t.jobs.values() if j["status"] == "server_busy")
        n_active = sum(
            1
            for j in t.jobs.values()
            if j["status"] in ("uploading", "processing") and j["pages_completed"] < j["total_pages"]
        )
        n_queued = len(t.jobs) - n_complete - n_failed_jobs - n_active - n_busy

        elapsed = time.monotonic() - self._t0
        pps = t.total_completed / elapsed if elapsed > 0 and t.total_completed > 0 else 0.0

        status_parts = [
            f"  [green]{n_complete} done[/green]",
            f"[cyan]{n_active} active[/cyan]",
        ]
        if n_busy > 0:
            status_parts.append(f"[red]{n_busy} server busy[/red]")
        if n_queued > 0:
            status_parts.append(f"[dim]{n_queued} queued[/dim]")
        if n_failed_jobs > 0:
            status_parts.append(f"[red]{n_failed_jobs} with failures[/red]")
        status_parts.append(f"[bold]{pps:.1f} pages/sec[/bold]")
        parts.append("  |  ".join(status_parts))

        # --- Active jobs table (only in-progress files) ---
        active_jobs = [
            (jid, info)
            for jid, info in t.jobs.items()
            if info["status"] in ("uploading", "processing", "server_busy")
            and info["pages_completed"] < info["total_pages"]
        ]
        if active_jobs:
            active_table = Table(
                show_header=True,
                expand=False,
                padding=(0, 1),
                title=f"Active ({len(active_jobs)})",
                title_style="bold cyan",
            )
            active_table.add_column("File", style="bold", max_width=35, no_wrap=True)
            active_table.add_column("Progress", width=20, no_wrap=True)
            active_table.add_column("Pages", justify="right", width=10)
            active_table.add_column("Status", width=12)

            for _jid, info in sorted(active_jobs, key=lambda x: x[1]["filename"]):
                name = info["filename"]
                if len(name) > 35:
                    name = "…" + name[-33:]
                completed = info["pages_completed"]
                total = info["total_pages"]
                pct = (completed / total * 100) if total > 0 else 0

                bar = self._mini_bar(completed, total)
                pages_text = f"{completed}/{total}"
                status = info["status"]
                if status == "server_busy":
                    status_text = Text("server busy", style="bold red")
                elif status == "uploading":
                    status_text = Text("uploading", style="dark_orange")
                else:
                    status_text = Text(f"{pct:.0f}%", style="green")

                active_table.add_row(name, bar, pages_text, status_text)

            parts.append("")
            parts.append(active_table)

        # --- Detection metrics table ---
        if t.metrics:
            total_det = sum(v["detections"] for v in t.metrics.values())
            total_inv = sum(v["invocations"] for v in t.metrics.values())

            metrics_table = Table(
                show_header=True,
                expand=False,
                padding=(0, 1),
                title=f"Detections ({total_det:,} total, {total_inv:,} invocations)",
                title_style="bold",
            )
            metrics_table.add_column("Model", style="cyan", min_width=20)
            metrics_table.add_column("Invocations", justify="right")
            metrics_table.add_column("Detections", justify="right")
            for model, vals in sorted(t.metrics.items()):
                metrics_table.add_row(
                    model,
                    f"{vals['invocations']:,}",
                    f"{vals['detections']:,}",
                )
            parts.append("")
            parts.append(metrics_table)

        return Group(*parts)


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------


class RetrieverServiceClient:
    """Submits documents to a running retriever service and tracks results.

    The SSE connection is opened **in parallel** with page uploads so that
    events emitted by the server during uploading are never missed.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7670",
        max_concurrency: int = 8,
        *,
        api_token: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._capacity_event = asyncio.Event()
        self._capacity_event.set()
        self._max_concurrency = max_concurrency
        self._api_token = (api_token or "").strip() or None

    @property
    def _auth_headers(self) -> dict[str, str]:
        """Return the Authorization header dict (empty if no token configured)."""
        return {"Authorization": f"Bearer {self._api_token}"} if self._api_token else {}

    def notify_capacity_available(self) -> None:
        self._capacity_event.set()

    # ------------------------------------------------------------------
    # Whole-document upload
    # ------------------------------------------------------------------

    async def _upload_document(
        self,
        client: httpx.AsyncClient,
        file_path: Path,
        tracker: _JobTracker | None = None,
    ) -> dict[str, Any]:
        """Upload a full document to /v1/ingest/job, retrying on 503.

        The server splits PDFs into pages internally. Returns the JSON
        response including ``job_id``, ``total_pages``, etc.
        """
        async with self._semaphore:
            backoff = _DEFAULT_RETRY_AFTER
            was_busy = False
            transport_attempts = 0
            file_bytes = file_path.read_bytes()
            filename = file_path.name

            while True:
                meta = {"filename": filename}
                files = {"file": (filename, file_bytes)}
                data = {"metadata": json.dumps(meta)}
                try:
                    resp = await client.post(f"{self._base_url}/v1/ingest/job", files=files, data=data)
                except _TRANSIENT_HTTPX_ERRORS as exc:
                    transport_attempts += 1
                    if transport_attempts > _UPLOAD_TRANSIENT_RETRIES:
                        logger.warning(
                            "Upload of %s failed after %d attempts: %s",
                            filename,
                            transport_attempts,
                            exc.__class__.__name__,
                        )
                        raise
                    delay = min(_RETRY_BASE_DELAY * (2 ** (transport_attempts - 1)), _MAX_BACKOFF)
                    logger.debug(
                        "Transient %s on upload of %s (attempt %d/%d), sleeping %.1fs",
                        exc.__class__.__name__,
                        filename,
                        transport_attempts,
                        _UPLOAD_TRANSIENT_RETRIES,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code != 503:
                    if was_busy and tracker is not None:
                        job_id = resp.json().get("job_id", "")
                        if job_id:
                            tracker.mark_not_busy(job_id)
                    resp.raise_for_status()
                    return resp.json()

                retry_after = float(resp.headers.get("Retry-After", backoff))
                self._capacity_event.clear()
                try:
                    await asyncio.wait_for(self._capacity_event.wait(), timeout=retry_after)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2.0, _MAX_BACKOFF)

    # ------------------------------------------------------------------
    # SSE job stream (runs concurrently with uploads)
    # ------------------------------------------------------------------

    async def _stream_jobs(
        self,
        client: httpx.AsyncClient,
        job_ids: list[str],
        tracker: _JobTracker,
        uploads_done: asyncio.Event,
    ) -> None:
        """Open one SSE connection for all jobs and update tracker.

        ``job_ids`` is a **live list** that grows as files are split.
        The method waits for at least one job to appear, then opens the
        SSE stream.  Newly added jobs are picked up by the catchup poll.

        The ``uploads_done`` event is set once all pages have been uploaded.
        After that signal the method cross-checks each pending job against
        the REST API to catch events that were missed.
        """
        url = f"{self._base_url}/v1/ingest/stream/jobs"

        while not job_ids:
            await asyncio.sleep(0.05)

        pending = set(job_ids)

        for attempt in range(_MAX_RETRIES):
            try:
                async with client.stream(
                    "POST",
                    url,
                    json={"job_ids": list(pending)},
                ) as resp:
                    resp.raise_for_status()

                    async def _poll_one_job(jid: str) -> None:
                        try:
                            r = await client.get(f"{self._base_url}/v1/ingest/job/{jid}")
                            r.raise_for_status()
                            body = r.json()
                            completed = body.get("pages_completed", 0)
                            tracker.set_pages_completed(jid, completed)
                            status = body.get("status", "")
                            if status in ("complete", "failed"):
                                tracker.job_done(jid, status)
                                self.notify_capacity_available()
                                pending.discard(jid)
                        except Exception:
                            pass

                    async def _catchup_poll() -> None:
                        await uploads_done.wait()
                        pending.update(job_ids)
                        while pending:
                            await asyncio.sleep(2.0)
                            batch = list(pending)
                            await asyncio.gather(
                                *(_poll_one_job(jid) for jid in batch),
                                return_exceptions=True,
                            )

                    catchup_task = asyncio.create_task(_catchup_poll())

                    buffer = ""
                    stream_iter = resp.aiter_text().__aiter__()
                    while pending or not uploads_done.is_set():
                        for jid in job_ids:
                            pending.add(jid)

                        try:
                            chunk = await asyncio.wait_for(
                                stream_iter.__anext__(),
                                timeout=5.0,
                            )
                        except asyncio.TimeoutError:
                            continue
                        except StopAsyncIteration:
                            break

                        buffer += chunk
                        while "\n\n" in buffer:
                            raw_event, buffer = buffer.split("\n\n", 1)
                            event = _parse_sse_event(raw_event)
                            if event is None:
                                continue

                            await self._handle_sse_event(
                                event,
                                tracker,
                                pending,
                            )

                    catchup_task.cancel()
                    try:
                        await catchup_task
                    except asyncio.CancelledError:
                        pass

                if not pending:
                    return

            except (httpx.ConnectError, httpx.PoolTimeout, httpx.RemoteProtocolError) as exc:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                if attempt < _MAX_RETRIES - 1:
                    logger.warning("SSE connection failed (%s), retrying in %.0fs", exc, delay)
                    await asyncio.sleep(delay)
                else:
                    logger.warning("SSE failed after %d attempts, falling back to polling", _MAX_RETRIES)
                    break

        for jid in list(pending):
            await self._poll_job_until_complete(client, jid, tracker, 3.0)
            pending.discard(jid)

    async def _handle_sse_event(
        self,
        event: dict[str, Any],
        tracker: _JobTracker,
        pending: set[str],
    ) -> None:
        """Dispatch a single parsed SSE event to the tracker."""
        event_type = event.get("event", "message")
        jid = event.get("job_id")

        if event_type == "page_complete" and jid:
            tracker.page_completed(jid)

        elif event_type == "job_complete" and jid:
            self.notify_capacity_available()
            tracker.job_done(jid, "complete")
            pending.discard(jid)

        elif event_type == "session_complete":
            pending.clear()

        elif event_type == "document_complete":
            self.notify_capacity_available()

    # ------------------------------------------------------------------
    # Polling fallback
    # ------------------------------------------------------------------

    async def _poll_job_until_complete(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        tracker: _JobTracker,
        poll_interval: float,
    ) -> None:
        consecutive_errors = 0
        while True:
            try:
                resp = await client.get(f"{self._base_url}/v1/ingest/job/{job_id}")
                resp.raise_for_status()
                body = resp.json()
                consecutive_errors = 0
            except (httpx.ConnectError, httpx.PoolTimeout, httpx.RemoteProtocolError) as exc:
                consecutive_errors += 1
                if consecutive_errors > _MAX_RETRIES:
                    raise
                delay = _RETRY_BASE_DELAY * (2 ** (consecutive_errors - 1))
                logger.warning("[%s] poll failed (%s), retry in %.0fs", job_id[:8], exc, delay)
                await asyncio.sleep(delay)
                continue

            tracker.set_pages_completed(job_id, body.get("pages_completed", 0))

            status = body.get("status", "")
            if status in ("complete", "failed"):
                self.notify_capacity_available()
                tracker.job_done(job_id, status)
                return
            await asyncio.sleep(poll_interval)

    async def _poll_all_jobs(
        self,
        client: httpx.AsyncClient,
        job_ids: list[str],
        tracker: _JobTracker,
        poll_interval: float,
    ) -> None:
        tasks = [
            self._poll_job_until_complete(client, jid, tracker, poll_interval)
            for jid in job_ids
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Text preview (post-processing validation)
    # ------------------------------------------------------------------

    async def _fetch_text_previews(
        self,
        client: httpx.AsyncClient,
    ) -> dict[str, str]:
        """Fetch per-file reassembled text from ingest_metrics + document status.

        Returns ``{source_file: full_text}`` for every completed file.
        """
        try:
            resp = await client.get(f"{self._base_url}/v1/ingest_metrics")
            resp.raise_for_status()
            metrics = resp.json()
        except Exception:
            return {}

        file_doc_ids: dict[str, list[tuple[int, str]]] = {}
        for file_info in metrics.get("files", []):
            source = file_info.get("source_file", "unknown")
            for page in file_info.get("pages", []):
                doc_id = page.get("document_id")
                page_num = page.get("page_number", 0)
                if doc_id:
                    file_doc_ids.setdefault(source, []).append((page_num, doc_id))

        for pages in file_doc_ids.values():
            pages.sort(key=lambda t: t[0])

        async def _get_page_text(doc_id: str) -> str:
            try:
                r = await client.get(f"{self._base_url}/v1/ingest/status/{doc_id}")
                r.raise_for_status()
                body = r.json()
                parts = []
                for page in sorted(body.get("pages", []), key=lambda p: p.get("page_number", 0)):
                    text = page.get("content", {}).get("text", "")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                return "\n".join(parts)
            except Exception:
                return ""

        previews: dict[str, str] = {}
        for source_file, pages in file_doc_ids.items():
            tasks = [_get_page_text(doc_id) for _, doc_id in pages]
            page_texts = await asyncio.gather(*tasks)
            previews[source_file] = "\n\n".join(t for t in page_texts if t)

        return previews

    # ------------------------------------------------------------------
    # Fetch failed pages from server (for poll-based flow)
    # ------------------------------------------------------------------

    async def _fetch_failed_pages(
        self,
        client: httpx.AsyncClient,
        tracker: _JobTracker,
    ) -> None:
        """Pull per-page failure details from /v1/ingest_metrics if any pages
        failed and we don't already have their details from SSE events."""
        if not any(j.get("pages_failed", 0) > 0 for j in tracker.jobs.values()):
            if not tracker.failed_pages:
                return

        try:
            resp = await client.get(f"{self._base_url}/v1/ingest_metrics")
            resp.raise_for_status()
            body = resp.json()
        except Exception:
            return

        known = {fp.page_id for fp in tracker.failed_pages}
        for file_info in body.get("files", []):
            for page in file_info.get("pages", []):
                page_id = page.get("id", "")
                if page_id and page_id not in known:
                    status = page.get("status", "complete") if "status" in page else "complete"
                    if status == "failed":
                        tracker.failed_pages.append(
                            _FailedPage(
                                source_file=page.get("source_file", "unknown"),
                                page_number=page.get("page_number", 0),
                                error=page.get("error_message", "unknown error"),
                            )
                        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def ingest_documents(
        self,
        files: list[Path],
        use_sse: bool = True,
        poll_interval: float = 2.0,
        *,
        on_file_submitted: Callable[[str, int], Any] | None = None,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """Upload whole documents and track progress until all jobs complete.

        Each file is uploaded as a whole document to ``POST /v1/ingest/job``;
        the server performs PDF splitting internally. The client then waits
        for every job to finish (via SSE or polling) and returns the full
        results for each document.

        Parameters
        ----------
        on_file_submitted
            Optional callback invoked with ``(filename, total_pages)`` each
            time a file is uploaded and accepted by the server.
        show_progress
            When ``True`` (default), show a Rich live progress display and
            print summary tables at the end.  Set to ``False`` for
            programmatic / library usage.
        """
        tracker = _JobTracker()
        display = _LiveDisplay(tracker)

        job_plans: list[dict[str, Any]] = []
        job_ids: list[str] = []

        pool_limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        timeout = httpx.Timeout(600.0, connect=30.0)

        processing_t0 = time.monotonic()

        async with httpx.AsyncClient(timeout=timeout, limits=pool_limits, headers=self._auth_headers) as client:
            live_ctx = Live(display.build_renderable(), refresh_per_second=4) if show_progress else nullcontext()
            with live_ctx as live:
                refresh_task: asyncio.Task[None] | None = None
                if show_progress and live is not None:

                    async def _refresh_loop() -> None:
                        while True:
                            live.update(display.build_renderable())
                            await asyncio.sleep(0.25)

                    refresh_task = asyncio.create_task(_refresh_loop())

                try:
                    uploads_done = asyncio.Event()

                    async def _upload_all() -> None:
                        for fpath in files:
                            resp_json = await self._upload_document(client, fpath, tracker=tracker)
                            job_id = resp_json["job_id"]
                            total_pages = resp_json.get("total_pages", 1)
                            tracker.register_job(job_id, fpath.name, total_pages)
                            display.add_job(job_id, fpath.name, total_pages)
                            if on_file_submitted is not None:
                                on_file_submitted(fpath.name, total_pages)
                            job_plans.append(
                                {
                                    "job_id": job_id,
                                    "filename": fpath.name,
                                    "total_pages": total_pages,
                                }
                            )
                            job_ids.append(job_id)
                        uploads_done.set()

                    if use_sse:
                        await asyncio.gather(
                            self._stream_jobs(
                                client,
                                job_ids,
                                tracker,
                                uploads_done,
                            ),
                            _upload_all(),
                        )
                    else:
                        await _upload_all()
                        await self._poll_all_jobs(
                            client,
                            job_ids,
                            tracker,
                            poll_interval,
                        )

                finally:
                    if refresh_task is not None:
                        refresh_task.cancel()
                        try:
                            await refresh_task
                        except asyncio.CancelledError:
                            pass
                    if show_progress and live is not None:
                        live.update(display.build_renderable())

        processing_elapsed = time.monotonic() - processing_t0

        final_results: list[dict[str, Any]] = []
        text_previews: dict[str, str] = {}

        async with httpx.AsyncClient(timeout=timeout, limits=pool_limits, headers=self._auth_headers) as client:
            for plan in job_plans:
                try:
                    resp = await client.get(f"{self._base_url}/v1/ingest/job/{plan['job_id']}/results")
                    resp.raise_for_status()
                    final_results.append(resp.json())
                except Exception as exc:
                    logger.error("Failed to fetch final results for job %s: %s", plan["job_id"][:8], exc)

            if show_progress:
                await self._fetch_failed_pages(client, tracker)
                text_previews = await self._fetch_text_previews(client)

        if show_progress:
            _print_failure_summary(tracker)
            if text_previews:
                _print_text_previews(text_previews)
            _print_throughput(tracker.total_pages, processing_elapsed)

        return final_results

    # ------------------------------------------------------------------
    # Streaming entry point — yields per-document results
    # ------------------------------------------------------------------

    async def aingest_documents_stream(
        self,
        files: list[Path],
    ) -> AsyncIterator[dict[str, Any]]:
        """Async generator: upload whole files, poll for completion, yield results.

        Pure polling approach — no SSE. Each file is uploaded via
        ``POST /v1/ingest/job``, then the client polls up to 16 jobs at
        a time via ``GET /v1/ingest/job/{id}`` until they reach a terminal
        state, at which point results are fetched from
        ``GET /v1/ingest/job/{id}/results``.

        Yields:

        * ``{"event": "job_started", "job_id": ..., "filename": ..., "total_pages": ...}``
        * ``{"event": "job_complete", "job_id": ..., "results": {...}}``
        """
        _POLL_CONCURRENCY = 16
        _POLL_INTERVAL_S = 2.0
        _STATUS_REPORT_INTERVAL_S = 30.0

        pool_limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        timeout = httpx.Timeout(600.0, connect=30.0)

        async with httpx.AsyncClient(timeout=timeout, limits=pool_limits, headers=self._auth_headers) as client:
            jobs: dict[str, dict[str, Any]] = {}

            for fpath in files:
                resp_json = await self._upload_document(client, fpath)
                job_id = resp_json["job_id"]
                total_pages = resp_json.get("total_pages", 1)
                jobs[job_id] = {
                    "filename": fpath.name,
                    "total_pages": total_pages,
                    "submitted_at": time.monotonic(),
                }
                print(f"  [upload] {fpath.name} -> job_id={job_id[:8]}  ({total_pages} pages)")
                yield {
                    "event": "job_started",
                    "job_id": job_id,
                    "filename": fpath.name,
                    "total_pages": total_pages,
                }

            print(f"  All {len(jobs)} documents uploaded. Polling for results...")
            pending = set(jobs.keys())
            last_status_report = time.monotonic()

            while pending:
                batch = list(pending)[:_POLL_CONCURRENCY]
                statuses = await asyncio.gather(
                    *(self._poll_single_job_status(client, jid) for jid in batch),
                    return_exceptions=True,
                )

                for jid, status in zip(batch, statuses):
                    if isinstance(status, Exception):
                        logger.debug("[poll] error for %s: %s", jid[:8], status)
                        continue

                    if status in ("complete", "failed"):
                        pending.discard(jid)
                        results = await self._fetch_single_job_results(client, jid)
                        yield {"event": "job_complete", "job_id": jid, "results": results}

                now = time.monotonic()
                if pending and (now - last_status_report) >= _STATUS_REPORT_INTERVAL_S:
                    last_status_report = now
                    print(f"\n  --- {len(pending)} jobs still pending ---")
                    for jid in sorted(pending, key=lambda j: jobs[j]["submitted_at"]):
                        elapsed = now - jobs[jid]["submitted_at"]
                        fname = jobs[jid]["filename"]
                        pages = jobs[jid]["total_pages"]
                        print(f"    {jid[:8]}  {fname:<40} {pages:>3} pages  {elapsed:>6.0f}s")
                    print()

                if pending:
                    await asyncio.sleep(_POLL_INTERVAL_S)

    async def _poll_single_job_status(
        self, client: httpx.AsyncClient, job_id: str
    ) -> str:
        """Return the current status string for a job, or 'unknown'."""
        r = await client.get(f"{self._base_url}/v1/ingest/job/{job_id}")
        if r.status_code == 404:
            return "failed"
        r.raise_for_status()
        return r.json().get("status", "unknown")

    async def _fetch_single_job_results(
        self, client: httpx.AsyncClient, job_id: str
    ) -> dict[str, Any] | None:
        """Fetch full results for a completed job."""
        try:
            r = await client.get(f"{self._base_url}/v1/ingest/job/{job_id}/results")
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.warning("Failed to fetch results for job %s: %s", job_id[:8], exc)
            return None


# ------------------------------------------------------------------
# End-of-run failure report
# ------------------------------------------------------------------


def _print_failure_summary(tracker: _JobTracker) -> None:
    """Print a clear summary of any failed pages after the live display ends."""
    if not tracker.failed_pages:
        console.print(f"\n[bold green]All {tracker.total_pages} pages processed successfully.[/bold green]")
        return

    n_failed = len(tracker.failed_pages)
    n_ok = tracker.total_pages - n_failed
    console.print()
    console.print(
        f"[bold yellow]Processing complete: "
        f"{n_ok} succeeded, {n_failed} failed "
        f"(out of {tracker.total_pages} total pages)[/bold yellow]"
    )
    console.print()

    fail_table = Table(
        title=f"[bold red]Failed Pages ({n_failed})[/bold red]",
        show_header=True,
        expand=False,
    )
    fail_table.add_column("Page", style="bold", min_width=30)
    fail_table.add_column("Error", style="red")

    for fp in sorted(tracker.failed_pages, key=lambda f: (f.source_file, f.page_number)):
        fail_table.add_row(fp.page_id, fp.error)

    console.print(fail_table)
    console.print()


def _print_throughput(total_pages: int, elapsed_s: float) -> None:
    """Print pages-per-second throughput as the final output line."""
    if elapsed_s <= 0 or total_pages == 0:
        return
    pps = total_pages / elapsed_s
    mins, secs = divmod(elapsed_s, 60)
    if mins >= 1:
        elapsed_fmt = f"{int(mins)}m {secs:.1f}s"
    else:
        elapsed_fmt = f"{elapsed_s:.1f}s"
    console.print(
        f"[bold]Throughput:[/bold] {total_pages:,} pages in {elapsed_fmt}"
        f" = [bold green]{pps:.2f} pages/sec[/bold green]"
    )
    console.print()


def _print_text_previews(previews: dict[str, str]) -> None:
    """Print a truncated text preview per file for quick validation."""
    if not previews:
        return

    table = Table(
        title="Extracted Text Preview (per document)",
        show_lines=True,
        expand=True,
    )
    table.add_column("File", style="bold cyan", max_width=35, no_wrap=True)
    table.add_column("Pages", justify="right", width=6)
    table.add_column("Chars", justify="right", width=8)
    table.add_column("Text Preview", ratio=1)

    for source_file in sorted(previews):
        text = previews[source_file]
        char_count = len(text)
        page_count = text.count("\n\n") + 1 if text else 0

        if not text.strip():
            preview = "[dim italic]<empty>[/dim italic]"
        elif char_count <= 200:
            preview = text.replace("\n", " ")
        else:
            head = text[:100].replace("\n", " ")
            tail = text[-100:].replace("\n", " ")
            preview = f"{head} [dim]…[/dim] {tail}"

        display_name = source_file
        if len(display_name) > 35:
            display_name = "…" + display_name[-33:]

        table.add_row(
            display_name,
            str(page_count),
            f"{char_count:,}",
            preview,
        )

    console.print()
    console.print(table)
    console.print()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_sse_event(raw: str) -> dict[str, Any] | None:
    data_lines: list[str] = []
    for line in raw.strip().splitlines():
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif line.startswith(":"):
            continue
    if not data_lines:
        return None
    try:
        return json.loads("".join(data_lines))
    except json.JSONDecodeError:
        return None
