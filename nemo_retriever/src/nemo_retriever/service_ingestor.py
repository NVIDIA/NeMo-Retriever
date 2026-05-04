# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingestor that submits work to a running ``retriever service`` HTTP server.

This is the third ``run_mode`` exposed via :func:`nemo_retriever.ingestor.create_ingestor`.
Where ``inprocess`` and ``batch`` execute the operator graph in the caller's
process / Ray cluster, ``service`` mode delegates execution to a separate
FastAPI server (see :mod:`nemo_retriever.service`) that runs its own pool of
worker processes and remote NIM endpoints.

Three execution surfaces are exposed:

1. :meth:`ServiceIngestor.ingest` — sync, blocks until every page of every
   file has finished, returns a :class:`ServiceIngestResult` (a ``list``
   subclass holding all per-page output rows in completion order, plus
   ``failures`` / ``metrics`` / ``job_ids`` / ``elapsed_s`` attributes).

2. :meth:`ServiceIngestor.ingest_stream` — sync generator yielding one
   ``dict`` per page as it lands.  Internally runs the async pipeline on a
   background thread and bridges via ``queue.Queue``.

3. :meth:`ServiceIngestor.aingest_stream` — true async generator for
   callers already inside an event loop.

The fluent pipeline-configuration methods (``.extract``, ``.embed``,
``.dedup``, ``.split``, ``.store``, ``.caption``, ``.webhook``, ``.udf``,
``.vdb_upload``, …) all raise :class:`NotImplementedError` with a clear
message: the server pipeline is configured at startup via
``retriever-service.yaml`` and cannot be overridden per-request today.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Union

import httpx

from nemo_retriever.ingestor import ingestor

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


class ServiceIngestResult(list):
    """Materialized result returned by :meth:`ServiceIngestor.ingest`.

    Subclasses ``list`` so it satisfies the existing
    ``ingestor.ingest()`` return-type annotation (``List[Any]``); callers
    can iterate it just like a normal list.  Additional metadata about
    the run is attached as attributes.

    Attributes
    ----------
    failures
        One entry per page that failed in the server pipeline.  Each
        entry contains ``job_id``, ``source_file``, ``page_number`` and
        ``error``.
    metrics
        Per-model aggregated detection / invocation counts, keyed by
        model name.
    job_ids
        Job identifiers created on the server, in submission order
        (one per input file).
    elapsed_s
        Wall-clock seconds from first upload to last result.
    """

    def __init__(self, pages: list[dict[str, Any]] | None = None) -> None:
        super().__init__(pages or [])
        self.failures: list[dict[str, Any]] = []
        self.metrics: dict[str, dict[str, int]] = {}
        self.job_ids: list[str] = []
        self.elapsed_s: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ServiceIngestResult(pages={len(self)}, "
            f"failures={len(self.failures)}, jobs={len(self.job_ids)}, "
            f"elapsed_s={self.elapsed_s:.2f})"
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

_FLUENT_NOT_SUPPORTED_TEMPLATE = (
    "ServiceIngestor.{method}() is not supported in run_mode='service'. "
    "The server-side pipeline is configured at service startup via "
    "retriever-service.yaml (see nim_endpoints + processing sections). "
    "To change extraction / embedding / store behaviour, restart the "
    "retriever service with an updated config; per-request overrides are "
    "not implemented today."
)


def _raise_unsupported(method: str) -> None:
    raise NotImplementedError(_FLUENT_NOT_SUPPORTED_TEMPLATE.format(method=method))


def _normalize_files(files: Union[str, List[str], List[Path]]) -> list[Path]:
    if isinstance(files, (str, Path)):
        return [Path(files)]
    return [Path(f) for f in files]


# ----------------------------------------------------------------------
# Async-to-sync queue bridge
# ----------------------------------------------------------------------


_SENTINEL = object()


class _AsyncToSyncBridge:
    """Run an async generator on a background thread and surface it as a sync iterator.

    The generator's items are funneled through a :class:`queue.Queue`; the
    sync side calls ``.get()`` blocking until the next item is ready.  The
    bridge owns its own asyncio event loop so the caller does not need
    one.
    """

    def __init__(self, agen_factory) -> None:
        self._agen_factory = agen_factory
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=64)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._exc: BaseException | None = None
        self._stop_event = threading.Event()

    def __iter__(self) -> Iterator[Any]:
        self._thread = threading.Thread(target=self._run, name="ServiceIngestorBridge", daemon=True)
        self._thread.start()
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    if self._exc is not None:
                        raise self._exc
                    return
                yield item
        finally:
            self._stop_event.set()
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5.0)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._drain())
        except BaseException as exc:  # noqa: BLE001 — we re-raise on the consumer side
            self._exc = exc
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            self._queue.put(_SENTINEL)

    async def _drain(self) -> None:
        agen = self._agen_factory()
        try:
            async for item in agen:
                while True:
                    if self._stop_event.is_set():
                        return
                    try:
                        self._queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            try:
                await agen.aclose()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass


# ----------------------------------------------------------------------
# ServiceIngestor
# ----------------------------------------------------------------------


class ServiceIngestor(ingestor):
    """Ingestor that submits work to a running ``retriever service``.

    Parameters
    ----------
    base_url
        Base URL of the retriever service (default ``http://localhost:7670``).
    documents
        Initial list of file paths to ingest; may also be set/extended via
        :meth:`files` and :meth:`buffers`.
    max_concurrency
        Maximum concurrent page uploads (default 8).
    request_timeout_s
        Per-request HTTP timeout (default 600s for large pages).
    poll_interval_s
        Used by :meth:`get_status` and the ``*_jobs`` counter methods to
        avoid re-querying the server too aggressively.
    """

    RUN_MODE = "service"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:7670",
        documents: Optional[List[str]] = None,
        max_concurrency: int = 8,
        request_timeout_s: float = 600.0,
        poll_interval_s: float = 2.0,
        api_token: str | None = None,
    ) -> None:
        super().__init__(documents=documents)
        self._base_url = base_url.rstrip("/")
        self._max_concurrency = max_concurrency
        self._request_timeout_s = request_timeout_s
        self._poll_interval_s = poll_interval_s
        self._api_token = (api_token or "").strip() or None
        self._job_ids: list[str] = []
        self._last_run_elapsed_s: float = 0.0

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_token}"} if self._api_token else {}

    # ------------------------------------------------------------------
    # Input configuration (these ARE meaningful client-side)
    # ------------------------------------------------------------------

    def files(self, documents: Union[str, List[str]]) -> "ServiceIngestor":
        """Add document paths/URIs for processing."""
        if isinstance(documents, str):
            self._documents.append(documents)
        else:
            self._documents.extend(documents)
        return self

    def buffers(
        self,
        buffers: Union[Tuple[str, BytesIO], List[Tuple[str, BytesIO]]],
    ) -> "ServiceIngestor":
        """Add in-memory buffers for processing.

        Each buffer must be ``(filename, BytesIO)`` so the server can record
        a meaningful source filename.  Buffers are written to a temporary
        file on disk before being uploaded so they share the same code
        path as on-disk documents.
        """
        if isinstance(buffers, tuple):
            buffers = [buffers]
        for name, buf in buffers:
            self._buffers.append((name, buf))
        return self

    def load(self) -> "ServiceIngestor":
        """No-op for service mode.

        Remote-URI fetching is not yet wired through the service ingestor;
        callers are expected to materialize remote documents to local paths
        before calling :meth:`files`.
        """
        return self

    # ------------------------------------------------------------------
    # Pipeline configuration — server-side only, not overridable per-request
    # ------------------------------------------------------------------

    def all_tasks(self) -> "ServiceIngestor":  # pragma: no cover - trivial
        _raise_unsupported("all_tasks")

    def dedup(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported("dedup")

    def embed(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported("embed")

    def extract(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported("extract")

    def extract_image_files(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported("extract_image_files")

    def filter(self) -> "ServiceIngestor":
        _raise_unsupported("filter")

    def split(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported("split")

    def store(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported("store")

    def store_embed(self) -> "ServiceIngestor":
        _raise_unsupported("store_embed")

    def udf(
        self,
        udf_function: str,
        udf_function_name: Optional[str] = None,
        phase: Optional[Union[int, str]] = None,
        target_stage: Optional[str] = None,
        run_before: bool = False,
        run_after: bool = False,
    ) -> "ServiceIngestor":
        _raise_unsupported("udf")

    def vdb_upload(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported("vdb_upload")

    def save_intermediate_results(self, output_dir: str) -> "ServiceIngestor":
        _raise_unsupported("save_intermediate_results")

    def save_to_disk(
        self,
        output_directory: Optional[str] = None,
        cleanup: bool = True,
        compression: Optional[str] = "gzip",
    ) -> "ServiceIngestor":
        _raise_unsupported("save_to_disk")

    def caption(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported("caption")

    def webhook(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported("webhook")

    def pdf_split_config(self, pages_per_chunk: int = 32) -> "ServiceIngestor":
        _raise_unsupported("pdf_split_config")

    # ------------------------------------------------------------------
    # Execution — sync materialized
    # ------------------------------------------------------------------

    def ingest(self, params: Any = None, **kwargs: Any) -> ServiceIngestResult:
        """Block until every document has finished processing on the server.

        Returns
        -------
        ServiceIngestResult
            A list (subclass of ``list[dict]``) of every per-page output row
            in completion order, with extra ``failures`` / ``metrics`` /
            ``job_ids`` / ``elapsed_s`` attributes.

        Notes
        -----
        - ``params`` and ``kwargs`` are accepted for parity with the base
          interface but ignored: there are no per-call execution knobs in
          service mode (use the YAML config / CLI flags on the server
          instead).
        - This method internally calls :meth:`ingest_stream` and collects
          results from ``job_complete`` events (which contain full document
          results fetched from the server).
        """
        del params, kwargs
        result = ServiceIngestResult()
        t0 = time.monotonic()

        event_counts: dict[str, int] = {}
        for evt in self.ingest_stream():
            event_type = evt.get("event")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            if event_type == "job_started":
                print(
                    f"[DEBUG] job_started: job_id={evt.get('job_id', '')[:8]}, "
                    f"filename={evt.get('filename')}, total_pages={evt.get('total_pages')}"
                )
            if event_type == "job_complete":
                job_results = evt.get("results")
                job_id_short = (evt.get("job_id") or "")[:8]
                if job_results and isinstance(job_results, dict):
                    input_pages = job_results.get("pages", [])
                    total_output_rows = sum(len(p.get("pages", [])) for p in input_pages)
                    print(
                        f"[DEBUG] job_complete {job_id_short}: "
                        f"filename={job_results.get('filename')}, "
                        f"status={job_results.get('status')}, "
                        f"input_pages={len(input_pages)}, "
                        f"output_rows={total_output_rows}, "
                        f"pages_completed={job_results.get('pages_completed')}"
                    )
                    for input_page in input_pages:
                        doc_id = input_page.get("document_id", "")
                        page_num = input_page.get("page_number", 0)
                        status = input_page.get("status", "")
                        for output_row in input_page.get("pages", []):
                            result.append(
                                {
                                    "job_id": evt.get("job_id"),
                                    "document_id": doc_id,
                                    "source_file": job_results.get("filename"),
                                    "page_number": page_num,
                                    "content": output_row.get("content", {}),
                                }
                            )
                        if status == "failed":
                            result.failures.append(
                                {
                                    "job_id": evt.get("job_id"),
                                    "source_file": job_results.get("filename"),
                                    "page_number": page_num,
                                    "error": f"page processing failed (status={status})",
                                }
                            )
                else:
                    print(
                        f"[DEBUG] job_complete {job_id_short}: "
                        f"results is None or not a dict! "
                        f"results_type={type(job_results)}, "
                        f"has_error={evt.get('error', 'no')}, "
                        f"synthetic={evt.get('synthetic', False)}"
                    )
            elif event_type == "page_result":
                for page in evt.get("pages", []):
                    result.append(
                        {
                            "job_id": evt.get("job_id"),
                            "document_id": evt.get("document_id"),
                            "source_file": evt.get("source_file"),
                            "page_number": evt.get("page_number"),
                            "content": page,
                        }
                    )
            elif event_type == "page_failed":
                result.failures.append(
                    {
                        "job_id": evt.get("job_id"),
                        "source_file": evt.get("source_file"),
                        "page_number": evt.get("page_number"),
                        "error": evt.get("error", "unknown error"),
                    }
                )
            elif event_type == "metrics_update":
                model = evt.get("model_name", "unknown")
                m = result.metrics.setdefault(model, {"invocations": 0, "detections": 0})
                m["invocations"] += int(evt.get("invocation_count", 0))
                m["detections"] += int(evt.get("detections_count", 0))
            elif event_type == "stream_overflow":
                logger.warning(
                    "Server SSE stream overflowed (subscriber too slow). "
                    "Falling back to polling for any remaining results."
                )
                self._poll_remaining_results(result)
                break
            elif event_type == "stream_error":
                logger.warning(
                    "Server SSE stream errored: %s. Falling back to polling.",
                    evt.get("error"),
                )
                self._poll_remaining_results(result)
                break

        print(f"[DEBUG] Stream finished. Event counts: {event_counts}")
        print(f"[DEBUG] Total result rows collected: {len(result)}, failures: {len(result.failures)}")
        result.job_ids = list(self._job_ids)
        result.elapsed_s = time.monotonic() - t0
        self._last_run_elapsed_s = result.elapsed_s
        return result

    # ------------------------------------------------------------------
    # Execution — sync streaming
    # ------------------------------------------------------------------

    def ingest_stream(self) -> Iterator[dict[str, Any]]:
        """Sync generator yielding one event dict per page as results land.

        See :meth:`aingest_stream` for the schema of yielded dicts.

        This wraps :meth:`aingest_stream` on a background thread so users
        can write straightforward synchronous code:

        >>> for evt in ingestor.ingest_stream():           # doctest: +SKIP
        ...     if evt["event"] == "page_result":
        ...         print(evt["source_file"], evt["page_number"])
        """
        files = self._collect_inputs()
        if not files:
            return iter(())  # empty

        self._job_ids.clear()

        def _record_job_id(evt: dict[str, Any]) -> None:
            if evt.get("event") == "job_started":
                jid = evt.get("job_id")
                if jid:
                    self._job_ids.append(jid)

        def _factory():
            return self._wrap_for_job_capture(self._aingest_stream_impl(files), _record_job_id)

        bridge = _AsyncToSyncBridge(_factory)
        return iter(bridge)

    # ------------------------------------------------------------------
    # Execution — async streaming
    # ------------------------------------------------------------------

    async def aingest_stream(self) -> AsyncIterator[dict[str, Any]]:
        """Async generator yielding one event dict per page as results land.

        Yields one of:

        * ``{"event": "job_started", "job_id": ..., "filename": ..., "total_pages": ...}``
        * ``{"event": "page_result", "job_id": ..., "source_file": ...,
              "page_number": <input page>, "document_id": ..., "pages": [<output rows>], ...}``
        * ``{"event": "page_failed", "job_id": ..., "source_file": ...,
              "page_number": ..., "error": ...}``
        * ``{"event": "metrics_update", ...}``
        * ``{"event": "job_complete", "job_id": ..., "filename": ...}``
        * ``{"event": "stream_overflow", ...}`` (server lost events; client
          should fall back to polling)
        * ``{"event": "stream_error", "error": ...}``
        """
        files = self._collect_inputs()
        if not files:
            return

        self._job_ids.clear()
        async for evt in self._aingest_stream_impl(files):
            if evt.get("event") == "job_started":
                jid = evt.get("job_id")
                if jid:
                    self._job_ids.append(jid)
            yield evt

    # ------------------------------------------------------------------
    # Async helper used by both sync and async streaming entry points
    # ------------------------------------------------------------------

    async def _aingest_stream_impl(
        self,
        files: list[Path],
    ) -> AsyncIterator[dict[str, Any]]:
        from nemo_retriever.service.client import RetrieverServiceClient

        client = RetrieverServiceClient(
            base_url=self._base_url,
            max_concurrency=self._max_concurrency,
            api_token=self._api_token,
        )
        async for evt in client.aingest_documents_stream(files=files):
            yield evt

    @staticmethod
    async def _wrap_for_job_capture(
        agen: AsyncIterator[dict[str, Any]],
        on_event,
    ) -> AsyncIterator[dict[str, Any]]:
        """Pass-through wrapper that lets the sync bridge capture job_ids."""
        async for evt in agen:
            on_event(evt)
            yield evt

    # ------------------------------------------------------------------
    # Async-future API
    # ------------------------------------------------------------------

    def ingest_async(
        self,
        *,
        return_failures: bool = False,
        return_traces: bool = False,
    ) -> Any:
        """Run :meth:`ingest` on a background thread; return a ``Future``.

        ``return_failures`` and ``return_traces`` are accepted for parity
        with the base interface; failures and metrics are always available
        via ``result.failures`` / ``result.metrics`` regardless.
        """
        del return_failures, return_traces
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ServiceIngestorAsync")
        return executor.submit(self.ingest)

    # ------------------------------------------------------------------
    # Status & job-counter accessors (use the new server endpoints)
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, str]:
        """Return ``{job_id: status}`` for every job submitted so far."""
        if not self._job_ids:
            return {}
        statuses: dict[str, str] = {}
        with httpx.Client(base_url=self._base_url, timeout=30.0, headers=self._auth_headers) as client:
            for jid in self._job_ids:
                try:
                    resp = client.get(f"/v1/ingest/job/{jid}")
                    resp.raise_for_status()
                    statuses[jid] = resp.json().get("status", "unknown")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not fetch status for job %s: %s", jid[:8], exc)
                    statuses[jid] = "unknown"
        return statuses

    def completed_jobs(self) -> int:
        return sum(1 for s in self.get_status().values() if s == "complete")

    def failed_jobs(self) -> int:
        return sum(1 for s in self.get_status().values() if s == "failed")

    def cancelled_jobs(self) -> int:
        return sum(1 for s in self.get_status().values() if s == "cancelled")

    def remaining_jobs(self) -> int:
        return sum(1 for s in self.get_status().values() if s in ("queued", "processing", "unknown"))

    # ------------------------------------------------------------------
    # Cancel — supported via POST /v1/ingest/job/{id}/cancel
    # ------------------------------------------------------------------

    def cancel(self, job_id: str | None = None) -> dict[str, Any]:
        """Cancel one job (by id) or every job submitted by this ingestor.

        Returns a dict ``{job_id: <server response>}``.  Pages already in
        flight on the server cannot be interrupted; any pages still queued
        or buffered are dropped before they reach a worker.
        """
        targets = [job_id] if job_id is not None else list(self._job_ids)
        if not targets:
            return {}

        results: dict[str, Any] = {}
        with httpx.Client(base_url=self._base_url, timeout=30.0, headers=self._auth_headers) as client:
            for jid in targets:
                try:
                    resp = client.post(f"/v1/ingest/job/{jid}/cancel")
                    resp.raise_for_status()
                    results[jid] = resp.json()
                except httpx.HTTPStatusError as exc:
                    results[jid] = {"error": str(exc), "status_code": exc.response.status_code}
                except Exception as exc:  # noqa: BLE001 — surfaces network errors
                    results[jid] = {"error": str(exc)}
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_inputs(self) -> list[Path]:
        """Gather both file paths and any in-memory buffers into Paths.

        Buffers are spilled to a temp directory; the temp directory lives
        for the lifetime of the process (deleted by the OS).
        """
        files = [Path(p) for p in self._documents]

        if self._buffers:
            import tempfile

            tmp_dir = Path(tempfile.mkdtemp(prefix="service_ingestor_"))
            for name, buf in self._buffers:
                target = tmp_dir / name
                target.write_bytes(buf.getvalue())
                files.append(target)

        return files

    def _poll_remaining_results(self, result: ServiceIngestResult) -> None:
        """Backfill any pages we missed via REST after an SSE failure."""
        if not self._job_ids:
            return

        seen_pages: set[tuple[str, int]] = {
            (p["document_id"], p["content"].get("_output_index", 0)) for p in result if p.get("document_id")
        }

        with httpx.Client(
            base_url=self._base_url,
            timeout=self._request_timeout_s,
            headers=self._auth_headers,
        ) as client:
            for jid in self._job_ids:
                try:
                    resp = client.get(f"/v1/ingest/job/{jid}/results")
                    resp.raise_for_status()
                    body = resp.json()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Polling fallback failed for job %s: %s", jid[:8], exc)
                    continue

                for input_page in body.get("pages", []):
                    doc_id = input_page.get("document_id")
                    for output_row in input_page.get("pages", []):
                        marker = (doc_id, output_row.get("page_number", 0))
                        if marker in seen_pages:
                            continue
                        seen_pages.add(marker)
                        result.append(
                            {
                                "job_id": jid,
                                "document_id": doc_id,
                                "source_file": body.get("filename"),
                                "page_number": input_page.get("page_number"),
                                "content": output_row.get("content", {}),
                            }
                        )
