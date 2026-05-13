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

1. :meth:`ServiceIngestor.ingest` — sync, blocks until every document has
   finished, returns a :class:`ServiceIngestResult` (a ``list`` subclass
   holding per-document completion events, plus ``failures`` /
   ``document_ids`` / ``elapsed_s`` attributes).

2. :meth:`ServiceIngestor.ingest_stream` — sync generator yielding one
   ``dict`` per event (upload_complete, document_complete, upload_failed).

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
    can iterate it just like a normal list.  Each entry is a per-document
    completion event dict.

    Attributes
    ----------
    failures
        ``(document_id_or_filename, error_message)`` pairs for documents
        that failed during upload or pipeline processing.
    document_ids
        Document identifiers returned by the server, in upload order.
    elapsed_s
        Wall-clock seconds from first upload to last result.
    """

    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        super().__init__(items or [])
        self.failures: list[tuple[str, str]] = []
        self.document_ids: list[str] = []
        self.elapsed_s: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ServiceIngestResult(documents={len(self)}, "
            f"failures={len(self.failures)}, "
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
        Maximum concurrent document uploads (default 8).
    request_timeout_s
        Per-request HTTP timeout (default 600s for large documents).
    api_token
        Optional bearer token for service authentication.
    """

    RUN_MODE = "service"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:7670",
        documents: Optional[List[str]] = None,
        max_concurrency: int = 8,
        request_timeout_s: float = 600.0,
        api_token: str | None = None,
    ) -> None:
        super().__init__(documents=documents)
        self._base_url = base_url.rstrip("/")
        self._max_concurrency = max_concurrency
        self._request_timeout_s = request_timeout_s
        self._api_token = (api_token or "").strip() or None
        self._document_ids: list[str] = []
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
        a meaningful source filename.
        """
        if isinstance(buffers, tuple):
            buffers = [buffers]
        for name, buf in buffers:
            self._buffers.append((name, buf))
        return self

    def load(self) -> "ServiceIngestor":
        """No-op for service mode."""
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
            A list of per-document completion events, with extra
            ``failures`` / ``document_ids`` / ``elapsed_s`` attributes.
        """
        del params, kwargs
        result = ServiceIngestResult()
        t0 = time.monotonic()

        documents_completed = 0
        documents_failed = 0
        total_uploaded = 0

        for evt in self.ingest_stream():
            event_type = evt.get("event")

            if event_type == "upload_complete":
                total_uploaded += 1
                print(
                    f"\r  Uploaded: {total_uploaded}  |  "
                    f"Completed: {documents_completed}  |  "
                    f"Failed: {documents_failed}",
                    end="",
                    flush=True,
                )

            elif event_type == "document_complete":
                status = evt.get("status", "completed")
                if status == "failed":
                    documents_failed += 1
                    error = evt.get("error", "unknown error")
                    doc_id = evt.get("document_id", "?")
                    result.failures.append((doc_id, error))
                else:
                    documents_completed += 1
                result.append(evt)
                print(
                    f"\r  Uploaded: {total_uploaded}  |  "
                    f"Completed: {documents_completed}  |  "
                    f"Failed: {documents_failed}",
                    end="",
                    flush=True,
                )

            elif event_type == "upload_failed":
                fname = evt.get("filename", "?")
                error = evt.get("error", "unknown")
                result.failures.append((fname, f"upload failed: {error}"))

        if total_uploaded > 0:
            print()

        result.document_ids = list(self._document_ids)
        result.elapsed_s = time.monotonic() - t0
        self._last_run_elapsed_s = result.elapsed_s
        return result

    # ------------------------------------------------------------------
    # Execution — sync streaming
    # ------------------------------------------------------------------

    def ingest_stream(self) -> Iterator[dict[str, Any]]:
        """Sync generator yielding events as documents are processed.

        Yields dicts with:

        * ``{"event": "upload_complete", "filename": ..., "document_id": ...}``
        * ``{"event": "document_complete", "document_id": ..., "status": ..., ...}``
        * ``{"event": "upload_failed", "filename": ..., "error": ...}``
        """
        files = self._collect_inputs()
        if not files:
            return iter(())

        self._document_ids.clear()

        def _record_doc_id(evt: dict[str, Any]) -> None:
            if evt.get("event") == "upload_complete":
                did = evt.get("document_id")
                if did:
                    self._document_ids.append(did)

        def _factory():
            return self._wrap_for_capture(self._aingest_stream_impl(files), _record_doc_id)

        bridge = _AsyncToSyncBridge(_factory)
        return iter(bridge)

    # ------------------------------------------------------------------
    # Execution — async streaming
    # ------------------------------------------------------------------

    async def aingest_stream(self) -> AsyncIterator[dict[str, Any]]:
        """Async generator yielding events as documents are processed."""
        files = self._collect_inputs()
        if not files:
            return

        self._document_ids.clear()
        async for evt in self._aingest_stream_impl(files):
            if evt.get("event") == "upload_complete":
                did = evt.get("document_id")
                if did:
                    self._document_ids.append(did)
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
    async def _wrap_for_capture(
        agen: AsyncIterator[dict[str, Any]],
        on_event,
    ) -> AsyncIterator[dict[str, Any]]:
        """Pass-through wrapper that lets the sync bridge capture document_ids."""
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
        """Run :meth:`ingest` on a background thread; return a ``Future``."""
        del return_failures, return_traces
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ServiceIngestorAsync")
        return executor.submit(self.ingest)

    # ------------------------------------------------------------------
    # Status & document-counter accessors
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, str]:
        """Return ``{document_id: status}`` for every document submitted so far."""
        if not self._document_ids:
            return {}
        url = f"{self._base_url}/v1/ingest/status/batch"
        with httpx.Client(timeout=30.0, headers=self._auth_headers) as client:
            try:
                resp = client.post(url, json={"ids": self._document_ids})
                resp.raise_for_status()
                items = resp.json().get("items", {})
                return {did: info.get("status", "unknown") for did, info in items.items()}
            except Exception as exc:
                logger.warning("Could not fetch bulk status: %s", exc)
                return {did: "unknown" for did in self._document_ids}

    def completed_jobs(self) -> int:
        return sum(1 for s in self.get_status().values() if s == "completed")

    def failed_jobs(self) -> int:
        return sum(1 for s in self.get_status().values() if s == "failed")

    def cancelled_jobs(self) -> int:
        return 0

    def remaining_jobs(self) -> int:
        return sum(1 for s in self.get_status().values() if s in ("processing", "unknown"))

    # ------------------------------------------------------------------
    # Cancel — not supported (no server endpoint)
    # ------------------------------------------------------------------

    def cancel(self, job_id: str | None = None) -> dict[str, Any]:
        """Not supported — the server does not expose a cancel endpoint."""
        raise NotImplementedError(
            "Cancel is not supported in service mode. " "The server does not currently expose a cancel endpoint."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_inputs(self) -> list[Path]:
        """Gather both file paths and any in-memory buffers into Paths."""
        files = [Path(p) for p in self._documents]

        if self._buffers:
            import tempfile

            tmp_dir = Path(tempfile.mkdtemp(prefix="service_ingestor_"))
            for name, buf in self._buffers:
                target = tmp_dir / name
                target.write_bytes(buf.getvalue())
                files.append(target)

        return files
