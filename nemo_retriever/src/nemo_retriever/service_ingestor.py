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

from nemo_retriever.ingestor import _merge_params, ingestor
from nemo_retriever.params import (
    DedupParams,
    EmbedParams,
    ExtractParams,
    PdfSplitParams,
)

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
    "ServiceIngestor.{method}() is not yet supported in run_mode='service'. "
    "{phase_hint}"
    "See retriever-service.yaml or `retriever service describe` for the "
    "current per-request override policy."
)


def _raise_unsupported(method: str, *, phase_hint: str = "") -> None:
    raise NotImplementedError(
        _FLUENT_NOT_SUPPORTED_TEMPLATE.format(method=method, phase_hint=phase_hint + " " if phase_hint else "")
    )


def _normalize_files(files: Union[str, List[str], List[Path]]) -> list[Path]:
    if isinstance(files, (str, Path)):
        return [Path(files)]
    return [Path(f) for f in files]


# ----------------------------------------------------------------------
# Client-side mirror of service.models.pipeline_spec.PipelineSpec
# ----------------------------------------------------------------------

# Keys this client treats as server-owned. Stripped from any params dict
# before it goes on the wire so users get a clear error if they try.
_SERVER_OWNED_KEYS: frozenset[str] = frozenset(
    {
        "invoke_url",
        "api_key",
        "page_elements_invoke_url",
        "page_elements_api_key",
        "ocr_invoke_url",
        "ocr_api_key",
        "graphic_elements_invoke_url",
        "table_structure_invoke_url",
        "nemotron_parse_invoke_url",
        "embed_invoke_url",
        "embedding_endpoint",
        "endpoint_url",
        "api_base",
        "auth_token",
        "lancedb_uri",
        "storage_uri",
    }
)


def _strip_server_owned(params_dict: dict[str, Any], method: str) -> dict[str, Any]:
    """Raise if the caller set a server-owned key; otherwise return as-is.

    We fail fast on the client so users see a useful message instead of
    a generic 403 from the server.
    """
    rejected = [k for k in params_dict if k in _SERVER_OWNED_KEYS]
    if rejected:
        raise ValueError(
            f"ServiceIngestor.{method}(): keys {rejected!r} are server-owned in "
            "run_mode='service'. Endpoint URLs and API keys are configured by "
            "the retriever service via retriever-service.yaml; they cannot be "
            "set per-request."
        )
    return params_dict


def _params_to_dict(value: Any) -> dict[str, Any]:
    """Normalise a fluent-method argument (model | dict | None) to a dict.

    Removes server-owned keys eagerly so they never leak into transport.
    Drops ``None`` values so the server's defaults can fill them in.
    """
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        d = value.model_dump(mode="json", exclude_none=True)
    elif isinstance(value, dict):
        d = {k: v for k, v in value.items() if v is not None}
    else:
        raise TypeError(f"Cannot serialise {type(value).__name__!r} to a params dict")
    return d


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
        self._pipeline_spec: dict[str, Any] = {
            "extraction_mode": "pdf",
            "stage_order": [],
        }

    # ------------------------------------------------------------------
    # Pipeline-spec helpers
    # ------------------------------------------------------------------

    def _record_stage(self, name: str) -> None:
        order = self._pipeline_spec["stage_order"]
        if name not in order:
            order.append(name)

    def _pipeline_payload(self) -> dict[str, Any] | None:
        """Return the spec dict to send on the wire, or ``None`` when empty.

        The "empty" check mirrors :meth:`PipelineSpec.is_empty` server-side
        so the worker can short-circuit identically.
        """
        spec = self._pipeline_spec
        is_empty = (
            spec.get("extraction_mode", "pdf") == "pdf"
            and not spec.get("stage_order")
            and not any(
                spec.get(k)
                for k in (
                    "extract_params",
                    "embed_params",
                    "dedup_params",
                    "caption_params",
                    "store_params",
                    "vdb_upload_params",
                    "webhook_params",
                    "split_config",
                    "pdf_split",
                )
            )
        )
        return None if is_empty else dict(spec)

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
    # Phase 1: pipeline-shape stages — sent via PipelineSpec
    # ------------------------------------------------------------------

    def all_tasks(self) -> "ServiceIngestor":
        """Record the default chain: extract → dedup → embed.

        Concrete params come from server config; ``all_tasks()`` only
        controls *stage order* and is the closest in-process equivalent
        of "run everything the server is configured to do".
        """
        self._record_stage("extract")
        self._record_stage("dedup")
        self._record_stage("embed")
        return self

    def dedup(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        """Record a dedup stage with optional :class:`DedupParams` overrides."""
        merged = _merge_params(params, kwargs) if (params or kwargs) else DedupParams()
        params_dict = _strip_server_owned(_params_to_dict(merged), "dedup")
        self._pipeline_spec["dedup_params"] = params_dict
        self._record_stage("dedup")
        return self

    def embed(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        """Record an embed stage with optional :class:`EmbedParams` overrides.

        Embedding endpoint URL and API key are server-owned and will be
        rejected if set here.
        """
        merged = _merge_params(params, kwargs) if (params or kwargs) else EmbedParams()
        params_dict = _strip_server_owned(_params_to_dict(merged), "embed")
        self._pipeline_spec["embed_params"] = params_dict
        self._record_stage("embed")
        return self

    def extract(
        self,
        params: Any = None,
        *,
        split_config: Optional[dict[str, Any]] = None,
        extraction_mode: str = "pdf",
        **kwargs: Any,
    ) -> "ServiceIngestor":
        """Record a generic extraction stage.

        ``extraction_mode`` selects the worker's extraction path
        (``'pdf'`` default, ``'auto'`` for mixed inputs, etc.).
        """
        merged = _merge_params(params, kwargs) if (params or kwargs) else ExtractParams()
        params_dict = _strip_server_owned(_params_to_dict(merged), "extract")
        self._pipeline_spec["extract_params"] = params_dict
        self._pipeline_spec["extraction_mode"] = extraction_mode
        if split_config is not None:
            self._pipeline_spec["split_config"] = split_config
        self._record_stage("extract")
        return self

    def extract_image_files(
        self, params: Any = None, *, split_config: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> "ServiceIngestor":
        """Record image-file extraction (``extraction_mode='image'``)."""
        merged = _merge_params(params, kwargs) if (params or kwargs) else ExtractParams()
        params_dict = _strip_server_owned(_params_to_dict(merged), "extract_image_files")
        self._pipeline_spec["extract_params"] = params_dict
        self._pipeline_spec["extraction_mode"] = "image"
        if split_config is not None:
            self._pipeline_spec["split_config"] = split_config
        self._record_stage("extract")
        return self

    def filter(self) -> "ServiceIngestor":
        """Record a filter stage."""
        self._record_stage("filter")
        return self

    def split(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        """Record post-extract split / chunking configuration.

        Accepts the same dict shape as :meth:`GraphIngestor.extract`'s
        ``split_config`` keyword (``{"<source_type>": {"max_tokens": …}}``).
        """
        merged: dict[str, Any]
        if isinstance(params, dict):
            merged = dict(params)
        elif params is None:
            merged = {}
        else:
            merged = _params_to_dict(params)
        merged.update(kwargs)
        self._pipeline_spec["split_config"] = merged
        return self

    def pdf_split_config(self, pages_per_chunk: int = 32) -> "ServiceIngestor":
        """Record PDF page-chunking config (per-request).

        The gateway uses this to decide realtime-vs-batch routing
        (chunked docs always go to batch).
        """
        PdfSplitParams.model_validate({})  # cheap sanity touch
        self._pipeline_spec["pdf_split"] = {"pages_per_chunk": int(pages_per_chunk)}
        return self

    # ------------------------------------------------------------------
    # Future-phase methods — informative errors only for now
    # ------------------------------------------------------------------

    def store(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported(
            "store",
            phase_hint=(
                "Scheduled for Phase 2 (remote sinks). The service "
                "today only writes image assets to its preconfigured store."
            ),
        )

    def store_embed(self) -> "ServiceIngestor":
        _raise_unsupported(
            "store_embed",
            phase_hint="Graph-mode semantics are still in flux; no service equivalent yet.",
        )

    def udf(
        self,
        udf_function: str,
        udf_function_name: Optional[str] = None,
        phase: Optional[Union[int, str]] = None,
        target_stage: Optional[str] = None,
        run_before: bool = False,
        run_after: bool = False,
    ) -> "ServiceIngestor":
        _raise_unsupported(
            "udf",
            phase_hint=(
                "Scheduled for Phase 5 (named UDFs). Operators will register "
                "callables in retriever-service.yaml and clients reference them by name."
            ),
        )

    def vdb_upload(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported(
            "vdb_upload",
            phase_hint=(
                "Scheduled for Phase 2 (remote sinks). The service writes "
                "to the vectordb pod configured via vectordb.vectordb_url."
            ),
        )

    def save_intermediate_results(self, output_dir: str) -> "ServiceIngestor":
        _raise_unsupported(
            "save_intermediate_results",
            phase_hint=(
                "Inherently in-process — the service runs stages in worker "
                "pods. Use run_mode='inprocess' for stage-by-stage debugging."
            ),
        )

    def save_to_disk(
        self,
        output_directory: Optional[str] = None,
        cleanup: bool = True,
        compression: Optional[str] = "gzip",
    ) -> "ServiceIngestor":
        _raise_unsupported(
            "save_to_disk",
            phase_hint=(
                "Scheduled for Phase 3 — will stream per-document JSON to "
                "the caller's output_directory via the existing SSE channel."
            ),
        )

    def caption(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported(
            "caption",
            phase_hint=(
                "Scheduled for Phase 4 (remote caption endpoint). Local-GPU "
                "captioning would not fit the CPU-only worker model."
            ),
        )

    def webhook(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        _raise_unsupported(
            "webhook",
            phase_hint=(
                "Scheduled for Phase 2 (remote sinks). Requires an operator-"
                "configured URL allowlist before clients can drive egress."
            ),
        )

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
        pipeline_payload = self._pipeline_payload()
        async for evt in client.aingest_documents_stream(files=files, pipeline_spec=pipeline_payload):
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
