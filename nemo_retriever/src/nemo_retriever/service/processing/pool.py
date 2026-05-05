# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ProcessPoolExecutor-based document processing for service mode.

Each worker process builds its own operator chain at startup via
``_worker_initializer`` and keeps it alive for the lifetime of the
process.  This eliminates all C-library thread-safety issues (pypdfium2,
image processing) because every chain runs in its own address space.

When multiple NIM endpoint URLs are configured (comma-separated), the
pool distributes them round-robin across workers so that each worker
targets exactly one URL per endpoint type, balancing traffic evenly.

Pages are accumulated in a ``_BatchBuffer`` and dispatched to worker
processes in batches (default 32) so that NIM endpoints and GPUs see
larger inference batches.  A configurable timeout ensures small jobs
are dispatched promptly without waiting for a full batch.

Workers return a lightweight ``BatchWorkerResult`` to the main process.
The main process publishes SSE events **immediately** on the callback
thread, then enqueues all DB persistence to a single ``_DbWriterThread``
that batches writes into one transaction — eliminating WAL lock
contention across 16+ worker processes.

Embedding vectors, base64 page images, and other large intermediates
are stripped from ``content_json`` (SQLite) via ``_SERIALIZE_SKIP``.
Embeddings are already persisted in LanceDB by the pipeline.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import multiprocessing
import os
import queue as _queue_mod
import signal
import threading
import time
import traceback as _traceback
import typing
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from nemo_retriever.service.config import NimEndpointsConfig, ServiceConfig
from nemo_retriever.service.db.engine import DatabaseEngine
from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_bus import EventBus
from nemo_retriever.service.event_logger import record_event
from nemo_retriever.service.failure_types import EventCategory, FailureType, categorize_exception
from nemo_retriever.service.models.document import ProcessingStatus
from nemo_retriever.service.models.event_log import EventOutcome, EventRecord, EventSeverity
from nemo_retriever.service.models.metrics import ProcessingMetric
from nemo_retriever.service.models.page_processing_log import PageProcessingLog
from nemo_retriever.service.models.page_result import PageResult
from nemo_retriever.service.spool import SpoolStore

logger = logging.getLogger(__name__)

_SERIALIZE_SKIP = frozenset(
    {
        # Internal provenance / routing columns
        "bytes",
        "_page_document_id",
        "_page_job_id",
        "_page_number",
        "_page_filename",
        # Embedding vectors — already persisted in LanceDB; duplicating
        # 2048-float JSON arrays into SQLite content_json and SSE payloads
        # causes massive write contention and memory pressure.
        "text_embeddings_1b_v2",
        "text_embeddings_1b_v2_dim",
        "text_embeddings_1b_v2_has_embedding",
        "embedding_v1_num_detections",
        "embedding_v1_counts_by_label",
        "_contains_embeddings",
        "_embed_modality",
        # metadata dict carries a duplicate copy of the embedding vector
        # under metadata["embedding"] as well as other large intermediates
        "metadata",
        # Large base64-encoded page/element images
        "page_image_b64",
        "page_image",
        "_image_b64",
    }
)

_SERIALIZE_SKIP_PREFIXES = (
    "text_embeddings_",
    "embedding_v",
)


# ======================================================================
# Pure helper functions (used in both main and worker processes)
# ======================================================================


def _safe_value(v: Any) -> Any:
    """Best-effort conversion to a JSON-serialisable type."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, bytes):
        return f"<bytes len={len(v)}>"
    if isinstance(v, (list, tuple)):
        return [_safe_value(i) for i in v]
    if isinstance(v, dict):
        return {str(k): _safe_value(val) for k, val in v.items()}
    return str(v)


def _row_to_page_content(row: dict[str, Any]) -> dict[str, Any]:
    return {
        k: _safe_value(v)
        for k, v in row.items()
        if k not in _SERIALIZE_SKIP and not k.startswith(_SERIALIZE_SKIP_PREFIXES)
    }


def _extract_metrics_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull detection/invocation counts and per-label breakdowns from pipeline columns."""
    by_model: dict[str, dict[str, Any]] = {}
    for col, val in row.items():
        if val is None:
            continue
        if col.endswith("_num_detections"):
            model = col[: -len("_num_detections")]
            entry = by_model.setdefault(
                model,
                {
                    "model_name": model,
                    "detections_count": 0,
                    "counts_by_label": {},
                    "invocation_count": 1,
                    "pages_processed": 1,
                },
            )
            entry["detections_count"] = int(val) if isinstance(val, (int, float)) else 0
        elif col.endswith("_counts_by_label"):
            model = col[: -len("_counts_by_label")]
            entry = by_model.setdefault(
                model,
                {
                    "model_name": model,
                    "detections_count": 0,
                    "counts_by_label": {},
                    "invocation_count": 1,
                    "pages_processed": 1,
                },
            )
            if isinstance(val, dict):
                entry["counts_by_label"] = {str(k): int(v) for k, v in val.items()}
                if entry["detections_count"] == 0:
                    entry["detections_count"] = sum(int(v) for v in val.values())
    return list(by_model.values())


_NIM_URL_FIELDS = (
    "page_elements_invoke_url",
    "ocr_invoke_url",
    "table_structure_invoke_url",
    "graphic_elements_invoke_url",
    "embed_invoke_url",
)


def _resolve_worker_nim_configs(
    nim: NimEndpointsConfig,
    num_workers: int,
) -> list[dict[str, Any]]:
    """Split comma-separated NIM URLs and round-robin them across workers.

    Returns *num_workers* plain dicts, each suitable for
    ``NimEndpointsConfig(**d)``.  Fields that contain a single URL (or
    are ``None``) are replicated identically to every worker.  Fields
    with multiple comma-separated URLs are distributed so that worker *i*
    gets ``urls[i % len(urls)]``.
    """
    base = nim.model_dump()

    parsed: dict[str, list[str]] = {}
    for field in _NIM_URL_FIELDS:
        raw = base.get(field)
        if raw and "," in raw:
            parsed[field] = [u.strip() for u in raw.split(",") if u.strip()]
        else:
            parsed[field] = [raw] if raw else []

    configs: list[dict[str, Any]] = []
    for i in range(num_workers):
        worker_dict = dict(base)
        for field in _NIM_URL_FIELDS:
            urls = parsed[field]
            if len(urls) > 1:
                worker_dict[field] = urls[i % len(urls)]
        configs.append(worker_dict)

    return configs


def _log_endpoint_distribution(nim: NimEndpointsConfig, num_workers: int) -> None:
    """Log a summary of how NIM endpoints are distributed across workers."""
    lines: list[str] = []
    for field in _NIM_URL_FIELDS:
        raw = getattr(nim, field, None)
        if not raw:
            continue
        urls = [u.strip() for u in raw.split(",") if u.strip()]
        n = len(urls)
        if n == 1:
            lines.append(f"  {field}: 1 endpoint -> all {num_workers} workers")
        else:
            per = num_workers // n
            remainder = num_workers % n
            if remainder == 0:
                lines.append(f"  {field}: {n} endpoints -> {per} workers each")
            else:
                lines.append(f"  {field}: {n} endpoints -> ~{per}-{per + 1} workers each")
            if n > num_workers:
                lines.append(
                    f"    WARNING: {n - num_workers} endpoint(s) will not be assigned (more URLs than workers)"
                )
    if lines:
        logger.info("NIM endpoint distribution across %d workers:\n%s", num_workers, "\n".join(lines))


def _build_params(nim: NimEndpointsConfig) -> tuple[Any, Any]:
    """Construct ``ExtractParams`` and (optionally) ``EmbedParams`` from NIM config."""
    from nemo_retriever.params import ExtractParams, EmbedParams

    api_key = nim.api_key or os.environ.get("NVIDIA_API_KEY")

    extract_kwargs: dict[str, Any] = {}
    if nim.page_elements_invoke_url:
        extract_kwargs["page_elements_invoke_url"] = nim.page_elements_invoke_url
    if nim.ocr_invoke_url:
        extract_kwargs["ocr_invoke_url"] = nim.ocr_invoke_url
    if nim.table_structure_invoke_url:
        extract_kwargs["table_structure_invoke_url"] = nim.table_structure_invoke_url
    if nim.graphic_elements_invoke_url:
        extract_kwargs["graphic_elements_invoke_url"] = nim.graphic_elements_invoke_url
    if api_key:
        extract_kwargs["api_key"] = api_key

    embed_params = None
    if nim.embed_invoke_url:
        embed_kwargs: dict[str, Any] = {"embed_invoke_url": nim.embed_invoke_url}
        if api_key:
            embed_kwargs["api_key"] = api_key
        embed_params = EmbedParams(**embed_kwargs)

    return ExtractParams(**extract_kwargs), embed_params


def _build_operator_chain(
    replica_id: int | str,
    nim_endpoints: NimEndpointsConfig,
    vector_store_config: dict[str, Any] | None = None,
) -> list[tuple[str, Any]]:
    """Build a linearised list of ``(name, operator_instance)`` pairs."""
    from nemo_retriever.graph.ingestor_runtime import build_graph
    from nemo_retriever.graph.executor import InprocessExecutor
    from nemo_retriever.graph.operator_resolution import resolve_graph
    from nemo_retriever.utils.ray_resource_hueristics import gather_local_resources

    extract_params, embed_params = _build_params(nim_endpoints)

    has_remote = any(
        [
            nim_endpoints.page_elements_invoke_url,
            nim_endpoints.ocr_invoke_url,
            nim_endpoints.table_structure_invoke_url,
            nim_endpoints.graphic_elements_invoke_url,
            nim_endpoints.embed_invoke_url,
        ]
    )
    mode_label = "remote NIM" if has_remote else "local GPU"
    logger.info("[pid %d] Building operator chain %s (%s)", os.getpid(), replica_id, mode_label)

    graph = build_graph(
        extraction_mode="pdf",
        extract_params=extract_params,
        embed_params=embed_params,
        stage_order=(),
    )
    resolved = resolve_graph(graph, gather_local_resources())
    nodes = InprocessExecutor._linearize(resolved)
    operators: list[tuple[str, Any]] = []
    for node in nodes:
        op = node.operator_class(**node.operator_kwargs)
        operators.append((node.name, op))

    if vector_store_config is not None and nim_endpoints.embed_invoke_url:
        from nemo_retriever.graph.lancedb_write_operator import LanceDBWriteOperator

        ldb_op = LanceDBWriteOperator(
            uri=vector_store_config.get("lancedb_uri", "/var/lib/nemo-retriever/lancedb"),
            table_name=vector_store_config.get("lancedb_table", "nv-ingest"),
        )
        operators.append(("lancedb_write", ldb_op))
        logger.info(
            "[pid %d] LanceDB write stage appended (uri=%s, table=%s)", os.getpid(), ldb_op._uri, ldb_op._table_name
        )

    logger.info("[pid %d] Operator chain %s ready (%d stages)", os.getpid(), replica_id, len(operators))
    return operators


# ======================================================================
# Worker-process state and entry point
# ======================================================================

_worker_chain: list[tuple[str, Any]] | None = None
_worker_db_path: str | None = None


def _record_probe_events(db_path: str) -> None:
    """Persist NIM probe failures collected during operator-chain init."""
    from nemo_retriever.nim.probe import drain_probe_results

    results = drain_probe_results()
    failures = [r for r in results if r.status != "ok"]
    if not failures:
        return

    try:
        engine = DatabaseEngine(db_path)
        repo = Repository(engine)
    except Exception:
        logger.warning("[pid %d] Could not open DB to record probe events", os.getpid())
        return

    for result in failures:
        category = (
            EventCategory.NIM_UNREACHABLE.value
            if result.status == "unreachable"
            else EventCategory.NIM_TIMEOUT.value
        )
        record_event(
            repo,
            category=category,
            severity=EventSeverity.ERROR,
            outcome=EventOutcome.FAILED,
            summary=f"{result.prefix}: {result.name} endpoint unreachable at startup",
            detail=result.detail,
            stage="probe",
            endpoint="pipeline",
            extra={"probe_url": result.url, "nim_name": result.name},
        )


def _worker_initializer(
    db_path: str,
    nim_config_queue: multiprocessing.Queue,  # type: ignore[type-arg]
    fallback_nim_config: dict[str, Any],
    vector_store_config: dict[str, Any] | None = None,
) -> None:
    """Called exactly once per worker process by ProcessPoolExecutor.

    Each worker pops a unique NIM config from the shared queue so that
    multi-endpoint URLs are distributed round-robin.  If the queue is
    empty (more processes than configs — shouldn't happen) the fallback
    config is used so the worker still functions.
    """
    global _worker_chain, _worker_db_path

    # Workers should not handle SIGINT — the main process owns shutdown.
    # Without this, Ctrl+C causes noisy KeyboardInterrupt tracebacks from
    # every worker blocked on call_queue.get().
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        import setproctitle

        setproctitle.setproctitle("nemo-retriever-worker")
    except ImportError:
        pass

    _worker_db_path = db_path
    try:
        nim_config_dict = nim_config_queue.get_nowait()
    except Exception:
        logger.warning("[pid %d] NIM config queue empty — using fallback config", os.getpid())
        nim_config_dict = fallback_nim_config

    nim = NimEndpointsConfig(**nim_config_dict)
    _worker_chain = _build_operator_chain(os.getpid(), nim, vector_store_config=vector_store_config)

    _record_probe_events(db_path)


@dataclasses.dataclass
class PageDescriptor:
    """Lightweight struct describing a single page to be batched.

    When ``spool_path`` is set, ``file_bytes`` is left empty and the
    worker subprocess reads the page bytes from disk.  This keeps the
    main↔worker IPC payload small (a few-KB descriptor instead of a
    multi-MB blob) AND means the worker can recover from a transient
    pickling failure by re-reading from the durable spool.

    When ``spool_path`` is unset, ``file_bytes`` carries the page
    inline — used for the unit-test path and for any caller that has
    explicitly disabled spooling in config.
    """

    document_id: str
    content_sha256: str
    file_bytes: bytes
    filename: str
    job_id: str | None = None
    page_number: int = 1
    spool_path: str | None = None


@dataclasses.dataclass
class WorkerResult:
    """Per-page result.  A batch returns a list of these."""

    document_id: str
    job_id: str | None
    source_file: str
    page_number: int
    success: bool
    error_message: str | None = None
    failure_type: str | None = None
    metrics: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    detection_count: int = 0
    processing_duration_ms: float = 0.0
    started_at: str = ""
    completed_at: str = ""
    total_pages: int = 0
    # Full per-output-row contents emitted by the pipeline for this input page.
    # Populated when the worker successfully runs the chain.  Plumbed back to
    # the main process so it can publish ``page_result`` SSE events without
    # making clients re-fetch the data over REST.
    page_contents: list[dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class BatchWorkerResult:
    """Picklable result returned from a worker process to the main process.

    Contains one :class:`WorkerResult` per page in the batch.
    """

    results: list[WorkerResult] = dataclasses.field(default_factory=list)


# ======================================================================
# Provenance column used to track which output rows belong to which page
# after the content-explosion step expands 1 input row into N output rows.
# ======================================================================

_PROVENANCE_COL = "_page_document_id"
_PROVENANCE_JOB_COL = "_page_job_id"
_PROVENANCE_NUM_COL = "_page_number"
_PROVENANCE_FILE_COL = "_page_filename"


def _run_pipeline_batch(
    page_descriptors: list[dict[str, Any]],
    *,
    db_path: str = "",
) -> BatchWorkerResult:
    """Execute the operator chain on a batch of pages.

    Builds a multi-row DataFrame (one row per page), runs the full
    pipeline once, then splits the output back into per-page results
    using the ``_page_document_id`` provenance column.
    """
    pid = os.getpid()
    engine = DatabaseEngine(db_path)
    repo = Repository(engine)

    pages = [PageDescriptor(**d) for d in page_descriptors]
    n_pages = len(pages)

    logger.info(
        "[worker %d] Processing batch of %d pages (%s)",
        pid,
        n_pages,
        ", ".join(f"{p.filename} p{p.page_number}" for p in pages[:5]) + (" ..." if n_pages > 5 else ""),
    )

    for p in pages:
        try:
            repo.update_document_status(p.document_id, ProcessingStatus.PROCESSING)
        except Exception:
            pass

    rows = []
    skipped: list[WorkerResult] = []
    for p in pages:
        page_bytes = p.file_bytes
        if not page_bytes and p.spool_path:
            try:
                with open(p.spool_path, "rb") as fh:
                    page_bytes = fh.read()
            except FileNotFoundError:
                logger.warning(
                    "[worker %d] spool file missing for doc %s (page %s); marking failed",
                    pid,
                    p.document_id,
                    p.page_number,
                )
                record_event(
                    repo,
                    category=EventCategory.SPOOL.value,
                    severity=EventSeverity.ERROR,
                    outcome=EventOutcome.FAILED,
                    summary=f"Spool file missing: {p.spool_path}",
                    detail=f"spool file missing: {p.spool_path}",
                    stage="byte_fetch",
                    endpoint="pipeline",
                    job_id=p.job_id,
                    document_id=p.document_id,
                    source_file=p.filename,
                    page_number=p.page_number,
                )
                skipped.append(
                    WorkerResult(
                        document_id=p.document_id,
                        job_id=p.job_id,
                        source_file=p.filename,
                        page_number=p.page_number,
                        success=False,
                        error_message=f"spool file missing: {p.spool_path}",
                        failure_type=FailureType.INTERNAL.value,
                    )
                )
                continue
        if not page_bytes:
            record_event(
                repo,
                category=EventCategory.INTERNAL.value,
                severity=EventSeverity.ERROR,
                outcome=EventOutcome.FAILED,
                summary="No page bytes available",
                detail="no page bytes (neither inline nor spooled)",
                stage="byte_fetch",
                endpoint="pipeline",
                job_id=p.job_id,
                document_id=p.document_id,
                source_file=p.filename,
                page_number=p.page_number,
            )
            skipped.append(
                WorkerResult(
                    document_id=p.document_id,
                    job_id=p.job_id,
                    source_file=p.filename,
                    page_number=p.page_number,
                    success=False,
                    error_message="no page bytes (neither inline nor spooled)",
                    failure_type=FailureType.INTERNAL.value,
                )
            )
            continue
        rows.append(
            {
                "bytes": page_bytes,
                "path": p.filename,
                _PROVENANCE_COL: p.document_id,
                _PROVENANCE_JOB_COL: p.job_id,
                _PROVENANCE_NUM_COL: p.page_number,
                _PROVENANCE_FILE_COL: p.filename,
            }
        )
    # The pages whose bytes are actually runnable through the pipeline.
    runnable_pages = [p for p in pages if not any(s.document_id == p.document_id for s in skipped)]

    if not rows:
        # Every page in the batch failed the byte-fetch step.  Mark each
        # one failed in the DB so the SSE consumer + REST status reflect
        # the truth, and return only the skipped results.
        completed_at = datetime.now(timezone.utc).isoformat()
        for s in skipped:
            _mark_page_failed(
                repo,
                next(p for p in pages if p.document_id == s.document_id),
                s.error_message or "page bytes unavailable",
                completed_at,
                completed_at,
                s.failure_type or FailureType.INTERNAL.value,
            )
        return BatchWorkerResult(results=skipped)

    df = pd.DataFrame(rows)

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    from nemo_retriever.nim.error_reporter import drain_errors

    if _worker_chain is None:
        raise RuntimeError("Worker operator chain not initialised — _worker_initializer was never called")

    failed_stage: str | None = None
    failed_exc: BaseException | None = None
    for stage_name, op in _worker_chain:
        try:
            df = op.run(df)
            if df is None:
                raise RuntimeError(
                    f"Operator '{stage_name}' ({type(op).__name__}) returned None instead of a DataFrame"
                )
        except BaseException as exc:
            failed_stage = stage_name
            failed_exc = exc
            break

        # Drain any exceptions caught internally by this operator
        for err in drain_errors():
            record_event(
                repo,
                category=EventCategory.from_failure_type(categorize_exception(ValueError(err.message))).value,
                severity=EventSeverity.WARNING,
                outcome=EventOutcome.RECOVERED,
                summary=f"{stage_name}: {err.exc_type}: {err.message[:150]}",
                detail=err.message,
                stack_trace=err.traceback,
                stage=err.stage or stage_name,
                endpoint="pipeline",
                job_id=runnable_pages[0].job_id if runnable_pages else None,
                document_id=runnable_pages[err.row_index].document_id
                if err.row_index is not None and err.row_index < len(runnable_pages)
                else None,
                source_file=runnable_pages[err.row_index].filename
                if err.row_index is not None and err.row_index < len(runnable_pages)
                else "",
                page_number=runnable_pages[err.row_index].page_number
                if err.row_index is not None and err.row_index < len(runnable_pages)
                else None,
            )

    if failed_exc is not None:
        # Drain any remaining errors from the failed stage
        drain_errors()

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        error_msg = f"{type(failed_exc).__name__}: {failed_exc}"
        failure_type = categorize_exception(failed_exc).value
        tb_str = "".join(_traceback.format_exception(type(failed_exc), failed_exc, failed_exc.__traceback__))
        logger.error(
            "[worker %d] BATCH FAILED at stage '%s' (%d pages, failure_type=%s): %s",
            pid,
            failed_stage,
            n_pages,
            failure_type,
            error_msg,
            exc_info=failed_exc,
        )
        completed_at = datetime.now(timezone.utc).isoformat()
        event_category = EventCategory.from_failure_type(categorize_exception(failed_exc)).value
        fail_results: list[WorkerResult] = list(skipped)
        for p in runnable_pages:
            _mark_page_failed(repo, p, error_msg, started_at, completed_at, failure_type)
            record_event(
                repo,
                category=event_category,
                severity=EventSeverity.ERROR,
                outcome=EventOutcome.FAILED,
                summary=f"Pipeline failed at stage '{failed_stage}': {error_msg[:180]}",
                detail=error_msg,
                stack_trace=tb_str,
                stage=failed_stage or "pipeline",
                endpoint="pipeline",
                job_id=p.job_id,
                document_id=p.document_id,
                source_file=p.filename,
                page_number=p.page_number,
            )
            fail_results.append(
                WorkerResult(
                    document_id=p.document_id,
                    job_id=p.job_id,
                    source_file=p.filename,
                    page_number=p.page_number,
                    success=False,
                    error_message=error_msg,
                    failure_type=failure_type,
                )
            )
        return BatchWorkerResult(results=fail_results)

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    completed_at = datetime.now(timezone.utc).isoformat()

    logger.info(
        "[worker %d] Batch of %d pages complete — %d output rows, %.0fms",
        pid,
        n_pages,
        len(df),
        elapsed_ms,
    )

    batch_result = _split_batch_results(
        repo,
        df,
        runnable_pages,
        elapsed_ms,
        started_at,
        completed_at,
    )
    if skipped:
        batch_result.results = list(skipped) + list(batch_result.results)
    return batch_result


def _mark_page_failed(
    repo: Repository,
    p: PageDescriptor,
    error_msg: str,
    started_at: str,
    completed_at: str,
    failure_type: str | None = None,
) -> None:
    """Mark a single page as failed in the DB."""
    try:
        repo.update_document_status(p.document_id, ProcessingStatus.FAILED)
        source_file = _resolve_source_file(repo, p)
        fail_log = PageProcessingLog(
            id=PageProcessingLog.make_id(source_file, p.page_number),
            document_id=p.document_id,
            job_id=p.job_id,
            source_file=source_file,
            page_number=p.page_number,
            status="failed",
            error_message=error_msg,
            failure_type=failure_type,
            detection_count=0,
            processing_duration_ms=0.0,
            started_at=started_at,
            completed_at=completed_at,
        )
        repo.insert_page_processing_log(fail_log)
    except Exception:
        pass


def _resolve_source_file(repo: Repository, p: PageDescriptor) -> str:
    """Look up the human-readable source filename for a page."""
    doc = repo.get_document(p.document_id)
    source_file = doc.filename if doc else p.filename
    if doc and doc.job_id:
        parent_job = repo.get_job(doc.job_id)
        if parent_job:
            source_file = parent_job.filename
    return source_file


def _split_batch_results(
    repo: Repository,
    df: pd.DataFrame,
    pages: list[PageDescriptor],
    elapsed_ms: float,
    started_at: str,
    completed_at: str,
) -> BatchWorkerResult:
    """Split the batched pipeline output back into per-page results.

    Groups output rows by ``_page_document_id``, extracts metrics from
    the first row of each group, and assembles per-page
    :class:`WorkerResult` objects.  **No SQLite writes happen here** —
    all persistence is deferred to :class:`_DbWriterThread` in the
    main process.
    """
    pid = os.getpid()
    page_lookup = {p.document_id: p for p in pages}

    if _PROVENANCE_COL not in df.columns:
        logger.warning("[worker %d] Provenance column missing — cannot split batch results", pid)
        results: list[WorkerResult] = []
        for p in pages:
            results.append(
                WorkerResult(
                    document_id=p.document_id,
                    job_id=p.job_id,
                    source_file=p.filename,
                    page_number=p.page_number,
                    success=False,
                    error_message="Provenance column lost during pipeline execution",
                    failure_type=FailureType.INTERNAL.value,
                )
            )
        return BatchWorkerResult(results=results)

    grouped = df.groupby(_PROVENANCE_COL, sort=False)
    results = []

    for doc_id, group_df in grouped:
        doc_id = str(doc_id)
        p = page_lookup.get(doc_id)
        if p is None:
            continue

        source_file = _resolve_source_file(repo, p)
        total_rows = len(group_df)

        page_metrics = _extract_metrics_from_row(group_df.iloc[0].to_dict())

        page_contents: list[dict[str, Any]] = []
        for _page_num, (_, row) in enumerate(group_df.iterrows()):
            row_dict = row.to_dict()
            content = _row_to_page_content(row_dict)
            page_contents.append(content)

        det_total = sum(m.get("detections_count", 0) for m in page_metrics)

        results.append(
            WorkerResult(
                document_id=doc_id,
                job_id=p.job_id,
                source_file=source_file,
                page_number=p.page_number,
                success=True,
                metrics=page_metrics,
                detection_count=det_total,
                processing_duration_ms=elapsed_ms,
                started_at=started_at,
                completed_at=completed_at,
                total_pages=total_rows,
                page_contents=page_contents,
            )
        )

    seen = {r.document_id for r in results}
    for p in pages:
        if p.document_id not in seen:
            results.append(
                WorkerResult(
                    document_id=p.document_id,
                    job_id=p.job_id,
                    source_file=p.filename,
                    page_number=p.page_number,
                    success=False,
                    error_message="Page produced no output rows after pipeline execution",
                    failure_type=FailureType.INTERNAL.value,
                    started_at=started_at,
                    completed_at=completed_at,
                )
            )

    return BatchWorkerResult(results=results)


# ======================================================================
# Main-process pool manager
# ======================================================================


class _BatchBuffer:
    """Thread-safe buffer that accumulates pages and auto-flushes.

    A batch is dispatched to the executor when either:
    - The buffer reaches ``batch_size`` pages, **or**
    - ``timeout_s`` seconds have passed since the first page was added
      to the current (non-empty) buffer.

    The buffer enforces a hard cap of ``max_buffered`` pages to bound
    memory usage.  When the cap is reached, ``enqueue`` returns ``False``
    so the caller can 503 the client.

    IMPORTANT: the dispatch function is called **outside** the lock so
    that pickling large batches and submitting to the executor never
    blocks the event-loop thread from enqueuing new pages or receiving
    SSE events.
    """

    def __init__(
        self,
        batch_size: int,
        timeout_s: float,
        max_buffered: int,
        dispatch_fn: "typing.Callable[[list[dict[str, Any]]], None]",
        db_engine: DatabaseEngine | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._timeout_s = timeout_s
        self._max_buffered = max_buffered
        self._dispatch_fn = dispatch_fn
        self._db_engine = db_engine
        self._lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []
        self._timer: threading.Timer | None = None
        self._closed = False

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def enqueue(self, page: dict[str, Any]) -> bool:
        """Add a page to the buffer.  Returns ``False`` if full."""
        batch: list[dict[str, Any]] | None = None
        with self._lock:
            if self._closed:
                return False
            if len(self._buffer) >= self._max_buffered:
                return False
            self._buffer.append(page)
            if len(self._buffer) >= self._batch_size:
                batch = self._drain_locked()
            elif len(self._buffer) == 1:
                self._start_timer_locked()
        if batch is not None:
            self._safe_dispatch(batch)
        return True

    def flush(self) -> None:
        """Force-flush whatever is in the buffer."""
        batch: list[dict[str, Any]] | None = None
        with self._lock:
            batch = self._drain_locked()
        if batch is not None:
            self._safe_dispatch(batch)

    def close(self) -> None:
        """Flush remaining pages and prevent further enqueues."""
        batch: list[dict[str, Any]] | None = None
        with self._lock:
            self._closed = True
            batch = self._drain_locked()
        if batch is not None:
            self._safe_dispatch(batch)

    def _drain_locked(self) -> list[dict[str, Any]] | None:
        """Extract the current buffer contents.  Must hold ``_lock``."""
        self._cancel_timer_locked()
        if not self._buffer:
            return None
        batch = self._buffer[:]
        self._buffer.clear()
        return batch

    def _safe_dispatch(self, batch: list[dict[str, Any]]) -> None:
        """Dispatch a batch outside the lock, handling errors."""
        try:
            self._dispatch_fn(batch)
        except Exception as exc:
            logger.exception("Failed to dispatch batch of %d pages", len(batch))
            if self._db_engine is not None:
                try:
                    repo = Repository(self._db_engine)
                    record_event(
                        repo,
                        category=EventCategory.DISPATCH.value,
                        severity=EventSeverity.ERROR,
                        outcome=EventOutcome.FAILED,
                        summary=f"Batch dispatch failed: {type(exc).__name__}: {exc}"[:200],
                        detail=str(exc),
                        stack_trace=_traceback.format_exc(),
                        stage="dispatch",
                        endpoint="pipeline",
                    )
                except Exception:
                    pass

    def _start_timer_locked(self) -> None:
        self._cancel_timer_locked()
        self._timer = threading.Timer(self._timeout_s, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _on_timeout(self) -> None:
        batch: list[dict[str, Any]] | None = None
        with self._lock:
            batch = self._drain_locked()
        if batch is not None:
            self._safe_dispatch(batch)


@dataclasses.dataclass
class _DbWriteItem:
    """All the data needed to persist one page's results to SQLite.

    Built by `_handle_single_result` and enqueued to `_DbWriterThread`.
    The background thread drains multiple items and writes them all
    in a single transaction — one lock acquisition for N pages instead
    of 7*N individual commits.
    """

    document_id: str
    job_id: str | None
    source_file: str
    page_number: int
    success: bool
    # Only populated on success:
    metrics: list[ProcessingMetric] = dataclasses.field(default_factory=list)
    page_results: list[PageResult] = dataclasses.field(default_factory=list)
    pages_received_increment: int = 0
    log_entry: PageProcessingLog | None = None
    total_pages: int = 0
    detection_count: int = 0
    processing_duration_ms: float = 0.0
    # Only populated on failure:
    error_message: str | None = None
    failure_type: str | None = None
    started_at: str = ""
    completed_at: str = ""


_SENTINEL = object()


class _DbWriterThread:
    """Single-threaded background writer that batches DB writes.

    All per-page SQLite persistence is funnelled through this thread
    so that worker processes and the callback thread never contend for
    the WAL write lock.  Items are batched: the thread wakes every
    ``flush_interval_s`` or when ``max_batch`` items accumulate.
    """

    def __init__(
        self,
        db_engine: DatabaseEngine,
        *,
        store_page_results: bool = False,
        flush_interval_s: float = 0.1,
        max_batch: int = 64,
    ) -> None:
        self._db_engine = db_engine
        self._store_page_results = store_page_results
        self._flush_interval_s = flush_interval_s
        self._max_batch = max_batch
        self._queue: _queue_mod.Queue[_DbWriteItem | object] = _queue_mod.Queue(maxsize=4096)
        self._thread = threading.Thread(target=self._run, name="db-writer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=timeout)

    def enqueue(self, item: _DbWriteItem) -> None:
        try:
            self._queue.put_nowait(item)
        except _queue_mod.Full:
            logger.warning("DB writer queue full — dropping write for doc %s", item.document_id[:8])

    def _run(self) -> None:
        repo = Repository(self._db_engine)
        while True:
            batch = self._drain_batch()
            if batch is None:
                return
            if batch:
                self._write_batch(repo, batch)

    def _drain_batch(self) -> list[_DbWriteItem] | None:
        """Block until at least one item arrives, then drain up to max_batch.

        Returns None when the sentinel is received (shutdown).
        """
        try:
            first = self._queue.get(timeout=self._flush_interval_s)
        except _queue_mod.Empty:
            return []
        if first is _SENTINEL:
            return None
        batch: list[_DbWriteItem] = [first]  # type: ignore[list-item]
        while len(batch) < self._max_batch:
            try:
                item = self._queue.get_nowait()
            except _queue_mod.Empty:
                break
            if item is _SENTINEL:
                self._queue.put(_SENTINEL)
                break
            batch.append(item)  # type: ignore[arg-type]
        return batch

    def _write_batch(self, repo: Repository, batch: list[_DbWriteItem]) -> None:
        from nemo_retriever.service.db.engine import execute_with_retry

        def _do() -> None:
            conn = repo._conn
            now = datetime.now(timezone.utc).isoformat()

            for item in batch:
                if item.success:
                    conn.execute(
                        "UPDATE documents SET processing_status = ?, total_pages = ?, updated_at = ? WHERE id = ?",
                        (ProcessingStatus.PROCESSING.value, item.total_pages, now, item.document_id),
                    )

                    for m in item.metrics:
                        row = m.to_row()
                        cols = ", ".join(row.keys())
                        placeholders = ", ".join(f":{k}" for k in row.keys())
                        conn.execute(f"INSERT INTO processing_metrics ({cols}) VALUES ({placeholders})", row)

                    if self._store_page_results:
                        for pr in item.page_results:
                            row = pr.to_row()
                            cols = ", ".join(row.keys())
                            placeholders = ", ".join(f":{k}" for k in row.keys())
                            conn.execute(f"INSERT INTO page_results ({cols}) VALUES ({placeholders})", row)

                    if item.pages_received_increment > 0:
                        conn.execute(
                            "UPDATE documents SET pages_received = pages_received + ?, updated_at = ? WHERE id = ?",
                            (item.pages_received_increment, now, item.document_id),
                        )

                    if item.log_entry is not None:
                        row = item.log_entry.to_row()
                        cols = ", ".join(row.keys())
                        placeholders = ", ".join(f":{k}" for k in row.keys())
                        conn.execute(
                            f"INSERT OR REPLACE INTO page_processing_log ({cols}) VALUES ({placeholders})", row
                        )

                    conn.execute(
                        "UPDATE documents SET processing_status = ?, updated_at = ? WHERE id = ?",
                        (ProcessingStatus.COMPLETE.value, now, item.document_id),
                    )

                    event = EventRecord(
                        category=EventCategory.INTERNAL.value,
                        severity=EventSeverity.INFO,
                        outcome=EventOutcome.IN_PROGRESS,
                        summary=f"Page processed ({item.detection_count} det, {item.processing_duration_ms:.0f}ms)",
                        stage="pipeline",
                        endpoint="pipeline",
                        job_id=item.job_id,
                        document_id=item.document_id,
                        source_file=item.source_file,
                        page_number=item.page_number,
                        extra_json=json.dumps({
                            "detection_count": item.detection_count,
                            "processing_duration_ms": item.processing_duration_ms,
                        }),
                    )
                    erow = event.to_row()
                    ecols = ", ".join(erow.keys())
                    eph = ", ".join(f":{k}" for k in erow.keys())
                    conn.execute(f"INSERT INTO event_log ({ecols}) VALUES ({eph})", erow)

                else:
                    conn.execute(
                        "UPDATE documents SET processing_status = ?, updated_at = ? WHERE id = ?",
                        (ProcessingStatus.FAILED.value, now, item.document_id),
                    )
                    if item.log_entry is not None:
                        row = item.log_entry.to_row()
                        cols = ", ".join(row.keys())
                        placeholders = ", ".join(f":{k}" for k in row.keys())
                        conn.execute(
                            f"INSERT OR REPLACE INTO page_processing_log ({cols}) VALUES ({placeholders})", row
                        )

                if item.job_id:
                    conn.execute(
                        "UPDATE jobs SET pages_completed = pages_completed + 1, updated_at = ? WHERE id = ?",
                        (now, item.job_id),
                    )

            conn.commit()

            for item in batch:
                if item.job_id:
                    cur = conn.execute(
                        "SELECT pages_completed, total_pages, processing_status FROM jobs WHERE id = ?",
                        (item.job_id,),
                    )
                    row = cur.fetchone()
                    if row and row["pages_completed"] >= row["total_pages"] > 0:
                        if row["processing_status"] not in ("complete", "failed", "cancelled"):
                            conn.execute(
                                "UPDATE jobs SET processing_status = ?, updated_at = ? WHERE id = ?",
                                (ProcessingStatus.COMPLETE.value, now, item.job_id),
                            )
                            conn.commit()

        try:
            execute_with_retry(_do)
        except Exception:
            logger.exception("DB writer batch failed (%d items)", len(batch))


class ProcessingPool:
    """Manages a ``ProcessPoolExecutor`` of isolated worker processes.

    Pages are accumulated in a ``_BatchBuffer`` and dispatched in
    batches (default 32) to worker processes.  Workers return a
    lightweight ``BatchWorkerResult`` to the main process for SSE
    event publishing.  All SQLite persistence is handled by a
    dedicated ``_DbWriterThread`` to eliminate WAL lock contention.
    """

    def __init__(
        self,
        config: ServiceConfig,
        db_engine: DatabaseEngine,
        event_bus: EventBus,
        event_loop: asyncio.AbstractEventLoop,
        spool_store: SpoolStore | None = None,
    ) -> None:
        self._config = config
        self._db_engine = db_engine
        self._event_bus = event_bus
        self._event_loop = event_loop
        self._spool_store = spool_store
        self._num_workers = config.processing.num_workers
        self._batch_size = config.processing.batch_size
        self._batch_timeout_s = config.processing.batch_timeout_s
        self._store_page_results = config.processing.store_page_results
        self._executor: ProcessPoolExecutor | None = None
        self._buffer: _BatchBuffer | None = None
        self._db_writer: _DbWriterThread | None = None
        self._in_flight = 0
        self._in_flight_lock = threading.Lock()
        # Cancellation + drain state
        self._cancelled_jobs: set[str] = set()
        self._cancel_lock = threading.Lock()
        self._draining = threading.Event()
        # In-memory job page counters for immediate SSE job_complete
        # detection without waiting for the DB writer thread.
        self._job_page_counts: dict[str, int] = {}
        self._job_page_counts_lock = threading.Lock()
        # Background spool sweeper (started in start()).
        self._spool_cleanup_task: asyncio.Task | None = None

    def start(self) -> None:
        nim = self._config.nim_endpoints
        has_remote = any(
            [
                nim.page_elements_invoke_url,
                nim.ocr_invoke_url,
                nim.table_structure_invoke_url,
                nim.graphic_elements_invoke_url,
                nim.embed_invoke_url,
            ]
        )
        mode_label = "remote NIM" if has_remote else "local GPU"
        logger.info(
            "Starting processing pool: %d worker processes, batch_size=%d, timeout=%.1fs (%s)",
            self._num_workers,
            self._batch_size,
            self._batch_timeout_s,
            mode_label,
        )

        worker_nim_dicts = _resolve_worker_nim_configs(nim, self._num_workers)
        _log_endpoint_distribution(nim, self._num_workers)

        db_path = str(self._db_engine._db_path)

        ctx = multiprocessing.get_context("spawn")
        nim_config_queue: multiprocessing.Queue[dict[str, Any]] = ctx.Queue()
        for cfg in worker_nim_dicts:
            nim_config_queue.put(cfg)
        fallback_nim_config = nim.model_dump()

        vs_dict: dict[str, Any] | None = None
        if self._config.vector_store is not None:
            vs_dict = self._config.vector_store.model_dump()

        logger.info("Initialising %d worker(s) — building operator chains", self._num_workers)
        self._executor = ProcessPoolExecutor(
            max_workers=self._num_workers,
            mp_context=ctx,
            initializer=_worker_initializer,
            initargs=(db_path, nim_config_queue, fallback_nim_config, vs_dict),
        )

        warmup_futures = [self._executor.submit(os.getpid) for _ in range(self._num_workers)]
        pids: set[int] = set()
        for fut in warmup_futures:
            pids.add(fut.result())
        logger.info("All %d worker process(es) initialised and ready", len(pids))

        self._db_writer = _DbWriterThread(
            self._db_engine,
            store_page_results=self._store_page_results,
        )
        self._db_writer.start()
        logger.info(
            "Background DB writer started (store_page_results=%s)",
            self._store_page_results,
        )

        max_buffered = self._num_workers * self._batch_size
        self._buffer = _BatchBuffer(
            batch_size=self._batch_size,
            timeout_s=self._batch_timeout_s,
            max_buffered=max_buffered,
            dispatch_fn=self._dispatch_batch,
            db_engine=self._db_engine,
        )
        logger.info(
            "Batch buffer ready — batch_size=%d, timeout=%.1fs, max_buffered=%d",
            self._batch_size,
            self._batch_timeout_s,
            max_buffered,
        )

    def shutdown(self) -> None:
        if self._spool_cleanup_task is not None and not self._spool_cleanup_task.done():
            self._spool_cleanup_task.cancel()
            self._spool_cleanup_task = None
        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None
        if self._executor is not None:
            logger.info("Shutting down processing pool …")
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        if self._db_writer is not None:
            self._db_writer.stop()
            self._db_writer = None

    # ------------------------------------------------------------------
    # Spool integration: recovery + periodic cleanup
    # ------------------------------------------------------------------

    def recover_from_spool(self) -> int:
        """Re-enqueue every spooled-but-unprocessed page from the durable store.

        Called once at startup, BEFORE HTTP starts accepting new ingest
        requests, so the recovered work hits the buffer first.  Returns
        the number of pages re-enqueued; logs a warning for any spool
        file that vanished (cleaned up between crash and restart).
        """
        if self._spool_store is None:
            return 0
        repo = Repository(self._db_engine)
        recovered = 0
        skipped = 0
        for doc in repo.list_recoverable_spooled_documents():
            spool_path = doc.spool_path
            if not spool_path:
                continue
            try:
                page_bytes = self._spool_store.read(spool_path)
            except FileNotFoundError:
                logger.warning(
                    "spool recovery: file %s missing for doc %s — marking failed",
                    spool_path,
                    doc.id,
                )
                repo.update_document_status(doc.id, ProcessingStatus.FAILED)
                repo.update_document_spool_path(doc.id, None)
                skipped += 1
                continue
            ok = self.try_submit(
                doc.id,
                doc.content_sha256,
                page_bytes,
                doc.filename,
                job_id=doc.job_id,
                page_number=doc.page_number or 1,
                spool_path=spool_path,
            )
            if ok:
                recovered += 1
            else:
                # Buffer full at recovery — extremely unlikely (no
                # ingest yet); log and let the next sweep retry.
                logger.warning(
                    "spool recovery: buffer rejected re-enqueue of doc %s; will retry on next sweep",
                    doc.id,
                )
                skipped += 1
        if recovered or skipped:
            logger.info(
                "spool recovery complete: re-enqueued=%d skipped=%d",
                recovered,
                skipped,
            )
        return recovered

    def start_spool_cleanup(self) -> None:
        """Start the background sweeper that removes terminal spool files.

        Safe to call multiple times — only starts a task when one isn't
        already running.  Must be called from inside the asyncio loop
        the service runs on.
        """
        if self._spool_store is None:
            return
        if self._spool_cleanup_task is not None and not self._spool_cleanup_task.done():
            return
        self._spool_cleanup_task = self._event_loop.create_task(self._spool_cleanup_loop())

    async def _spool_cleanup_loop(self) -> None:
        """Periodically delete spool files for documents in a terminal state."""
        assert self._spool_store is not None
        interval = float(self._config.spool.cleanup_interval_s)
        batch = int(self._config.spool.cleanup_batch_size)
        while True:
            try:
                await asyncio.sleep(interval)
                evictable = await asyncio.to_thread(
                    Repository(self._db_engine).list_evictable_spooled_documents,
                    batch,
                )
                if not evictable:
                    continue
                deleted = await asyncio.to_thread(self._spool_store.cleanup_terminal, evictable)
                # Clear spool_path on the rows whose files we successfully
                # removed (re-using the same repo instance per call so the
                # connection sticks to this thread).
                if deleted:
                    repo_main = Repository(self._db_engine)
                    for doc_id, _path in evictable:
                        await asyncio.to_thread(repo_main.update_document_spool_path, doc_id, None)
            except asyncio.CancelledError:
                logger.info("Spool cleanup loop stopped")
                return
            except Exception:  # noqa: BLE001 — sweeper must be resilient
                logger.exception("Spool cleanup loop iteration failed (continuing)")

    @property
    def pool_size(self) -> int:
        """Number of worker processes configured for this pool."""
        return self._num_workers

    # Backwards-compatible alias kept for callers that already use the
    # private name; new code should prefer ``pool_size``.
    @property
    def _pool_size(self) -> int:
        return self._num_workers

    @property
    def capacity(self) -> int:
        buf_count = self._buffer.buffered_count if self._buffer else 0
        max_buf = self._num_workers * self._batch_size
        return max(0, max_buf - buf_count)

    def has_capacity(self) -> bool:
        return self.capacity > 0

    # ------------------------------------------------------------------
    # Cancellation + drain (main-process state only)
    # ------------------------------------------------------------------

    def cancel_job(self, job_id: str) -> None:
        """Mark *job_id* as cancelled.

        Pages already in flight will run to completion (worker subprocesses
        cannot be safely interrupted mid-pipeline), but any pages still
        sitting in the batch buffer or still being uploaded will be dropped
        before they reach an executor.
        """
        with self._cancel_lock:
            self._cancelled_jobs.add(job_id)

    def is_job_cancelled(self, job_id: str | None) -> bool:
        if not job_id:
            return False
        with self._cancel_lock:
            return job_id in self._cancelled_jobs

    @property
    def is_draining(self) -> bool:
        return self._draining.is_set()

    def begin_drain(self) -> None:
        """Stop accepting new pages.  Existing batches still drain through."""
        self._draining.set()
        if self._buffer is not None:
            # Force-flush any partial batch so it isn't held by the timer.
            self._buffer.flush()

    def in_flight_batches(self) -> int:
        with self._in_flight_lock:
            return self._in_flight

    async def drain(self, timeout_s: float) -> bool:
        """Mark the pool draining and wait for in-flight batches to finish.

        Returns ``True`` if drain completed within ``timeout_s``, ``False``
        if the timeout fired with batches still running.  Callers (the
        lifespan hook, primarily) should still call :meth:`shutdown` after
        :meth:`drain` returns to release the executor.
        """
        self.begin_drain()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout_s)
        while True:
            in_flight = self.in_flight_batches()
            if in_flight == 0:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning(
                    "Drain timed out with %d batch(es) still in flight after %.1fs",
                    in_flight,
                    timeout_s,
                )
                return False
            await asyncio.sleep(min(0.5, remaining))

    # ------------------------------------------------------------------
    # SSE event publishing (main process only)
    # ------------------------------------------------------------------

    def _publish_event(self, document_id: str, event: dict[str, Any]) -> None:
        """Fan an event out under the configured overflow policy.

        Under ``drop_low_priority`` (default) this is fire-and-forget —
        the future is ignored and the worker callback thread returns
        immediately.  Under ``backpressure`` or ``block`` the call WAITS
        on the publish future so a slow consumer naturally back-presses
        the worker pool.  We bound the wait at twice the configured
        publish timeout (or 5 minutes for ``block``) as a safety net so
        a truly stuck consumer can never deadlock the worker thread
        forever.
        """
        # We always submit both publish coroutines; whether to wait
        # depends on the bus's policy.
        doc_fut = asyncio.run_coroutine_threadsafe(
            self._event_bus.publish(document_id, event),
            self._event_loop,
        )
        job_id = event.get("job_id")
        job_fut = (
            asyncio.run_coroutine_threadsafe(
                self._event_bus.publish(job_id, event),
                self._event_loop,
            )
            if job_id
            else None
        )

        if self._event_bus.overflow_policy == "drop_low_priority":
            return

        # Bound the worker-side wait independently of the bus's own
        # timeout so a misconfigured ``block`` mode can't pin a worker
        # thread forever.
        if self._event_bus.overflow_policy == "block":
            wait_s: float | None = 300.0
        else:
            wait_s = max(self._config.event_bus.publish_timeout_s * 2.0, 1.0)

        for fut in (doc_fut, job_fut):
            if fut is None:
                continue
            try:
                fut.result(timeout=wait_s)
            except Exception:  # noqa: BLE001 — publish exceptions logged inside the bus
                pass

    # ------------------------------------------------------------------
    # Batch dispatch (called by _BatchBuffer from its timer/flush)
    # ------------------------------------------------------------------

    def _dispatch_batch(self, batch: list[dict[str, Any]]) -> None:
        """Submit a batch of pages to the executor as a single task.

        Pages whose job has been cancelled since enqueue are filtered out
        and a synthetic cancelled ``WorkerResult`` is published so SSE
        subscribers see the terminal event.
        """
        if self._executor is None:
            raise RuntimeError("ProcessingPool has not been started")

        runnable: list[dict[str, Any]] = []
        cancelled: list[dict[str, Any]] = []
        for page in batch:
            if self.is_job_cancelled(page.get("job_id")):
                cancelled.append(page)
            else:
                runnable.append(page)

        for page in cancelled:
            self._publish_cancelled(page)

        if not runnable:
            return

        with self._in_flight_lock:
            self._in_flight += 1

        logger.debug(
            "Dispatching batch of %d pages to executor (skipped %d cancelled)",
            len(runnable),
            len(cancelled),
        )
        fut: Future[BatchWorkerResult] = self._executor.submit(
            _run_pipeline_batch,
            runnable,
            db_path=str(self._db_engine._db_path),
        )
        fut.add_done_callback(self._on_batch_result)

    def _publish_cancelled(self, page: dict[str, Any]) -> None:
        """Mark a buffered page as cancelled in the DB and emit SSE event."""
        repo = Repository(self._db_engine)
        doc_id = page.get("document_id", "")
        job_id = page.get("job_id")
        try:
            repo.update_document_status(doc_id, ProcessingStatus.CANCELLED)
        except Exception:  # noqa: BLE001 — best effort
            pass

        record_event(
            repo,
            category=EventCategory.CANCELLED.value,
            severity=EventSeverity.INFO,
            outcome=EventOutcome.FAILED,
            summary="Page cancelled before processing",
            detail="job_cancelled",
            stage="dispatch",
            endpoint="pipeline",
            job_id=job_id,
            document_id=doc_id,
            source_file=page.get("filename", ""),
            page_number=page.get("page_number"),
        )

        if job_id:
            self._handle_job_completion(job_id)

    # ------------------------------------------------------------------
    # Result callback (runs in main process on a callback thread)
    # ------------------------------------------------------------------

    def _on_batch_result(self, future: Future[BatchWorkerResult]) -> None:
        """Called in the main process when a batch finishes."""
        with self._in_flight_lock:
            self._in_flight -= 1

        try:
            batch_result = future.result()
        except Exception as exc:
            logger.exception("Worker process raised an unhandled exception: %s", exc)
            try:
                repo = Repository(self._db_engine)
                record_event(
                    repo,
                    category=EventCategory.INTERNAL.value,
                    severity=EventSeverity.ERROR,
                    outcome=EventOutcome.FAILED,
                    summary=f"Worker process crashed: {type(exc).__name__}: {exc}"[:200],
                    detail=str(exc),
                    stack_trace=_traceback.format_exc(),
                    stage="worker_process",
                    endpoint="pipeline",
                )
            except Exception:
                pass
            return

        for result in batch_result.results:
            self._handle_single_result(result)

    def _handle_single_result(self, result: WorkerResult) -> None:
        """Process one per-page result: publish SSE events immediately, then enqueue DB writes."""
        doc_id = result.document_id
        job_id = result.job_id

        if result.success:
            page_complete_payload: dict[str, Any] = {
                "event": "page_complete",
                "document_id": doc_id,
                "pages_received": result.total_pages,
                "total_pages": result.total_pages,
            }
            if job_id:
                page_complete_payload["job_id"] = job_id
            self._publish_event(doc_id, page_complete_payload)

            self._publish_event(
                doc_id,
                {
                    "event": "document_complete",
                    "document_id": doc_id,
                    "job_id": job_id,
                    "total_pages": result.total_pages,
                },
            )

            metric_objs = [
                ProcessingMetric(
                    document_id=doc_id,
                    model_name=m["model_name"],
                    invocation_count=m.get("invocation_count", 1),
                    pages_processed=m.get("pages_processed", 1),
                    detections_count=m.get("detections_count", 0),
                    counts_by_label_json=json.dumps(m.get("counts_by_label", {})),
                )
                for m in result.metrics
            ]

            page_result_objs = [
                PageResult(
                    document_id=doc_id,
                    page_number=result.page_number,
                    content_json=json.dumps(c),
                )
                for c in result.page_contents
            ]

            log_entry = PageProcessingLog(
                id=PageProcessingLog.make_id(result.source_file, result.page_number),
                document_id=doc_id,
                job_id=job_id,
                source_file=result.source_file,
                page_number=result.page_number,
                detection_count=result.detection_count,
                processing_duration_ms=result.processing_duration_ms,
                started_at=result.started_at,
                completed_at=result.completed_at,
            )

            write_item = _DbWriteItem(
                document_id=doc_id,
                job_id=job_id,
                source_file=result.source_file,
                page_number=result.page_number,
                success=True,
                metrics=metric_objs,
                page_results=page_result_objs,
                pages_received_increment=len(result.page_contents),
                log_entry=log_entry,
                total_pages=result.total_pages,
                detection_count=result.detection_count,
                processing_duration_ms=result.processing_duration_ms,
                started_at=result.started_at,
                completed_at=result.completed_at,
            )
        else:
            fail_log = PageProcessingLog(
                id=PageProcessingLog.make_id(result.source_file, result.page_number),
                document_id=doc_id,
                job_id=job_id,
                source_file=result.source_file,
                page_number=result.page_number,
                status="failed",
                error_message=result.error_message,
                failure_type=result.failure_type,
                detection_count=0,
                processing_duration_ms=0.0,
                started_at=result.started_at,
                completed_at=result.completed_at,
            )

            write_item = _DbWriteItem(
                document_id=doc_id,
                job_id=job_id,
                source_file=result.source_file,
                page_number=result.page_number,
                success=False,
                error_message=result.error_message,
                failure_type=result.failure_type,
                log_entry=fail_log,
                started_at=result.started_at,
                completed_at=result.completed_at,
            )

        if self._db_writer is not None:
            self._db_writer.enqueue(write_item)

        if job_id:
            self._handle_job_completion(job_id)

    def _handle_job_completion(self, job_id: str) -> None:
        """Track page completion in-memory and publish job_complete SSE immediately.

        Uses an in-memory counter so the SSE fires as soon as the last
        page result arrives, without waiting for the DB writer thread.
        """
        with self._job_page_counts_lock:
            self._job_page_counts[job_id] = self._job_page_counts.get(job_id, 0) + 1
            pages_completed = self._job_page_counts[job_id]

        repo = Repository(self._db_engine)
        job = repo.get_job(job_id)
        if job is None:
            return

        if pages_completed >= job.total_pages and job.total_pages > 0:
            with self._job_page_counts_lock:
                self._job_page_counts.pop(job_id, None)
            logger.info("[job %s] All %d pages of %s complete", job_id[:8], job.total_pages, job.filename)
            asyncio.run_coroutine_threadsafe(
                self._event_bus.publish(
                    job_id,
                    {
                        "event": "job_complete",
                        "job_id": job_id,
                        "filename": job.filename,
                        "total_pages": job.total_pages,
                    },
                ),
                self._event_loop,
            )

    # ------------------------------------------------------------------
    # Submit (public API — enqueues into the batch buffer)
    # ------------------------------------------------------------------

    def try_submit(
        self,
        document_id: str,
        content_sha256: str,
        file_bytes: bytes,
        filename: str,
        *,
        job_id: str | None = None,
        page_number: int = 1,
        spool_path: str | None = None,
    ) -> bool:
        """Enqueue a page for batched processing.

        Returns ``True`` if the page was accepted, ``False`` if the buffer
        is full or the pool is draining / job is cancelled (caller should
        translate to a 503 with the appropriate detail).

        When ``spool_path`` is provided the page bytes are NOT carried
        in-memory through the IPC payload — the worker reads them from
        the spool file inside its own process.  This keeps the
        cross-process descriptor small (a few KB) and means a worker
        crash mid-batch leaves the bytes available for re-enqueue.
        """
        if self._executor is None:
            raise RuntimeError("ProcessingPool has not been started")
        if self._buffer is None:
            raise RuntimeError("Batch buffer has not been initialised")

        if self._draining.is_set():
            return False
        if self.is_job_cancelled(job_id):
            return False

        # When the bytes have been spooled, do NOT also pickle them into
        # the descriptor — the worker will read them from disk.
        descriptor = PageDescriptor(
            document_id=document_id,
            content_sha256=content_sha256,
            file_bytes=b"" if spool_path else file_bytes,
            filename=filename,
            job_id=job_id,
            page_number=page_number,
            spool_path=spool_path,
        )
        return self._buffer.enqueue(dataclasses.asdict(descriptor))
