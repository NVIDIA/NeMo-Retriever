# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge between the service layer and the nemo-retriever pipeline.

Builds ``ExtractParams`` / ``EmbedParams`` from :class:`ServiceConfig` and
returns async work functions suitable for :class:`_Pool` worker loops.

Each work function:

1. Constructs a fresh :class:`GraphIngestor` per item (cheap — just sets
   Python attributes).
2. Feeds the raw bytes via ``.buffers()`` so no temp files are needed.
3. Runs the synchronous ``InprocessExecutor`` pipeline in a **child
   process** via :class:`concurrent.futures.ProcessPoolExecutor` to
   isolate PDFium's non-thread-safe C library.
4. Returns a lightweight summary of the result rows for status polling.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from io import BytesIO
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import numpy as np

if TYPE_CHECKING:
    from nemo_retriever.service.config import NimEndpointsConfig, ServiceConfig
    from nemo_retriever.service.services.pipeline_pool import WorkItem

logger = logging.getLogger(__name__)

_MP_CONTEXT = mp.get_context("forkserver")
_MAX_TASKS_PER_CHILD = 100

_SENSITIVE_PATTERNS = frozenset(
    {
        "api_key",
        "password",
        "secret",
        "token",
        "credential",
    }
)


def _redact_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of *d* with sensitive-looking values masked."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if any(pat in k.lower() for pat in _SENSITIVE_PATTERNS):
            out[k] = "***REDACTED***" if v else None
        elif isinstance(v, dict):
            out[k] = _redact_dict(v)
        else:
            out[k] = v
    return out


def _params_to_dict(params: Any) -> dict[str, Any]:
    """Serialize a Pydantic params model to a redacted dict."""
    if params is None:
        return {}
    raw = params.model_dump(mode="json") if hasattr(params, "model_dump") else {}
    return _redact_dict(raw)


_pipeline_configs: dict[str, dict[str, Any]] = {}


def get_pipeline_configs() -> dict[str, dict[str, Any]]:
    """Return the captured pipeline configurations (populated at startup)."""
    return _pipeline_configs


_LARGE_COLUMNS = frozenset(
    {
        "bytes",
        "page_image",
        "image_b64",
        "images",
        "charts",
        "infographics",
        "tables",
    }
)

_MAX_STR_LEN = 500


def _sanitize_value(val: Any) -> Any:
    """Convert a single cell value to a JSON-safe, memory-friendly form."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return f"<ndarray shape={val.shape} dtype={val.dtype}>"
    if isinstance(val, (list, tuple)) and len(val) > 20:
        return f"<{type(val).__name__} len={len(val)}>"
    if isinstance(val, bytes):
        return f"<bytes len={len(val)}>"
    if isinstance(val, str) and len(val) > _MAX_STR_LEN:
        return val[:_MAX_STR_LEN] + f"…[{len(val)} chars total]"
    return val


def _sanitize_result_data(df: Any) -> list[dict[str, Any]]:
    """Convert a pipeline DataFrame to lightweight JSON-safe dicts.

    Drops large binary/image columns entirely and truncates remaining
    values so the result can be stored in memory and returned via the
    status endpoint without risk of OOM.
    """
    cols_to_keep = [c for c in df.columns if c not in _LARGE_COLUMNS]
    light_df = df[cols_to_keep]
    records = light_df.to_dict(orient="records")
    return [{k: _sanitize_value(v) for k, v in row.items()} for row in records]


# ── Process pool registry ────────────────────────────────────────────

_process_executors: list[ProcessPoolExecutor] = []


def shutdown_process_executors() -> None:
    """Shut down all process pool executors created by work-function factories.

    Called during application shutdown (before the asyncio pool is torn down)
    so that child processes are reaped cleanly.  Actively kills running
    child processes so shutdown is not blocked by long-running pipelines.
    """
    import os
    import signal

    for executor in _process_executors:
        # Kill running child processes immediately so blocked
        # run_in_executor() futures unblock.
        pids: list[int] = []
        if hasattr(executor, "_processes"):
            pids = list(executor._processes.keys())
        executor.shutdown(wait=False, cancel_futures=True)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    _process_executors.clear()
    logger.info("All pipeline process executors shut down")


def _post_rows_to_vectordb(rows: list[dict[str, Any]], vectordb_url: str, filename: str) -> None:
    """Fire-and-forget POST of LanceDB rows to the vectordb service."""
    import json
    import urllib.request
    import urllib.error

    if not rows:
        return

    url = vectordb_url.rstrip("/") + "/internal/vectordb/write"
    body = json.dumps({"rows": rows}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            logging.getLogger(__name__).info(
                "Posted %d rows to vectordb for %s — HTTP %d",
                len(rows),
                filename,
                resp.status,
            )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Failed to POST %d rows to vectordb for %s: %s",
            len(rows),
            filename,
            exc,
        )


def _run_pipeline_in_process(
    filename: str,
    payload: bytes,
    extract_params_dict: dict[str, Any],
    embed_params_dict: dict[str, Any] | None,
    vectordb_url: str | None = None,
) -> tuple[int, list[dict[str, Any]], float]:
    """Execute one pipeline run inside a child process.

    This is a **top-level module function** so it can be pickled by
    :class:`ProcessPoolExecutor`.  All heavy imports happen here so
    that the parent process stays lightweight.
    """
    from nemo_retriever.graph_ingestor import GraphIngestor
    from nemo_retriever.params import EmbedParams, ExtractParams

    t0 = time.monotonic()

    extract_params = ExtractParams(**extract_params_dict)
    embed_params = EmbedParams(**embed_params_dict) if embed_params_dict else None

    ingestor = GraphIngestor(run_mode="inprocess", show_progress=False)
    ingestor = ingestor.buffers([(filename, BytesIO(payload))])
    ingestor = ingestor.extract(extract_params)
    if embed_params is not None:
        ingestor = ingestor.embed(embed_params)

    result_df = ingestor.ingest()
    elapsed = time.monotonic() - t0

    row_count = len(result_df)

    if vectordb_url and row_count > 0:
        from nemo_retriever.vdb.lancedb_schema import build_lancedb_rows

        lancedb_rows = build_lancedb_rows(result_df)
        _post_rows_to_vectordb(lancedb_rows, vectordb_url, filename)

    result_data = _sanitize_result_data(result_df)
    return row_count, result_data, elapsed


def build_extract_params(nim: NimEndpointsConfig) -> Any:
    """Derive :class:`ExtractParams` from service NIM endpoint config.

    The ``ExtractParams`` model validator auto-enables
    ``use_graphic_elements`` / ``use_table_structure`` when the
    corresponding invoke URLs are provided.
    """
    from nemo_retriever.params import ExtractParams

    kwargs: dict[str, Any] = {}
    if nim.page_elements_invoke_url:
        kwargs["page_elements_invoke_url"] = nim.page_elements_invoke_url
    if nim.ocr_invoke_url:
        kwargs["ocr_invoke_url"] = nim.ocr_invoke_url
    if nim.graphic_elements_invoke_url:
        kwargs["graphic_elements_invoke_url"] = nim.graphic_elements_invoke_url
    if nim.table_structure_invoke_url:
        kwargs["table_structure_invoke_url"] = nim.table_structure_invoke_url
    if nim.api_key:
        kwargs["api_key"] = nim.api_key

    return ExtractParams(**kwargs)


def build_embed_params(nim: NimEndpointsConfig) -> Any | None:
    """Derive :class:`EmbedParams` from service NIM endpoint config.

    Returns ``None`` when no embedding endpoint is configured, signalling
    that the embed stage should be skipped.
    """
    if not nim.embed_invoke_url:
        return None

    from nemo_retriever.params import EmbedParams

    kwargs: dict[str, Any] = {"embed_invoke_url": nim.embed_invoke_url}
    if nim.api_key:
        kwargs["api_key"] = nim.api_key

    return EmbedParams(**kwargs)


def _make_work_fn(
    config: ServiceConfig,
    *,
    label: str,
) -> Callable[[WorkItem], Awaitable[tuple[int, list[dict[str, Any]]]]]:
    """Factory that captures pipeline params once and returns an async worker.

    Each invocation creates a :class:`ProcessPoolExecutor` so that every
    pipeline run is isolated in its own child process — this eliminates
    PDFium thread-safety issues (the C library has global mutable state
    that corrupts under concurrent thread access).
    """
    extract_params = build_extract_params(config.nim_endpoints)
    embed_params = build_embed_params(config.nim_endpoints)

    vectordb_url: str | None = None
    if config.vectordb.enabled:
        vectordb_url = config.vectordb.vectordb_url
        logger.info("VectorDB write enabled for %s workers → %s", label, vectordb_url)

    num_workers = config.pipeline.realtime_workers if label.lower() == "realtime" else config.pipeline.batch_workers

    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=_MP_CONTEXT,
        max_tasks_per_child=_MAX_TASKS_PER_CHILD,
    )
    _process_executors.append(executor)

    extract_params_dict = extract_params.model_dump(mode="json")
    embed_params_dict = embed_params.model_dump(mode="json") if embed_params else None

    _pipeline_configs[label.lower()] = {
        "label": label,
        "run_mode": "inprocess",
        "execution": "process-isolated",
        "show_progress": False,
        "extract_params": _params_to_dict(extract_params),
        "embed_params": _params_to_dict(embed_params) if embed_params else None,
        "embed_enabled": embed_params is not None,
        "pool": {
            "workers": num_workers,
            "queue_size": (
                config.pipeline.realtime_queue_size if label.lower() == "realtime" else config.pipeline.batch_queue_size
            ),
            "max_tasks_per_child": _MAX_TASKS_PER_CHILD,
        },
        "nim_endpoints": _redact_dict(config.nim_endpoints.model_dump(mode="json")),
    }

    logger.info(
        "Pipeline work function created (%s): extract=%s, embed=%s, " "process_pool_workers=%d, max_tasks_per_child=%d",
        label,
        type(extract_params).__name__,
        type(embed_params).__name__ if embed_params else "disabled",
        num_workers,
        _MAX_TASKS_PER_CHILD,
    )

    # Mutable holder so the BrokenProcessPool handler can replace the
    # executor while the closure keeps a stable reference.
    executor_ref: list[ProcessPoolExecutor] = [executor]

    async def _work(item: WorkItem) -> tuple[int, list[dict[str, Any]]]:
        filename = item.filename or item.id
        loop = asyncio.get_running_loop()

        try:
            row_count, result_data, elapsed = await loop.run_in_executor(
                executor_ref[0],
                _run_pipeline_in_process,
                filename,
                item.payload,
                extract_params_dict,
                embed_params_dict,
                vectordb_url,
            )
        except BrokenProcessPool:
            logger.error(
                "%s process pool broken (worker crash) while processing " "id=%s file=%s — recreating pool",
                label,
                item.id,
                filename,
            )
            old = executor_ref[0]
            try:
                old.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            if old in _process_executors:
                _process_executors.remove(old)
            new_executor = ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=_MP_CONTEXT,
                max_tasks_per_child=_MAX_TASKS_PER_CHILD,
            )
            executor_ref[0] = new_executor
            _process_executors.append(new_executor)
            raise

        logger.info(
            "%s pipeline completed: id=%s file=%s rows=%d elapsed=%.2fs",
            label,
            item.id,
            filename,
            row_count,
            elapsed,
        )
        return row_count, result_data

    return _work


def create_realtime_work_fn(
    config: ServiceConfig,
) -> Callable[[WorkItem], Awaitable[tuple[int, list[dict[str, Any]]]]]:
    """Build the async work function for the **realtime** pool.

    Processes single pages — the extract operator finds one page and the
    pipeline runs with minimal latency.
    """
    return _make_work_fn(config, label="Realtime")


def create_batch_work_fn(
    config: ServiceConfig,
) -> Callable[[WorkItem], Awaitable[tuple[int, list[dict[str, Any]]]]]:
    """Build the async work function for the **batch** pool.

    Processes full documents — the extract operator splits internally
    into N pages and processes them in one pass for better throughput.
    """
    return _make_work_fn(config, label="Batch")
