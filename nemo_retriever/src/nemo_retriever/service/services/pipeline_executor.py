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


_TRUST_OWNED_EXTRACT_KEYS: tuple[str, ...] = (
    "invoke_url",
    "api_key",
    "page_elements_invoke_url",
    "page_elements_api_key",
    "ocr_invoke_url",
    "ocr_api_key",
    "graphic_elements_invoke_url",
    "table_structure_invoke_url",
    "nemotron_parse_invoke_url",
)
_TRUST_OWNED_EMBED_KEYS: tuple[str, ...] = (
    "embed_invoke_url",
    "embedding_endpoint",
    "api_key",
    "embed_model_name",
    "model_name",
)
# Trust-owned caption keys. ``endpoint_url`` / ``api_key`` /
# ``model_name`` are all set by the operator via NimEndpointsConfig and
# can never be redirected per-request.
_TRUST_OWNED_CAPTION_KEYS: tuple[str, ...] = (
    "endpoint_url",
    "api_key",
    "model_name",
)


def _merge_server_owned(
    base: dict[str, Any], override: dict[str, Any] | None, owned: tuple[str, ...]
) -> dict[str, Any]:
    """Merge *override* on top of *base* while preserving server-owned keys.

    The denylist enforced by :mod:`nemo_retriever.service.policy` already
    rejects requests with these keys, but we apply a belt-and-suspenders
    overwrite here so a misconfigured policy can never cause a request
    to redirect endpoint URLs or replace API keys.
    """
    merged = dict(base)
    if override:
        merged.update(override)
    for k in owned:
        if k in base:
            merged[k] = base[k]
    return merged


def _resolve_sidecar_in_spec(spec: dict[str, Any] | None) -> dict[str, Any] | None:
    """Resolve ``vdb_upload_params.meta_dataframe_id`` to in-band bytes.

    The pipeline runs in a child process that cannot reach the
    ``SidecarStore`` directly, so the parent process consumes the
    sidecar (or fails the request) before submitting the work item.
    The returned spec stays pickleable: ``meta_dataframe_id`` becomes
    ``_meta_dataframe_bytes`` + ``_meta_dataframe_content_type``,
    which :func:`_build_graph_ingestor_from_spec` resolves to a
    pandas DataFrame inside the worker.
    """
    if spec is None:
        return None
    vdb = spec.get("vdb_upload_params")
    if not vdb:
        return spec
    sidecar_id = vdb.get("meta_dataframe_id")
    if not sidecar_id:
        return spec

    from nemo_retriever.service.services.sidecar_store import get_sidecar_store

    store = get_sidecar_store()
    if store is None:
        raise RuntimeError(
            "vdb_upload_params.meta_dataframe_id was set but the SidecarStore " "is not initialised on this pod."
        )
    entry = store.consume(sidecar_id)
    if entry is None:
        raise RuntimeError(
            f"Sidecar id {sidecar_id!r} not found. The sidecar may have "
            "expired (default TTL is 1h) or already been consumed. "
            "Re-upload via POST /v1/ingest/sidecar."
        )

    resolved = dict(spec)
    vdb_copy = dict(vdb)
    vdb_copy.pop("meta_dataframe_id", None)
    vdb_copy["_meta_dataframe_bytes"] = entry.payload
    vdb_copy["_meta_dataframe_content_type"] = entry.content_type
    vdb_copy["_meta_dataframe_filename"] = entry.filename
    resolved["vdb_upload_params"] = vdb_copy
    return resolved


def _materialize_sidecar_bytes(vdb_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert resolved sidecar bytes into a pandas DataFrame in place.

    Runs *inside* the child process. ``_meta_dataframe_bytes`` is the
    payload uploaded via ``POST /v1/ingest/sidecar``; the content-type
    (or filename suffix as fallback) picks the right pandas reader.
    """
    payload = vdb_kwargs.pop("_meta_dataframe_bytes", None)
    if payload is None:
        return vdb_kwargs
    content_type = vdb_kwargs.pop("_meta_dataframe_content_type", "") or ""
    filename = vdb_kwargs.pop("_meta_dataframe_filename", "") or ""

    from io import BytesIO

    import pandas as pd

    ct_lower = content_type.lower()
    fname_lower = filename.lower()
    if "parquet" in ct_lower or fname_lower.endswith(".parquet") or fname_lower.endswith(".pq"):
        df = pd.read_parquet(BytesIO(payload))
    elif "json" in ct_lower or fname_lower.endswith(".json") or fname_lower.endswith(".jsonl"):
        df = pd.read_json(BytesIO(payload), lines=fname_lower.endswith(".jsonl"))
    else:
        df = pd.read_csv(BytesIO(payload))
    vdb_kwargs["meta_dataframe"] = df
    return vdb_kwargs


def _build_graph_ingestor_from_spec(
    filename: str,
    payload: bytes,
    base_extract: dict[str, Any],
    base_embed: dict[str, Any] | None,
    spec: dict[str, Any] | None,
    base_caption: dict[str, Any] | None = None,
) -> "tuple[Any, str, bool]":
    """Construct a :class:`GraphIngestor` reflecting the per-request *spec*.

    Returns ``(ingestor, extraction_mode, has_per_request_vdb)``. The
    last value tells the caller to skip the legacy out-of-graph
    vectordb fan-out — the in-graph ``IngestVdbOperator`` already
    handles persistence when ``vdb_upload_params`` is present.
    """
    from nemo_retriever.graph_ingestor import GraphIngestor
    from nemo_retriever.params import (
        CaptionParams,
        DedupParams,
        EmbedParams,
        ExtractParams,
        StoreParams,
        VdbUploadParams,
        WebhookParams,
    )

    spec = spec or {}
    extraction_mode = spec.get("extraction_mode", "pdf")

    extract_kwargs = _merge_server_owned(base_extract, spec.get("extract_params"), _TRUST_OWNED_EXTRACT_KEYS)
    extract_params = ExtractParams(**extract_kwargs)

    embed_override = spec.get("embed_params")
    if base_embed is None and embed_override is None:
        embed_params = None
    else:
        embed_base = base_embed or {}
        embed_kwargs = _merge_server_owned(embed_base, embed_override, _TRUST_OWNED_EMBED_KEYS)
        embed_params = EmbedParams(**embed_kwargs) if embed_kwargs.get("embed_invoke_url") else None

    # Caption baseline + per-request overrides. The base dict carries
    # the server-owned endpoint/API key/model name; the override carries
    # behavioural knobs (prompt, system_prompt, batch_size, …).
    caption_override = spec.get("caption_params")
    if base_caption is None and caption_override is None:
        caption_params = None
    elif base_caption is None and caption_override is not None:
        raise RuntimeError(
            "caption_params provided but no caption endpoint is configured on "
            "this worker. The policy layer should have rejected this earlier."
        )
    else:
        caption_kwargs = _merge_server_owned(base_caption or {}, caption_override, _TRUST_OWNED_CAPTION_KEYS)
        caption_params = CaptionParams(**caption_kwargs) if caption_kwargs.get("endpoint_url") else None

    ingestor = GraphIngestor(run_mode="inprocess", show_progress=False)
    ingestor = ingestor.buffers([(filename, BytesIO(payload))])

    if extraction_mode == "image":
        ingestor = ingestor.extract_image_files(extract_params, split_config=spec.get("split_config"))
    else:
        ingestor = ingestor.extract(
            extract_params,
            split_config=spec.get("split_config"),
            extraction_mode=extraction_mode,
        )

    stage_order = spec.get("stage_order") or []
    seen_post_extract: set[str] = set()

    def _apply_store_if_requested() -> None:
        nonlocal ingestor
        store_kwargs = spec.get("store_params")
        if store_kwargs is not None:
            ingestor = ingestor.store(StoreParams(**store_kwargs))

    def _apply_webhook_if_requested() -> None:
        nonlocal ingestor
        webhook_kwargs = spec.get("webhook_params")
        if webhook_kwargs is not None:
            ingestor = ingestor.webhook(WebhookParams(**webhook_kwargs))

    def _apply_caption_if_requested() -> None:
        nonlocal ingestor
        if caption_params is not None:
            ingestor = ingestor.caption(caption_params)

    for stage_name in stage_order:
        if stage_name in ("extract",) or stage_name in seen_post_extract:
            continue
        seen_post_extract.add(stage_name)
        if stage_name == "dedup":
            dedup_kwargs = spec.get("dedup_params") or {}
            ingestor = ingestor.dedup(DedupParams(**dedup_kwargs))
        elif stage_name == "embed":
            if embed_params is not None:
                ingestor = ingestor.embed(embed_params)
        elif stage_name == "filter":
            ingestor = ingestor.filter()
        elif stage_name == "store":
            _apply_store_if_requested()
        elif stage_name == "webhook":
            _apply_webhook_if_requested()
        elif stage_name == "caption":
            _apply_caption_if_requested()

    if embed_params is not None and "embed" not in seen_post_extract:
        ingestor = ingestor.embed(embed_params)

    # ``store`` / ``webhook`` / ``caption`` may be present in params
    # without an explicit stage_order entry (matches the GraphIngestor
    # pattern where the params model triggers the stage). Auto-append.
    if "caption" not in seen_post_extract:
        _apply_caption_if_requested()
    if "store" not in seen_post_extract:
        _apply_store_if_requested()
    if "webhook" not in seen_post_extract:
        _apply_webhook_if_requested()

    # vdb_upload is not a stage_order entry in GraphIngestor either — the
    # operator is always appended after embed/store from the params model.
    has_per_request_vdb = False
    vdb_kwargs = spec.get("vdb_upload_params")
    if vdb_kwargs is not None:
        # Sidecar metadata (Phase 6): the parent process placed the
        # uploaded bytes on the spec; turn them into a DataFrame here.
        vdb_kwargs = _materialize_sidecar_bytes(dict(vdb_kwargs))
        ingestor = ingestor.vdb_upload(VdbUploadParams(**vdb_kwargs))
        has_per_request_vdb = True

    return ingestor, extraction_mode, has_per_request_vdb


def _run_pipeline_in_process(
    filename: str,
    payload: bytes,
    extract_params_dict: dict[str, Any],
    embed_params_dict: dict[str, Any] | None,
    vectordb_url: str | None = None,
    pipeline_spec: dict[str, Any] | None = None,
    caption_params_dict: dict[str, Any] | None = None,
) -> tuple[int, list[dict[str, Any]], float]:
    """Execute one pipeline run inside a child process.

    This is a **top-level module function** so it can be pickled by
    :class:`ProcessPoolExecutor`.  All heavy imports happen here so
    that the parent process stays lightweight.

    The pipeline shape comes from two layers:

    * ``extract_params_dict`` / ``embed_params_dict`` — server-owned
      defaults derived from :class:`ServiceConfig.nim_endpoints` at
      startup. Carry the endpoint URLs and API keys.
    * ``pipeline_spec`` — optional per-request override validated by
      :func:`nemo_retriever.service.policy.validate_pipeline_spec`.
      Carries "shape" knobs (chunk sizes, output flags, stage order, …).

    When ``pipeline_spec`` is ``None`` (or empty) the behaviour exactly
    matches the original closure-baked pipeline.
    """
    t0 = time.monotonic()

    ingestor, _extraction_mode, has_per_request_vdb = _build_graph_ingestor_from_spec(
        filename,
        payload,
        extract_params_dict,
        embed_params_dict,
        pipeline_spec,
        caption_params_dict,
    )

    result_df = ingestor.ingest()
    elapsed = time.monotonic() - t0

    row_count = len(result_df)

    if vectordb_url and row_count > 0 and not has_per_request_vdb:
        # Skip the out-of-graph fan-out when the client already wired
        # IngestVdbOperator into the spec — that operator handles
        # persistence itself.
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


def build_caption_params(nim: NimEndpointsConfig) -> Any | None:
    """Derive :class:`CaptionParams` from service NIM endpoint config.

    Returns ``None`` when no caption endpoint is configured — clients
    that request the ``caption`` stage will hit the policy's
    ``caption_enabled`` guard before reaching this point.
    """
    from nemo_retriever.params import CaptionParams

    if not nim.caption_invoke_url:
        return None

    kwargs: dict[str, Any] = {"endpoint_url": nim.caption_invoke_url}
    if nim.caption_model_name:
        kwargs["model_name"] = nim.caption_model_name
    if nim.api_key:
        kwargs["api_key"] = nim.api_key
    return CaptionParams(**kwargs)


def build_embed_params(nim: NimEndpointsConfig) -> Any | None:
    """Derive :class:`EmbedParams` from service NIM endpoint config.

    Returns ``None`` when no embedding endpoint is configured, signalling
    that the embed stage should be skipped.
    """
    if not nim.embed_invoke_url:
        return None

    from nemo_retriever.params import EmbedParams

    kwargs: dict[str, Any] = {"embed_invoke_url": nim.embed_invoke_url}
    if nim.embed_model_name:
        kwargs["model_name"] = nim.embed_model_name
        kwargs["embed_model_name"] = nim.embed_model_name
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
    caption_params = build_caption_params(config.nim_endpoints)

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
    caption_params_dict = caption_params.model_dump(mode="json") if caption_params else None

    _pipeline_configs[label.lower()] = {
        "label": label,
        "run_mode": "inprocess",
        "execution": "process-isolated",
        "show_progress": False,
        "extract_params": _params_to_dict(extract_params),
        "embed_params": _params_to_dict(embed_params) if embed_params else None,
        "embed_enabled": embed_params is not None,
        "caption_params": _redact_dict(_params_to_dict(caption_params)) if caption_params else None,
        "caption_enabled": caption_params is not None,
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

        resolved_spec = _resolve_sidecar_in_spec(item.pipeline_spec)

        try:
            row_count, result_data, elapsed = await loop.run_in_executor(
                executor_ref[0],
                _run_pipeline_in_process,
                filename,
                item.payload,
                extract_params_dict,
                embed_params_dict,
                vectordb_url,
                resolved_spec,
                caption_params_dict,
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
