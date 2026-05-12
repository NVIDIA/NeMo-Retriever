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
3. Runs the synchronous ``InprocessExecutor`` pipeline off the event loop
   via :func:`asyncio.to_thread`.
4. Logs a summary of the result and discards it (storage can be added
   later by replacing the tail of the work function).
"""

from __future__ import annotations

import asyncio
import logging
import time
from io import BytesIO
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from nemo_retriever.service.config import NimEndpointsConfig, ServiceConfig
    from nemo_retriever.service.services.pipeline_pool import WorkItem

logger = logging.getLogger(__name__)


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
) -> Callable[[WorkItem], Awaitable[None]]:
    """Factory that captures pipeline params once and returns an async worker.

    The returned coroutine is safe for concurrent use: each invocation
    creates its own ``GraphIngestor`` and operator instances.
    """
    extract_params = build_extract_params(config.nim_endpoints)
    embed_params = build_embed_params(config.nim_endpoints)

    logger.info(
        "Pipeline work function created (%s): extract=%s, embed=%s",
        label,
        type(extract_params).__name__,
        type(embed_params).__name__ if embed_params else "disabled",
    )

    async def _work(item: WorkItem) -> None:
        from nemo_retriever.graph_ingestor import GraphIngestor

        filename = item.filename or item.id

        def _run() -> Any:
            t0 = time.monotonic()
            ingestor = GraphIngestor(run_mode="inprocess", show_progress=False)
            ingestor = ingestor.buffers([(filename, BytesIO(item.payload))])
            ingestor = ingestor.extract(extract_params)
            if embed_params is not None:
                ingestor = ingestor.embed(embed_params)
            result = ingestor.ingest()
            elapsed = time.monotonic() - t0
            return result, elapsed

        result, elapsed = await asyncio.to_thread(_run)
        logger.info(
            "%s pipeline completed: id=%s file=%s rows=%d elapsed=%.2fs",
            label,
            item.id,
            filename,
            len(result),
            elapsed,
        )

    return _work


def create_realtime_work_fn(
    config: ServiceConfig,
) -> Callable[[WorkItem], Awaitable[None]]:
    """Build the async work function for the **realtime** pool.

    Processes single pages — the extract operator finds one page and the
    pipeline runs with minimal latency.
    """
    return _make_work_fn(config, label="Realtime")


def create_batch_work_fn(
    config: ServiceConfig,
) -> Callable[[WorkItem], Awaitable[None]]:
    """Build the async work function for the **batch** pool.

    Processes full documents — the extract operator splits internally
    into N pages and processes them in one pass for better throughput.
    """
    return _make_work_fn(config, label="Batch")
