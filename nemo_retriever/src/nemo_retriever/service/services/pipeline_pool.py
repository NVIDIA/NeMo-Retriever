# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline pool manager for low-latency and batch workloads.

Maintains two independent worker pools:

- **realtime pool** — sized for low-latency, one-at-a-time page processing.
  Small number of workers, short queue, prioritises fast turnaround.
- **batch pool** — sized for throughput-oriented bulk uploads.
  Larger worker count, deep queue, optimised for sustained saturation.

Both pools expose the same submission interface so callers don't need to
know which pool handles their work — routing is decided at the service
layer based on the ingest path that accepted the request.

Singleton access follows the same optional pattern as the metrics service::

    if (pool := get_pipeline_pool()) is not None:
        accepted = await pool.submit(PoolType.REALTIME, item)
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any, Callable

from pydantic import ConfigDict

from nemo_retriever.service.config import PipelinePoolConfig
from nemo_retriever.service.models.base import RichModel

logger = logging.getLogger(__name__)


class PoolType(str, Enum):
    REALTIME = "realtime"
    BATCH = "batch"


class WorkItem(RichModel):
    """A unit of work submitted to a pool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    payload: Any = None
    filename: str | None = None
    callback: Callable[[Any], None] | None = None
    callback_url: str | None = None


async def _fire_gateway_callback(
    callback_url: str,
    item_id: str,
    status: str,
    *,
    result_rows: int = 0,
    result_data: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> None:
    """POST job completion data back to the originating gateway pod."""
    import httpx

    payload: dict[str, Any] = {
        "id": item_id,
        "status": status,
        "result_rows": result_rows,
    }
    if result_data is not None:
        payload["result_data"] = result_data
    if error:
        payload["error"] = error

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(callback_url, json=payload)
            if resp.status_code != 200:
                logger.warning(
                    "Gateway callback returned HTTP %d for item %s",
                    resp.status_code,
                    item_id,
                )
    except Exception as exc:
        logger.warning("Failed to fire gateway callback for item %s: %s", item_id, exc)


class _Pool:
    """A single bounded worker pool backed by an asyncio.Queue.

    Workers are spawned at :meth:`start` and drain the queue continuously.
    The ``work_fn`` callback is called for each item; when ``None`` (the
    default) items are acknowledged and discarded immediately (useful for
    benchmarking upload throughput before real pipeline stages are wired in).
    """

    def __init__(
        self,
        name: str,
        num_workers: int,
        max_queue_size: int,
        work_fn: Callable[[WorkItem], Any] | None = None,
    ) -> None:
        self._name = name
        self._num_workers = num_workers
        self._max_queue_size = max_queue_size
        self._work_fn = work_fn
        self._queue: asyncio.Queue[WorkItem | None] | None = None
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._processed: int = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def queue_depth(self) -> int:
        if self._queue is None:
            return 0
        return self._queue.qsize()

    @property
    def max_queue_size(self) -> int:
        return self._max_queue_size

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def processed(self) -> int:
        return self._processed

    def start(self) -> None:
        if self._running:
            return
        self._queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._running = True
        self._workers = [asyncio.create_task(self._worker_loop(i)) for i in range(self._num_workers)]
        logger.info(
            "Pool '%s' started: workers=%d queue_size=%d work_fn=%s",
            self._name,
            self._num_workers,
            self._max_queue_size,
            self._work_fn.__name__ if self._work_fn else "noop",
        )

    async def _worker_loop(self, worker_id: int) -> None:
        """Consume items until a ``None`` sentinel is received.

        When an item has a ``callback_url`` (set by the gateway), the
        worker POSTs completion data back to the gateway instead of
        updating a local job tracker.  In standalone mode (no callback),
        the local tracker is updated directly.
        """
        from nemo_retriever.service.services.job_tracker import get_job_tracker

        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                tracker = get_job_tracker()
                if tracker is not None:
                    tracker.mark_processing(item.id)
                result_rows = 0
                result_data = None
                if self._work_fn is not None:
                    result = self._work_fn(item)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, tuple) and len(result) == 2:
                        result_rows, result_data = result
                    elif isinstance(result, int):
                        result_rows = result

                if item.callback_url:
                    await _fire_gateway_callback(
                        item.callback_url,
                        item.id,
                        "completed",
                        result_rows=result_rows,
                        result_data=result_data,
                    )
                elif tracker is not None:
                    tracker.mark_completed(
                        item.id,
                        result_rows=result_rows,
                        result_data=result_data,
                    )
                self._processed += 1
            except Exception as exc:
                if item.callback_url:
                    await _fire_gateway_callback(
                        item.callback_url,
                        item.id,
                        "failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    tracker = get_job_tracker()
                    if tracker is not None:
                        tracker.mark_failed(item.id, f"{type(exc).__name__}: {exc}")
                logger.exception("Pool '%s' worker %d failed on item %s", self._name, worker_id, item.id)
            finally:
                self._queue.task_done()

    async def submit(self, item: WorkItem) -> bool:
        """Enqueue a work item.  Returns ``False`` if the queue is full."""
        if not self._running or self._queue is None:
            return False
        try:
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            return False

    def has_capacity(self) -> bool:
        if self._queue is None:
            return False
        return not self._queue.full()

    async def shutdown(self, timeout: float = 5.0) -> None:
        if not self._running:
            return
        self._running = False

        # Cancel all worker tasks immediately — don't bother draining
        # the queue with sentinels since active workers may be blocked
        # on long-running child processes.  The process executors are
        # already shut down by the time we get here, so the blocked
        # run_in_executor() futures will raise quickly.
        for task in self._workers:
            task.cancel()

        if self._workers:
            done, still_pending = await asyncio.wait(
                self._workers,
                timeout=timeout,
            )
            if still_pending:
                logger.warning(
                    "Pool '%s': %d workers did not exit within %.1fs — " "force-cancelling",
                    self._name,
                    len(still_pending),
                    timeout,
                )
                for task in still_pending:
                    task.cancel()

        self._workers.clear()
        self._queue = None
        logger.info("Pool '%s' shut down (processed=%d)", self._name, self._processed)

    def stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "num_workers": self._num_workers,
            "max_queue_size": self._max_queue_size,
            "queue_depth": self.queue_depth,
            "processed": self._processed,
            "running": self._running,
        }


class PipelinePool:
    """Manages separate realtime and batch worker pools.

    Constructed from the ``pipeline`` section of ``ServiceConfig``.
    When *mode* is ``realtime`` or ``batch``, only the corresponding pool
    is created; the other is ``None`` and submissions to it are rejected.
    """

    def __init__(
        self,
        config: PipelinePoolConfig,
        *,
        mode: str = "standalone",
        realtime_work_fn: Callable[[WorkItem], Any] | None = None,
        batch_work_fn: Callable[[WorkItem], Any] | None = None,
    ) -> None:
        self._config = config
        self._mode = mode
        self._realtime: _Pool | None = None
        self._batch: _Pool | None = None

        if mode in ("standalone", "realtime"):
            self._realtime = _Pool(
                name="realtime",
                num_workers=config.realtime_workers,
                max_queue_size=config.realtime_queue_size,
                work_fn=realtime_work_fn,
            )
        if mode in ("standalone", "batch"):
            self._batch = _Pool(
                name="batch",
                num_workers=config.batch_workers,
                max_queue_size=config.batch_queue_size,
                work_fn=batch_work_fn,
            )

    @property
    def mode(self) -> str:
        return self._mode

    def start(self) -> None:
        if self._realtime is not None:
            self._realtime.start()
        if self._batch is not None:
            self._batch.start()

    async def shutdown(self) -> None:
        if self._realtime is not None:
            await self._realtime.shutdown()
        if self._batch is not None:
            await self._batch.shutdown()

    def pool_for(self, pool_type: PoolType) -> _Pool | None:
        if pool_type is PoolType.REALTIME:
            return self._realtime
        return self._batch

    async def submit(self, pool_type: PoolType, item: WorkItem) -> bool:
        pool = self.pool_for(pool_type)
        if pool is None:
            return False
        return await pool.submit(item)

    def has_capacity(self, pool_type: PoolType) -> bool:
        pool = self.pool_for(pool_type)
        if pool is None:
            return False
        return pool.has_capacity()

    def stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": self._mode}
        if self._realtime is not None:
            result["realtime"] = self._realtime.stats()
        if self._batch is not None:
            result["batch"] = self._batch.stats()
        return result


# ── Module-level singleton access ────────────────────────────────────

_instance: PipelinePool | None = None


def init_pipeline_pool(
    config: PipelinePoolConfig,
    *,
    mode: str = "standalone",
    realtime_work_fn: Callable[[WorkItem], Any] | None = None,
    batch_work_fn: Callable[[WorkItem], Any] | None = None,
) -> PipelinePool:
    """Create and start the global pipeline pool (call once at startup).

    *mode* controls which pools are started:

    * ``standalone`` — both realtime and batch (default).
    * ``realtime`` — only the realtime pool.
    * ``batch`` — only the batch pool.
    * ``gateway`` — should not be called (gateway has no local pools).
    """
    global _instance
    pool = PipelinePool(
        config,
        mode=mode,
        realtime_work_fn=realtime_work_fn,
        batch_work_fn=batch_work_fn,
    )
    pool.start()
    _instance = pool
    logger.info(
        "Pipeline pool initialised (mode=%s, realtime=%dw/%dq, batch=%dw/%dq)",
        mode,
        config.realtime_workers,
        config.realtime_queue_size,
        config.batch_workers,
        config.batch_queue_size,
    )
    return pool


def get_pipeline_pool() -> PipelinePool | None:
    """Return the pipeline pool singleton, or ``None`` if not initialised.

    Optional usage pattern::

        if (pool := get_pipeline_pool()) is not None:
            if not await pool.submit(PoolType.BATCH, item):
                raise HTTPException(429, ...)
    """
    return _instance


async def shutdown_pipeline_pool() -> None:
    """Shut down the singleton (call during app shutdown)."""
    global _instance
    if _instance is not None:
        await _instance.shutdown()
        logger.info("Pipeline pool shut down")
        _instance = None
