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
import time
from enum import Enum
from typing import Any, Callable

from pydantic import ConfigDict

from nemo_retriever.service.config import PipelinePoolConfig
from nemo_retriever.service.models.base import RichModel
from nemo_retriever.service.services.prometheus import (
    POOL_MAX_QUEUE_SIZE,
    POOL_PROCESSED_TOTAL,
    POOL_PROCESSING_DURATION,
    POOL_QUEUE_DEPTH,
    POOL_QUEUE_DEPTH_RATIO,
    POOL_WORKERS,
)

logger = logging.getLogger(__name__)

# Cadence for the periodic queue-depth reporter. One second is more than
# enough resolution for an HPA that polls every 15s; faster than that and
# we just generate redundant samples for prometheus_client to overwrite.
_QUEUE_DEPTH_REPORT_INTERVAL_S = 1.0


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
    # Owning job aggregate (J1+). Always set today since the only
    # admission path is /v1/ingest/job/{job_id}/document.
    job_id: str | None = None
    # Validated per-request pipeline overrides (PipelineSpec serialised
    # to a dict). ``None`` means: run the legacy startup-baked pipeline.
    pipeline_spec: dict[str, Any] | None = None


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
        self._reporter_task: asyncio.Task[None] | None = None
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

        # Publish startup-constant metrics so prometheus-adapter can join
        # depth (Gauge) with capacity (Gauge) at query time.
        POOL_MAX_QUEUE_SIZE.labels(pool=self._name).set(self._max_queue_size)
        POOL_WORKERS.labels(pool=self._name).set(self._num_workers)
        POOL_QUEUE_DEPTH.labels(pool=self._name).set(0)
        POOL_QUEUE_DEPTH_RATIO.labels(pool=self._name).set(0.0)

        # Periodic gauge reporter — keeps the queue-depth series live so
        # HPA decisions don't lag behind reality between submissions. We
        # publish here (not on every submit/get) so the metric reflects
        # the *steady-state* fill rather than a noisy edge-triggered one.
        #
        # ``start()`` is called from inside the FastAPI lifespan (an
        # async context). Tests or scripts that instantiate the pool
        # without a running loop still get the constant gauges above,
        # they just won't get the periodic depth updates — that's an
        # acceptable degradation, and is preferable to crashing the
        # pool startup when no loop is available.
        try:
            self._reporter_task = asyncio.create_task(self._report_metrics_loop())
        except RuntimeError:
            self._reporter_task = None
            logger.debug(
                "Pool '%s' started outside a running event loop; " "skipping periodic queue-depth reporter",
                self._name,
            )

        logger.info(
            "Pool '%s' started: workers=%d queue_size=%d work_fn=%s",
            self._name,
            self._num_workers,
            self._max_queue_size,
            self._work_fn.__name__ if self._work_fn else "noop",
        )

    async def _report_metrics_loop(self) -> None:
        """Publish queue-depth gauges at a steady cadence.

        Runs until :meth:`shutdown` cancels it. Exceptions are logged
        and swallowed so a transient error in prometheus_client (e.g.
        a re-registration race in tests) never tears down the pool.
        """
        depth_g = POOL_QUEUE_DEPTH.labels(pool=self._name)
        ratio_g = POOL_QUEUE_DEPTH_RATIO.labels(pool=self._name)
        max_qs = max(1, self._max_queue_size)
        try:
            while self._running:
                try:
                    depth = self.queue_depth
                    depth_g.set(depth)
                    ratio_g.set(depth / max_qs)
                except Exception:
                    logger.exception(
                        "Pool '%s' metrics reporter raised; continuing",
                        self._name,
                    )
                await asyncio.sleep(_QUEUE_DEPTH_REPORT_INTERVAL_S)
        except asyncio.CancelledError:
            pass

    async def _worker_loop(self, worker_id: int) -> None:
        """Consume items until a ``None`` sentinel is received.

        When an item has a ``callback_url`` (set by the gateway), the
        worker POSTs completion data back to the gateway instead of
        updating a local job tracker.  In standalone mode (no callback),
        the local tracker is updated directly.
        """
        from nemo_retriever.service.services.job_tracker import get_job_tracker

        assert self._queue is not None
        duration_h = POOL_PROCESSING_DURATION.labels(pool=self._name)
        processed_ok = POOL_PROCESSED_TOTAL.labels(pool=self._name, outcome="completed")
        processed_err = POOL_PROCESSED_TOTAL.labels(pool=self._name, outcome="failed")
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            # Per-item timer covers the *useful* work — tracker bookkeeping
            # is excluded so the histogram reflects pipeline cost only.
            t0 = time.monotonic()
            outcome = "completed"
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
                outcome = "failed"
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
                # Always observe; cheaper to keep latency series complete
                # than to gate on outcome. Bucketed histogram, so even
                # very-failed-fast items show up in the low buckets.
                duration_h.observe(time.monotonic() - t0)
                if outcome == "completed":
                    processed_ok.inc()
                else:
                    processed_err.inc()
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

        # Stop the metrics reporter first so it doesn't observe queue==0
        # in the middle of the worker-cancellation race below.
        if self._reporter_task is not None:
            self._reporter_task.cancel()
            try:
                await self._reporter_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reporter_task = None

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
        # Reset depth gauges so a terminating pod doesn't keep its last
        # high-water mark live on the scraper. We deliberately leave the
        # *configuration* gauges (max_queue_size, workers) untouched —
        # those are pod identity, not runtime state.
        POOL_QUEUE_DEPTH.labels(pool=self._name).set(0)
        POOL_QUEUE_DEPTH_RATIO.labels(pool=self._name).set(0.0)
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
