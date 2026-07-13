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
import os
import random
import time
from enum import Enum
from typing import Any, Callable, Mapping

from pydantic import ConfigDict, Field

from nemo_retriever.service.config import PipelinePoolConfig
from nemo_retriever.common.schemas.base import RichModel
from nemo_retriever.service.services.prometheus import (
    POOL_CALLBACK_BACKPRESSURE_TOTAL,
    POOL_DEFERRED_CALLBACKS,
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
_CALLBACK_RETRY_DELAYS_S = (0.5, 1.0, 2.0, 4.0, 8.0)
_CALLBACK_DEFERRED_INITIAL_DELAY_S = 16.0
_CALLBACK_DEFERRED_MAX_DELAY_S = 300.0


class PoolType(str, Enum):
    REALTIME = "realtime"
    BATCH = "batch"


class _CallbackDeliveryOutcome(str, Enum):
    ACKNOWLEDGED = "acknowledged"
    RETRYABLE = "retryable"
    PERMANENT_FAILURE = "permanent_failure"


def _callback_status_is_retryable(status_code: int) -> bool:
    return status_code in (408, 425, 429) or 500 <= status_code < 600


def _safe_extract_trace_context(carrier: Mapping[str, str] | None, *, pool_name: str, item_id: str) -> Any | None:
    """Best-effort W3C trace context extraction for worker processing."""
    from nemo_retriever.service import tracing

    try:
        return tracing.extract_trace_context(carrier)
    except Exception as exc:
        logger.warning(
            "Pool '%s' trace context extraction failed for item %s; continuing without parent context: %s",
            pool_name,
            item_id,
            exc,
        )
        return None


class WorkItem(RichModel):
    """A unit of work submitted to a pool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    payload: Any = None
    filename: str | None = None
    callback: Callable[[Any], None] | None = None
    callback_url: str | None = None
    callback_headers: dict[str, str] = Field(default_factory=dict)
    # Owning job aggregate (J1+). Always set today since the only
    # admission path is /v1/ingest/job/{job_id}/document.
    job_id: str | None = None
    retain_results: bool = False
    # Validated per-request pipeline overrides (PipelineSpec serialised
    # to a dict). ``None`` means: run the legacy startup-baked pipeline.
    pipeline_spec: dict[str, Any] | None = None
    trace_context: dict[str, str] = Field(default_factory=dict)
    enqueued_at_monotonic_s: float | None = None


async def _fire_gateway_callback(
    callback_url: str,
    item_id: str,
    status: str,
    *,
    result_rows: int = 0,
    error: str | None = None,
    result_worker_ip: str | None = None,
    callback_headers: Mapping[str, str] | None = None,
    retry_after_cap_s: float = _CALLBACK_RETRY_DELAYS_S[-1],
) -> _CallbackDeliveryOutcome:
    """POST a lightweight completion notification to the gateway pod.

    ``result_data`` is never included. For retained results, the worker
    advertises its pod IP so the gateway can copy rows from the exact owner
    before acknowledging completion.
    """
    import httpx

    payload: dict[str, Any] = {
        "id": item_id,
        "status": status,
        "result_rows": result_rows,
    }
    if error:
        payload["error"] = error
    if result_worker_ip:
        payload["result_worker_ip"] = result_worker_ip

    try:
        async with httpx.AsyncClient(timeout=10.0, headers=dict(callback_headers or {})) as client:
            for attempt in range(len(_CALLBACK_RETRY_DELAYS_S) + 1):
                retry_after_s: float | None = None
                try:
                    resp = await client.post(callback_url, json=payload)
                    if resp.status_code == 200:
                        return _CallbackDeliveryOutcome.ACKNOWLEDGED
                    logger.warning(
                        "Gateway callback returned HTTP %d for item %s (attempt %d)",
                        resp.status_code,
                        item_id,
                        attempt + 1,
                    )
                    if not _callback_status_is_retryable(resp.status_code):
                        logger.error(
                            "Gateway callback permanently rejected item %s with HTTP %d",
                            item_id,
                            resp.status_code,
                        )
                        return _CallbackDeliveryOutcome.PERMANENT_FAILURE
                    retry_after = getattr(resp, "headers", {}).get("Retry-After")
                    if retry_after is not None:
                        try:
                            parsed = float(retry_after)
                        except (TypeError, ValueError):
                            parsed = -1.0
                        if parsed >= 0.0:
                            retry_after_s = min(parsed, retry_after_cap_s)
                except Exception as exc:
                    logger.warning(
                        "Failed to fire gateway callback for item %s (attempt %d): %s",
                        item_id,
                        attempt + 1,
                        exc,
                    )
                if attempt < len(_CALLBACK_RETRY_DELAYS_S):
                    delay_s = retry_after_s if retry_after_s is not None else _CALLBACK_RETRY_DELAYS_S[attempt]
                    await asyncio.sleep(delay_s)
    except Exception as exc:
        logger.warning("Unable to initialize gateway callback client for item %s: %s", item_id, exc)
    return _CallbackDeliveryOutcome.RETRYABLE


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
        self._handoff_tasks: dict[str, asyncio.Task[None]] = {}
        self._handoff_slots: asyncio.BoundedSemaphore | None = None
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
        self._handoff_slots = asyncio.BoundedSemaphore(self._num_workers)
        self._running = True
        self._workers = [asyncio.create_task(self._worker_loop(i)) for i in range(self._num_workers)]

        # Publish startup-constant metrics so prometheus-adapter can join
        # depth (Gauge) with capacity (Gauge) at query time.
        POOL_MAX_QUEUE_SIZE.labels(pool=self._name).set(self._max_queue_size)
        POOL_WORKERS.labels(pool=self._name).set(self._num_workers)
        POOL_QUEUE_DEPTH.labels(pool=self._name).set(0)
        POOL_QUEUE_DEPTH_RATIO.labels(pool=self._name).set(0.0)
        POOL_DEFERRED_CALLBACKS.labels(pool=self._name).set(0)

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

    async def _schedule_gateway_callback_retry(
        self,
        *,
        callback_url: str,
        item_id: str,
        status: str,
        result_rows: int = 0,
        error: str | None = None,
        result_worker_ip: str | None = None,
        callback_headers: Mapping[str, str] | None = None,
        retain_results: bool = False,
    ) -> None:
        """Continue callback delivery with bounded, backpressured concurrency."""
        if not self._running or item_id in self._handoff_tasks:
            return
        slots = self._handoff_slots
        if slots is None:
            return
        if slots.locked():
            POOL_CALLBACK_BACKPRESSURE_TOTAL.labels(pool=self._name).inc()
        await slots.acquire()
        if not self._running or item_id in self._handoff_tasks:
            slots.release()
            return
        task = asyncio.create_task(
            self._retry_gateway_callback_until_expired(
                callback_url=callback_url,
                item_id=item_id,
                status=status,
                result_rows=result_rows,
                error=error,
                result_worker_ip=result_worker_ip,
                callback_headers=callback_headers,
                retain_results=retain_results,
            )
        )
        self._handoff_tasks[item_id] = task
        POOL_DEFERRED_CALLBACKS.labels(pool=self._name).inc()

        def _remove_finished(finished: asyncio.Task[None]) -> None:
            if self._handoff_tasks.get(item_id) is finished:
                self._handoff_tasks.pop(item_id, None)
                POOL_DEFERRED_CALLBACKS.labels(pool=self._name).dec()
                slots.release()
            if not finished.cancelled():
                try:
                    finished.result()
                except Exception:
                    logger.exception("Deferred gateway callback task failed for item %s", item_id)

        task.add_done_callback(_remove_finished)

    async def _retry_gateway_callback_until_expired(
        self,
        *,
        callback_url: str,
        item_id: str,
        status: str,
        result_rows: int,
        error: str | None,
        result_worker_ip: str | None,
        callback_headers: Mapping[str, str] | None,
        retain_results: bool,
    ) -> None:
        from nemo_retriever.service.services.worker_result_store import (
            discard_local_result_data,
            result_retention_seconds,
        )

        deadline = time.monotonic() + result_retention_seconds()
        delay_s = _CALLBACK_DEFERRED_INITIAL_DELAY_S
        while self._running:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            await asyncio.sleep(min(remaining_s, delay_s * random.uniform(0.8, 1.2)))
            delivery_outcome = await _fire_gateway_callback(
                callback_url,
                item_id,
                status,
                result_rows=result_rows,
                error=error,
                result_worker_ip=result_worker_ip,
                callback_headers=callback_headers,
                retry_after_cap_s=_CALLBACK_DEFERRED_MAX_DELAY_S,
            )
            if delivery_outcome == _CallbackDeliveryOutcome.ACKNOWLEDGED:
                if retain_results:
                    discard_local_result_data(item_id)
                logger.info("Deferred gateway callback succeeded for item %s", item_id)
                return
            if delivery_outcome == _CallbackDeliveryOutcome.PERMANENT_FAILURE:
                logger.error(
                    "Deferred gateway callback permanently failed for item %s; result remains unacknowledged",
                    item_id,
                )
                return
            delay_s = min(delay_s * 2.0, _CALLBACK_DEFERRED_MAX_DELAY_S)

        if self._running:
            logger.error(
                "Gateway callback delivery expired for item %s; result remains unacknowledged",
                item_id,
            )

    async def _worker_loop(self, worker_id: int) -> None:
        """Consume items until a ``None`` sentinel is received.

        When an item has a ``callback_url`` (set by the gateway), the
        worker POSTs completion data back to the gateway instead of
        updating a local job tracker.  In standalone mode (no callback),
        the local tracker is updated directly.
        """
        from nemo_retriever.service import tracing
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
            ctx = _safe_extract_trace_context(item.trace_context, pool_name=self._name, item_id=item.id)
            with tracing.start_span(
                f"pool.{self._name}.process",
                context=ctx,
                attributes={
                    "pool": self._name,
                    "document.id": item.id,
                    "job.id": item.job_id or "",
                },
            ) as span:
                if item.enqueued_at_monotonic_s is not None and hasattr(span, "set_attribute"):
                    span.set_attribute(
                        "queue.wait_ms",
                        (time.monotonic() - item.enqueued_at_monotonic_s) * 1000.0,
                    )
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

                    retain_results = item.retain_results
                    if not retain_results and item.job_id:
                        tracker_lookup = get_job_tracker()
                        if tracker_lookup is not None:
                            retain_results = tracker_lookup.should_retain_results(item.job_id)

                    if item.callback_url:
                        if retain_results:
                            from nemo_retriever.service.services.worker_result_store import (
                                store_result_data,
                            )

                            store_result_data(item.id, result_data)
                        callback_outcome = await _fire_gateway_callback(
                            item.callback_url,
                            item.id,
                            "completed",
                            result_rows=result_rows,
                            result_worker_ip=(os.environ.get("POD_IP") if retain_results and result_rows > 0 else None),
                            callback_headers=item.callback_headers,
                        )
                        if callback_outcome == _CallbackDeliveryOutcome.ACKNOWLEDGED:
                            if retain_results:
                                from nemo_retriever.service.services.worker_result_store import (
                                    discard_local_result_data,
                                )

                                discard_local_result_data(item.id)
                        elif callback_outcome == _CallbackDeliveryOutcome.RETRYABLE:
                            await self._schedule_gateway_callback_retry(
                                callback_url=item.callback_url,
                                item_id=item.id,
                                status="completed",
                                result_rows=result_rows,
                                result_worker_ip=(
                                    os.environ.get("POD_IP") if retain_results and result_rows > 0 else None
                                ),
                                callback_headers=item.callback_headers,
                                retain_results=retain_results,
                            )
                    elif tracker is not None:
                        tracker.mark_completed(
                            item.id,
                            result_rows=result_rows,
                            result_data=result_data if retain_results else None,
                        )
                    self._processed += 1
                except Exception as exc:
                    outcome = "failed"
                    if item.callback_url:
                        error = f"{type(exc).__name__}: {exc}"
                        callback_outcome = await _fire_gateway_callback(
                            item.callback_url,
                            item.id,
                            "failed",
                            error=error,
                            callback_headers=item.callback_headers,
                        )
                        if callback_outcome == _CallbackDeliveryOutcome.RETRYABLE:
                            await self._schedule_gateway_callback_retry(
                                callback_url=item.callback_url,
                                item_id=item.id,
                                status="failed",
                                error=error,
                                callback_headers=item.callback_headers,
                            )
                    else:
                        tracker = get_job_tracker()
                        if tracker is not None:
                            tracker.mark_failed(item.id, f"{type(exc).__name__}: {exc}")
                    logger.exception(
                        "Pool '%s' worker %d failed on item %s",
                        self._name,
                        worker_id,
                        item.id,
                    )
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
            if item.enqueued_at_monotonic_s is None:
                item.enqueued_at_monotonic_s = time.monotonic()
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

        for task in self._handoff_tasks.values():
            task.cancel()
        if self._handoff_tasks:
            await asyncio.gather(*self._handoff_tasks.values(), return_exceptions=True)
        self._handoff_tasks.clear()

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
        self._handoff_slots = None
        # Reset depth gauges so a terminating pod doesn't keep its last
        # high-water mark live on the scraper. We deliberately leave the
        # *configuration* gauges (max_queue_size, workers) untouched —
        # those are pod identity, not runtime state.
        POOL_QUEUE_DEPTH.labels(pool=self._name).set(0)
        POOL_QUEUE_DEPTH_RATIO.labels(pool=self._name).set(0.0)
        POOL_DEFERRED_CALLBACKS.labels(pool=self._name).set(0)
        logger.info("Pool '%s' shut down (processed=%d)", self._name, self._processed)

    def stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "num_workers": self._num_workers,
            "max_queue_size": self._max_queue_size,
            "queue_depth": self.queue_depth,
            "processed": self._processed,
            "deferred_callbacks": len(self._handoff_tasks),
            "max_deferred_callbacks": self._num_workers,
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
