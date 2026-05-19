# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the H1 pool-metrics wiring (queue depth + processing duration).

These tests pin three contracts that downstream HPA / prometheus-adapter
configurations depend on:

1. Startup publishes the constant ``pool_max_queue_size`` and
   ``pool_workers`` gauges with the correct ``pool`` label.
2. The periodic reporter keeps ``pool_queue_depth`` /
   ``pool_queue_depth_ratio`` close to the live queue size.
3. Each work item bumps ``pool_processing_duration_seconds`` (a
   histogram) and the per-outcome ``pool_processed_total`` counter.

We exercise ``_Pool`` directly rather than the full FastAPI app because
this layer is where the metric writes live and we want a fast, focused
regression net that doesn't depend on the lifespan/tracker plumbing.

The project doesn't depend on ``pytest-asyncio``, so each async test is
driven through a dedicated ``asyncio.Runner`` rather than the marker.
"""

from __future__ import annotations

import asyncio
import time

from prometheus_client import REGISTRY

from nemo_retriever.service.services import pipeline_pool as pp
from nemo_retriever.service.services.pipeline_pool import _Pool, WorkItem


def _sample_value(name: str, labels: dict[str, str]) -> float | None:
    """Read a single sample from the global Prometheus registry."""
    return REGISTRY.get_sample_value(name, labels)


def _run(coro):
    """Drive an async test body to completion on a fresh event loop."""
    return asyncio.new_event_loop().run_until_complete(coro)


def test_startup_publishes_capacity_gauges():
    """``pool_max_queue_size`` + ``pool_workers`` are set at start()."""

    async def body():
        pool = _Pool(name="rt-cap", num_workers=3, max_queue_size=17, work_fn=None)
        pool.start()
        try:
            assert _sample_value("nemo_retriever_pool_max_queue_size", {"pool": "rt-cap"}) == 17.0
            assert _sample_value("nemo_retriever_pool_workers", {"pool": "rt-cap"}) == 3.0
            # Depth starts at zero — explicit set on startup so even an
            # idle pool shows a fresh sample to scrapers.
            assert _sample_value("nemo_retriever_pool_queue_depth", {"pool": "rt-cap"}) == 0.0
            assert _sample_value("nemo_retriever_pool_queue_depth_ratio", {"pool": "rt-cap"}) == 0.0
        finally:
            await pool.shutdown()

    _run(body())


def test_queue_depth_ratio_reflects_live_size(monkeypatch):
    """Filling the queue moves the ratio gauge proportionally."""
    # Tighten the report interval so we don't have to sleep a full
    # second per assertion.
    monkeypatch.setattr(pp, "_QUEUE_DEPTH_REPORT_INTERVAL_S", 0.02)

    async def body():
        blocker = asyncio.Event()

        async def slow_work(_item):
            # Hold each worker in-flight so submitted items accumulate
            # in the queue rather than draining instantly.
            await blocker.wait()
            return 0

        pool = _Pool(
            name="rt-depth",
            num_workers=1,
            max_queue_size=4,
            work_fn=slow_work,
        )
        pool.start()
        try:
            # First item gets picked up by the only worker; subsequent
            # submits accumulate in the queue.
            for i in range(4):
                await pool.submit(WorkItem(id=f"depth-{i}"))

            # Give the reporter at least one tick to publish.
            await asyncio.sleep(0.1)

            depth = _sample_value("nemo_retriever_pool_queue_depth", {"pool": "rt-depth"})
            ratio = _sample_value("nemo_retriever_pool_queue_depth_ratio", {"pool": "rt-depth"})
            # 1 of 4 submits was picked up by the worker → 3 remain queued.
            # Ratio is depth / max_queue_size = 3/4 = 0.75. We allow a
            # one-slot tolerance for the worker-pickup race.
            assert depth is not None and 2.0 <= depth <= 3.0
            assert ratio is not None and 0.5 <= ratio <= 0.75
        finally:
            blocker.set()
            await pool.shutdown()
            # On shutdown the depth gauges are explicitly zeroed so a
            # terminating pod doesn't keep a stale high-water mark live.
            assert _sample_value("nemo_retriever_pool_queue_depth", {"pool": "rt-depth"}) == 0.0
            assert _sample_value("nemo_retriever_pool_queue_depth_ratio", {"pool": "rt-depth"}) == 0.0

    _run(body())


def test_processing_duration_and_outcome_counters():
    """Histogram count + completed/failed counters track real work."""

    async def body():
        done = asyncio.Event()
        seen = 0

        async def work(_item):
            nonlocal seen
            seen += 1
            # A tiny but non-zero duration so the 0.01 histogram bucket
            # gets a hit (it doesn't matter which bucket — we only
            # assert on the _count and _sum aggregates).
            time.sleep(0.001)
            if seen == 3:
                done.set()
            return 1

        pool = _Pool(name="rt-obs", num_workers=2, max_queue_size=16, work_fn=work)
        pool.start()
        try:
            for i in range(3):
                await pool.submit(WorkItem(id=f"obs-{i}"))
            await asyncio.wait_for(done.wait(), timeout=2.0)
            # Let the workers finish their task_done() bookkeeping
            # before we read counters.
            await asyncio.sleep(0.05)

            count = _sample_value(
                "nemo_retriever_pool_processing_duration_seconds_count",
                {"pool": "rt-obs"},
            )
            ok = _sample_value(
                "nemo_retriever_pool_processed_total",
                {"pool": "rt-obs", "outcome": "completed"},
            )
            # All three items completed, no failures.
            assert count == 3.0
            assert ok == 3.0
            assert _sample_value(
                "nemo_retriever_pool_processed_total",
                {"pool": "rt-obs", "outcome": "failed"},
            ) in (None, 0.0)
        finally:
            await pool.shutdown()

    _run(body())


def test_failed_work_increments_failed_outcome_counter():
    """Exceptions inside ``work_fn`` route to ``outcome="failed"``."""

    async def body():
        async def boom(_item):
            raise RuntimeError("oops")

        pool = _Pool(name="rt-fail", num_workers=1, max_queue_size=4, work_fn=boom)
        pool.start()
        try:
            for i in range(2):
                await pool.submit(WorkItem(id=f"fail-{i}"))
            # Two failures need a brief moment to land.
            for _ in range(50):
                failed = _sample_value(
                    "nemo_retriever_pool_processed_total",
                    {"pool": "rt-fail", "outcome": "failed"},
                )
                if failed == 2.0:
                    break
                await asyncio.sleep(0.02)
            assert (
                _sample_value(
                    "nemo_retriever_pool_processed_total",
                    {"pool": "rt-fail", "outcome": "failed"},
                )
                == 2.0
            )
            # The histogram count must still rise on failed items —
            # latency observation runs in the ``finally`` block,
            # regardless of outcome.
            assert (
                _sample_value(
                    "nemo_retriever_pool_processing_duration_seconds_count",
                    {"pool": "rt-fail"},
                )
                == 2.0
            )
        finally:
            await pool.shutdown()

    _run(body())
