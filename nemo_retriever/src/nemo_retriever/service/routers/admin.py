# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admin endpoints exposed by worker pods (realtime / batch / standalone).

These are NOT part of the public ingest contract — they exist so the
gateway's dashboard overview can fan out and ask each worker pod
"what's your live queue depth?" without going through Prometheus.
Available without auth so the cluster's internal Service DNS can reach
them; the routes are not advertised on the OpenAPI schema.

The same data is published as Prometheus metrics
(:func:`nemo_retriever_pool_queue_depth*`) — this endpoint is the
human-readable, JSON-shaped sibling for use cases that can't or
don't want to go through Prometheus.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin"], include_in_schema=False)


@router.get("/admin/pool_stats")
async def pool_stats(request: Request) -> dict[str, Any]:
    """Return live worker-pool state for the current pod.

    Response shape::

        {
          "mode": "realtime" | "batch" | "standalone" | "gateway",
          "pools": {
            "realtime": {
              "queue_depth": int,
              "queue_depth_ratio": float,
              "max_queue_size": int,
              "num_workers": int,
              "processed": int,
              "is_running": bool,
            },
            ...
          }
        }

    Gateway pods have no local pool and return ``{"mode": "gateway",
    "pools": {}}`` — the caller is expected to fan out to the worker
    Services themselves for live numbers.
    """
    from nemo_retriever.service.services.pipeline_pool import (
        PoolType,
        get_pipeline_pool,
    )

    pool = get_pipeline_pool()
    config = getattr(request.app.state, "config", None)
    mode = config.mode if config is not None else "unknown"

    pools: dict[str, dict[str, Any]] = {}
    if pool is not None:
        for pt in (PoolType.REALTIME, PoolType.BATCH):
            p = pool.pool_for(pt)
            if p is None:
                continue
            depth = p.queue_depth
            max_qs = max(1, p.max_queue_size)
            pools[pt.value] = {
                "queue_depth": depth,
                "queue_depth_ratio": round(depth / max_qs, 4),
                "max_queue_size": p.max_queue_size,
                "num_workers": p.num_workers,
                "processed": p.processed,
                "is_running": p.is_running,
            }

    return {"mode": mode, "pools": pools}
