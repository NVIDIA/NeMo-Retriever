# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test endpoint for the retriever service.

Provides a lightweight, no-dependency health-check route that validates
the Python runtime and the current service mode are reachable at all.
"""

from __future__ import annotations

import logging
import platform
import sys

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["test"], include_in_schema=True)


@router.get("/test", summary="Health-check that validates the Python runtime")
async def test(request: Request) -> dict:
    """Return a JSON blob describing the current process environment.

    Response shape::

        {
          "status": "ok",
          "mode": "gateway" | "realtime" | "batch" | "standalone",
          "python": "3.12.1+linux",
        }

    Intended for cluster probes, load-balancer heart-beats, and manual
    smoke-tests -- it has no external dependencies (no DB, no pipeline
    pool, no media binaries).
    """
    config = getattr(request.app.state, "config", None)
    mode = config.mode if config is not None else "unknown"
    runtime = f"{sys.version.split()[0]}; {'/'.join(platform.system().split())}"
    return {"status": "ok", "mode": mode, "python": runtime}
