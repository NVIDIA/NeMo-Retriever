# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .executor import run_mode_ingest
from .factory import RunMode, create_runmode_ingestor

__all__ = [
    "RunMode",
    "create_runmode_ingestor",
    "run_mode_ingest",
    "run_batch",
    "run_fused",
    "run_inprocess",
    "run_online",
]


def __getattr__(name: str):
    if name == "run_batch":
        from .run_batch import run_batch

        return run_batch
    if name == "run_fused":
        from .run_fused import run_fused

        return run_fused
    if name == "run_inprocess":
        from .run_inprocess import run_inprocess

        return run_inprocess
    if name == "run_online":
        from .run_online import run_online

        return run_online
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
