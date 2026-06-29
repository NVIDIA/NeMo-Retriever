# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for portal runners that still import ``nemo_retriever.harness.run``."""

from __future__ import annotations

import socket
import sys
from importlib import metadata
from typing import Any

from nemo_retriever.harness.environment import _gpu_metadata
from nemo_retriever.harness.portal_job import run_portal_job_entry as _run_entry

__all__ = ["_run_entry", "_collect_run_metadata"]


def _collect_run_metadata() -> dict[str, Any]:
    """Collect host/GPU metadata for runner registration with the portal."""
    try:
        host = socket.gethostname().strip() or "unknown"
    except OSError:
        host = "unknown"

    version_info = getattr(sys, "version_info", None)
    if version_info is None:
        python_version = "unknown"
    else:
        python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    try:
        ray_version = metadata.version("ray")
    except metadata.PackageNotFoundError:
        ray_version = "unknown"

    gpu_count, cuda_driver = _gpu_metadata()
    return {
        "host": host,
        "gpu_count": gpu_count,
        "cuda_driver": cuda_driver,
        "ray_version": ray_version,
        "python_version": python_version,
    }
