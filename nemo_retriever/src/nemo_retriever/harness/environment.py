# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import metadata
import os
import platform
import socket
import subprocess
from typing import Any

from nemo_retriever.harness.artifacts import last_commit

_RECORDED_RUNTIME_ENV_KEYS = ("VLLM_USE_DEEP_GEMM",)


def _safe_package_version() -> str:
    for name in ("nemo-retriever", "nemo_retriever"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "unknown"


def _gpu_metadata() -> tuple[int | None, str | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None
    combined = f"{result.stdout}\n{result.stderr}"
    if "No devices were found" in combined:
        return 0, None
    if result.returncode != 0:
        return None, None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return 0, None
    return len(lines), lines[0]


def collect_environment() -> dict[str, Any]:
    try:
        ray_version = metadata.version("ray")
    except metadata.PackageNotFoundError:
        ray_version = "unknown"
    gpu_count, cuda_driver = _gpu_metadata()
    runtime_environment = {key: os.environ[key] for key in _RECORDED_RUNTIME_ENV_KEYS if key in os.environ}
    return {
        "git_sha": last_commit(),
        "package_version": _safe_package_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "gpu_count": gpu_count,
        "cuda_driver": cuda_driver,
        "ray_version": ray_version,
        "runtime_environment": runtime_environment,
    }
