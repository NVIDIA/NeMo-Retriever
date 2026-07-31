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

_RECORDED_RUNTIME_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "NEMO_RETRIEVER_HF_CACHE_DIR",
    "TRANSFORMERS_CACHE",
    "TRANSFORMERS_OFFLINE",
    "VLLM_DEEP_GEMM_WARMUP",
    "VLLM_MOE_USE_DEEP_GEMM",
    "VLLM_USE_DEEP_GEMM",
)


def _workload_gpu_count(physical_gpu_count: int | None) -> int | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return physical_gpu_count

    normalized = visible_devices.strip().lower()
    if normalized in {"", "-1", "none", "void"}:
        return 0
    if normalized == "all":
        return physical_gpu_count
    return len([device for device in visible_devices.split(",") if device.strip()])


def _safe_package_version() -> str:
    for name in ("nemo-retriever", "nemo_retriever"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "unknown"


def _gpu_metadata() -> tuple[int | None, str | None, str | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None, None
    combined = f"{result.stdout}\n{result.stderr}"
    if "No devices were found" in combined:
        return 0, None, None
    if result.returncode != 0:
        return None, None, None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return 0, None, None

    gpu_names: list[str] = []
    driver_versions: list[str] = []
    for line in lines:
        name, separator, driver_version = line.rpartition(",")
        if not separator:
            return len(lines), None, None
        gpu_names.append(name.strip())
        driver_versions.append(driver_version.strip())

    unique_gpu_names = list(dict.fromkeys(gpu_names))
    unique_driver_versions = list(dict.fromkeys(driver_versions))
    gpu_sku = "; ".join(unique_gpu_names) if unique_gpu_names else None
    cuda_driver = "; ".join(unique_driver_versions) if unique_driver_versions else None
    return len(lines), gpu_sku, cuda_driver


def collect_environment() -> dict[str, Any]:
    try:
        ray_version = metadata.version("ray")
    except metadata.PackageNotFoundError:
        ray_version = "unknown"
    gpu_count, gpu_sku, cuda_driver = _gpu_metadata()
    runtime_environment = {key: os.environ[key] for key in _RECORDED_RUNTIME_ENV_KEYS if key in os.environ}
    return {
        "git_sha": last_commit(),
        "package_version": _safe_package_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "gpu_count": gpu_count,
        "workload_gpu_count": _workload_gpu_count(gpu_count),
        "gpu_sku": gpu_sku,
        "cuda_driver": cuda_driver,
        "ray_version": ray_version,
        "runtime_environment": runtime_environment,
    }
