# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixin flag for operators that require GPU resources."""

import ctypes
import logging
import os


logger = logging.getLogger(__name__)

_CUDA_BLOCKING_SYNC_ENV = "NRL_CUDA_BLOCKING_SYNC"
_CU_CTX_SCHED_BLOCKING_SYNC = 0x4
_cuda_blocking_sync_configured = False


def _check(result: int, name: str) -> None:
    if result != 0:
        raise RuntimeError(f"{name} returned CUresult={result}")


def _configure_cuda_blocking_sync() -> None:
    """Make CUDA waits in this process sleep instead of spinning a CPU core."""
    global _cuda_blocking_sync_configured
    if os.environ.get(_CUDA_BLOCKING_SYNC_ENV) != "1" or _cuda_blocking_sync_configured:
        return
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        _check(cuda.cuInit(0), "cuInit")
        count = ctypes.c_int()
        _check(cuda.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        for ordinal in range(count.value):
            device = ctypes.c_int()
            _check(cuda.cuDeviceGet(ctypes.byref(device), ordinal), "cuDeviceGet")
            _check(
                cuda.cuDevicePrimaryCtxSetFlags_v2(device, _CU_CTX_SCHED_BLOCKING_SYNC),
                "cuDevicePrimaryCtxSetFlags_v2",
            )
        _cuda_blocking_sync_configured = True
        logger.info("%s=1: blocking sync set on %d CUDA device(s)", _CUDA_BLOCKING_SYNC_ENV, count.value)
    except (OSError, AttributeError, RuntimeError) as exc:
        logger.warning("%s=1 ignored: %s", _CUDA_BLOCKING_SYNC_ENV, exc)


class GPUOperator:
    """Mixin flag indicating an operator requires GPU resources.

    Operators that load torch models or perform CUDA-based inference
    should inherit from both :class:`AbstractOperator` and this class::

        class MyGPUActor(AbstractOperator, GPUOperator):
            ...

    Executors can inspect ``isinstance(op, GPUOperator)`` to allocate
    GPU resources or route work to GPU-capable workers.

    """
