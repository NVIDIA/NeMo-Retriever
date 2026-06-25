# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for starting local Ray runtimes from Retriever processes."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager


_UV_ENV_VARS = ("UV", "UV_RUN_RECURSION_DEPTH")
_RAY_UV_RUNTIME_ENV_FLAG = "RAY_ENABLE_UV_RUN_RUNTIME_ENV"


@contextmanager
def without_uv_run_env() -> Iterator[None]:
    """Prevent Ray workers from recursively bootstrapping through ``uv run``."""

    previous = {name: os.environ.pop(name) for name in _UV_ENV_VARS if name in os.environ}
    previous_ray_flag = os.environ.get(_RAY_UV_RUNTIME_ENV_FLAG)
    os.environ[_RAY_UV_RUNTIME_ENV_FLAG] = "0"
    try:
        yield
    finally:
        if previous_ray_flag is None:
            os.environ.pop(_RAY_UV_RUNTIME_ENV_FLAG, None)
        else:
            os.environ[_RAY_UV_RUNTIME_ENV_FLAG] = previous_ray_flag
        os.environ.update(previous)


def disable_ray_uv_runtime_env_hook(ray: object) -> None:
    """Disable Ray's parent-process uv hook when Ray was imported earlier."""

    ray._private.ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
