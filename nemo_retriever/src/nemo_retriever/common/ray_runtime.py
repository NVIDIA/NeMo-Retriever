# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for starting local Ray runtimes from Retriever processes."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from nemo_retriever.common.remote_auth import collect_remote_auth_runtime_env
from nemo_retriever.models.hf_cache import collect_hf_runtime_env


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


def build_local_ray_runtime_env() -> dict[str, Any]:
    """Build the runtime env that lets local Ray workers reuse this process environment."""

    venv = os.path.dirname(os.path.dirname(sys.executable))
    venv_bin = os.path.join(venv, "bin")
    pypath = os.pathsep.join(p for p in sys.path if p)
    ray_env_vars: dict[str, str] = {
        "VIRTUAL_ENV": venv,
        "PATH": venv_bin + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONPATH": pypath,
    }
    ray_env_vars.update(collect_hf_runtime_env())
    ray_env_vars.update(collect_remote_auth_runtime_env())
    if "HF_HUB_OFFLINE" in ray_env_vars:
        os.environ["HF_HUB_OFFLINE"] = ray_env_vars["HF_HUB_OFFLINE"]
    return {"env_vars": ray_env_vars, "py_executable": sys.executable}


def ensure_local_ray_runtime(ray_address: str | None = None, *, log_to_driver: bool | None = None) -> object:
    """Import Ray and initialize it with Retriever's local worker runtime env."""

    with without_uv_run_env():
        import ray

        disable_ray_uv_runtime_env_hook(ray)
        if ray_address or not ray.is_initialized():
            init_kwargs: dict[str, Any] = {
                "address": ray_address,
                "ignore_reinit_error": True,
                "runtime_env": build_local_ray_runtime_env(),
            }
            if log_to_driver is not None:
                init_kwargs["log_to_driver"] = log_to_driver
            ray.init(**init_kwargs)
        return ray
