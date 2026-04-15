# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared hooks for service tests (Ray workers, import layout)."""

from __future__ import annotations

import os


def pytest_configure(config) -> None:
    # Tests under this tree are imported as ``util....`` (see ``tests/service_tests/util``).
    # Ray workers need the same PYTHONPATH as the pytest driver before ``ray.init`` runs,
    # or cloudpickled callables from test modules fail to import on workers.
    root = os.path.dirname(__file__)
    prev = os.environ.get("PYTHONPATH", "")
    merged = f"{root}{os.pathsep}{prev}" if prev else root
    os.environ["PYTHONPATH"] = merged
