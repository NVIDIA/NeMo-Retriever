# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ensure Ray workers can import ``api_tests`` (package lives under ``api/``)."""

from __future__ import annotations

import os


def pytest_configure(config) -> None:
    api_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{api_root}{os.pathsep}{prev}" if prev else api_root
