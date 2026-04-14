# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Page element detection stage (Nemotron Page Elements v3).

This package provides:
- `detect_page_elements_v3`: pure-Python batch function (pandas.DataFrame in/out)
- `PageElementDetectionActor`: Ray-friendly callable that initializes the model once
"""

from .gpu_actor import PageElementDetectionActor
from .shared import detect_page_elements_v3

__all__ = [
    "detect_page_elements_v3",
    "PageElementDetectionActor",
]
