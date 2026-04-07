# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .__main__ import app

try:
    from .render import (
        render_page_element_detections_for_dir,
        render_page_element_detections_for_image,
    )

    __all__ = [
        "app",
        "render_page_element_detections_for_image",
        "render_page_element_detections_for_dir",
    ]
except ModuleNotFoundError:
    __all__ = ["app"]
