# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.cli.image.__main__ import app

try:
    from nemo_retriever.cli.image.render import (
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
