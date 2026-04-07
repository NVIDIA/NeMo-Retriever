# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.ocr.shared import (
    _blocks_to_pseudo_markdown,
    _blocks_to_text,
    _crop_all_from_page,
    _crop_b64_image_by_norm_bbox,
    _extract_remote_ocr_item,
    _np_rgb_to_b64_png,
    _parse_ocr_result,
    nemotron_parse_page_elements,
    ocr_page_elements,
)

__all__ = [
    "ocr_page_elements",
    "nemotron_parse_page_elements",
    "_blocks_to_pseudo_markdown",
    "_blocks_to_text",
    "_crop_all_from_page",
    "_crop_b64_image_by_norm_bbox",
    "_extract_remote_ocr_item",
    "_np_rgb_to_b64_png",
    "_parse_ocr_result",
]


def __getattr__(name: str):
    if name == "OCRActor":
        from nemo_retriever.ocr.gpu_ocr import OCRActor

        return OCRActor
    if name == "OCRCPUActor":
        from nemo_retriever.ocr.cpu_ocr import OCRCPUActor

        return OCRCPUActor
    if name == "NemotronParseActor":
        from nemo_retriever.ocr.gpu_parse import NemotronParseActor

        return NemotronParseActor
    if name == "NemotronParseCPUActor":
        from nemo_retriever.ocr.cpu_parse import NemotronParseCPUActor

        return NemotronParseCPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
