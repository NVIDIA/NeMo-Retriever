# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.operator_archetype import ArchetypeOperator
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


class OCRActor(ArchetypeOperator):
    """Graph-facing OCR archetype."""

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        kwargs = operator_kwargs or {}
        return bool(str(kwargs.get("ocr_invoke_url") or kwargs.get("invoke_url") or "").strip())

    @classmethod
    def cpu_variant_class(cls):
        from nemo_retriever.ocr.cpu_ocr import OCRCPUActor

        return OCRCPUActor

    @classmethod
    def gpu_variant_class(cls):
        from nemo_retriever.ocr.gpu_ocr import OCRActor as OCRGPUActor

        return OCRGPUActor

    def __init__(self, **ocr_kwargs: Any) -> None:
        super().__init__(**ocr_kwargs)


def __getattr__(name: str):
    if name == "OCRCPUActor":
        from nemo_retriever.ocr.cpu_ocr import OCRCPUActor

        return OCRCPUActor
    if name == "OCRGPUActor":
        from nemo_retriever.ocr.gpu_ocr import OCRActor as OCRGPUActor

        return OCRGPUActor
    if name == "NemotronParseCPUActor":
        from nemo_retriever.ocr.cpu_parse import NemotronParseCPUActor

        return NemotronParseCPUActor
    if name == "NemotronParseGPUActor":
        from nemo_retriever.ocr.gpu_parse import NemotronParseActor as NemotronParseGPUActor

        return NemotronParseGPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
