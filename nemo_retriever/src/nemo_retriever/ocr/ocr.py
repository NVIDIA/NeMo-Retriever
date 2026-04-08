# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from nemo_retriever.nim.nim import invoke_image_inference_batches
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.ocr import shared as _shared
from nemo_retriever.ocr.shared import (
    _blocks_to_pseudo_markdown,
    _blocks_to_text,
    _crop_all_from_page,
    _crop_b64_image_by_norm_bbox,
    _extract_remote_ocr_item,
    _np_rgb_to_b64_png,
    _parse_ocr_result,
)

__all__ = [
    "ocr_page_elements",
    "nemotron_parse_page_elements",
    "invoke_image_inference_batches",
    "_blocks_to_pseudo_markdown",
    "_blocks_to_text",
    "_crop_all_from_page",
    "_crop_b64_image_by_norm_bbox",
    "_extract_remote_ocr_item",
    "_np_rgb_to_b64_png",
    "_parse_ocr_result",
]


@contextmanager
def _patched_shared_runtime() -> Any:
    original_np_rgb_to_b64_png = _shared._np_rgb_to_b64_png
    original_invoke = _shared.invoke_image_inference_batches
    _shared._np_rgb_to_b64_png = _np_rgb_to_b64_png
    _shared.invoke_image_inference_batches = invoke_image_inference_batches
    try:
        yield
    finally:
        _shared._np_rgb_to_b64_png = original_np_rgb_to_b64_png
        _shared.invoke_image_inference_batches = original_invoke


def ocr_page_elements(*args: Any, **kwargs: Any):
    with _patched_shared_runtime():
        return _shared.ocr_page_elements(*args, **kwargs)


def nemotron_parse_page_elements(*args: Any, **kwargs: Any):
    with _patched_shared_runtime():
        return _shared.nemotron_parse_page_elements(*args, **kwargs)


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
