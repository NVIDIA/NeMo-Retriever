# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from nemo_retriever.nim.nim import invoke_image_inference_batches
from nemo_retriever.graph.designer import designer_component
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


@designer_component(
    name="OCR",
    category="Detection & OCR",
    compute="gpu",
    description="Performs optical character recognition on document images",
)
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


@designer_component(
    name="OCR V2",
    category="Detection & OCR",
    compute="gpu",
    description="Performs multilingual optical character recognition using Nemotron OCR v2",
)
class OCRV2Actor(ArchetypeOperator):
    """Graph-facing OCR v2 archetype (multilingual, higher throughput)."""

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        kwargs = operator_kwargs or {}
        return bool(str(kwargs.get("ocr_invoke_url") or kwargs.get("invoke_url") or "").strip())

    @classmethod
    def cpu_variant_class(cls):
        from nemo_retriever.ocr.cpu_ocrv2 import OCRV2CPUActor

        return OCRV2CPUActor

    @classmethod
    def gpu_variant_class(cls):
        from nemo_retriever.ocr.gpu_ocrv2 import OCRV2Actor as OCRV2GPUActor

        return OCRV2GPUActor

    def __init__(self, **ocr_kwargs: Any) -> None:
        super().__init__(**ocr_kwargs)


def resolve_ocr_archetype(extract_params: Any) -> type:
    """Pick the OCR archetype class based on ExtractParams.ocr_version.

    Default is v2 (OCRV2Actor). Pass ocr_version="v1" on ExtractParams
    to fall back to the legacy English-only OCRActor.

    Args:
        extract_params: An object exposing an ``ocr_version`` attribute.
            When the attribute is missing, v2 is assumed.

    Returns:
        The OCR archetype class (``OCRActor`` for v1, ``OCRV2Actor`` for v2).
    """
    return OCRActor if getattr(extract_params, "ocr_version", "v2") == "v1" else OCRV2Actor


def __getattr__(name: str):
    if name == "OCRCPUActor":
        from nemo_retriever.ocr.cpu_ocr import OCRCPUActor

        return OCRCPUActor
    if name == "OCRGPUActor":
        from nemo_retriever.ocr.gpu_ocr import OCRActor as OCRGPUActor

        return OCRGPUActor
    if name == "OCRV2CPUActor":
        from nemo_retriever.ocr.cpu_ocrv2 import OCRV2CPUActor

        return OCRV2CPUActor
    if name == "OCRV2GPUActor":
        from nemo_retriever.ocr.gpu_ocrv2 import OCRV2Actor as OCRV2GPUActor

        return OCRV2GPUActor
    if name == "NemotronParseCPUActor":
        from nemo_retriever.ocr.cpu_parse import NemotronParseCPUActor

        return NemotronParseCPUActor
    if name == "NemotronParseGPUActor":
        from nemo_retriever.ocr.gpu_parse import NemotronParseActor as NemotronParseGPUActor

        return NemotronParseGPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
