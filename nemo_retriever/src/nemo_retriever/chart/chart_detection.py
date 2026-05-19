# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.chart.shared import _prediction_to_detections, graphic_elements_ocr_page_elements
from nemo_retriever.graph.designer import designer_component

__all__ = [
    "graphic_elements_ocr_page_elements",
    "_prediction_to_detections",
]

logger = logging.getLogger(__name__)


@designer_component(
    name="Graphic Elements Detection (CPU)",
    category="Detection & OCR",
    compute="cpu",
    description="Detects charts and graphic elements using CPU",
)
class GraphicElementsActor(ArchetypeOperator):
    """Graph-facing graphic-elements archetype."""

    _LOCAL_OCR_SELECTOR_KEYS = frozenset({"ocr_version", "ocr_lang"})

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        kwargs = operator_kwargs or {}
        graphic_url = str(kwargs.get("graphic_elements_invoke_url") or "").strip()
        ocr_url = str(kwargs.get("ocr_invoke_url") or kwargs.get("invoke_url") or "").strip()
        if bool(graphic_url) != bool(ocr_url):
            missing = "ocr_invoke_url" if graphic_url else "graphic_elements_invoke_url"
            configured = "graphic_elements_invoke_url" if graphic_url else "ocr_invoke_url"
            logger.warning(
                "GraphicElementsActor received %s without %s; GPU-capable runs will use the local model "
                "for the missing graphic/OCR stage. Configure both URLs to force the remote CPU variant.",
                configured,
                missing,
            )
        return bool(graphic_url and ocr_url)

    @classmethod
    def cpu_variant_class(cls):
        from nemo_retriever.chart.cpu_actor import GraphicElementsCPUActor

        return GraphicElementsCPUActor

    @classmethod
    def gpu_variant_class(cls):
        from nemo_retriever.chart.gpu_actor import GraphicElementsActor as GraphicElementsGPUActor

        return GraphicElementsGPUActor

    @classmethod
    def variant_operator_kwargs(cls, operator_class, operator_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs = super().variant_operator_kwargs(operator_class, operator_kwargs)
        if operator_class is cls.cpu_variant_class():
            for key in cls._LOCAL_OCR_SELECTOR_KEYS:
                kwargs.pop(key, None)
        return kwargs

    def __init__(self, **detect_kwargs: Any) -> None:
        super().__init__(**detect_kwargs)


def __getattr__(name: str):
    if name == "GraphicElementsCPUActor":
        from nemo_retriever.chart.cpu_actor import GraphicElementsCPUActor

        return GraphicElementsCPUActor
    if name == "GraphicElementsGPUActor":
        from nemo_retriever.chart.gpu_actor import GraphicElementsActor as GraphicElementsGPUActor

        return GraphicElementsGPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
