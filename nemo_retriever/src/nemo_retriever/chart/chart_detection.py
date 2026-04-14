# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.chart.shared import _prediction_to_detections, graphic_elements_ocr_page_elements

__all__ = [
    "graphic_elements_ocr_page_elements",
    "_prediction_to_detections",
]


class GraphicElementsActor(ArchetypeOperator):
    """Graph-facing graphic-elements archetype."""

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        kwargs = operator_kwargs or {}
        return bool(
            str(kwargs.get("graphic_elements_invoke_url") or "").strip()
            or str(kwargs.get("ocr_invoke_url") or kwargs.get("invoke_url") or "").strip()
        )

    @classmethod
    def cpu_variant_class(cls):
        from nemo_retriever.chart.cpu_actor import GraphicElementsCPUActor

        return GraphicElementsCPUActor

    @classmethod
    def gpu_variant_class(cls):
        from nemo_retriever.chart.gpu_actor import GraphicElementsActor as GraphicElementsGPUActor

        return GraphicElementsGPUActor

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
