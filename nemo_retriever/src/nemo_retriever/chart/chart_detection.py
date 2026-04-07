# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.chart.shared import _prediction_to_detections, graphic_elements_ocr_page_elements

__all__ = [
    "graphic_elements_ocr_page_elements",
    "_prediction_to_detections",
]


def __getattr__(name: str):
    if name == "GraphicElementsActor":
        from nemo_retriever.chart.gpu_actor import GraphicElementsActor

        return GraphicElementsActor
    if name == "GraphicElementsCPUActor":
        from nemo_retriever.chart.cpu_actor import GraphicElementsCPUActor

        return GraphicElementsCPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
