# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.page_elements.shared import detect_page_elements_v3
from nemo_retriever.graph.designer import designer_component

__all__ = [
    "detect_page_elements_v3",
]


@designer_component(
    name="Page Element Detection (CPU)",
    category="Detection & OCR",
    compute="cpu",
    description="Detects page elements using CPU inference",
)
class PageElementDetectionActor(ArchetypeOperator):
    """Graph-facing page element detection archetype."""

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        kwargs = operator_kwargs or {}
        return bool(str(kwargs.get("page_elements_invoke_url") or kwargs.get("invoke_url") or "").strip())

    @classmethod
    def cpu_variant_class(cls):
        from nemo_retriever.page_elements.cpu_actor import PageElementDetectionCPUActor

        return PageElementDetectionCPUActor

    @classmethod
    def gpu_variant_class(cls):
        from nemo_retriever.page_elements.gpu_actor import PageElementDetectionActor as PageElementDetectionGPUActor

        return PageElementDetectionGPUActor

    def __init__(self, **detect_kwargs: Any) -> None:
        super().__init__(**detect_kwargs)


def __getattr__(name: str):
    if name == "PageElementDetectionCPUActor":
        from nemo_retriever.page_elements.cpu_actor import PageElementDetectionCPUActor

        return PageElementDetectionCPUActor
    if name == "PageElementDetectionGPUActor":
        from nemo_retriever.page_elements.gpu_actor import PageElementDetectionActor as PageElementDetectionGPUActor

        return PageElementDetectionGPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
