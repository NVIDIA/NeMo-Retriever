# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.table.shared import table_structure_ocr_page_elements

__all__ = [
    "table_structure_ocr_page_elements",
]


class TableStructureActor(ArchetypeOperator):
    """Graph-facing table-structure archetype."""

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        kwargs = operator_kwargs or {}
        return bool(str(kwargs.get("table_structure_invoke_url") or kwargs.get("invoke_url") or "").strip())

    @classmethod
    def cpu_variant_class(cls):
        from nemo_retriever.table.cpu_actor import TableStructureCPUActor

        return TableStructureCPUActor

    @classmethod
    def gpu_variant_class(cls):
        from nemo_retriever.table.gpu_actor import TableStructureActor as TableStructureGPUActor

        return TableStructureGPUActor

    def __init__(self, **detect_kwargs: Any) -> None:
        super().__init__(**detect_kwargs)


def __getattr__(name: str):
    if name == "TableStructureCPUActor":
        from nemo_retriever.table.cpu_actor import TableStructureCPUActor

        return TableStructureCPUActor
    if name == "TableStructureGPUActor":
        from nemo_retriever.table.gpu_actor import TableStructureActor as TableStructureGPUActor

        return TableStructureGPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
