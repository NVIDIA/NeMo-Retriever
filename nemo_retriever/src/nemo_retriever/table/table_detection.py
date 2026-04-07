# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.table.shared import table_structure_ocr_page_elements

__all__ = [
    "table_structure_ocr_page_elements",
]


def __getattr__(name: str):
    if name == "TableStructureActor":
        from nemo_retriever.table.gpu_actor import TableStructureActor

        return TableStructureActor
    if name == "TableStructureCPUActor":
        from nemo_retriever.table.cpu_actor import TableStructureCPUActor

        return TableStructureCPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
