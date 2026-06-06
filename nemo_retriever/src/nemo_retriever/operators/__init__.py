# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical home for shared pipeline operator classes.

Re-exports are resolved lazily so that importing an ``operators`` submodule does
not eagerly pull the compatibility aliases (which keeps package initialization
free of import cycles during the bucket reorganization).
"""
from __future__ import annotations

__all__ = ["AbstractOperator", "CPUOperator", "GPUOperator", "ExplodeContentActor", "_BatchEmbedActor"]

_LAZY = {
    "AbstractOperator": "nemo_retriever.operators.abstract_operator",
    "CPUOperator": "nemo_retriever.operators.cpu_operator",
    "GPUOperator": "nemo_retriever.operators.gpu_operator",
    "ExplodeContentActor": "nemo_retriever.operators.graph_ops.content_operators",
    "_BatchEmbedActor": "nemo_retriever.operators.embed.operators",
}


def __getattr__(name: str):
    import importlib

    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
