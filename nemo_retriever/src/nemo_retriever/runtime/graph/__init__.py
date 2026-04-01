# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility wrappers for the canonical ``nemo_retriever.graph`` package."""

from __future__ import annotations

from nemo_retriever.graph import (
    AbstractExecutor,
    AbstractOperator,
    CPUOperator,
    FileListLoaderOperator,
    GPUOperator,
    Graph,
    InprocessExecutor,
    Node,
    RayDataExecutor,
    UDFOperator,
)

__all__ = [
    "AbstractExecutor",
    "AbstractOperator",
    "CPUOperator",
    "FileListLoaderOperator",
    "GPUOperator",
    "Graph",
    "InprocessExecutor",
    "MultiTypeExtractOperator",
    "Node",
    "RayDataExecutor",
    "UDFOperator",
]


def __getattr__(name: str):
    if name == "MultiTypeExtractOperator":
        from nemo_retriever.graph import MultiTypeExtractOperator

        return MultiTypeExtractOperator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
