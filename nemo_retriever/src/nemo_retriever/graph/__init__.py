# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical graph-execution package for operators, graphs, and executors."""

from __future__ import annotations

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.custom_operator import UDFOperator
from nemo_retriever.graph.executor import AbstractExecutor, InprocessExecutor, RayDataExecutor
from nemo_retriever.graph.file_loader_operator import FileListLoaderOperator
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.graph.graph_pipeline_registry import GraphPipelineRegistry, default_registry
from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.graph.store_operator import StoreOperator

__all__ = [
    "AbstractExecutor",
    "AbstractOperator",
    "ArchetypeOperator",
    "CPUOperator",
    "FileListLoaderOperator",
    "GPUOperator",
    "Graph",
    "GraphPipelineRegistry",
    "InprocessExecutor",
    "MultiTypeExtractOperator",
    "Node",
    "RayDataExecutor",
    "StoreOperator",
    "UDFOperator",
    "default_registry",
]


def __getattr__(name: str):
    if name == "MultiTypeExtractOperator":
        from nemo_retriever.graph.multi_type_extract_operator import MultiTypeExtractOperator

        return MultiTypeExtractOperator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
