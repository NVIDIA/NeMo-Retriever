# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backwards-compatible re-export — canonical location is graph.pipeline_graph."""

from nemo_retriever.graph.pipeline_graph import Graph, Node, _ensure_node

__all__ = ["Graph", "Node", "_ensure_node"]
