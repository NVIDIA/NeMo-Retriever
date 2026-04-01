# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility wrappers for graph-backed ingestor builders."""

from __future__ import annotations

from nemo_retriever.graph.ingestor_runtime import build_batch_graph, build_inprocess_graph

__all__ = ["build_batch_graph", "build_inprocess_graph"]
