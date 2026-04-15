# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for graph text-embedding operators."""

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.designer import designer_component
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.text_embed.runtime import embed_text_main_text_embed

__all__ = ["_BatchEmbedActor", "embed_text_main_text_embed"]

@designer_component(
    name="Batch Embedder",
    category="Embeddings & Ranking",
    compute="gpu",
    description="Generates embeddings in batches using configurable embedding parameters",
    category_color="#e06cff",
)
class _BatchEmbedActor(ArchetypeOperator):
    """Graph-facing batch embedding archetype."""

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        params = (operator_kwargs or {}).get("params")
        endpoint = getattr(params, "embed_invoke_url", None) or getattr(params, "embedding_endpoint", None)
        return bool(str(endpoint or "").strip())

    @classmethod
    def cpu_variant_class(cls):
        from nemo_retriever.text_embed.cpu_operator import _BatchEmbedCPUActor

        return _BatchEmbedCPUActor

    @classmethod
    def gpu_variant_class(cls):
        from nemo_retriever.text_embed.gpu_operator import _BatchEmbedActor as _BatchEmbedGPUActor

        return _BatchEmbedGPUActor

    def __init__(self, params: Any) -> None:
        super().__init__(params=params)


def __getattr__(name: str):
    if name == "_BatchEmbedCPUActor":
        from nemo_retriever.text_embed.cpu_operator import _BatchEmbedCPUActor

        return _BatchEmbedCPUActor
    if name == "_BatchEmbedGPUActor":
        from nemo_retriever.text_embed.gpu_operator import _BatchEmbedActor as _BatchEmbedGPUActor

        return _BatchEmbedGPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
