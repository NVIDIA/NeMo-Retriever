# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for graph text-embedding operators."""

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.designer import designer_component
from nemo_retriever.operators.operator_archetype import ArchetypeOperator
from nemo_retriever.models.inference.runtime import embed_text_main_text_embed

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
    def resolve_operator_class(
        cls,
        resources: Any = None,
        operator_kwargs: dict[str, Any] | None = None,
    ):
        kwargs = operator_kwargs or {}
        if kwargs.get("force_local") and not cls.prefers_cpu_variant(kwargs):
            return cls.gpu_variant_class()
        return super().resolve_operator_class(resources, operator_kwargs=kwargs)

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        params = (operator_kwargs or {}).get("params")
        endpoint = getattr(params, "embed_invoke_url", None) or getattr(params, "embedding_endpoint", None)
        return bool(str(endpoint or "").strip())

    @classmethod
    def cpu_variant_class(cls):
        from nemo_retriever.operators.embed.cpu_operator import _BatchEmbedCPUActor

        return _BatchEmbedCPUActor

    @classmethod
    def gpu_variant_class(cls):
        from nemo_retriever.operators.embed.gpu_operator import _BatchEmbedActor as _BatchEmbedGPUActor

        return _BatchEmbedGPUActor

    @classmethod
    def variant_operator_kwargs(
        cls,
        operator_class: type,
        operator_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs = super().variant_operator_kwargs(operator_class, operator_kwargs)
        kwargs.pop("force_local", None)
        return kwargs

    def __init__(self, params: Any, *, force_local: bool = False) -> None:
        super().__init__(params=params, force_local=force_local)


def __getattr__(name: str):
    if name == "_BatchEmbedCPUActor":
        from nemo_retriever.operators.embed.cpu_operator import _BatchEmbedCPUActor

        return _BatchEmbedCPUActor
    if name == "_BatchEmbedGPUActor":
        from nemo_retriever.operators.embed.gpu_operator import _BatchEmbedActor as _BatchEmbedGPUActor

        return _BatchEmbedGPUActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
