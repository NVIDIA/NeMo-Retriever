# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_retriever.model.model import BaseModel

VL_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
VL_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"

_VL_EMBED_MODEL_IDS = frozenset(
    {
        VL_EMBED_MODEL,
        "llama-nemotron-embed-vl-1b-v2",
    }
)

_VL_RERANK_MODEL_IDS = frozenset(
    {
        VL_RERANK_MODEL,
        "llama-nemotron-rerank-vl-1b-v2",
    }
)

# Short name → full HF repo ID.
_EMBED_MODEL_ALIASES: dict[str, str] = {
    "nemo_retriever_v1": "nvidia/llama-nemotron-embed-1b-v2",
    "llama-nemotron-embed-vl-1b-v2": VL_EMBED_MODEL,
}

_DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"


def resolve_embed_model(model_name: str | None) -> str:
    """Resolve a model name/alias to a full HF repo ID.

    Returns ``_DEFAULT_EMBED_MODEL`` when *model_name* is ``None`` or empty.
    """
    if not model_name:
        return _DEFAULT_EMBED_MODEL
    return _EMBED_MODEL_ALIASES.get(model_name, model_name)


def is_vl_embed_model(model_name: str | None) -> bool:
    """Return True if *model_name* refers to the VL embedding model."""
    return resolve_embed_model(model_name) in _VL_EMBED_MODEL_IDS


def is_vl_rerank_model(model_name: str | None) -> bool:
    """Return True if *model_name* refers to the VL reranker model."""
    return (model_name or "") in _VL_RERANK_MODEL_IDS


def create_local_embedder(
    model_name: str | None = None,
    *,
    device: str | None = None,
    hf_cache_dir: str | None = None,
    normalize: bool = True,
    max_length: int = 8192,
):
    """Create the appropriate local embedding model (VL or non-VL).

    Centralises the resolve -> branch -> construct pattern that was previously
    duplicated across batch, inprocess, fused, gpu_pool, recall, retriever,
    and text_embed code paths.
    """
    model_id = resolve_embed_model(model_name)

    if is_vl_embed_model(model_name):
        from nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder import (
            LlamaNemotronEmbedVL1BV2Embedder,
        )

        return LlamaNemotronEmbedVL1BV2Embedder(
            device=device,
            hf_cache_dir=hf_cache_dir,
            model_id=model_id,
        )

    from nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder import (
        LlamaNemotronEmbed1BV2Embedder,
    )

    return LlamaNemotronEmbed1BV2Embedder(
        device=device,
        hf_cache_dir=hf_cache_dir,
        normalize=normalize,
        max_length=max_length,
        model_id=model_id,
    )


def create_local_reranker(
    model_name: str | None = None,
    *,
    device: str | None = None,
    hf_cache_dir: str | None = None,
    backend: str = "vllm",
    gpu_memory_utilization: float = 0.5,
) -> "BaseModel":
    """Create the appropriate local reranker model (VL or text-only).

    Dispatches to ``NemotronRerankVLV2VLLM`` (default) or
    ``NemotronRerankVLV2`` when *model_name* matches a VL reranker ID,
    depending on *backend*.  Otherwise returns the text-only
    ``NemotronRerankV2``.

    Parameters
    ----------
    backend:
        ``"vllm"`` (default) uses vLLM's pooling runner for the VL
        reranker.  ``"transformers"`` uses HuggingFace
        ``AutoModelForSequenceClassification``.  Only affects VL reranker
        dispatch; the text-only reranker always uses transformers.
    gpu_memory_utilization:
        Fraction of GPU memory for the vLLM engine (only used when
        *backend* is ``"vllm"``).
    """
    if is_vl_rerank_model(model_name):
        if backend == "vllm":
            from nemo_retriever.model.local.nemotron_rerank_vl_v2_vllm import NemotronRerankVLV2VLLM

            return NemotronRerankVLV2VLLM(
                model_name=model_name,
                device=device,
                hf_cache_dir=hf_cache_dir,
                gpu_memory_utilization=gpu_memory_utilization,
            )

        from nemo_retriever.model.local.nemotron_rerank_vl_v2 import NemotronRerankVLV2

        return NemotronRerankVLV2(
            model_name=model_name,
            device=device,
            hf_cache_dir=hf_cache_dir,
        )

    from nemo_retriever.model.local.nemotron_rerank_v2 import NemotronRerankV2

    return NemotronRerankV2(
        model_name=model_name or "nvidia/llama-nemotron-rerank-1b-v2",
        device=device,
        hf_cache_dir=hf_cache_dir,
    )
