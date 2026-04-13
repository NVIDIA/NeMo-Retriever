# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

_VL_EMBED_MODEL_IDS = frozenset(
    {
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        "llama-nemotron-embed-vl-1b-v2",
    }
)

# Short name → full HF repo ID.
_EMBED_MODEL_ALIASES: dict[str, str] = {
    "nemo_retriever_v1": "nvidia/llama-nemotron-embed-1b-v2",
    "llama-nemotron-embed-vl-1b-v2": "nvidia/llama-nemotron-embed-vl-1b-v2",
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


def create_local_embedder(
    model_name: str | None = None,
    *,
    device: str | None = None,
    hf_cache_dir: str | None = None,
    gpu_memory_utilization: float = 0.45,
    enforce_eager: bool = False,
    dimensions: int | None = None,
    normalize: bool = True,
    max_length: int = 8192,
) -> Any:
    """Create the appropriate local embedding model (VL or non-VL).

    VL models always use HuggingFace (supports image + text+image modalities).
    Non-VL models use vLLM via ``LlamaNemotronEmbed1BV2Embedder`` in
    ``nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder``.

    ``device`` applies only to the VL (HuggingFace) path. For non-VL text models,
    ``device`` is forwarded for compatibility but deprecated and ignored (vLLM
    placement is process-level); passing it emits ``DeprecationWarning``.

    Note: ``gpu_memory_utilization``, ``enforce_eager``, ``dimensions``,
    ``normalize``, and ``max_length`` apply to the non-VL (vLLM) path only;
    VL models ignore them.
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
        model_id=model_id,
        hf_cache_dir=hf_cache_dir,
        device=device,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        dimensions=dimensions,
        normalize=normalize,
        max_length=int(max_length),
    )


_LOCAL_QUERY_BACKENDS = frozenset({"auto", "hf", "vllm"})


def create_local_query_embedder(
    model_name: str | None = None,
    *,
    backend: str = "auto",
    device: str | None = None,
    hf_cache_dir: str | None = None,
    gpu_memory_utilization: float = 0.45,
    enforce_eager: bool = False,
    dimensions: int | None = None,
    normalize: bool = True,
    max_length: int = 8192,
) -> Any:
    """Create a local embedder for *query* vectors in retrieval (Retriever / recall).

    For non-VL text models:

    - ``backend="auto"`` or ``"vllm"``: same as :func:`create_local_embedder` (vLLM on this branch).
    - ``backend="hf"``: HuggingFace mean pooling with ``query:`` / ``passage:`` prefixes
      (``LlamaNemotronEmbed1BV2HFEmbedder``), for recall when LanceDB was built with vLLM
      document vectors but you want cheaper or better-aligned query embeddings.

    VL models always use the HuggingFace VL embedder (same as :func:`create_local_embedder`);
    *backend* does not change that path.
    """
    b = (backend or "auto").strip().lower()
    if b not in _LOCAL_QUERY_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_LOCAL_QUERY_BACKENDS)}, got {backend!r}")
    model_id = resolve_embed_model(model_name)

    if is_vl_embed_model(model_name):
        return create_local_embedder(
            model_name,
            device=device,
            hf_cache_dir=hf_cache_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            dimensions=dimensions,
            normalize=normalize,
            max_length=int(max_length),
        )

    if b == "hf":
        from nemo_retriever.model.local.llama_nemotron_embed_1b_v2_hf_embedder import (
            LlamaNemotronEmbed1BV2HFEmbedder,
        )

        return LlamaNemotronEmbed1BV2HFEmbedder(
            device=device,
            hf_cache_dir=hf_cache_dir,
            normalize=normalize,
            max_length=int(max_length),
            model_id=model_id,
        )

    return create_local_embedder(
        model_name,
        hf_cache_dir=hf_cache_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        dimensions=dimensions,
        normalize=normalize,
        max_length=int(max_length),
    )
