# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    backend: str = "vllm",
    device: str | None = None,
    hf_cache_dir: str | None = None,
    gpu_memory_utilization: float = 0.45,
    enforce_eager: bool = False,
    dimensions: int | None = None,
    normalize: bool = True,
    max_length: int = 8192,
) -> Any:
    """Create the appropriate local embedding model (VL or non-VL).

    *backend* must be ``"vllm"`` or ``"hf"``.

    For non-VL models:

    - ``backend="vllm"`` (default): vLLM via ``LlamaNemotronEmbed1BV2Embedder``.
    - ``backend="hf"``: HuggingFace via ``LlamaNemotronEmbed1BV2HFEmbedder``.

    For VL models:

    - ``backend="vllm"`` (default): vLLM via ``LlamaNemotronEmbedVL1BV2VLLMEmbedder``.
    - ``backend="hf"``: HuggingFace via ``LlamaNemotronEmbedVL1BV2Embedder``.

    ``device`` applies only to HuggingFace paths. For vLLM paths, ``device`` is
    forwarded for compatibility but deprecated and ignored (vLLM placement is
    process-level); passing it emits ``DeprecationWarning``.

    Note: ``gpu_memory_utilization``, ``enforce_eager``, ``dimensions``,
    ``normalize``, and ``max_length`` apply to vLLM paths only; the HF VL path ignores them.
    """
    b = (backend or "vllm").strip().lower()
    if b not in ("vllm", "hf"):
        raise ValueError(f"backend must be 'vllm' or 'hf', got {backend!r}")
    model_id = resolve_embed_model(model_name)

    if is_vl_embed_model(model_name):
        if b == "hf":
            from nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder import (
                LlamaNemotronEmbedVL1BV2Embedder,
            )

            return LlamaNemotronEmbedVL1BV2Embedder(
                device=device,
                hf_cache_dir=hf_cache_dir,
                model_id=model_id,
            )

        from nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder import (
            LlamaNemotronEmbedVL1BV2VLLMEmbedder,
        )

        return LlamaNemotronEmbedVL1BV2VLLMEmbedder(
            model_id=model_id,
            device=device,
            hf_cache_dir=hf_cache_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
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


_LOCAL_QUERY_BACKENDS = frozenset({"hf", "vllm"})


def create_local_query_embedder(
    model_name: str | None = None,
    *,
    backend: str = "hf",
    device: str | None = None,
    hf_cache_dir: str | None = None,
    gpu_memory_utilization: float = 0.45,
    enforce_eager: bool = False,
    dimensions: int | None = None,
    normalize: bool = True,
    max_length: int = 8192,
) -> Any:
    """Create a local embedder for *query* vectors in retrieval (Retriever / recall).

    *backend* must be ``"hf"`` (default) or ``"vllm"``.

    - ``backend="hf"``: HuggingFace for both VL and non-VL models.
    - ``backend="vllm"``: vLLM for both VL and non-VL models.
    """
    b = (backend or "hf").strip().lower()
    if b not in _LOCAL_QUERY_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_LOCAL_QUERY_BACKENDS)}, got {backend!r}")

    return create_local_embedder(
        model_name,
        backend=b,
        device=device,
        hf_cache_dir=hf_cache_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        dimensions=dimensions,
        normalize=normalize,
        max_length=int(max_length),
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
        ``"auto"`` / ``"vllm"`` (default) uses vLLM's pooling runner for
        the VL reranker.  ``"hf"`` uses HuggingFace
        ``AutoModelForSequenceClassification``.  Only affects VL reranker
        dispatch; the text-only reranker always uses HuggingFace.
    gpu_memory_utilization:
        Fraction of GPU memory for the vLLM engine (only used when
        *backend* resolves to ``"vllm"``).
    """
    b = (backend or "auto").strip().lower()
    if b not in ("auto", "vllm", "hf"):
        raise ValueError(f"backend must be 'auto', 'vllm', or 'hf'; got {backend!r}")
    if is_vl_rerank_model(model_name):
        if b in ("auto", "vllm"):
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
