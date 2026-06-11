# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_retriever.models.model import BaseModel

VL_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
VL_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"

_VL_EMBED_MODEL_IDS = frozenset(
    {
        VL_EMBED_MODEL,
        "llama-nemotron-embed-vl-1b-v2",
        "llama-3.2-nemoretriever-1b-vlm-embed-v1",
        "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
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
    "llama-3.2-nemoretriever-1b-vlm-embed-v1": VL_EMBED_MODEL,
    "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1": VL_EMBED_MODEL,
}

_DEFAULT_EMBED_MODEL = VL_EMBED_MODEL


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


LOCAL_EMBED_ARCH_ENV = "NRL_LOCAL_EMBED_ARCH"
_VALID_LOCAL_EMBED_ARCHS = frozenset({"vl", "text"})


def _is_local_checkpoint_dir(model_name: str | None) -> bool:
    """Return True if *model_name* points at an on-disk checkpoint directory."""
    return bool(model_name) and os.path.isdir(str(model_name))


def _resolve_local_embed_arch(model_arch: str | None) -> bool:
    """Return True (VL) / False (text) for a local checkpoint directory.

    The architecture is never inferred. It must be declared explicitly via the
    *model_arch* argument or the ``NRL_LOCAL_EMBED_ARCH`` environment variable,
    so a local checkpoint can never be silently routed to the wrong embedder.

    Raises:
        ValueError: when the architecture is unset or not one of ``vl``/``text``.
    """
    raw = model_arch if model_arch is not None else os.getenv(LOCAL_EMBED_ARCH_ENV)
    arch = (raw or "").strip().lower()
    if arch not in _VALID_LOCAL_EMBED_ARCHS:
        raise ValueError(
            "A local embedding checkpoint directory requires its architecture to be "
            f"declared explicitly: set {LOCAL_EMBED_ARCH_ENV}='vl'|'text' (or pass "
            f"model_arch) so it routes to the correct embedder. Got {raw!r}."
        )
    return arch == "vl"


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
    query_max_length: int = 128,
    model_arch: str | None = None,
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

    A local checkpoint *directory* (e.g. a fine-tuned drop-in or proxy model)
    is supported on both the text and VL paths. Because a directory carries no
    registry entry, its architecture (``vl``/``text``) must be declared via
    *model_arch* or ``NRL_LOCAL_EMBED_ARCH``; it is never inferred.
    """
    b = (backend or "vllm").strip().lower()
    if b not in ("vllm", "hf"):
        raise ValueError(f"backend must be 'vllm' or 'hf', got {backend!r}")
    model_id = resolve_embed_model(model_name)

    # Registered Hub ids select VL vs text by the id allow-list (unchanged). A
    # local checkpoint dir is not in the allow-list, so it must declare its
    # architecture explicitly (fail-loud rather than guess).
    if _is_local_checkpoint_dir(model_name):
        use_vl = _resolve_local_embed_arch(model_arch)
    else:
        use_vl = is_vl_embed_model(model_name)

    if use_vl:
        if b == "hf":
            from nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder import (
                LlamaNemotronEmbedVL1BV2Embedder,
            )

            return LlamaNemotronEmbedVL1BV2Embedder(
                device=device,
                hf_cache_dir=hf_cache_dir,
                model_id=model_id,
            )

        from nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder import (
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
        from nemo_retriever.models.local.llama_nemotron_embed_1b_v2_hf_embedder import (
            LlamaNemotronEmbed1BV2HFEmbedder,
        )

        return LlamaNemotronEmbed1BV2HFEmbedder(
            device=device,
            hf_cache_dir=hf_cache_dir,
            normalize=normalize,
            max_length=int(max_length),
            query_max_length=int(query_max_length),
            model_id=model_id,
        )

    from nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder import (
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
_LOCAL_RERANKER_BACKENDS = frozenset({"hf", "vllm"})
_LOCAL_INGEST_EMBED_BACKENDS = frozenset({"hf", "vllm"})


def normalize_backend(value: str | None, valid: frozenset[str], *, field_name: str, default: str) -> str:
    """Normalize *value* (strip + lowercase) and validate against *valid*.

    Raises ``ValueError`` referencing *field_name* on invalid input.
    Falsy *value* is replaced by *default* before validation.
    """
    v = (value or default).strip().lower()
    if v not in valid:
        raise ValueError(f"{field_name} must be one of {sorted(valid)}; got {value!r}")
    return v


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
    query_max_length: int = 128,
    model_arch: str | None = None,
) -> Any:
    """Create a local embedder for *query* vectors in retrieval (Retriever / recall).

    *backend* must be ``"hf"`` (default) or ``"vllm"``.

    - ``backend="hf"``: HuggingFace for both VL and non-VL models.
    - ``backend="vllm"``: vLLM for both VL and non-VL models.

    *model_arch* (``vl``/``text``) declares the architecture of a local
    checkpoint directory; see :func:`create_local_embedder`.
    """
    b = normalize_backend(backend, _LOCAL_QUERY_BACKENDS, field_name="backend", default="hf")

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
        query_max_length=int(query_max_length),
        model_arch=model_arch,
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
        reranker.  ``"hf"`` uses HuggingFace
        ``AutoModelForSequenceClassification``.  Only affects VL reranker
        dispatch; the text-only reranker always uses HuggingFace.
    gpu_memory_utilization:
        Fraction of GPU memory for the vLLM engine (only used when
        *backend* is ``"vllm"``).
    """
    b = normalize_backend(backend, _LOCAL_RERANKER_BACKENDS, field_name="backend", default="vllm")
    if is_vl_rerank_model(model_name):
        if b == "vllm":
            from nemo_retriever.models.local.nemotron_rerank_vl_v2 import NemotronRerankVLV2VLLM

            return NemotronRerankVLV2VLLM(
                model_name=model_name,
                device=device,
                hf_cache_dir=hf_cache_dir,
                gpu_memory_utilization=gpu_memory_utilization,
            )

        from nemo_retriever.models.local.nemotron_rerank_vl_v2_hf import NemotronRerankVLV2

        return NemotronRerankVLV2(
            model_name=model_name,
            device=device,
            hf_cache_dir=hf_cache_dir,
        )

    from nemo_retriever.models.local.nemotron_rerank_v2 import NemotronRerankV2

    return NemotronRerankV2(
        model_name=model_name or "nvidia/llama-nemotron-rerank-1b-v2",
        device=device,
        hf_cache_dir=hf_cache_dir,
    )
