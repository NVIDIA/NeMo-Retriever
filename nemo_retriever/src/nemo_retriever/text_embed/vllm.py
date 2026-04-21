# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM batched embedding inference.

Uses vLLM's Python API (LLM with runner="pooling" and llm.embed())
to compute embeddings without running a vLLM server. Use this when you want
the same embedding model (e.g. nvidia/llama-3.2-nv-embedqa-1b-v2) with vLLM's batched
inference and no HTTP server.

Uses bfloat16 and FLASH_ATTN backend by default for best throughput.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

VLLM_DTYPE = "bfloat16"
VLLM_ATTENTION_BACKEND = "FLASH_ATTN"


def create_vllm_llm(
    model: str,
    *,
    revision: Optional[str] = None,
    dimensions: Optional[int] = None,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.45,
    enforce_eager: bool = False,
    hf_overrides: Optional[dict] = None,
    limit_mm_per_prompt: Optional[dict] = None,
) -> Any:
    """
    Create and return a vLLM LLM instance for embedding (pooling runner).
    Caller can reuse it across many embed batches to avoid repeated model load and CUDA graph capture.

    Uses bfloat16 and FLASH_ATTN backend (fixed for this module).

    """
    try:
        from vllm import LLM
    except ImportError as e:
        raise RuntimeError("vLLM is not installed. Install with: uv pip install -e '.' or pip install -e '.'") from e

    pooler_config = None
    try:
        from vllm.config.pooler import PoolerConfig

        try:
            pooler_config = PoolerConfig(seq_pooling_type="MEAN", dimensions=dimensions)
        except TypeError:
            try:
                pooler_config = PoolerConfig(pooling_type="MEAN", dimensions=dimensions)
            except (TypeError, ValueError):
                pooler_config = PoolerConfig()
    except ImportError:
        pooler_config = None

    kwargs: dict = {
        "model": model,
        "trust_remote_code": trust_remote_code,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": VLLM_DTYPE,
        "runner": "pooling",
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "attention_backend": VLLM_ATTENTION_BACKEND,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if hf_overrides is not None:
        kwargs["hf_overrides"] = hf_overrides
    if pooler_config is not None:
        kwargs["pooler_config"] = pooler_config
    if limit_mm_per_prompt is not None:
        kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt

    return LLM(**kwargs)


def embed_with_vllm_llm(
    prompts: List[str],
    llm: Any,
    *,
    batch_size: int = 256,
    prefix: Optional[str] = None,
) -> List[List[float]]:
    """
    Compute embeddings using an existing vLLM LLM instance (no new model load).
    Use this when the caller holds a shared LLM (e.g. one per Ray actor).
    """
    if prefix:
        prompts = [str(prefix) + p for p in prompts]
    if not prompts:
        return []

    all_embeddings: List[List[float]] = []
    for i in range(0, len(prompts), max(1, batch_size)):
        batch = prompts[i : i + max(1, batch_size)]
        outputs = llm.embed(batch)
        for out in outputs:
            emb = getattr(getattr(out, "outputs", None), "embedding", None)
            if emb is not None:
                if hasattr(emb, "tolist"):
                    all_embeddings.append(emb.tolist())
                elif isinstance(emb, list):
                    all_embeddings.append([float(x) for x in emb])
                else:
                    all_embeddings.append(list(emb))
            else:
                all_embeddings.append([])
    return all_embeddings


def embed_multimodal_with_vllm_llm(
    prompt_dicts: List[dict],
    llm: Any,
    *,
    batch_size: int = 64,
) -> List[List[float]]:
    """
    Compute embeddings for multimodal prompts using an existing vLLM LLM instance.

    Each element of ``prompt_dicts`` must be a dict with:
      - ``"prompt"``: text string including the ``<image>`` placeholder
      - ``"multi_modal_data"``: ``{"image": PIL.Image.Image}``

    The LLM must have been created with ``limit_mm_per_prompt={"image": 1}``.
    Returns one embedding vector (list of floats) per input; ``[]`` for failures.
    """
    if not prompt_dicts:
        return []

    all_embeddings: List[List[float]] = []
    for i in range(0, len(prompt_dicts), max(1, batch_size)):
        batch = prompt_dicts[i : i + max(1, batch_size)]
        outputs = llm.embed(batch)
        for out in outputs:
            emb = getattr(getattr(out, "outputs", None), "embedding", None)
            if emb is not None:
                if hasattr(emb, "tolist"):
                    all_embeddings.append(emb.tolist())
                elif isinstance(emb, list):
                    all_embeddings.append([float(x) for x in emb])
                else:
                    all_embeddings.append(list(emb))
            else:
                all_embeddings.append([])
    return all_embeddings


__all__ = ["create_vllm_llm", "embed_with_vllm_llm", "embed_multimodal_with_vllm_llm"]
