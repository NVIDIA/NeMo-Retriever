# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import torch

from nemo_retriever.models.hf_cache import configure_global_hf_cache_base
from nemo_retriever.models.embed_errors import report_lost_rows
from nemo_retriever.models.embed_model_spec import resolve_embed_model_revision


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float()
    denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


@dataclass
class LlamaNemotronEmbed1BV2Embedder:
    """
    vLLM embedder for compatible dense LlamaBidirectional Nemotron checkpoints.

    Uses vLLM's pooling runner (``llm.embed()``) for throughput. No HTTP remote
    calls — load and inference stay in-process.

    vLLM device placement follows the process environment (see vLLM docs).
    ``max_length`` is passed to vLLM as ``max_model_len`` when set positive.

    ``device`` is deprecated and ignored; it remains only for backward compatibility
    with callers that constructed the former HuggingFace embedder with ``device=``.
    The legacy class name is retained for compatibility.
    """

    model_id: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    device: Optional[str] = None
    gpu_memory_utilization: float = 0.45
    enforce_eager: bool = False
    dimensions: Optional[int] = None
    normalize: bool = True
    max_length: int = 8192
    revision: Optional[str] = None
    query_prefix: str = "query: "
    document_prefix: str = "passage: "

    _llm: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.device is not None:
            warnings.warn(
                "LlamaNemotronEmbed1BV2Embedder no longer uses 'device'; "
                "vLLM follows process-level GPU placement. Use CUDA_VISIBLE_DEVICES "
                "or vLLM's tensor_parallel_size instead. 'device' will be removed "
                "in a future release.",
                DeprecationWarning,
                stacklevel=4,
            )

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from nemo_retriever.models import _DEFAULT_EMBED_MODEL
        from nemo_retriever.models.inference.vllm import create_vllm_llm

        configure_global_hf_cache_base(self.hf_cache_dir)
        model_id = self.model_id or _DEFAULT_EMBED_MODEL
        max_model_len = int(self.max_length) if int(self.max_length) > 0 else None
        self._llm = create_vllm_llm(
            str(model_id),
            revision=resolve_embed_model_revision(model_id, self.revision),
            dimensions=self.dimensions,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            max_model_len=max_model_len,
        )

    @property
    def is_remote(self) -> bool:
        return False

    def _finalize_vectors(self, vectors: List[List[float]]) -> torch.Tensor:
        # See the mirror of this function in
        # ``models/local/llama_nemotron_embed_vl_1b_v2_embedder.py``.
        report_lost_rows(vectors, embedder=type(self).__name__)
        valid = [v for v in vectors if v]
        if not valid:
            return torch.empty((0, 0), dtype=torch.float32)
        dim = len(valid[0])
        padded: List[List[float]] = [v if v else [0.0] * dim for v in vectors]
        t = torch.tensor(padded, dtype=torch.float32)
        if self.normalize:
            return _l2_normalize(t)
        return t

    def embed(self, texts: Sequence[str], *, batch_size: int = 64, prefix: str | None = None) -> torch.Tensor:
        """Embed texts. Returns CPU tensor ``[N, D]``.

        ``prefix`` overrides the checkpoint-declared document prefix when set.
        """
        self._ensure_loaded()
        from nemo_retriever.models.inference.vllm import embed_with_vllm_llm

        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)
        vectors = embed_with_vllm_llm(
            texts_list,
            self._llm,
            batch_size=max(1, int(batch_size)),
            prefix=self.document_prefix if prefix is None else prefix,
            normalize=self.normalize,
        )
        return self._finalize_vectors(vectors)

    def embed_queries(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed query strings. Returns CPU tensor ``[N, D]``."""
        self._ensure_loaded()
        from nemo_retriever.models.inference.vllm import embed_with_vllm_llm

        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)
        vectors = embed_with_vllm_llm(
            texts_list,
            self._llm,
            batch_size=max(1, int(batch_size)),
            prefix=self.query_prefix,
            normalize=self.normalize,
        )
        return self._finalize_vectors(vectors)

    def unload(self) -> None:
        """Release GPU memory held by the vLLM engine."""
        del self._llm
        self._llm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
