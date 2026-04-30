# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace-only text embedder for ``nvidia/llama-nemotron-embed-1b-v2``.

Used when local query embedding should match classic HF pooling (e.g. recall
evaluation) while document ingestion uses vLLM elsewhere.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from nemo_retriever.utils.hf_model_registry import get_hf_revision


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float()
    denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


@dataclass
class LlamaNemotronEmbed1BV2HFEmbedder:
    """Mean-pooled HF embeddings with Nemotron-style ``query:`` / ``passage:`` prefixes."""

    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    normalize: bool = True
    max_length: int = 8192
    model_id: Optional[str] = None

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from nemo_retriever.model import _DEFAULT_EMBED_MODEL
        from transformers import AutoModel, AutoTokenizer

        model_id = self.model_id or _DEFAULT_EMBED_MODEL
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hf_cache_dir = configure_global_hf_cache_base(self.hf_cache_dir)
        _revision = get_hf_revision(model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=_revision,
            cache_dir=hf_cache_dir,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            model_id,
            revision=_revision,
            trust_remote_code=True,
            cache_dir=hf_cache_dir,
            torch_dtype=torch.bfloat16,
        )
        self._model = self._model.to(dev)
        self._model.eval()
        self._device = dev

    @property
    def is_remote(self) -> bool:
        return False

    def _embed_local(self, texts: List[str], *, batch_size: int) -> torch.Tensor:
        self._ensure_loaded()
        dev = self._device
        bs = max(1, int(batch_size))
        max_len = max(1, int(self.max_length))

        outs: List[torch.Tensor] = []
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            # Tokenize per chunk (same as vLLM microbatches) so padding length matches
            # within-batch max, not global max over all strings — bidirectional attention
            # can otherwise shift vectors vs vLLM ingest / recall.
            for i in range(0, len(texts), bs):
                chunk = texts[i : i + bs]
                batch = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                batch = {k: v.to(dev) for k, v in batch.items()}
                out = self._model(**batch, output_hidden_states=True)
                lhs = getattr(out, "last_hidden_state", None)
                if lhs is None:
                    hs = getattr(out, "hidden_states", None)
                    if hs is not None:
                        lhs = hs[-1]
                    else:
                        raise AttributeError(
                            f"Model output ({type(out).__name__}) has neither "
                            "'last_hidden_state' nor 'hidden_states'."
                        )
                lhs = lhs.float()
                mask = batch["attention_mask"].unsqueeze(-1).float()
                vec = (lhs * mask).sum(dim=1) / mask.sum(dim=1)
                vec = vec.detach().to("cpu")
                if self.normalize:
                    vec = _l2_normalize(vec)
                outs.append(vec)

        return torch.cat(outs, dim=0) if outs else torch.empty((0, 0), dtype=torch.float32)

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Document strings; each line is prefixed with ``passage:`` for parity with vLLM."""
        texts_list = []
        for t in texts:
            raw = str(t)
            if not raw.strip():
                continue
            if not raw.lower().startswith("passage:"):
                raw = "passage: " + raw
            texts_list.append(raw)
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)
        return self._embed_local(texts_list, batch_size=batch_size)

    def embed_queries(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Query strings; each line is prefixed with ``query:`` (same rules as vLLM embed_queries)."""
        texts_list = []
        for t in texts:
            raw = str(t)
            if not raw.strip():
                continue
            if not raw.lower().startswith("query:"):
                raw = "query: " + raw
            texts_list.append(raw)
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)
        return self._embed_local(texts_list, batch_size=batch_size)

    def unload(self) -> None:
        """Release GPU memory held by the HF model."""
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
