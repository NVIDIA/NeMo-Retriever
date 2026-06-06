# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace-only text embedder for ``nvidia/llama-nemotron-embed-1b-v2``.

Used when local query embedding should match classic HF pooling (e.g. recall
evaluation) while document ingestion uses vLLM elsewhere.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from nemo_retriever.utils.hf_model_registry import get_hf_revision

logger = logging.getLogger(__name__)


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
    # Fixed query padding length. Queries longer than this are truncated; larger
    # values preserve more text but increase query embedding cost.
    query_max_length: int = 128
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

    def _embed_local(
        self,
        texts: List[str],
        *,
        batch_size: int,
        padding: bool | str = True,
        max_length: int | None = None,
    ) -> torch.Tensor:
        self._ensure_loaded()
        dev = self._device
        bs = max(1, int(batch_size))
        max_len = max(1, int(max_length if max_length is not None else self.max_length))

        outs: List[torch.Tensor] = []
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            for i in range(0, len(texts), bs):
                chunk = texts[i : i + bs]
                batch = self._tokenizer(
                    chunk,
                    padding=padding,
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

    def _warn_if_queries_truncated(self, texts: Sequence[str], *, max_length: int) -> None:
        self._ensure_loaded()
        tokenizer = self._tokenizer
        truncated = 0
        for text in texts:
            try:
                encode = getattr(tokenizer, "encode", None)
                if callable(encode):
                    input_ids = encode(str(text), add_special_tokens=True, truncation=False)
                else:
                    encoded = tokenizer(str(text), add_special_tokens=True, truncation=False)
                    input_ids = (
                        encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
                    )
            except Exception:  # noqa: BLE001 - best-effort warning only
                return

            if isinstance(input_ids, list):
                length = len(input_ids[0]) if input_ids and isinstance(input_ids[0], list) else len(input_ids)
                if length > max_length:
                    truncated += 1

        if truncated:
            logger.warning(
                "Truncating %d/%d HF query embeddings to query_max_length=%d tokens; "
                "increase query_max_length (or --local-query-max-length in pipeline eval) "
                "if long queries lose recall.",
                truncated,
                len(texts),
                max_length,
            )

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
        # The Nemotron text embedder is sensitive to padding length. Use fixed
        # query padding so retrieval vectors do not change with batch grouping.
        query_max_length = max(1, int(self.query_max_length))
        self._warn_if_queries_truncated(texts_list, max_length=query_max_length)
        return self._embed_local(
            texts_list,
            batch_size=batch_size,
            padding="max_length",
            max_length=query_max_length,
        )

    def unload(self) -> None:
        """Release GPU memory held by the HF model."""
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
