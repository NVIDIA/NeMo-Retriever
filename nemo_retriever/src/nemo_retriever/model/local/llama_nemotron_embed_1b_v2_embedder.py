# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
class LlamaNemotronEmbed1BV2Embedder:
    """
    Minimal embedder wrapper for local HuggingFace execution.

    This intentionally contains **no remote invocation logic**.
    """

    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    normalize: bool = True
    # IMPORTANT: Some HF tokenizers set an effectively "infinite" model_max_length.
    # If we rely on that, `truncation=True` may still allow extremely long sequences,
    # which can explode attention-mask memory (O(seq_len^2)) and OOM the GPU.
    # max_length: int = 4096
    max_length: int = 8192
    model_id: Optional[str] = None

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = None

        from nemo_retriever.model import _DEFAULT_EMBED_MODEL
        from transformers import AutoModel, AutoTokenizer

        MODEL_ID = self.model_id or _DEFAULT_EMBED_MODEL
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hf_cache_dir = configure_global_hf_cache_base(self.hf_cache_dir)
        _revision = get_hf_revision(MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=_revision,
            cache_dir=hf_cache_dir,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            MODEL_ID,
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

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """
        Returns a CPU tensor of shape [N, D].
        """
        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)

        return self._embed_local(texts_list, batch_size=batch_size)

    def _embed_local(self, texts: List[str], *, batch_size: int) -> torch.Tensor:
        if self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("Local embedder was not initialized.")
        dev = self._device
        bs = max(1, int(batch_size))

        # Tokenize all texts in a single call to avoid repeated setup overhead.
        full_batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max(1, int(self.max_length)),
            return_tensors="pt",
        )

        outs: List[torch.Tensor] = []
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            for i in range(0, len(texts), bs):
                batch = {k: v[i : i + bs].to(dev) for k, v in full_batch.items()}
                out = self._model(**batch, output_hidden_states=True)
                # The bidirectional model returns BaseModelOutputWithPast
                # (last_hidden_state), but some transformers versions or
                # model revisions return CausalLMOutputWithPast (hidden_states).
                lhs = getattr(out, "last_hidden_state", None)
                if lhs is None:
                    # CausalLMOutputWithPast: use the last layer's hidden state.
                    hs = getattr(out, "hidden_states", None)
                    if hs is not None:
                        lhs = hs[-1]
                    else:
                        raise AttributeError(
                            f"Model output ({type(out).__name__}) has neither "
                            "'last_hidden_state' nor 'hidden_states'. "
                            "Ensure the model is loaded with trust_remote_code=True."
                        )
                # Pool in float32 to avoid accumulation errors in bf16.
                lhs = lhs.float()  # [B, S, D]
                mask = batch["attention_mask"].unsqueeze(-1).float()  # [B, S, 1]
                vec = (lhs * mask).sum(dim=1) / mask.sum(dim=1)  # [B, D]
                vec = vec.detach().to("cpu")
                if self.normalize:
                    vec = _l2_normalize(vec)
                outs.append(vec)

        return torch.cat(outs, dim=0) if outs else torch.empty((0, 0), dtype=torch.float32)

    # Intentionally no remote embedding method.


@dataclass
class LlamaNemotronEmbed1BV2VLLMEmbedder:
    """
    vLLM-backed embedder for ``nvidia/llama-nemotron-embed-1b-v2``.

    Always uses vLLM's Python API (bfloat16 + FLASH_ATTN, pooling runner).
    Exposes the same ``embed()`` interface as ``LlamaNemotronEmbed1BV2Embedder``
    so the two are drop-in substitutes from the caller's perspective.
    """

    model_id: Optional[str] = None
    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    gpu_memory_utilization: float = 0.45
    enforce_eager: bool = False
    compile_cache_dir: Optional[str] = None
    dimensions: Optional[int] = None

    def __post_init__(self) -> None:
        from nemo_retriever.text_embed.vllm import create_vllm_llm

        if self.device is not None:
            import os

            dev_id = self.device.split(":")[-1] if ":" in self.device else self.device
            os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
        configure_global_hf_cache_base(self.hf_cache_dir)

        from nemo_retriever.model import _DEFAULT_EMBED_MODEL

        model_id = self.model_id or _DEFAULT_EMBED_MODEL
        self._llm = create_vllm_llm(
            str(model_id),
            revision=get_hf_revision(model_id),
            dimensions=self.dimensions,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            compile_cache_dir=self.compile_cache_dir,
        )

    @property
    def is_remote(self) -> bool:
        return False

    def embed(self, texts: Sequence[str], *, batch_size: int = 64, prefix: str = "passage: ") -> torch.Tensor:
        """Embed texts. Returns CPU tensor ``[N, D]``.

        ``prefix`` is prepended to every text before encoding; defaults to ``"passage: "`` for
        document embeddings.  Pass ``prefix="query: "`` for query embeddings.
        """
        from nemo_retriever.text_embed.vllm import embed_with_vllm_llm

        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)
        vectors = embed_with_vllm_llm(texts_list, self._llm, batch_size=max(1, int(batch_size)), prefix=prefix)
        if not vectors:
            return torch.empty((0, 0), dtype=torch.float32)
        return _l2_normalize(torch.tensor(vectors, dtype=torch.float32))

    def embed_queries(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed query strings. Returns CPU tensor ``[N, D]``."""
        from nemo_retriever.text_embed.vllm import embed_with_vllm_llm

        texts_list = [str(t) for t in texts]
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)
        vectors = embed_with_vllm_llm(texts_list, self._llm, batch_size=max(1, int(batch_size)), prefix="query: ")
        if not vectors:
            return torch.empty((0, 0), dtype=torch.float32)
        return _l2_normalize(torch.tensor(vectors, dtype=torch.float32))
