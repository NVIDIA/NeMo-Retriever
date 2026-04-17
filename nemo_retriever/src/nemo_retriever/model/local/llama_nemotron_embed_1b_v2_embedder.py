# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch

from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from nemo_retriever.utils.hf_model_registry import get_hf_revision


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float()
    denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


def _is_cuda_oom_like_error(exc: RuntimeError) -> bool:
    msg = str(exc).upper()
    return any(
        token in msg
        for token in (
            "CUDA OUT OF MEMORY",
            "CUBLAS_STATUS_ALLOC_FAILED",
            "CUDNN_STATUS_INTERNAL_ERROR",
            "CUDNN_STATUS_ALLOC_FAILED",
        )
    )


def _batch_length_summary(
    chunk: Sequence[str],
    *,
    tokenizer: object | None,
    max_length: int,
) -> dict[str, Any]:
    if not chunk:
        return {
            "failed_batch": 0,
            "char_max": 0,
            "char_p95": 0,
            "tok_max": None,
            "tok_p95": None,
            "token_lengths": [],
        }

    char_lengths = [len(text) for text in chunk]
    sorted_char_lengths = sorted(char_lengths)
    p95_char_idx = int(0.95 * (len(sorted_char_lengths) - 1))
    summary: dict[str, Any] = {
        "failed_batch": len(chunk),
        "char_max": sorted_char_lengths[-1],
        "char_p95": sorted_char_lengths[p95_char_idx],
        "tok_max": None,
        "tok_p95": None,
        "token_lengths": [],
    }

    if tokenizer is None:
        return summary

    try:
        tokenized = tokenizer(
            list(chunk),
            padding=False,
            truncation=True,
            max_length=max(1, int(max_length)),
            return_length=True,
        )
        token_lengths = tokenized.get("length")
    except Exception:
        token_lengths = None

    if not token_lengths:
        return summary

    token_lengths_int = [int(length) for length in token_lengths]
    sorted_token_lengths = sorted(token_lengths_int)
    p95_tok_idx = int(0.95 * (len(sorted_token_lengths) - 1))
    summary["tok_max"] = sorted_token_lengths[-1]
    summary["tok_p95"] = sorted_token_lengths[p95_tok_idx]
    summary["token_lengths"] = token_lengths_int
    summary["max_length"] = int(max_length)
    return summary


def _batch_length_diagnostics(
    chunk: Sequence[str],
    *,
    tokenizer: object | None,
    max_length: int,
) -> str:
    summary = _batch_length_summary(chunk, tokenizer=tokenizer, max_length=max_length)
    return _batch_length_diagnostics_from_summary(summary, max_length=max_length)


def _batch_length_diagnostics_from_summary(summary: dict[str, Any], *, max_length: int) -> str:
    diag = (
        f"failed_batch={summary.get('failed_batch', 0)}, "
        f"char_max={summary.get('char_max', 0)}, "
        f"char_p95={summary.get('char_p95', 0)}"
    )
    tok_max = summary.get("tok_max")
    tok_p95 = summary.get("tok_p95")
    if tok_max is not None and tok_p95 is not None:
        diag = f"{diag}, tok_max={tok_max}, tok_p95={tok_p95}, max_length={int(max_length)}"
    return diag


@dataclass
class LlamaNemotronEmbed1BV2Embedder:
    """
    Minimal embedder wrapper for local-only HuggingFace execution.

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
        self._adaptive_batch_size: int | None = None

        from transformers import AutoModel, AutoTokenizer

        MODEL_ID = self.model_id or "nvidia/llama-nemotron-embed-1b-v2"
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
        texts_list = [str(text) for text in texts if str(text).strip()]
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)

        return self._embed_local(texts_list, batch_size=batch_size)

    def _embed_chunk(self, chunk: List[str]) -> torch.Tensor:
        dev = self._device
        batch = self._tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max(1, int(self.max_length)),
            return_tensors="pt",
        ).to(dev)
        out = self._model(**batch, output_hidden_states=True)
        lhs = getattr(out, "last_hidden_state", None)
        if lhs is None:
            hs = getattr(out, "hidden_states", None)
            if hs is not None:
                lhs = hs[-1]
            else:
                raise AttributeError(
                    f"Model output ({type(out).__name__}) has neither "
                    "'last_hidden_state' nor 'hidden_states'. "
                    "Ensure the model is loaded with trust_remote_code=True."
                )
        lhs = lhs.float()
        mask = batch["attention_mask"].unsqueeze(-1).float()
        vec = (lhs * mask).sum(dim=1) / mask.sum(dim=1)
        vec = vec.detach().to("cpu")
        if self.normalize:
            vec = _l2_normalize(vec)
        return vec

    def _embed_local(
        self,
        texts: List[str],
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("Local embedder was not initialized.")

        indexed_texts: list[tuple[int, str]] = []
        for idx, text in enumerate(texts):
            indexed_texts.append((idx, text))
        indexed_texts.sort(key=lambda pair: len(pair[1]), reverse=True)
        sorted_texts = [text for _, text in indexed_texts]
        sorted_to_original = [idx for idx, _ in indexed_texts]

        outs: List[torch.Tensor] = []
        target_bs = max(1, int(batch_size))
        current_bs = min(target_bs, self._adaptive_batch_size) if self._adaptive_batch_size is not None else target_bs
        success_streak = 0
        autocast_ctx = torch.autocast(device_type="cuda") if self._device.type == "cuda" else nullcontext()
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            with autocast_ctx:
                i = 0
                while i < len(sorted_texts):
                    chunk = sorted_texts[i : i + current_bs]
                    try:
                        outs.append(self._embed_chunk(chunk))
                        i += current_bs
                        if current_bs < target_bs:
                            success_streak += 1
                            if success_streak >= 3:
                                current_bs = min(target_bs, max(current_bs + 1, current_bs * 2))
                                success_streak = 0
                                self._adaptive_batch_size = current_bs
                        else:
                            success_streak = 0
                            self._adaptive_batch_size = target_bs
                    except torch.cuda.OutOfMemoryError:
                        if self._device.type != "cuda":
                            raise
                        torch.cuda.empty_cache()
                        success_streak = 0
                        if current_bs <= 1:
                            raise
                        current_bs = max(1, current_bs // 2)
                        self._adaptive_batch_size = current_bs
                        length_summary = _batch_length_summary(
                            chunk,
                            tokenizer=self._tokenizer,
                            max_length=self.max_length,
                        )
                        diag = _batch_length_diagnostics_from_summary(length_summary, max_length=self.max_length)
                        warnings.warn(
                            f"CUDA OOM during embedding; retrying with batch_size={current_bs} "
                            f"(requested={target_bs}, {diag})"
                        )
                    except RuntimeError as exc:
                        if self._device.type != "cuda" or not _is_cuda_oom_like_error(exc):
                            raise
                        torch.cuda.empty_cache()
                        success_streak = 0
                        if current_bs <= 1:
                            raise
                        current_bs = max(1, current_bs // 2)
                        self._adaptive_batch_size = current_bs
                        length_summary = _batch_length_summary(
                            chunk,
                            tokenizer=self._tokenizer,
                            max_length=self.max_length,
                        )
                        diag = _batch_length_diagnostics_from_summary(length_summary, max_length=self.max_length)
                        warnings.warn(
                            f"CUDA alloc/runtime error during embedding; retrying with batch_size={current_bs} "
                            f"(requested={target_bs}, {diag}): {exc}"
                        )

        if not outs:
            return torch.empty((0, 0), dtype=torch.float32)

        sorted_embeddings = torch.cat(outs, dim=0)
        reordered_embeddings: List[torch.Tensor | None] = [None] * len(sorted_to_original)
        for sorted_idx, original_idx in enumerate(sorted_to_original):
            reordered_embeddings[original_idx] = sorted_embeddings[sorted_idx]
        if any(emb is None for emb in reordered_embeddings):
            raise RuntimeError("Failed to reconstruct embedding order after length sorting.")
        return torch.stack([emb for emb in reordered_embeddings if emb is not None], dim=0)

    # Intentionally no remote embedding method.
