# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from nemo_retriever.utils.hf_model_registry import get_hf_revision
from nemo_retriever.utils.nvtx import gpu_inference_range


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float()
    denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


def _b64_to_pil(b64_str: str):
    import base64
    import io

    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


@dataclass
class LlamaNemotronEmbedVL1BV2Embedder:
    """
    Multimodal embedder wrapper for ``nvidia/llama-nemotron-embed-vl-1b-v2``.

    The VL model exposes ``encode_queries()`` and ``encode_documents()``
    instead of the standard tokenizer + forward pass used by the embedqa
    model.  This class supports text, image, and text+image modalities.
    """

    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    model_id: Optional[str] = None

    # Populated in __post_init__
    _model: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from transformers import AutoModel

        from nemo_retriever.model import VL_EMBED_MODEL

        model_id = self.model_id or VL_EMBED_MODEL
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hf_cache_dir = configure_global_hf_cache_base(self.hf_cache_dir)

        # flash_attention_2 requires the model on GPU at init time, so use
        # device_map when requesting it.  Fall back to sdpa/eager on CPU or
        # when flash-attn is not installed.
        use_gpu = dev.type == "cuda"
        _revision = get_hf_revision(model_id)
        for attn_impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                kwargs: dict[str, Any] = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": attn_impl,
                    "cache_dir": hf_cache_dir,
                    "revision": _revision,
                }
                if attn_impl == "flash_attention_2" and use_gpu:
                    kwargs["device_map"] = dev
                self._model = AutoModel.from_pretrained(model_id, **kwargs)
                break
            except (ValueError, ImportError):
                if attn_impl == "eager":
                    raise
                continue

        if not hasattr(self._model, "device_map"):
            self._model = self._model.to(dev)
        self._model.eval()
        self._device = dev

    @property
    def is_remote(self) -> bool:
        return False

    def _set_p_max_length(self, modality: str) -> None:
        _RECOMMENDED = {"text": 8192, "image": 2048, "text_image": 10240}
        p = _RECOMMENDED.get(modality)
        if p is not None and hasattr(self._model, "processor"):
            self._model.processor.p_max_length = p

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed document texts. Returns CPU tensor ``[N, 2048]``."""
        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 2048), dtype=torch.float32)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            self._set_p_max_length("text")
            with gpu_inference_range("LlamaNemotronEmbedVL1B", batch_size=len(texts_list), mode="doc_text"):
                out = self._model.encode_documents(texts=texts_list)
        if isinstance(out, torch.Tensor):
            return _l2_normalize(out.detach().cpu())
        return _l2_normalize(torch.as_tensor(out, dtype=torch.float32).cpu())

    def embed_queries(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed query strings. Returns CPU tensor ``[N, 2048]``."""
        texts_list = [str(t) for t in texts]
        if not texts_list:
            return torch.empty((0, 2048), dtype=torch.float32)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            with gpu_inference_range("LlamaNemotronEmbedVL1B", batch_size=len(texts_list), mode="query"):
                out = self._model.encode_queries(texts_list)
        if isinstance(out, torch.Tensor):
            return _l2_normalize(out.detach().cpu())
        return _l2_normalize(torch.as_tensor(out, dtype=torch.float32).cpu())

    def embed_images(self, images_b64: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed images (base64-encoded). Returns CPU tensor ``[N, 2048]``."""
        image_dicts = [{"base64": b64} for b64 in images_b64 if b64]
        if not image_dicts:
            return torch.empty((0, 2048), dtype=torch.float32)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            self._set_p_max_length("image")
            with gpu_inference_range("LlamaNemotronEmbedVL1B", batch_size=len(image_dicts), mode="doc_image"):
                out = self._model.encode_documents(images=image_dicts)
        if isinstance(out, torch.Tensor):
            return _l2_normalize(out.detach().cpu())
        return _l2_normalize(torch.as_tensor(out, dtype=torch.float32).cpu())

    def embed_text_image(
        self, texts: Sequence[str], images_b64: Sequence[str], *, batch_size: int = 64
    ) -> torch.Tensor:
        """Embed paired text+image inputs. Returns CPU tensor ``[N, 2048]``."""
        paired_texts: list[str] = []
        paired_images: list[dict[str, str]] = []
        for t, b64 in zip(texts, images_b64):
            if b64:
                paired_texts.append(str(t))
                paired_images.append({"base64": b64})
        if not paired_images:
            return torch.empty((0, 2048), dtype=torch.float32)
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            self._set_p_max_length("text_image")
            with gpu_inference_range("LlamaNemotronEmbedVL1B", batch_size=len(paired_images), mode="doc_text_image"):
                out = self._model.encode_documents(texts=paired_texts, images=paired_images)
        if isinstance(out, torch.Tensor):
            return _l2_normalize(out.detach().cpu())
        return _l2_normalize(torch.as_tensor(out, dtype=torch.float32).cpu())


@dataclass
class LlamaNemotronEmbedVL1BV2VLLMEmbedder:
    """
    vLLM-backed embedder for ``nvidia/llama-nemotron-embed-vl-1b-v2``.

    Supports text, image, and text+image modalities via vLLM's Python API
    (bfloat16 + FLASH_ATTN, pooling runner). Requires vLLM >= 0.17.0.

    .. warning::
        When ``device`` is set, ``CUDA_VISIBLE_DEVICES`` is overwritten for the entire
        process. Constructing multiple instances with different devices in the same process
        (e.g. in a Ray actor pool) will produce incorrect behaviour. Pass device constraints
        via ``tensor_parallel_size`` or ensure each actor is launched in an isolated process.
    """

    model_id: Optional[str] = None
    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    gpu_memory_utilization: float = 0.45
    enforce_eager: bool = False

    def __post_init__(self) -> None:
        try:
            import vllm as _vllm_mod

            from nemo_retriever.text_embed.vllm import create_vllm_llm
        except ImportError as e:
            raise RuntimeError(
                "vLLM is not installed. Install with: uv pip install -e '.[local]'"
            ) from e

        from packaging.version import Version

        _vllm_version = Version(_vllm_mod.__version__)
        if _vllm_version < Version("0.17.0"):
            raise RuntimeError(
                f"VLM embedding via vLLM requires vLLM >= 0.17.0, "
                f"but {_vllm_version} is installed. "
                "Update with: uv pip install 'vllm>=0.17.0'"
            )

        if self.device is not None:
            import os

            dev_id = self.device.split(":")[-1] if ":" in self.device else self.device
            os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
        configure_global_hf_cache_base(self.hf_cache_dir)

        model_id = self.model_id or "nvidia/llama-nemotron-embed-vl-1b-v2"
        self._llm = create_vllm_llm(
            str(model_id),
            revision=get_hf_revision(model_id),
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            limit_mm_per_prompt={"image": 1},
        )

    @property
    def is_remote(self) -> bool:
        return False

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed document texts. Returns CPU tensor ``[N, 2048]``."""
        from nemo_retriever.text_embed.vllm import embed_with_vllm_llm

        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 2048), dtype=torch.float32)
        vectors = embed_with_vllm_llm(texts_list, self._llm, batch_size=max(1, int(batch_size)), prefix="passage: ")
        valid = [v for v in vectors if v]
        if not valid:
            return torch.empty((0, 2048), dtype=torch.float32)
        dim = len(valid[0])
        padded = [v if v else [0.0] * dim for v in vectors]
        return _l2_normalize(torch.tensor(padded, dtype=torch.float32))

    def embed_queries(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed query strings. Returns CPU tensor ``[N, 2048]``."""
        from nemo_retriever.text_embed.vllm import embed_with_vllm_llm

        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 2048), dtype=torch.float32)
        vectors = embed_with_vllm_llm(texts_list, self._llm, batch_size=max(1, int(batch_size)), prefix="query: ")
        valid = [v for v in vectors if v]
        if not valid:
            return torch.empty((0, 2048), dtype=torch.float32)
        dim = len(valid[0])
        padded = [v if v else [0.0] * dim for v in vectors]
        return _l2_normalize(torch.tensor(padded, dtype=torch.float32))

    def embed_images(self, images_b64: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed images (base64-encoded). Returns CPU tensor ``[N, 2048]``."""
        from nemo_retriever.text_embed.vllm import embed_multimodal_with_vllm_llm

        valid_b64 = [b64 for b64 in images_b64 if b64 and str(b64).strip()]
        if not valid_b64:
            return torch.empty((0, 2048), dtype=torch.float32)

        prompt_dicts = [
            {"prompt": "passage: <image> ", "multi_modal_data": {"image": _b64_to_pil(b64)}} for b64 in valid_b64
        ]
        vectors = embed_multimodal_with_vllm_llm(prompt_dicts, self._llm, batch_size=max(1, int(batch_size)))
        valid = [v for v in vectors if v]
        if not valid:
            return torch.empty((0, 2048), dtype=torch.float32)
        dim = len(valid[0])
        padded = [v if v else [0.0] * dim for v in vectors]
        return _l2_normalize(torch.tensor(padded, dtype=torch.float32))

    def embed_text_image(
        self, texts: Sequence[str], images_b64: Sequence[str], *, batch_size: int = 64
    ) -> torch.Tensor:
        """Embed paired text+image inputs. Returns CPU tensor ``[N, 2048]``."""
        from nemo_retriever.text_embed.vllm import embed_multimodal_with_vllm_llm

        paired_texts: list = []
        paired_b64: list = []
        for t, b64 in zip(texts, images_b64):
            if b64 and str(b64).strip():
                paired_texts.append(str(t))
                paired_b64.append(b64)

        if not paired_b64:
            return torch.empty((0, 2048), dtype=torch.float32)

        prompt_dicts = [
            {"prompt": f"passage: <image> {text}", "multi_modal_data": {"image": _b64_to_pil(b64)}}
            for text, b64 in zip(paired_texts, paired_b64)
        ]
        vectors = embed_multimodal_with_vllm_llm(prompt_dicts, self._llm, batch_size=max(1, int(batch_size)))
        valid = [v for v in vectors if v]
        if not valid:
            return torch.empty((0, 2048), dtype=torch.float32)
        dim = len(valid[0])
        padded = [v if v else [0.0] * dim for v in vectors]
        return _l2_normalize(torch.tensor(padded, dtype=torch.float32))
