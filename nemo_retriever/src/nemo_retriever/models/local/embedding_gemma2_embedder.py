# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text and single-image document embeddings with native vLLM EmbeddingGemma2."""

import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from nemo_retriever.models.embed_model_spec import resolve_embed_model_revision
from nemo_retriever.models.hf_cache import configure_global_hf_cache_base
from nemo_retriever.models.inference.image import decode_base64_image
from nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder


@dataclass
class EmbeddingGemma2Embedder(LlamaNemotronEmbed1BV2Embedder):
    output_dimension: int = 768
    query_max_length: int = 128
    _processor: Any = field(default=None, init=False, repr=False)
    _processor_lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dimensions not in {None, self.output_dimension}:
            raise ValueError("This checkpoint profile supports only its full embedding dimension.")
        if not 0 < self.max_length <= 8192:
            raise ValueError("EmbeddingGemma 2 max_length must be between 1 and 8192.")

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from nemo_retriever.models.inference.vllm import create_vllm_llm

        configure_global_hf_cache_base(self.hf_cache_dir)
        self._llm = create_vllm_llm(
            self.model_id,
            revision=resolve_embed_model_revision(self.model_id, self.revision),
            dimensions=self.dimensions,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            max_model_len=self.max_length,
            limit_mm_per_prompt={"image": 1, "audio": 0, "video": 0},
            attention_backend="TRITON_ATTN",
        )

    def _ensure_processor(self) -> None:
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                revision=resolve_embed_model_revision(self.model_id, self.revision),
                cache_dir=self.hf_cache_dir,
            )

    def _image_prompt(self, text: str) -> str:
        return f"{self.document_prefix}{text + ' ' if text else ''}<|image|>"

    def _image_prompt_length(self, prompt: str, image: Any) -> int:
        self._ensure_processor()
        batch = self._processor(text=[prompt], images=[image], truncation=False, return_tensors="pt")
        return int(batch["attention_mask"].sum())

    def plan_text_image(self, text: str, image_b64: str, *, query: bool = False):
        """Plan lossless text splitting with the image repeated in every child."""
        from nemo_retriever.models.inference.embedding_input import EmbeddingInputPolicy

        if query:
            raise ValueError("EmbeddingGemma 2 queries must be text-only.")
        image = decode_base64_image(image_b64)
        try:
            with self._processor_lock:
                self._ensure_processor()
                policy = EmbeddingInputPolicy(
                    tokenizer=self._processor.tokenizer,
                    max_tokens=self.max_length,
                    prefix="",
                    formatted_token_counter=lambda text: self._image_prompt_length(self._image_prompt(text), image),
                )
                if policy._formatted_token_count("") >= self.max_length:
                    raise ValueError("Image and document prefix leave no text budget within max_length.")
                return policy.plan([text])[0]
        finally:
            image.close()

    def embed_images(self, images_b64: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed one full image per document."""
        return self.embed_text_image([""] * len(images_b64), images_b64, batch_size=batch_size)

    def embed_text_image(
        self, texts: Sequence[str], images_b64: Sequence[str], *, batch_size: int = 64
    ) -> torch.Tensor:
        """Embed aligned document text/image pairs without truncating either."""
        from nemo_retriever.models.inference.vllm import embed_multimodal_with_vllm_llm

        if len(texts) != len(images_b64):
            raise ValueError("Text and image counts must match.")
        if not texts:
            return torch.empty((0, self.output_dimension), dtype=torch.float32)
        images = []
        try:
            requests = []
            for text, image_b64 in zip(texts, images_b64):
                image = decode_base64_image(image_b64)
                images.append(image)
                prompt = self._image_prompt(str(text))
                with self._processor_lock:
                    if self._image_prompt_length(prompt, image) > self.max_length:
                        raise ValueError("Text+image embedding exceeds max_length after image expansion.")
                requests.append({"prompt": prompt, "multi_modal_data": {"image": image}})
            self._ensure_loaded()
            vectors = embed_multimodal_with_vllm_llm(
                requests, self._llm, batch_size=max(1, int(batch_size)), normalize=self.normalize
            )
            if len(vectors) != len(requests):
                raise RuntimeError("vLLM did not return one embedding per image document.")
            return self._finalize_vectors(vectors)
        finally:
            for image in images:
                image.close()
