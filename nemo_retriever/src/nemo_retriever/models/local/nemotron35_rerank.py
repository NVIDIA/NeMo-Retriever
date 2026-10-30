# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-native N3.5 reranking with HF and in-process vLLM backends."""

from __future__ import annotations

from collections.abc import Sequence
import math
import os
from typing import Any

from nemo_retriever.models import NEMOTRON_35_RERANK_MODEL
from nemo_retriever.models.hf_cache import configure_global_hf_cache_base
from nemo_retriever.models.hf_model_registry import get_hf_revision
from nemo_retriever.models.inference.image import decode_base64_image
from nemo_retriever.models.model import BaseModel, ModelRunMode


def rerank_messages(query: str, document: str, image: Any = None) -> list[dict[str, Any]]:
    """Build checkpoint-native query/document roles without escaping user text.

    Args:
        query: Unmodified query text.
        document: Passage text, possibly empty for image-only scoring.
        image: Optional decoded PIL image; checkpoint templates handle insertion.

    Returns:
        Two message dictionaries for the checkpoint's processor/chat template.
    """
    content = [{"type": "text", "text": document}]
    if image is not None:
        content.append({"type": "image", "image": image})
    return [
        {"role": "query", "content": [{"type": "text", "text": query}]},
        {"role": "document", "content": content},
    ]


class Nemotron35RerankerHF(BaseModel):
    """Score query/document/image pairs using the trained mean-pooled BF16 head."""

    def __init__(self, *, device: str | None = None, hf_cache_dir: str | None = None) -> None:
        """Load the immutable checkpoint with native Transformers and SDPA.

        Args:
            device: HF inference device; defaults to CUDA when available, else CPU.
            hf_cache_dir: Optional Hugging Face cache shared by all model assets.

        Raises:
            ValueError: If checkpoint attention, pooling, or label count is invalid.
            RuntimeError: If weights do not match exactly or device loading fails.
            OSError, ImportError: If assets or runtime dependencies cannot load.
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoProcessor

        super().__init__()
        self._model: Any = None
        self._processor: Any = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._revision = get_hf_revision(NEMOTRON_35_RERANK_MODEL)
        cache = configure_global_hf_cache_base(hf_cache_dir)
        kwargs = {"revision": self._revision, "trust_remote_code": True, "cache_dir": cache}
        self._processor = AutoProcessor.from_pretrained(NEMOTRON_35_RERANK_MODEL, **kwargs)
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            NEMOTRON_35_RERANK_MODEL,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            output_loading_info=True,
            **kwargs,
        )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Nemotron 3.5 reranker weights did not load exactly: {loading}")
        config = model.config
        if (
            config.model_type != "mistral3"
            or config.text_config.model_type != "ministral3"
            or getattr(config.text_config, "is_causal", None) is not False
            or getattr(config.text_config, "pooling", None) != "avg"
            or config.num_labels != 1
        ):
            raise ValueError("Unsupported N3.5 reranker architecture, attention, or pooling configuration.")
        for module in model.model.language_model.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = False
        self._model = model
        try:
            model.eval().to(self._device)
        except (RuntimeError, ValueError):
            self.unload()
            raise
        self._processor.tokenizer.padding_side = "left"

    @property
    def model_name(self) -> str:
        """Return the canonical checkpoint identity."""
        return NEMOTRON_35_RERANK_MODEL

    @property
    def model_type(self) -> str:
        """Return the multimodal reranker category."""
        return "vl_reranker"

    @property
    def model_runmode(self) -> ModelRunMode:
        """Return the local inference mode."""
        return "local"

    @property
    def input(self) -> str:
        """Return the query/document/optional-image input schema."""
        return "List[Tuple[str, str, Optional[str]]]"

    @property
    def output(self) -> str:
        """Return the raw-logit output schema."""
        return "List[float]"

    @property
    def input_batch_size(self) -> int:
        """Return the conservative default microbatch size."""
        return 8

    def _features(self, conversations: list[list[dict[str, Any]]]) -> Any:
        return self._processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True, "truncation": False, "add_special_tokens": False},
        )

    def _text_length(self, conversation: list[dict[str, Any]]) -> int:
        """Count rendered tokens without decoding or preprocessing the image."""
        prompt = self._processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        return len(self._processor.tokenizer.encode(prompt, add_special_tokens=False))

    def _fit_document(self, query: str, document: str, image: Any, max_length: int) -> list[dict[str, Any]]:
        conversation = rerank_messages(query, document, image)
        expanded_length = int(self._features([conversation])["attention_mask"].sum())
        if expanded_length <= max_length:
            return conversation
        # Image expansion depends on the unchanged image geometry. Measure its
        # token cost once, retaining the image placeholder and exact chat template
        # for text-only search probes. Scoring still checks the fully expanded input.
        image_token_cost = expanded_length - self._text_length(conversation)
        if self._text_length(rerank_messages(query, "", image)) + image_token_cost > max_length:
            raise ValueError("Reranker query and image exceed max_length even without document text.")
        tokenizer = self._processor.tokenizer
        tokens = tokenizer.encode(document, add_special_tokens=False)
        low, high = 0, len(tokens)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = rerank_messages(query, tokenizer.decode(tokens[:middle]), image)
            if self._text_length(candidate) + image_token_cost <= max_length:
                low = middle
            else:
                high = middle - 1
        return rerank_messages(query, tokenizer.decode(tokens[:low]), image)

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        images_b64: Sequence[str | None] | None = None,
        max_length: int = 10240,
        batch_size: int = 8,
    ) -> list[float]:
        """Return relevance logits for one query and aligned candidate documents.

        Args:
            query: Full query text, never truncated by this adapter.
            documents: Candidate passages; empty text permits image-only scoring.
            images_b64: Optional base64 images, one per candidate, or None entries.
            max_length: Expanded input budget from 1 through 10,240 tokens.
            batch_size: Positive microbatch size.

        Returns:
            Finite raw logits in candidate order. Higher means more relevant.

        Raises:
            ValueError: If limits or image alignment are invalid, or the query
                and image cannot fit even after removing all document text.
            RuntimeError: If unloaded, input expansion changes, or inference fails.
            OSError: If checkpoint processing or image decoding fails.
        """
        return self.score_pairs(
            [(query, document) for document in documents],
            images_b64=images_b64,
            max_length=max_length,
            batch_size=batch_size,
        )

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        images_b64: Sequence[str | None] | None = None,
        max_length: int = 10240,
        batch_size: int = 8,
    ) -> list[float]:
        """Score query/passage pairs, retaining every query and full image.

        Args:
            pairs: Query/passage tuples in desired output order.
            images_b64: Optional aligned base64 image strings or None entries.
            max_length: Expanded input budget from 1 through 10,240 tokens.
                Only passage text is shortened when necessary.
            batch_size: Positive microbatch size.

        Returns:
            Finite raw relevance logits aligned with pairs; [] for empty input.

        Raises:
            ValueError: If limits, images, or protected query/image lengths are invalid.
            RuntimeError: If unloaded, expanded lengths drift, or model inference fails.
            OSError: If image decoding or checkpoint processing fails.
        """
        if not isinstance(max_length, int) or not 1 <= max_length <= 10240:
            raise ValueError("max_length must be an integer from 1 through 10240.")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if images_b64 is not None and len(images_b64) != len(pairs):
            raise ValueError("images_b64 must have exactly one entry per query/document pair.")
        if not pairs:
            return []
        if self._model is None or self._processor is None:
            raise RuntimeError("Reranker is unloaded; construct a new instance before scoring.")
        results = []
        for start in range(0, len(pairs), batch_size):
            pictures = []
            try:
                conversations = []
                for offset, (query, document) in enumerate(pairs[start : start + batch_size]):
                    encoded = images_b64[start + offset] if images_b64 is not None else None
                    image = decode_base64_image(encoded) if encoded else None
                    if image is not None:
                        pictures.append(image)
                    conversations.append(self._fit_document(query, document, image, max_length))
                results.extend(self._score_conversations(conversations, max_length))
            finally:
                for picture in pictures:
                    picture.close()
        return results

    def _score_conversations(self, conversations: list[list[dict[str, Any]]], max_length: int) -> list[float]:
        import torch

        inputs = self._features(conversations).to(self._device)
        mask = inputs["attention_mask"]
        if int(mask.sum(-1).max()) > max_length:
            raise RuntimeError("Expanded reranker inputs exceed max_length; refusing silent truncation.")
        inputs["position_ids"] = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
        with torch.inference_mode():
            scores = self._model(**inputs).logits.float().flatten().cpu().tolist()
        self._validate_scores(scores, len(conversations))
        return scores

    @staticmethod
    def _validate_scores(scores: list[float], expected: int) -> None:
        if len(scores) != expected or not all(math.isfinite(score) for score in scores):
            raise RuntimeError("Reranker returned missing or nonfinite relevance scores.")

    def unload(self) -> None:
        """Release owned tensors and CUDA cache; repeated calls are safe.

        Returns:
            None. Further scoring requires construction of a new instance.

        Raises:
            RuntimeError: If moving tensors off the device or cache cleanup fails.
        """
        import torch

        if self._model is not None:
            self._model.to("cpu")
            self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Nemotron35Reranker(Nemotron35RerankerHF):
    """Default local N3.5 reranker using in-process vLLM classification."""

    def __init__(
        self,
        *,
        device: str | None = None,
        hf_cache_dir: str | None = None,
        gpu_memory_utilization: float = 0.5,
    ) -> None:
        """Load BF16 weights with bidirectional attention and the trained head.

        Args:
            device: None, cuda, or cuda:0 in the worker's visible GPU set.
            hf_cache_dir: Optional cache for the pinned checkpoint snapshot.
            gpu_memory_utilization: Fraction of GPU memory available to vLLM.

        Raises:
            ValueError: If device or GPU memory fraction is unsupported.
            RuntimeError: If CUDA is unavailable or engine initialization fails.
            OSError, ImportError: If assets or runtime dependencies cannot load.
        """
        import torch

        BaseModel.__init__(self)
        if device not in (None, "cuda", "cuda:0"):
            raise ValueError("vLLM reranking requires the first visible CUDA device; use backend='hf' otherwise.")
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1].")
        if not torch.cuda.is_available():
            raise RuntimeError("Local N3.5 vLLM reranking requires CUDA; select backend='hf' for CPU inference.")
        self._model = None
        self._processor = None
        self._revision = get_hf_revision(NEMOTRON_35_RERANK_MODEL)
        cache = configure_global_hf_cache_base(hf_cache_dir)
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor
        from nemo_retriever.models.inference.nemotron35_rerank_vllm import register_nemotron35_rerank_vllm
        from nemo_retriever.models.inference.vllm import apply_vllm_startup_defaults

        self._processor = AutoProcessor.from_pretrained(
            NEMOTRON_35_RERANK_MODEL,
            revision=self._revision,
            trust_remote_code=True,
            cache_dir=cache,
        )
        snapshot = snapshot_download(NEMOTRON_35_RERANK_MODEL, revision=self._revision, cache_dir=cache)
        apply_vllm_startup_defaults()
        previous_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
        try:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            register_nemotron35_rerank_vllm()
            from vllm import LLM

            self._model = LLM(
                model=snapshot,
                runner="pooling",
                max_model_len=10240,
                trust_remote_code=True,
                dtype="bfloat16",
                enforce_eager=True,
                gpu_memory_utilization=gpu_memory_utilization,
                download_dir=cache,
                hf_overrides={"architectures": ["NRLNemotron35RerankModel"]},
                pooler_config={"seq_pooling_type": "MEAN", "use_activation": False},
                limit_mm_per_prompt={"image": 1},
            )
        finally:
            if previous_method is None:
                os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
            else:
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = previous_method

    def _score_conversations(self, conversations: list[list[dict[str, Any]]], max_length: int) -> list[float]:
        requests, lengths = [], []
        for conversation in conversations:
            lengths.append(int(self._features([conversation])["attention_mask"].sum()))
            request = {
                "prompt": self._processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                ),
                "mm_processor_kwargs": {"text_kwargs": {"truncation": False, "add_special_tokens": False}},
            }
            for item in conversation[1]["content"]:
                if item["type"] == "image":
                    request["multi_modal_data"] = {"image": item["image"]}
            requests.append(request)
        if any(length > max_length for length in lengths):
            raise RuntimeError("Expanded reranker input exceeds max_length.")
        outputs = self._model.encode(requests, pooling_task="classify", use_tqdm=False)
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM reranker returned missing relevance scores.")
        scores = []
        for output, expected in zip(outputs, lengths):
            if len(output.prompt_token_ids) != expected:
                raise RuntimeError("vLLM reranker changed expanded input length; refusing silent truncation.")
            scores.append(float(output.outputs.data.item()))
        self._validate_scores(scores, len(requests))
        return scores

    def unload(self) -> None:
        """Shut down owned vLLM workers before clearing model references.

        Returns:
            None. Repeated calls are safe; construct a new instance to reload.

        Raises:
            RuntimeError: If engine shutdown or CUDA cleanup fails. References
                remain available if shutdown fails, so cleanup can be retried.
        """
        import torch

        if self._model is not None:
            self._model.llm_engine.engine_core.shutdown(timeout=30.0)
            self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
