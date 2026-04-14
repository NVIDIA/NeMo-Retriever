# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local wrapper for nvidia/llama-nemotron-rerank-vl-1b-v2 VL cross-encoder reranker."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from nemo_retriever.utils.hf_model_registry import get_hf_revision
from ..model import BaseModel, RunMode


from nemo_retriever.model import VL_RERANK_MODEL

_DEFAULT_MODEL = VL_RERANK_MODEL
_DEFAULT_MAX_LENGTH = 10240
_DEFAULT_BATCH_SIZE = 1


class NemotronRerankVLV2(BaseModel):
    """
    Local VL cross-encoder reranker wrapping nvidia/llama-nemotron-rerank-vl-1b-v2.

    Scores (query, document, image) triplets and returns raw logits; higher
    values indicate greater relevance.  When an image is ``None`` for a given
    document, the model falls back to text-only scoring for that pair.

    Unlike the text-only :class:`NemotronRerankV2` which uses
    ``AutoTokenizer`` and a manual prompt template, this model uses
    ``AutoProcessor`` with ``process_queries_documents_crossencoder()``
    to handle vision token insertion.

    Example::

        reranker = NemotronRerankVLV2()
        scores = reranker.score(
            "What is ML?",
            ["Machine learning is…", "Paris is…"],
            images_b64=["iVBOR...", None],
        )
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoProcessor

        configure_global_hf_cache_base()

        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if hf_cache_dir:
            kwargs["cache_dir"] = hf_cache_dir

        revision = get_hf_revision(model_name, strict=False)
        if revision:
            kwargs["revision"] = revision

        self._processor = AutoProcessor.from_pretrained(
            model_name,
            use_thumbnail=True,
            **kwargs,
        )
        if hasattr(self._processor, "max_input_tiles"):
            self._processor.max_input_tiles = 6
        if hasattr(self._processor, "rerank_max_length"):
            self._processor.rerank_max_length = 10240

        self._model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                **kwargs,
            )
            .eval()
            .to(self._device)
        )

        if self._model.config.pad_token_id is None:
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            self._model.config.pad_token_id = getattr(tokenizer, "eos_token_id", 0)

    def unload(self) -> None:
        """Release GPU memory held by the model and processor."""
        import torch

        del self._model
        del self._processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # BaseModel abstract properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_type(self) -> str:
        return "vl_reranker"

    @property
    def model_runmode(self) -> RunMode:
        return "local"

    @property
    def input(self):
        return "List[Tuple[str, str, Optional[str]]]"

    @property
    def output(self):
        return "List[float]"

    @property
    def input_batch_size(self) -> int:
        return _DEFAULT_BATCH_SIZE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_images(
        images_b64: Optional[Sequence[Optional[str]]],
        n: int,
    ) -> list[Optional[dict[str, str]]]:
        """Normalise *images_b64* into a list of ``{"base64": ...}`` dicts."""
        if images_b64 is None:
            return [None] * n
        out: list[Optional[dict[str, str]]] = []
        for img in images_b64:
            if isinstance(img, str) and img:
                out.append({"base64": img})
            else:
                out.append(None)
        # Pad to length n if caller passed a shorter list.
        while len(out) < n:
            out.append(None)
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        query: str,
        documents: List[str],
        *,
        images_b64: Optional[Sequence[Optional[str]]] = None,
        max_length: int = _DEFAULT_MAX_LENGTH,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> List[float]:
        """
        Score relevance of *documents* (with optional images) to *query*.

        Parameters
        ----------
        query:
            The search query.
        documents:
            Candidate passages/documents to score.
        images_b64:
            Optional base64-encoded images aligned with *documents*.  Entries
            may be ``None`` for documents without images (text-only fallback).
        max_length:
            Processor truncation length.
        batch_size:
            Number of triplets to process per GPU forward pass.

        Returns
        -------
        List[float]
            Raw logit scores aligned with *documents* (higher = more relevant).
        """
        import torch

        if not documents:
            return []

        image_dicts = self._prepare_images(images_b64, len(documents))
        all_scores: List[float] = []

        if hasattr(self._processor, "rerank_max_length"):
            self._processor.rerank_max_length = max_length

        with torch.inference_mode():
            for start in range(0, len(documents), batch_size):
                end = start + batch_size
                batch_docs = documents[start:end]
                batch_images = image_dicts[start:end]

                features = [
                    {
                        "question": query,
                        "doc_text": doc,
                        "doc_image": img,
                    }
                    for doc, img in zip(batch_docs, batch_images)
                ]
                inputs = self._processor.process_queries_documents_crossencoder(features)
                inputs = {k: v.to(self._device) for k, v in inputs.items() if hasattr(v, "to")}
                logits = self._model(**inputs).logits
                all_scores.extend(logits.view(-1).cpu().tolist())

        return all_scores

    def score_pairs(
        self,
        pairs: List[tuple],
        *,
        images_b64: Optional[Sequence[Optional[str]]] = None,
        max_length: int = _DEFAULT_MAX_LENGTH,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> List[float]:
        """
        Score a list of (query, document) pairs with optional images.

        Parameters
        ----------
        pairs:
            Sequence of ``(query, document)`` tuples.
        images_b64:
            Optional base64-encoded images aligned with *pairs*.
        max_length:
            Processor truncation length.
        batch_size:
            GPU forward-pass batch size.

        Returns
        -------
        List[float]
            Raw logit scores (higher = more relevant).
        """
        import torch

        if not pairs:
            return []

        image_dicts = self._prepare_images(images_b64, len(pairs))
        all_scores: List[float] = []

        if hasattr(self._processor, "rerank_max_length"):
            self._processor.rerank_max_length = max_length

        with torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                end = start + batch_size
                batch_pairs = pairs[start:end]
                batch_images = image_dicts[start:end]

                features = [
                    {
                        "question": q,
                        "doc_text": d,
                        "doc_image": img,
                    }
                    for (q, d), img in zip(batch_pairs, batch_images)
                ]
                inputs = self._processor.process_queries_documents_crossencoder(features)
                inputs = {k: v.to(self._device) for k, v in inputs.items() if hasattr(v, "to")}
                logits = self._model(**inputs).logits
                all_scores.extend(logits.view(-1).cpu().tolist())

        return all_scores
