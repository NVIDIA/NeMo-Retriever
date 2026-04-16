# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-backed local wrapper for nvidia/llama-nemotron-rerank-vl-1b-v2 VL cross-encoder reranker."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from nemo_retriever.utils.hf_model_registry import get_hf_revision
from ..model import BaseModel, RunMode

from nemo_retriever.model import VL_RERANK_MODEL

_DEFAULT_MODEL = VL_RERANK_MODEL
_DEFAULT_MAX_LENGTH = 10240
_DEFAULT_BATCH_SIZE = 32

# Jinja2 chat template for vLLM's ``llm.score()`` API.
# The model expects ``question:`` / ``passage:`` labels with an optional
# leading ``<image>`` token when the document contains an image.
SCORE_TEMPLATE = """\
{%- set query_msg = (messages | selectattr('role', 'equalto', 'query') | list | first) -%}
{%- set doc_msg   = (messages | selectattr('role', 'equalto', 'document') | list | first) -%}
{%- set q = query_msg['content'] -%}
{%- set d = doc_msg['content'] -%}
{%- set has_image = ("<image>" in d) -%}
{%- set d_clean = d | replace("<image>", "") -%}
{%- set q_clean = q | replace("<image>", "") -%}
{%- if has_image -%}<image>{{ " " }}{%- endif -%}
question:{{ q_clean }}{{ " " }}
{{ " " }}
{{ " " }}passage:{{ d_clean }}"""


class NemotronRerankVLV2VLLM(BaseModel):
    """
    vLLM-backed VL cross-encoder reranker wrapping nvidia/llama-nemotron-rerank-vl-1b-v2.

    Uses vLLM's pooling runner (``llm.score()``) instead of HuggingFace
    ``AutoModelForSequenceClassification``.  This provides better throughput
    through continuous batching and optimised attention kernels.

    The public API (``score()``, ``score_pairs()``) is identical to
    :class:`NemotronRerankVLV2` so callers need not change.

    Example::

        reranker = NemotronRerankVLV2VLLM()
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
        gpu_memory_utilization: float = 0.5,
    ) -> None:
        super().__init__()

        try:
            from vllm import LLM  # noqa: F401
        except ImportError as e:
            raise ImportError("vLLM reranker backend requires vllm. " 'Install with: pip install "vllm>=0.17"') from e

        self._model_name = model_name

        configure_global_hf_cache_base(hf_cache_dir)

        if device is not None:
            dev_id = device.split(":")[-1] if ":" in device else device
            os.environ["CUDA_VISIBLE_DEVICES"] = dev_id

        revision = get_hf_revision(model_name, strict=False)

        engine_kwargs: dict[str, Any] = {}
        if revision:
            engine_kwargs["revision"] = revision

        self._llm = LLM(
            model=model_name,
            runner="pooling",
            max_model_len=_DEFAULT_MAX_LENGTH,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            **engine_kwargs,
        )

    def unload(self) -> None:
        """Release GPU memory held by the vLLM engine."""
        import torch

        del self._llm
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
    def _build_document(text: str, image_b64: Optional[str] = None) -> Any:
        """Build a vLLM-compatible document representation.

        For text-only documents, returns the plain string.  For multimodal
        documents (text + image), returns a dict with a ``content`` list in
        OpenAI-style format that vLLM understands.
        """
        if not image_b64:
            return text

        content: list[dict[str, Any]] = []
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )
        if text:
            content.append({"type": "text", "text": text})
        return {"content": content}

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
            Kept for API compatibility with the transformers variant.
            vLLM truncates prompts to ``max_model_len`` set at engine init
            via ``truncate_prompt_tokens``.
        batch_size:
            Unused (kept for API compatibility). vLLM handles batching
            internally via continuous batching.

        Returns
        -------
        List[float]
            Raw logit scores aligned with *documents* (higher = more relevant).
        """
        if not documents:
            return []

        doc_inputs = []
        for i, doc in enumerate(documents):
            img = None
            if images_b64 is not None and i < len(images_b64):
                img = images_b64[i]
            doc_inputs.append(self._build_document(doc, img))

        outputs = self._llm.score(
            query,
            doc_inputs,
            chat_template=SCORE_TEMPLATE,
            tokenization_kwargs={"truncate_prompt_tokens": -1},
        )
        return [out.outputs.score for out in outputs]

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
            Kept for API compatibility. vLLM truncates prompts to
            ``max_model_len`` via ``truncate_prompt_tokens``.
        batch_size:
            Unused (API compatibility).

        Returns
        -------
        List[float]
            Raw logit scores (higher = more relevant).
        """
        if not pairs:
            return []

        all_scores: List[float] = []
        for i, (q, d) in enumerate(pairs):
            img = None
            if images_b64 is not None and i < len(images_b64):
                img = images_b64[i]
            doc_input = self._build_document(d, img)
            outputs = self._llm.score(
                q,
                [doc_input],
                chat_template=SCORE_TEMPLATE,
                tokenization_kwargs={"truncate_prompt_tokens": -1},
            )
            all_scores.append(outputs[0].outputs.score)

        return all_scores
