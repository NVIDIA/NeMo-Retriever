# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VLM-capable LLM generation client for multimodal RAG.

LiteVLMClient extends LiteLLMClient with generate_multimodal(), which
accepts MultimodalChunk objects. Chunks that carry a stored_image_uri have
their image loaded, base64-encoded, and injected into the OpenAI vision
content array alongside the text caption/description.

Supported URI schemes for image_uri:
- Local paths  : /abs/path/to/image.png  or  file:///abs/path/to/image.png
- HTTP(S) URLs : https://host/path/image.png
- S3 URIs      : s3://bucket/key  (requires boto3; skipped with a warning if absent)
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Optional

from nemo_retriever.models.llm.clients.litellm import (
    LiteLLMClient,
    _format_rag_system_prompt,
    _with_no_reasoning_controls,
    _NO_REASONING_EXTRA_PARAMS,
)
from nemo_retriever.models.llm.text_utils import strip_think_tags
from nemo_retriever.models.llm.types import GenerationResult, MultimodalChunk
from nemo_retriever.common.params.models import LLMInferenceParams, LLMRemoteClientParams

logger = logging.getLogger(__name__)

_VISUAL_CONTENT_TYPES = frozenset({"image", "chart", "infographic", "table"})

_VLM_RAG_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the information provided in the context below, "
    "which may include text passages and images. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Be concise and factual."
)


def _load_image_as_base64(uri: str) -> Optional[str]:
    """Load an image from a local path, file://, http(s)://, or s3:// URI.

    Returns the raw base64 string (no data-URI prefix), or None if loading
    fails so that callers can gracefully degrade to text-only.
    """
    try:
        if uri.startswith("s3://"):
            try:
                import boto3

                parts = uri[5:].split("/", 1)
                bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
                s3 = boto3.client("s3")
                response = s3.get_object(Bucket=bucket, Key=key)
                data = response["Body"].read()
            except ImportError:
                logger.warning("boto3 not installed; skipping image at %s", uri)
                return None
        elif uri.startswith("http://") or uri.startswith("https://"):
            import urllib.request

            with urllib.request.urlopen(uri, timeout=10) as resp:  # noqa: S310
                data = resp.read()
        else:
            # Local path — strip optional file:// scheme
            path = uri.removeprefix("file://")
            with open(path, "rb") as fh:
                data = fh.read()

        return base64.b64encode(data).decode("ascii")
    except Exception as exc:
        logger.warning("Failed to load image %s: %s", uri, exc)
        return None


def _build_multimodal_rag_prompt(
    query: str,
    chunks: list[MultimodalChunk],
    *,
    formatted_rag_system_prompt: str,
) -> list[dict]:
    """Build an OpenAI-style vision messages list for multimodal RAG.

    Each chunk becomes one or more content blocks in the user message:
    - Text-only chunks: a single ``{"type": "text", ...}`` block.
    - Visual chunks: a text block (caption/description) followed by an
      ``{"type": "image_url", ...}`` block carrying the base64 data-URI.
      If the image cannot be loaded the chunk falls back to text-only.
    """
    user_content: list[dict[str, Any]] = []

    if not chunks:
        user_content.append({"type": "text", "text": "(no context retrieved)"})
    else:
        user_content.append({"type": "text", "text": "Context:\n"})
        for i, chunk in enumerate(chunks):
            label = f"[{i + 1}] ({chunk.content_type})"
            if chunk.image_uri and chunk.content_type in _VISUAL_CONTENT_TYPES:
                b64 = _load_image_as_base64(chunk.image_uri)
                if b64:
                    if chunk.text:
                        user_content.append({"type": "text", "text": f"{label} {chunk.text}\n"})
                    else:
                        user_content.append({"type": "text", "text": f"{label}\n"})
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )
                else:
                    # Image load failed — fall back to caption text only
                    user_content.append({"type": "text", "text": f"{label} {chunk.text}\n"})
            else:
                user_content.append({"type": "text", "text": f"{label} {chunk.text}\n"})
            if i < len(chunks) - 1:
                user_content.append({"type": "text", "text": "\n---\n\n"})

    user_content.append({"type": "text", "text": f"\nQuestion: {query}\n\nAnswer:"})

    return [
        {"role": "system", "content": formatted_rag_system_prompt},
        {"role": "user", "content": user_content},
    ]


class LiteVLMClient(LiteLLMClient):
    """LiteLLM client extended with multimodal (vision) generation.

    Inherits all text-only generation from LiteLLMClient and adds
    generate_multimodal(), which accepts list[MultimodalChunk] and wires
    images into the OpenAI vision content array.

    Usage::

        client = LiteVLMClient.from_kwargs(
            model="openai/gpt-4o",
            api_base="https://integrate.api.nvidia.com/v1",
            api_key="nvapi-...",
        )
        result = retriever.answer_multimodal(query, llm=client)
    """

    _DEFAULT_MODEL: str = "openai/meta/llama-4-scout-17b-16e-instruct"

    def __init__(
        self,
        transport: LLMRemoteClientParams,
        sampling: Optional[LLMInferenceParams] = None,
    ):
        super().__init__(transport=transport, sampling=sampling)
        # Use VLM-specific system prompt when no custom prompt was configured
        if not transport.rag_system_prompt and not transport.rag_system_prompt_prefix:
            self._formatted_rag_system_prompt = _format_rag_system_prompt(
                rag_system_prompt=_VLM_RAG_SYSTEM_PROMPT,
            )

    def generate_multimodal(
        self,
        query: str,
        chunks: list[MultimodalChunk],
        *,
        reasoning_enabled: Optional[bool] = None,
    ) -> GenerationResult:
        """Generate an answer using both text and image context from chunks."""
        messages = _build_multimodal_rag_prompt(
            query,
            chunks,
            formatted_rag_system_prompt=self._formatted_rag_system_prompt,
        )
        request_extra_params: dict[str, Any] | None = None
        effective_reasoning_enabled = (
            self.transport.reasoning_enabled if reasoning_enabled is None else reasoning_enabled
        )
        if not effective_reasoning_enabled:
            messages = _with_no_reasoning_controls(messages)
            request_extra_params = _NO_REASONING_EXTRA_PARAMS
        try:
            raw_answer, latency = self.complete(messages, extra_params=request_extra_params)
            answer = strip_think_tags(raw_answer)
            if not answer:
                return GenerationResult(
                    answer="",
                    latency_s=latency,
                    model=self.transport.model,
                    error="thinking_truncated",
                )
            return GenerationResult(answer=answer, latency_s=latency, model=self.transport.model)
        except Exception as exc:
            logger.debug("VLM generation failed for model=%s: %s", self.transport.model, exc)
            return GenerationResult(
                answer="",
                latency_s=0.0,
                model=self.transport.model,
                error=str(exc),
            )
