# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, List, Tuple

import pandas as pd
from PIL import Image

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.params import CaptionParams

_DEFAULT_MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
_MAX_CONTEXT_TEXT_CHARS = 4096
_MIN_IMAGE_DIMENSION = 32
_cached_local_model = None


def _image_meets_min_size(b64: str) -> bool:
    """Return True if the base64 image is at least _MIN_IMAGE_DIMENSION on both sides."""
    try:
        img = Image.open(BytesIO(base64.b64decode(b64)))
        w, h = img.size
        return w >= _MIN_IMAGE_DIMENSION and h >= _MIN_IMAGE_DIMENSION
    except Exception:
        return False


def _create_local_model(kwargs: dict) -> "Any":
    from nemo_retriever.model.local import NemotronVLMCaptioner

    return NemotronVLMCaptioner(
        model_path=kwargs.get("model_name", _DEFAULT_MODEL_NAME),
        device=kwargs.get("device"),
        hf_cache_dir=kwargs.get("hf_cache_dir"),
        tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
        gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.5),
    )


def _get_cached_local_model(kwargs: dict) -> "Any":
    global _cached_local_model
    if _cached_local_model is None:
        _cached_local_model = _create_local_model(kwargs)
    return _cached_local_model


class CaptionActor(AbstractOperator, GPUOperator):
    """Ray Data actor that holds a local VLM captioner on a single GPU.

    When ``endpoint_url`` is provided, the actor delegates to a remote VLM
    endpoint and no local model is loaded.
    """

    def __init__(self, params: CaptionParams) -> None:
        super().__init__(params=params)
        self._params = params
        self._kwargs = params.model_dump(mode="python")
        endpoint = (self._kwargs.get("endpoint_url") or "").strip()
        if endpoint:
            self._model = None
        else:
            self._model = _create_local_model(self._kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        return caption_images(batch_df, model=self._model, **self._kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def _build_prompt_with_context(base_prompt: str, context_text: str) -> str:
    """Prepend surrounding page text to the base VLM prompt.

    If *context_text* is empty the *base_prompt* is returned unchanged.
    """
    if not context_text:
        return base_prompt
    return f"Text near this image:\n---\n{context_text}\n---\n\n{base_prompt}"


def _create_remote_client(endpoint_url: str, api_key: str | None) -> Any:
    """Create a reusable NIM inference client for a remote VLM endpoint."""
    from nv_ingest_api.internal.primitives.nim.model_interface.vlm import VLMModelInterface
    from nv_ingest_api.util.nim import create_inference_client

    return create_inference_client(
        model_interface=VLMModelInterface(),
        endpoints=(None, endpoint_url),
        auth_token=api_key,
        infer_protocol="http",
    )


def _caption_batch_remote(
    base64_images: List[str],
    *,
    nim_client: Any,
    model_name: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
) -> List[str]:
    """Send a batch of images to a remote VLM endpoint and return captions."""
    from nv_ingest_api.util.image_processing.transforms import scale_image_to_encoding_size

    scaled = [scale_image_to_encoding_size(b64)[0] for b64 in base64_images]

    data: Dict[str, Any] = {
        "base64_images": scaled,
        "prompt": prompt,
    }
    if system_prompt:
        data["system_prompt"] = system_prompt

    return nim_client.infer(data, model_name=model_name, temperature=temperature)


def _caption_batch_local(
    base64_images: List[str],
    *,
    model: Any,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
) -> List[str]:
    """Generate captions using a local ``NemotronVLMCaptioner`` model."""
    return model.caption_batch(
        base64_images,
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
    )


def _caption_one(
    b64: str,
    *,
    model: Any,
    nim_client: Any | None,
    model_name: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
) -> str:
    """Caption a single image (used when each image gets a unique prompt)."""
    if model is not None:
        captions = _caption_batch_local(
            [b64],
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    else:
        captions = _caption_batch_remote(
            [b64],
            nim_client=nim_client,
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    return captions[0] if captions else ""


def caption_images(
    batch_df: pd.DataFrame,
    *,
    model: Any = None,
    endpoint_url: str | None = None,
    model_name: str = _DEFAULT_MODEL_NAME,
    api_key: str | None = None,
    prompt: str = "Caption the content of this image:",
    system_prompt: str | None = "/no_think",
    temperature: float = 1.0,
    batch_size: int = 8,
    context_text_max_chars: int = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Caption images in the ``images`` column using a VLM.

    Supports two modes:

    * **Remote** (``endpoint_url`` is set): sends images to an HTTP VLM
      endpoint via ``create_inference_client`` / ``VLMModelInterface``.
    * **Local** (``model`` is set): runs inference through a local
      ``NemotronVLMCaptioner`` instance loaded from Hugging Face.

    When ``context_text_max_chars`` is greater than zero, the page's ``text``
    column is prepended to the prompt for each image so the VLM can use
    surrounding OCR text as context.  In this mode images are captioned
    one at a time (each gets its own enriched prompt).

    For each row, any item in the ``images`` list whose ``text`` field is
    empty will be captioned.  The returned caption is written back into
    ``images[i]["text"]``.
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df
    if "images" not in batch_df.columns:
        return batch_df

    if model is None and not endpoint_url:
        model = _get_cached_local_model(kwargs)

    nim_client = _create_remote_client(endpoint_url, api_key) if endpoint_url and model is None else None

    use_context = context_text_max_chars > 0
    effective_max = min(context_text_max_chars, _MAX_CONTEXT_TEXT_CHARS) if use_context else 0

    pending: List[Tuple[int, int, str]] = []
    for row_idx, row in batch_df.iterrows():
        images = row.get("images")
        if not isinstance(images, list):
            continue
        for item_idx, item in enumerate(images):
            if not isinstance(item, dict):
                continue
            if item.get("text"):
                continue  # already captioned
            b64 = item.get("image_b64")
            if b64 and _image_meets_min_size(b64):
                pending.append((row_idx, item_idx, b64))

    if not pending:
        return batch_df

    if use_context:
        for row_idx, item_idx, b64 in pending:
            page_text = batch_df.at[row_idx, "text"] if "text" in batch_df.columns else ""
            context = (page_text or "")[:effective_max]
            enriched_prompt = _build_prompt_with_context(prompt, context)
            caption = _caption_one(
                b64,
                model=model,
                nim_client=nim_client,
                model_name=model_name,
                prompt=enriched_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            batch_df.at[row_idx, "images"][item_idx]["text"] = caption
    else:
        all_b64 = [b64 for _, _, b64 in pending]

        if model is not None:
            all_captions = _caption_batch_local(
                all_b64,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        else:
            all_captions: List[str] = []
            for start in range(0, len(all_b64), batch_size):
                captions = _caption_batch_remote(
                    all_b64[start : start + batch_size],
                    nim_client=nim_client,
                    model_name=model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )
                all_captions.extend(captions)

        for (row_idx, item_idx, _), caption in zip(pending, all_captions):
            batch_df.at[row_idx, "images"][item_idx]["text"] = caption

    return batch_df
