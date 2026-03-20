# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from nemo_retriever.params import CaptionParams

_MAX_CONTEXT_TEXT_CHARS = 4096


class CaptionActor:
    """Ray Data actor that holds a local VLM captioner on a single GPU.

    When ``endpoint_url`` is provided, the actor delegates to a remote VLM
    endpoint and no local model is loaded.
    """

    def __init__(self, params: CaptionParams) -> None:
        self._params = params
        self._kwargs = params.model_dump(mode="python")
        endpoint = (self._kwargs.get("endpoint_url") or "").strip()
        if endpoint:
            self._model = None
        else:
            from nemo_retriever.model.local import NemotronVLMCaptioner

            self._model = NemotronVLMCaptioner(
                model_path=self._kwargs.get("model_name", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"),
                device=self._kwargs.get("device"),
                hf_cache_dir=self._kwargs.get("hf_cache_dir"),
                tensor_parallel_size=self._kwargs.get("tensor_parallel_size", 1),
                gpu_memory_utilization=self._kwargs.get("gpu_memory_utilization", 0.9),
            )

    def __call__(self, batch_df: Any) -> Any:
        return caption_images(batch_df, model=self._model, **self._kwargs)


def _build_prompt_with_context(base_prompt: str, context_text: str) -> str:
    """Prepend surrounding page text to the base VLM prompt.

    If *context_text* is empty the *base_prompt* is returned unchanged.
    """
    if not context_text:
        return base_prompt
    return f"Text near this image:\n---\n{context_text}\n---\n\n{base_prompt}"


def _caption_batch_remote(
    base64_images: List[str],
    *,
    endpoint_url: str,
    model_name: str,
    api_key: str | None,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
) -> List[str]:
    """Send a batch of images to a remote VLM endpoint and return captions."""
    from nv_ingest_api.internal.primitives.nim.model_interface.vlm import VLMModelInterface
    from nv_ingest_api.util.nim import create_inference_client
    from nv_ingest_api.util.image_processing.transforms import scale_image_to_encoding_size

    scaled = [scale_image_to_encoding_size(b64)[0] for b64 in base64_images]

    data: Dict[str, Any] = {
        "base64_images": scaled,
        "prompt": prompt,
    }
    if system_prompt:
        data["system_prompt"] = system_prompt

    nim_client = create_inference_client(
        model_interface=VLMModelInterface(),
        endpoints=(None, endpoint_url),
        auth_token=api_key,
        infer_protocol="http",
    )
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
    endpoint_url: str | None,
    model_name: str,
    api_key: str | None,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
) -> str:
    """Caption a single image (used when each image gets a unique prompt)."""
    if model is not None:
        captions = _caption_batch_local(
            [b64], model=model, prompt=prompt,
            system_prompt=system_prompt, temperature=temperature,
        )
    else:
        captions = _caption_batch_remote(
            [b64], endpoint_url=endpoint_url,  # type: ignore[arg-type]
            model_name=model_name, api_key=api_key, prompt=prompt,
            system_prompt=system_prompt, temperature=temperature,
        )
    return captions[0] if captions else ""


def caption_images(
    batch_df: pd.DataFrame,
    *,
    model: Any = None,
    endpoint_url: str | None = None,
    model_name: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
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
        # Lazy model creation for the sequential (no GPU pool) fallback.
        from nemo_retriever.model.local import NemotronVLMCaptioner

        model = NemotronVLMCaptioner(
            model_path=kwargs.get("model_name", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"),
            device=kwargs.get("device"),
            hf_cache_dir=kwargs.get("hf_cache_dir"),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
        )

    use_context = context_text_max_chars > 0
    effective_max = min(context_text_max_chars, _MAX_CONTEXT_TEXT_CHARS) if use_context else 0

    # Collect all (row_idx, item_idx, image_b64) needing captions.
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
            if b64:
                pending.append((row_idx, item_idx, b64))

    if not pending:
        return batch_df

    if use_context:
        # Each image gets a per-page enriched prompt, so caption one at a time.
        for row_idx, item_idx, b64 in pending:
            page_text = batch_df.at[row_idx, "text"] if "text" in batch_df.columns else ""
            context = (page_text or "")[:effective_max]
            enriched_prompt = _build_prompt_with_context(prompt, context)
            caption = _caption_one(
                b64, model=model, endpoint_url=endpoint_url,
                model_name=model_name, api_key=api_key,
                prompt=enriched_prompt, system_prompt=system_prompt,
                temperature=temperature,
            )
            batch_df.at[row_idx, "images"][item_idx]["text"] = caption
    else:
        # Batch mode: all images share the same prompt.
        all_captions: List[str] = []
        for start in range(0, len(pending), batch_size):
            chunk_b64 = [b64 for _, _, b64 in pending[start : start + batch_size]]

            if model is not None:
                captions = _caption_batch_local(
                    chunk_b64, model=model, prompt=prompt,
                    system_prompt=system_prompt, temperature=temperature,
                )
            else:
                captions = _caption_batch_remote(
                    chunk_b64, endpoint_url=endpoint_url,  # type: ignore[arg-type]
                    model_name=model_name, api_key=api_key, prompt=prompt,
                    system_prompt=system_prompt, temperature=temperature,
                )
            all_captions.extend(captions)

        for (row_idx, item_idx, _), caption in zip(pending, all_captions):
            batch_df.at[row_idx, "images"][item_idx]["text"] = caption

    return batch_df
