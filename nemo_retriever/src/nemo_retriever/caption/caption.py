# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


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
    **kwargs: Any,
) -> pd.DataFrame:
    """Caption images in the ``images`` column using a VLM.

    Supports two modes:

    * **Remote** (``endpoint_url`` is set): sends images to an HTTP VLM
      endpoint via ``create_inference_client`` / ``VLMModelInterface``.
    * **Local** (``model`` is set): runs inference through a local
      ``NemotronVLMCaptioner`` instance loaded from Hugging Face.

    For each row, any item in the ``images`` list whose ``text`` field is
    empty will be captioned.  The returned caption is written back into
    ``images[i]["text"]``.

    Parameters
    ----------
    batch_df : pd.DataFrame
        DataFrame with an ``images`` column containing lists of dicts with
        keys ``image_b64``, ``text``, and ``bbox_xyxy_norm``.
    model : NemotronVLMCaptioner | None
        Pre-loaded local VLM model.  When provided, ``endpoint_url`` is
        ignored and inference runs in-process.
    endpoint_url : str | None
        URL of a remote VLM HTTP endpoint.
    model_name : str
        Model identifier passed to the remote VLM endpoint (ignored for
        local mode).
    api_key : str | None
        Bearer token for the remote VLM endpoint.
    prompt : str
        Text prompt sent alongside each image.
    system_prompt : str | None
        Optional system prompt for the VLM.
    temperature : float
        Sampling temperature.
    batch_size : int
        Number of images per remote VLM request (local mode processes
        images one at a time).
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

    # Generate captions.
    all_captions: List[str] = []
    for start in range(0, len(pending), batch_size):
        chunk_b64 = [b64 for _, _, b64 in pending[start : start + batch_size]]

        if model is not None:
            captions = _caption_batch_local(
                chunk_b64,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        else:
            captions = _caption_batch_remote(
                chunk_b64,
                endpoint_url=endpoint_url,  # type: ignore[arg-type]
                model_name=model_name,
                api_key=api_key,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        all_captions.extend(captions)

    # Write captions back into the DataFrame.
    for (row_idx, item_idx, _), caption in zip(pending, all_captions):
        batch_df.at[row_idx, "images"][item_idx]["text"] = caption

    return batch_df
