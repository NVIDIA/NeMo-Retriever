# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VideoFrameOCRActor: full-frame OCR for video frames.

Calls Nemotron OCR v1 directly on each frame's ``image_b64`` (no
page-elements detection — we want to OCR everything visible in a frame,
not crop by document-layout regions). Output rows have a populated
``text`` column; rows with empty OCR text are dropped.
"""

from __future__ import annotations

import logging
from typing import Any, List

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.nim.nim import NIMClient
from nemo_retriever.ocr.shared import (
    _blocks_to_text,
    _extract_remote_ocr_item,
    _parse_ocr_result,
)
from nemo_retriever.params import RemoteRetryParams, VideoOCRParams

logger = logging.getLogger(__name__)


def _ocr_response_to_text(preds: Any) -> str:
    """Extract joined OCR text, returning ``""`` when no text is detected.

    Wraps :func:`_parse_ocr_result` + :func:`_blocks_to_text` from the existing
    OCR pipeline but suppresses ``_parse_ocr_result``'s last-resort stringify
    fallback (it stringifies the raw response dict when no shape matches,
    which produces noise rows for frames with no on-screen text).
    """
    blocks = _parse_ocr_result(preds)
    if not blocks:
        return ""
    if len(blocks) == 1:
        only = blocks[0]
        if only.get("sort_x", 0.0) == 0.0 and only.get("sort_y", 0.0) == 0.0:
            text = (only.get("text") or "").strip()
            if text.startswith("[") or text.startswith("{"):
                # Stringified Python repr — treat as no text detected.
                return ""
    return _blocks_to_text(blocks)


def _params_from_kwargs(ocr_kwargs: dict[str, Any]) -> VideoOCRParams:
    params = ocr_kwargs.get("params")
    if isinstance(params, VideoOCRParams):
        return params
    return VideoOCRParams(
        ocr_invoke_url=ocr_kwargs.get("ocr_invoke_url") or ocr_kwargs.get("invoke_url"),
        api_key=ocr_kwargs.get("api_key"),
        batch_size=int(ocr_kwargs.get("batch_size", 8)),
        merge_level=str(ocr_kwargs.get("merge_level", "paragraph")),
        request_timeout_s=float(ocr_kwargs.get("request_timeout_s", 120.0)),
    )


class VideoFrameOCRGPUActor(AbstractOperator, GPUOperator):
    """Local Nemotron OCR v1 on full video frames (one frame per ``invoke()`` call)."""

    def __init__(self, **ocr_kwargs: Any) -> None:
        super().__init__(**ocr_kwargs)
        self._params = _params_from_kwargs(ocr_kwargs)
        self._model = None  # lazily loaded on first call

    def _ensure_model(self) -> None:
        if self._model is None:
            from nemo_retriever.model.local import NemotronOCRV1

            self._model = NemotronOCRV1()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame()
        frame_df, passthrough = _split_frame_rows(batch_df)
        if frame_df.empty:
            return passthrough
        self._ensure_model()
        out = frame_df.copy()
        texts: List[str] = []
        for image_b64 in out.get("image_b64", []):
            if not isinstance(image_b64, str) or not image_b64:
                texts.append("")
                continue
            try:
                preds = self._model.invoke(  # type: ignore[union-attr]
                    image_b64.encode("utf-8"),
                    merge_level=self._params.merge_level,
                )
                texts.append(_ocr_response_to_text(preds))
            except Exception as exc:
                logger.exception("Local OCR failed on frame: %s", exc)
                texts.append("")
        out["text"] = texts
        out = out[out["text"].astype(bool)].reset_index(drop=True)
        return _concat_with_passthrough(out, passthrough)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def _split_frame_rows(batch_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition rows into ``video_frame`` rows (to OCR) and the rest (passthrough)."""
    if "_content_type" not in batch_df.columns:
        # Backward compat: no discriminator column → treat the whole batch as frames.
        return batch_df.copy(), pd.DataFrame()
    is_frame = batch_df["_content_type"].astype(str) == "video_frame"
    return (
        batch_df[is_frame].reset_index(drop=True),
        batch_df[~is_frame].reset_index(drop=True),
    )


def _concat_with_passthrough(processed: pd.DataFrame, passthrough: pd.DataFrame) -> pd.DataFrame:
    """Concat the actor's output with the passthrough rows, harmonising columns."""
    if passthrough is None or passthrough.empty:
        return processed
    if processed is None or processed.empty:
        return passthrough
    for col in processed.columns:
        if col not in passthrough.columns:
            passthrough = passthrough.assign(**{col: None})
    for col in passthrough.columns:
        if col not in processed.columns:
            processed = processed.assign(**{col: None})
    return pd.concat([processed[passthrough.columns.tolist()], passthrough], ignore_index=True, sort=False)


class VideoFrameOCRCPUActor(AbstractOperator, CPUOperator):
    """Remote Nemotron OCR v1 (NIM) on full video frames, batched per call."""

    DEFAULT_INVOKE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1"

    def __init__(self, **ocr_kwargs: Any) -> None:
        super().__init__(**ocr_kwargs)
        self._params = _params_from_kwargs(ocr_kwargs)
        invoke_url = (self._params.ocr_invoke_url or self.DEFAULT_INVOKE_URL).strip()
        self._invoke_url = invoke_url
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(ocr_kwargs.get("remote_max_pool_workers", 16)),
            remote_max_retries=int(ocr_kwargs.get("remote_max_retries", 10)),
            remote_max_429_retries=int(ocr_kwargs.get("remote_max_429_retries", 5)),
        )
        self._nim_client = NIMClient(
            max_pool_workers=int(self._remote_retry.remote_max_pool_workers),
        )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame()
        frame_df, passthrough = _split_frame_rows(batch_df)
        if frame_df.empty:
            return passthrough

        out = frame_df.copy()
        b64_list: list[str] = []
        idxs: list[int] = []
        for i, image_b64 in enumerate(out.get("image_b64", [])):
            if isinstance(image_b64, str) and image_b64:
                b64_list.append(image_b64)
                idxs.append(i)

        texts = [""] * len(out.index)
        if b64_list:
            try:
                response_items = self._nim_client.invoke_image_inference_batches(
                    invoke_url=self._invoke_url,
                    image_b64_list=b64_list,
                    api_key=self._params.api_key,
                    timeout_s=float(self._params.request_timeout_s),
                    max_batch_size=int(self._params.batch_size),
                    max_retries=int(self._remote_retry.remote_max_retries),
                    max_429_retries=int(self._remote_retry.remote_max_429_retries),
                )
            except Exception as exc:
                logger.exception("Remote OCR call failed: %s", exc)
                response_items = [None] * len(b64_list)

            if len(response_items) != len(b64_list):
                logger.warning("OCR response count mismatch: expected %d, got %d", len(b64_list), len(response_items))
            for resp, dst in zip(response_items, idxs):
                try:
                    preds = _extract_remote_ocr_item(resp)
                    texts[dst] = _ocr_response_to_text(preds)
                except Exception as exc:
                    logger.warning("Failed to parse OCR response for frame %d: %s", dst, exc)
                    texts[dst] = ""

        out["text"] = texts
        out = out[out["text"].astype(bool)].reset_index(drop=True)
        return _concat_with_passthrough(out, passthrough)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


@designer_component(
    name="Video Frame OCR",
    category="Video",
    compute="gpu",
    description="Runs Nemotron OCR v1 directly on full video frames",
)
class VideoFrameOCRActor(ArchetypeOperator):
    """Graph-facing archetype that resolves to GPU or CPU variant.

    Routes to the CPU (NIM) variant when ``ocr_invoke_url`` (or
    ``invoke_url``) is provided; otherwise loads the local Nemotron OCR
    model on GPU.
    """

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        kwargs = operator_kwargs or {}
        params = kwargs.get("params")
        if isinstance(params, VideoOCRParams) and params.ocr_invoke_url:
            return True
        return bool(str(kwargs.get("ocr_invoke_url") or kwargs.get("invoke_url") or "").strip())

    @classmethod
    def cpu_variant_class(cls):
        return VideoFrameOCRCPUActor

    @classmethod
    def gpu_variant_class(cls):
        return VideoFrameOCRGPUActor

    def __init__(self, **ocr_kwargs: Any) -> None:
        super().__init__(**ocr_kwargs)
