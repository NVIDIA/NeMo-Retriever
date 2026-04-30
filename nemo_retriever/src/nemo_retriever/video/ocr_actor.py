# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VideoFrameOCRActor: full-frame OCR for video frames.

Thin wrapper around the shared :func:`ocr_b64_to_text` helper in
:mod:`nemo_retriever.ocr.shared`. The CPU/GPU variants only own the
model lifecycle (NIM client init, lazy ``NemotronOCRV1`` load); the
batch-shape handling and OCR call are reused from the page-elements
OCR pipeline so response parsing, the empty-text fallback fix, and
selective passthrough live in exactly one place.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.graph.operator_archetype import ArchetypeOperator
from nemo_retriever.nim.nim import NIMClient
from nemo_retriever.ocr.shared import (
    concat_with_passthrough,
    full_image_ocr_df,
    split_ocrable_rows,
)
from nemo_retriever.params import RemoteRetryParams, VideoOCRParams

logger = logging.getLogger(__name__)


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
        ocr_df, passthrough = split_ocrable_rows(batch_df)
        if ocr_df.empty:
            return passthrough
        self._ensure_model()
        out = full_image_ocr_df(ocr_df, model=self._model, merge_level=self._params.merge_level)
        return concat_with_passthrough(out, passthrough)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class VideoFrameOCRCPUActor(AbstractOperator, CPUOperator):
    """Remote Nemotron OCR v1 (NIM) on full video frames, batched per call."""

    DEFAULT_INVOKE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1"

    def __init__(self, **ocr_kwargs: Any) -> None:
        super().__init__(**ocr_kwargs)
        self._params = _params_from_kwargs(ocr_kwargs)
        self._invoke_url = (self._params.ocr_invoke_url or self.DEFAULT_INVOKE_URL).strip()
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(ocr_kwargs.get("remote_max_pool_workers", 16)),
            remote_max_retries=int(ocr_kwargs.get("remote_max_retries", 10)),
            remote_max_429_retries=int(ocr_kwargs.get("remote_max_429_retries", 5)),
        )
        self._nim_client = NIMClient(max_pool_workers=int(self._remote_retry.remote_max_pool_workers))

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame()
        ocr_df, passthrough = split_ocrable_rows(batch_df)
        if ocr_df.empty:
            return passthrough
        out = full_image_ocr_df(
            ocr_df,
            invoke_url=self._invoke_url,
            api_key=self._params.api_key,
            nim_client=self._nim_client,
            merge_level=self._params.merge_level,
            batch_size=int(self._params.batch_size),
            timeout_s=float(self._params.request_timeout_s),
            retry=self._remote_retry,
        )
        return concat_with_passthrough(out, passthrough)

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
