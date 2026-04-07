# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.params import RemoteRetryParams
from nemo_retriever.ocr.shared import ocr_page_elements


class OCRCPUActor(AbstractOperator, CPUOperator):
    """CPU-only variant of :class:`OCRActor`."""

    DEFAULT_INVOKE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1"

    def __init__(self, **ocr_kwargs: Any) -> None:
        super().__init__(**ocr_kwargs)
        self.ocr_kwargs = dict(ocr_kwargs)
        invoke_url = str(
            self.ocr_kwargs.get("ocr_invoke_url") or self.ocr_kwargs.get("invoke_url") or self.DEFAULT_INVOKE_URL
        ).strip()
        if "invoke_url" not in self.ocr_kwargs:
            self.ocr_kwargs["invoke_url"] = invoke_url

        self.ocr_kwargs["extract_text"] = bool(self.ocr_kwargs.get("extract_text", False))
        self.ocr_kwargs["extract_tables"] = bool(self.ocr_kwargs.get("extract_tables", False))
        self.ocr_kwargs["extract_charts"] = bool(self.ocr_kwargs.get("extract_charts", False))
        self.ocr_kwargs["extract_infographics"] = bool(self.ocr_kwargs.get("extract_infographics", False))
        self.ocr_kwargs["use_graphic_elements"] = bool(self.ocr_kwargs.get("use_graphic_elements", False))
        self.ocr_kwargs["request_timeout_s"] = float(self.ocr_kwargs.get("request_timeout_s", 120.0))
        self.ocr_kwargs["inference_batch_size"] = int(self.ocr_kwargs.get("inference_batch_size", 8))

        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(self.ocr_kwargs.get("remote_max_pool_workers", 16)),
            remote_max_retries=int(self.ocr_kwargs.get("remote_max_retries", 10)),
            remote_max_429_retries=int(self.ocr_kwargs.get("remote_max_429_retries", 5)),
        )
        self._model = None

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return ocr_page_elements(
            data,
            model=self._model,
            remote_retry=self._remote_retry,
            **self.ocr_kwargs,
            **kwargs,
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
