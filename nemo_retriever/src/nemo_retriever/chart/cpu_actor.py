# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.nim.nim import NIMClient
from nemo_retriever.nim.probe import probe_endpoint
from nemo_retriever.params import RemoteRetryParams
from nemo_retriever.chart.shared import graphic_elements_ocr_page_elements

logger = logging.getLogger(__name__)


class GraphicElementsCPUActor(AbstractOperator, CPUOperator):
    """CPU-only variant of :class:`GraphicElementsActor`.

    Defaults to build.nvidia.com endpoints for ``nemotron-graphic-elements-v1``
    and ``nemotron-ocr-v1``. No local GPU models are loaded.
    """

    DEFAULT_GRAPHIC_ELEMENTS_INVOKE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1"
    DEFAULT_OCR_INVOKE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1"

    def __init__(
        self,
        *,
        graphic_elements_invoke_url: Optional[str] = None,
        ocr_invoke_url: Optional[str] = None,
        load_ocr_v2: bool = True,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
        inference_batch_size: int = 8,
    ) -> None:
        super().__init__()
        self._graphic_elements_invoke_url = (
            graphic_elements_invoke_url or self.DEFAULT_GRAPHIC_ELEMENTS_INVOKE_URL
        ).strip()
        self._ocr_invoke_url = (ocr_invoke_url or invoke_url or self.DEFAULT_OCR_INVOKE_URL).strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        self._inference_batch_size = int(inference_batch_size)
        self._graphic_elements_model = None
        self._ocr_model = None
        self._nim_client = NIMClient(
            max_pool_workers=int(remote_max_pool_workers),
        )

        logger.info(
            "GraphicElementsCPUActor initialized: graphic_elements_url=%s, ocr_url=%s",
            self._graphic_elements_invoke_url,
            self._ocr_invoke_url,
        )
        probe_endpoint(
            self._graphic_elements_invoke_url,
            name="graphic-elements",
            prefix="GraphicElementsCPUActor",
            api_key=self._api_key,
        )
        probe_endpoint(self._ocr_invoke_url, name="ocr", prefix="GraphicElementsCPUActor", api_key=self._api_key)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        n_rows = len(data) if hasattr(data, "__len__") else "?"
        logger.info("GraphicElementsCPUActor.process: received batch of %s rows", n_rows)
        t0 = time.perf_counter()
        result = graphic_elements_ocr_page_elements(
            data,
            graphic_elements_model=self._graphic_elements_model,
            ocr_model=self._ocr_model,
            graphic_elements_invoke_url=self._graphic_elements_invoke_url,
            ocr_invoke_url=self._ocr_invoke_url,
            api_key=self._api_key,
            request_timeout_s=self._request_timeout_s,
            remote_retry=self._remote_retry,
            nim_client=self._nim_client,
            inference_batch_size=self._inference_batch_size,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0
        logger.info("GraphicElementsCPUActor.process: finished batch of %s rows in %.2fs", n_rows, elapsed)
        return result

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return self.run(batch_df, **override_kwargs)
        except BaseException as exc:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = {
                    "timing": None,
                    "error": {
                        "stage": "chart_graphic_elements_cpu_actor_call",
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }
                n = len(out.index)
                out["chart"] = [[] for _ in range(n)]
                out["graphic_elements_ocr_v1"] = [payload for _ in range(n)]
                out["graphic_elements_v1_num_detections"] = [0 for _ in range(n)]
                out["graphic_elements_v1_counts_by_label"] = [{} for _ in range(n)]
                return out
            return [
                {
                    "graphic_elements_ocr_v1": {
                        "timing": None,
                        "error": {
                            "stage": "chart_graphic_elements_cpu_actor_call",
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                    }
                }
            ]
