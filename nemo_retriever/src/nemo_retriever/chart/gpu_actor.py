# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.fusion import ProcessOnlyFusionSafe
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.params import RemoteRetryParams
from nemo_retriever.chart.shared import graphic_elements_ocr_page_elements


class GraphicElementsActor(ProcessOnlyFusionSafe, AbstractOperator, GPUOperator):
    """
    Ray-friendly callable that initializes both graphic-elements and OCR
    models once per actor and runs the combined stage.
    """

    fusion_stage_id = "graphic_elements"
    fusion_next_stage_ids = ("ocr",)
    fusion_can_start_segment = False

    def __init__(
        self,
        *,
        graphic_elements_invoke_url: Optional[str] = None,
        ocr_invoke_url: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
        inference_batch_size: int = 8,
    ) -> None:
        super().__init__()
        self._graphic_elements_invoke_url = (graphic_elements_invoke_url or "").strip()
        self._ocr_invoke_url = (ocr_invoke_url or invoke_url or "").strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        self._inference_batch_size = int(inference_batch_size)

        if self._graphic_elements_invoke_url:
            self._graphic_elements_model = None
        else:
            from nemo_retriever.model.local import NemotronGraphicElementsV1

            self._graphic_elements_model = NemotronGraphicElementsV1()

        if self._ocr_invoke_url:
            self._ocr_model = None
        else:
            from nemo_retriever.model.local import NemotronOCRV1

            self._ocr_model = NemotronOCRV1()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return graphic_elements_ocr_page_elements(
            data,
            graphic_elements_model=self._graphic_elements_model,
            ocr_model=self._ocr_model,
            graphic_elements_invoke_url=self._graphic_elements_invoke_url,
            ocr_invoke_url=self._ocr_invoke_url,
            api_key=self._api_key,
            request_timeout_s=self._request_timeout_s,
            remote_retry=self._remote_retry,
            inference_batch_size=self._inference_batch_size,
            **kwargs,
        )

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
                        "stage": "chart_graphic_elements_ocr_actor_call",
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }
                n = len(out.index)
                out["chart"] = [[] for _ in range(n)]
                out["graphic_elements_ocr_v1"] = [payload for _ in range(n)]
                return out
            return [
                {
                    "graphic_elements_ocr_v1": {
                        "timing": None,
                        "error": {
                            "stage": "chart_graphic_elements_ocr_actor_call",
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                    }
                }
            ]
