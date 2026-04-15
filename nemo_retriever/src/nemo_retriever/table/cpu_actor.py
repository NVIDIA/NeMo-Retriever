# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.params import RemoteRetryParams
from nemo_retriever.table.shared import table_structure_ocr_page_elements


class TableStructureCPUActor(AbstractOperator, CPUOperator):
    """CPU-only variant of :class:`TableStructureActor`.

    Defaults to the build.nvidia.com endpoint for
    ``nemotron-table-structure-v1``. No local GPU models are loaded.
    """

    DEFAULT_TABLE_STRUCTURE_INVOKE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1"

    def __init__(
        self,
        *,
        table_structure_invoke_url: Optional[str] = None,
        invoke_url: Optional[str] = None,
        ocr_invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        table_output_format: Optional[str] = None,
        request_timeout_s: float = 120.0,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        super().__init__()
        self._table_structure_invoke_url = (
            table_structure_invoke_url or invoke_url or self.DEFAULT_TABLE_STRUCTURE_INVOKE_URL
        ).strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._table_output_format = table_output_format
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        self._table_structure_model = None

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return table_structure_ocr_page_elements(
            data,
            table_structure_model=self._table_structure_model,
            table_structure_invoke_url=self._table_structure_invoke_url,
            api_key=self._api_key,
            table_output_format=self._table_output_format,
            request_timeout_s=self._request_timeout_s,
            remote_retry=self._remote_retry,
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
                        "stage": "table_structure_cpu_actor_call",
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["table_structure_ocr_v1"] = [payload for _ in range(n)]
                return out
            return [
                {
                    "table_structure_ocr_v1": {
                        "timing": None,
                        "error": {
                            "stage": "table_structure_cpu_actor_call",
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                    }
                }
            ]
