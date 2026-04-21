# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.nim.nim import NIMClient
from nemo_retriever.params import RemoteRetryParams
from nemo_retriever.ocr.shared import nemotron_parse_page_elements


class NemotronParseCPUActor(AbstractOperator, CPUOperator):
    """CPU-only variant of :class:`NemotronParseActor`."""

    DEFAULT_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        *,
        extract_text: bool = False,
        extract_tables: bool = False,
        extract_charts: bool = False,
        extract_infographics: bool = False,
        nemotron_parse_invoke_url: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        task_prompt: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        super().__init__()
        self._invoke_url = (nemotron_parse_invoke_url or invoke_url or self.DEFAULT_INVOKE_URL).strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._task_prompt = str(task_prompt)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        self._model = None
        self._nim_client = NIMClient(
            max_pool_workers=int(remote_max_pool_workers),
        )
        self._extract_text = bool(extract_text)
        self._extract_tables = bool(extract_tables)
        self._extract_charts = bool(extract_charts)
        self._extract_infographics = bool(extract_infographics)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return nemotron_parse_page_elements(
            data,
            model=self._model,
            invoke_url=self._invoke_url,
            api_key=self._api_key,
            request_timeout_s=self._request_timeout_s,
            task_prompt=self._task_prompt,
            extract_text=self._extract_text,
            extract_tables=self._extract_tables,
            extract_charts=self._extract_charts,
            extract_infographics=self._extract_infographics,
            remote_retry=self._remote_retry,
            nim_client=self._nim_client,
            **kwargs,
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
