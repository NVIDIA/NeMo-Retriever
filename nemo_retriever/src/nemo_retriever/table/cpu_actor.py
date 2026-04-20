# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import pandas as pd
import requests

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.nim.nim import NIMClient
from nemo_retriever.params import RemoteRetryParams
from nemo_retriever.table.shared import table_structure_ocr_page_elements

logger = logging.getLogger(__name__)


def _probe_endpoint(url: str, *, name: str, timeout: float = 5.0) -> None:
    """Fire a lightweight request to verify the endpoint is reachable.

    Tries a GET against the base URL (strips trailing ``/infer`` etc.) first,
    falling back to a HEAD against the full URL.  Logs success or failure but
    never raises — this is a best-effort diagnostic.
    """
    import urllib.parse

    parsed = urllib.parse.urlparse(url)
    health_url = f"{parsed.scheme}://{parsed.netloc}/v1/health/ready"

    for probe_url, method in [(health_url, "GET"), (url, "HEAD")]:
        try:
            t0 = time.perf_counter()
            resp = requests.request(method, probe_url, timeout=timeout)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "TableStructureCPUActor: %s endpoint %s responded %d in %.0fms",
                name,
                probe_url,
                resp.status_code,
                elapsed_ms,
            )
            return
        except requests.ConnectionError:
            logger.warning(
                "TableStructureCPUActor: %s endpoint %s is UNREACHABLE (connection refused). "
                "Processing will stall until this endpoint becomes available.",
                name,
                probe_url,
            )
            return
        except requests.Timeout:
            logger.warning(
                "TableStructureCPUActor: %s endpoint %s timed out after %.1fs. "
                "The endpoint may be overloaded or not ready.",
                name,
                probe_url,
                timeout,
            )
            return
        except Exception as exc:
            logger.debug(
                "TableStructureCPUActor: %s endpoint probe %s failed: %s",
                name,
                probe_url,
                exc,
            )


class TableStructureCPUActor(AbstractOperator, CPUOperator):
    """CPU-only variant of :class:`TableStructureActor`.

    Defaults to the build.nvidia.com endpoint for
    ``nemotron-table-structure-v1``. No local GPU models are loaded.
    """

    DEFAULT_TABLE_STRUCTURE_INVOKE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1"
    DEFAULT_OCR_INVOKE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1"

    def __init__(
        self,
        *,
        table_structure_invoke_url: Optional[str] = None,
        ocr_invoke_url: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        table_output_format: Optional[str] = None,
        request_timeout_s: float = 120.0,
        inference_batch_size: int = 8,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        super().__init__()
        self._table_structure_invoke_url = (
            table_structure_invoke_url or invoke_url or self.DEFAULT_TABLE_STRUCTURE_INVOKE_URL
        ).strip()
        self._ocr_invoke_url = (ocr_invoke_url or self.DEFAULT_OCR_INVOKE_URL).strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._inference_batch_size = int(inference_batch_size)
        self._table_output_format = table_output_format
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        self._table_structure_model = None
        self._ocr_model = None
        self._nim_client = NIMClient(
            max_pool_workers=int(remote_max_pool_workers),
        )

        logger.info(
            "TableStructureCPUActor initialized: table_structure_url=%s, ocr_url=%s",
            self._table_structure_invoke_url,
            self._ocr_invoke_url,
        )
        _probe_endpoint(self._table_structure_invoke_url, name="table-structure")
        _probe_endpoint(self._ocr_invoke_url, name="ocr")

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        n_rows = len(data) if hasattr(data, "__len__") else "?"
        logger.info("TableStructureCPUActor.process: received batch of %s rows", n_rows)
        t0 = time.perf_counter()
        result = table_structure_ocr_page_elements(
            data,
            table_structure_model=self._table_structure_model,
            table_structure_invoke_url=self._table_structure_invoke_url,
            api_key=self._api_key,
            table_output_format=self._table_output_format,
            request_timeout_s=self._request_timeout_s,
            inference_batch_size=self._inference_batch_size,
            remote_retry=self._remote_retry,
            nim_client=self._nim_client,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0
        logger.info("TableStructureCPUActor.process: finished batch of %s rows in %.2fs", n_rows, elapsed)
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
