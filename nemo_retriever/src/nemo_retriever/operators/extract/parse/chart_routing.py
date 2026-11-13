# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local and remote chart-crop routing stages."""

from __future__ import annotations

import atexit
from typing import Any

import pandas as pd

from nemo_retriever.common.modality.chart_routing import _CHART_ROUTING_SAMPLING, route_chart_regions
from nemo_retriever.common.params import ParseParams, RemoteRetryParams
from nemo_retriever.models.nim.nim import NIMClient
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.operators.operator_archetype import ArchetypeOperator


class _ChartRoutingBase(AbstractOperator):
    def __init__(
        self,
        *,
        nemotron_parse_invoke_url: str | None = None,
        nemotron_parse_table_format: str | None = None,
        parse_sampling_params: dict[str, Any] | None = None,
        use_dflash: bool | None = None,
        num_speculative_tokens: int | None = None,
        api_key: str | None = None,
        request_timeout_s: float = 120.0,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        options = ParseParams(table_format=nemotron_parse_table_format, **(parse_sampling_params or {}))
        self._sampling = {**_CHART_ROUTING_SAMPLING, **options._sampling_overrides()}
        super().__init__(
            nemotron_parse_invoke_url=nemotron_parse_invoke_url,
            nemotron_parse_table_format=options.table_format or "simple_html",
            parse_sampling_params=options._sampling_overrides(),
            use_dflash=False if use_dflash is None else use_dflash,
            num_speculative_tokens=32 if num_speculative_tokens is None else num_speculative_tokens,
            api_key=api_key,
            request_timeout_s=request_timeout_s,
            remote_max_pool_workers=remote_max_pool_workers,
            remote_max_retries=remote_max_retries,
            remote_max_429_retries=remote_max_429_retries,
        )
        self._model = None
        self._client = None
        self._cleanup_registered = False
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=remote_max_pool_workers,
            remote_max_retries=remote_max_retries,
            remote_max_429_retries=remote_max_429_retries,
        )

    def preprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Pass extracted page rows to chart routing without modification."""
        return data

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Replace chart content using the configured local or remote Parse 3.5 service."""
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
        if self.nemotron_parse_invoke_url:
            if self._client is None:
                self._client = NIMClient(max_pool_workers=self._remote_retry.remote_max_pool_workers)
        elif self._model is None:
            from nemo_retriever.models.local.nemotron_parse35 import NemotronParse35

            self._model = NemotronParse35(
                backend="vllm",
                use_dflash=self.use_dflash,
                num_speculative_tokens=self.num_speculative_tokens,
                max_tokens=self._sampling["max_tokens"],
                sampling_params={k: v for k, v in self._sampling.items() if k != "max_tokens"},
            )
        return route_chart_regions(
            data,
            model=self._model,
            invoke_url=self.nemotron_parse_invoke_url,
            api_key=self.api_key,
            request_timeout_s=self.request_timeout_s,
            remote_retry=self._remote_retry,
            nim_client=self._client,
            table_format=self.nemotron_parse_table_format,
            sampling_params=self._sampling,
        )

    def cleanup(self) -> None:
        """Release the local engine and owned HTTP client; repeated calls are safe."""
        try:
            if self._model is not None:
                self._model.unload()
                self._model = None
        finally:
            if self._client is not None:
                self._client.shutdown()
                self._client = None
            if self._model is None and self._client is None and self._cleanup_registered:
                atexit.unregister(self.cleanup)
                self._cleanup_registered = False

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Return routed page rows, including per-crop diagnostics."""
        return data


class ChartRoutingCPUActor(_ChartRoutingBase, CPUOperator):
    """Route chart crops through an explicitly configured remote Parse 3.5 endpoint.

    Accepts the endpoint, credential, timeout, and retry arguments of the routing
    stage. Processing preserves non-chart columns and records failed crops
    without replacing their original OCR text. Configure a chat-completions URL;
    no local model is loaded by this CPU variant.
    """

    def __init__(self, **kwargs: Any) -> None:
        if not kwargs.get("nemotron_parse_invoke_url"):
            raise ValueError("ChartRoutingCPUActor requires nemotron_parse_invoke_url for Parse 3.5.")
        super().__init__(**kwargs)


class ChartRoutingGPUActor(_ChartRoutingBase, GPUOperator):
    """Lazily run local Parse 3.5 on chart crops with configurable crop generation and Table rendering settings."""


class ChartRoutingActor(ArchetypeOperator):
    """Select local GPU routing or remote CPU routing from nemotron_parse_invoke_url.

    Constructor arguments are the Parse 3.5 endpoint, api_key, request_timeout_s,
    generation overrides, Table rendering format, and remote retry limits.
    The stage accepts extracted page DataFrames and
    returns copies with replaced chart text and per-crop diagnostics. Other
    content and coordinates are preserved; inference errors retain original OCR.
    """

    _cpu_variant_class = ChartRoutingCPUActor
    _gpu_variant_class = ChartRoutingGPUActor

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        """Return whether an explicit Parse endpoint selects remote CPU routing."""
        return bool((operator_kwargs or {}).get("nemotron_parse_invoke_url"))

    def cleanup(self) -> None:
        """Release resources held by a delegate without constructing a new one."""
        delegate = self._resolved_delegate
        if delegate is not None:
            delegate.cleanup()

    def __init__(self, **routing_kwargs: Any) -> None:
        super().__init__(**routing_kwargs)
