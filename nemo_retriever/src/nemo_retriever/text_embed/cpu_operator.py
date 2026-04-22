# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU graph operator for remote-only text embeddings."""

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.nim.probe import probe_endpoint
from nemo_retriever.params import EmbedParams
from nemo_retriever.text_embed.runtime import embed_text_main_text_embed
from nemo_retriever.text_embed.shared import build_embed_kwargs


class _BatchEmbedCPUActor(AbstractOperator, CPUOperator):
    """CPU-only embedding actor that always targets a remote endpoint."""

    DEFAULT_EMBED_INVOKE_URL = "https://integrate.api.nvidia.com/v1/embeddings"

    def __init__(self, params: EmbedParams) -> None:
        super().__init__()
        self._params = params
        self._kwargs = build_embed_kwargs(params)
        if "embedding_endpoint" not in self._kwargs:
            self._kwargs["embedding_endpoint"] = self._kwargs.get("embed_invoke_url") or self.DEFAULT_EMBED_INVOKE_URL

        endpoint = (self._kwargs.get("embedding_endpoint") or self._kwargs.get("embed_invoke_url") or "").strip()
        if not endpoint:
            self._kwargs["embedding_endpoint"] = self.DEFAULT_EMBED_INVOKE_URL
        self._model = None

        api_key = self._kwargs.get("api_key")
        if endpoint and api_key:
            # Probe the /embeddings path with a model-name-only body — auth is
            # checked before body validation so a bad key returns 401 without
            # triggering inference. A valid key with an empty input returns 400.
            model_name = self._kwargs.get("model_name", "")
            probe_endpoint(
                endpoint,
                name="embed",
                prefix="_BatchEmbedCPUActor",
                api_key=api_key,
                post_url=endpoint.rstrip("/") + "/embeddings",
                post_body={"input": [], "model": model_name},
            )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return embed_text_main_text_embed(data, model=self._model, **self._kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
