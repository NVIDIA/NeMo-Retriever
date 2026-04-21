# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU graph operator for embedding text and multimodal content."""

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.params import EmbedParams
from nemo_retriever.text_embed.runtime import embed_text_main_text_embed
from nemo_retriever.text_embed.shared import build_embed_kwargs


class _BatchEmbedActor(AbstractOperator, GPUOperator):
    """Graph embedding actor that loads a local embedder or calls a remote endpoint."""

    def __init__(self, params: EmbedParams) -> None:
        super().__init__()
        import warnings

        warnings.filterwarnings(
            "ignore",
            message=r".*`input_embeds` is deprecated.*create_bidirectional_mask.*",
            category=FutureWarning,
        )

        self._params = params
        self._kwargs = build_embed_kwargs(params)

        endpoint = (self._kwargs.get("embedding_endpoint") or self._kwargs.get("embed_invoke_url") or "").strip()
        if endpoint:
            self._model = None
            return

        ingest_backend = (self._kwargs.get("local_ingest_backend") or "vllm").strip().lower()
        hf_cache = str(self._kwargs["hf_cache_dir"]) if self._kwargs.get("hf_cache_dir") else None
        if ingest_backend == "hf":
            from nemo_retriever.model import create_local_query_embedder

            self._model = create_local_query_embedder(
                self._kwargs.get("model_name"),
                backend="hf",
                hf_cache_dir=hf_cache,
            )
        else:
            from nemo_retriever.model import create_local_embedder

            self._model = create_local_embedder(
                self._kwargs.get("model_name"),
                hf_cache_dir=hf_cache,
                gpu_memory_utilization=float(self._kwargs.get("gpu_memory_utilization", 0.45)),
                enforce_eager=bool(self._kwargs.get("enforce_eager", False)),
            )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return embed_text_main_text_embed(data, model=self._model, **self._kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any) -> Any:
        return self.run(batch_df)
