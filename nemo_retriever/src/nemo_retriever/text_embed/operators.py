# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operators for embedding text and multimodal content."""

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.params import EmbedParams
from nemo_retriever.text_embed.runtime import embed_text_main_text_embed


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
        self._kwargs = {
            **params.model_dump(mode="python", exclude={"runtime", "batch_tuning", "fused_tuning"}, exclude_none=True),
            **params.runtime.model_dump(mode="python", exclude_none=True),
        }
        if "embedding_endpoint" not in self._kwargs and self._kwargs.get("embed_invoke_url"):
            self._kwargs["embedding_endpoint"] = self._kwargs.get("embed_invoke_url")

        endpoint = (self._kwargs.get("embedding_endpoint") or self._kwargs.get("embed_invoke_url") or "").strip()
        if endpoint:
            self._model = None
            return

        from nemo_retriever.model import create_local_embedder

        self._model = create_local_embedder(
            self._kwargs.get("model_name"),
            device=str(self._kwargs["device"]) if self._kwargs.get("device") else None,
            hf_cache_dir=str(self._kwargs["hf_cache_dir"]) if self._kwargs.get("hf_cache_dir") else None,
            normalize=bool(self._kwargs.get("normalize", True)),
            max_length=int(self._kwargs.get("max_length", 8192)),
        )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return embed_text_main_text_embed(data, model=self._model, **self._kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any) -> Any:
        return self.run(batch_df)


class _BatchEmbedCPUActor(AbstractOperator, CPUOperator):
    """CPU-only embedding actor that always targets a remote endpoint."""

    DEFAULT_EMBED_INVOKE_URL = "https://integrate.api.nvidia.com/v1/embeddings"

    def __init__(self, params: EmbedParams) -> None:
        super().__init__()
        self._params = params
        self._kwargs = {
            **params.model_dump(mode="python", exclude={"runtime", "batch_tuning", "fused_tuning"}, exclude_none=True),
            **params.runtime.model_dump(mode="python", exclude_none=True),
        }
        if "embedding_endpoint" not in self._kwargs:
            self._kwargs["embedding_endpoint"] = self._kwargs.get("embed_invoke_url") or self.DEFAULT_EMBED_INVOKE_URL

        endpoint = (self._kwargs.get("embedding_endpoint") or self._kwargs.get("embed_invoke_url") or "").strip()
        if not endpoint:
            self._kwargs["embedding_endpoint"] = self.DEFAULT_EMBED_INVOKE_URL
        self._model = None

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return embed_text_main_text_embed(data, model=self._model, **self._kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
