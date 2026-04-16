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
        import os
        import warnings

        warnings.filterwarnings(
            "ignore",
            message=r".*`input_embeds` is deprecated.*create_bidirectional_mask.*",
            category=FutureWarning,
        )

        self._params = params
        self._kwargs = build_embed_kwargs(params)

        # Propagate HF_TOKEN into the worker environment so that
        # huggingface_hub API calls are authenticated.
        hf_token = self._kwargs.pop("hf_token", None)
        if hf_token and "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = hf_token

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
