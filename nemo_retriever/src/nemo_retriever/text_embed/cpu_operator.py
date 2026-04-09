# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU graph operator for remote-only text embeddings."""

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
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

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        import logging as _logging
        import pandas as pd

        _log = _logging.getLogger(__name__)
        if isinstance(data, pd.DataFrame):
            text_col = self._kwargs.get("text_column", "text")
            n_total = len(data)
            n_with_text = int(data[text_col].notna().sum()) if text_col in data.columns else -1
            _log.debug(
                "[embed] input: %d rows, %d with non-null '%s', endpoint=%s",
                n_total,
                n_with_text,
                text_col,
                self._kwargs.get("embedding_endpoint") or self._kwargs.get("embed_invoke_url"),
            )
        out = embed_text_main_text_embed(data, model=self._model, **self._kwargs)
        if isinstance(out, pd.DataFrame):
            dim_col = self._kwargs.get("embedding_dim_column", "text_embeddings_1b_v2_dim")
            has_col = self._kwargs.get("has_embedding_column", "text_embeddings_1b_v2_has_embedding")
            n_embedded = int(out[has_col].sum()) if has_col in out.columns else -1
            dims = out[dim_col].unique().tolist() if dim_col in out.columns else []
            _log.debug("[embed] output: %d rows, %d with embeddings, dims=%s", len(out), n_embedded, dims)
        return out

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
