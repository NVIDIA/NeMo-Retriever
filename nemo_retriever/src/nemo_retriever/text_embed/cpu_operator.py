# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU graph operator for text embeddings (remote HTTP or local HF on CPU)."""

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.params import EmbedParams
from nemo_retriever.text_embed.runtime import embed_text_main_text_embed
from nemo_retriever.text_embed.shared import build_embed_kwargs


class _BatchEmbedCPUActor(AbstractOperator, CPUOperator):
    """CPU-only embedding: remote when ``embed_invoke_url`` / ``embedding_endpoint`` is set, else local HF."""

    def __init__(self, params: EmbedParams) -> None:
        super().__init__()
        self._params = params
        self._kwargs = build_embed_kwargs(params)

        endpoint = (self._kwargs.get("embedding_endpoint") or self._kwargs.get("embed_invoke_url") or "").strip()
        if endpoint:
            self._model = None
            if "embedding_endpoint" not in self._kwargs and self._kwargs.get("embed_invoke_url"):
                self._kwargs["embedding_endpoint"] = self._kwargs.get("embed_invoke_url")
            return

        from nemo_retriever.model import create_local_embedder

        device = self._kwargs.get("device") or "cpu"
        self._model = create_local_embedder(
            self._kwargs.get("model_name"),
            device=str(device) if device else None,
            hf_cache_dir=str(self._kwargs["hf_cache_dir"]) if self._kwargs.get("hf_cache_dir") else None,
            normalize=bool(self._kwargs.get("normalize", True)),
            max_length=int(self._kwargs.get("max_length", 8192)),
        )
        self._kwargs["embedding_endpoint"] = None
        self._kwargs["embed_invoke_url"] = None

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
