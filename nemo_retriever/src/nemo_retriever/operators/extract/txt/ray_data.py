# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ray Data adapter for .txt: TxtSplitActor turns bytes+path batches into chunk rows.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List  # noqa: F401

import pandas as pd

from nemo_retriever.common.modality.txt.tokenizer_provider import (
    TokenizerUnavailableError,
)
from nemo_retriever.common.params import TextChunkParams
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.operators.operator_archetype import ArchetypeOperator

from nemo_retriever.common.modality.txt.split import empty_text_chunks_df, text_to_chunks_df, txt_bytes_to_chunks_df

logger = logging.getLogger(__name__)


@designer_component(
    name="Text Chunker",
    category="Text & Content",
    compute="cpu",
    description="Chunks text documents into smaller segments",
    category_color="#42d6a4",
)
class TextChunkCPUActor(AbstractOperator, CPUOperator):
    """
    Ray Data map_batches callable: re-chunk existing ``text`` column by token count.

    This is the batch-mode equivalent of :func:`~nemo_retriever.txt.split.split_df`.
    Constructor takes :class:`TextChunkParams`; ``__call__`` receives a pandas batch
    and returns the split result.
    """

    # Chunk expansion mixes retained Arrow-backed nested values with empty
    # Python collections on continuation rows. Keep that heterogeneous result
    # in pandas so Ray does not infer incompatible extension arrays.
    PRESERVE_PANDAS_OUTPUT = True

    def __init__(self, params: TextChunkParams | None = None) -> None:
        super().__init__()
        self._params = params or TextChunkParams()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        from nemo_retriever.common.modality.txt.split import split_df

        if not isinstance(data, pd.DataFrame) or data.empty:
            return data

        kw = self._params.model_dump(mode="python")
        kw.pop("encoding", None)
        return split_df(data, **kw)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        return self.run(batch_df)


@designer_component(
    name="Text Splitter",
    category="Text & Content",
    compute="cpu",
    description="Splits raw text files for processing",
    category_color="#42d6a4",
)
class TxtSplitCPUActor(AbstractOperator, CPUOperator):
    """
    Ray Data map_batches callable: DataFrame with bytes/text, path -> DataFrame of chunks.

    Each output row has: text, content, path, page_number, metadata (same shape as txt_file_to_chunks_df).
    """

    def __init__(self, params: TextChunkParams | None = None) -> None:
        super().__init__()
        self._params = params or TextChunkParams()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return empty_text_chunks_df()
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return data

        params = self._params
        out_dfs: List[pd.DataFrame] = []
        for _, row in data.iterrows():
            raw = row.get("bytes")
            text = row.get("text")
            path = row.get("path")
            if (not isinstance(raw, (bytes, bytearray)) and not isinstance(text, str)) or path is None:
                continue
            path_str = str(path) if path is not None else ""
            try:
                if isinstance(raw, (bytes, bytearray)):
                    chunk_df = txt_bytes_to_chunks_df(bytes(raw), path_str, params=params)
                elif isinstance(text, str):
                    chunk_df = text_to_chunks_df(text, path_str, params=params)
                else:
                    continue
                if not chunk_df.empty:
                    out_dfs.append(chunk_df)
            except TokenizerUnavailableError:
                raise
            except Exception:
                logger.warning("Failed to split text source %r", path_str, exc_info=True)
                continue
        if not out_dfs:
            return empty_text_chunks_df()
        return pd.concat(out_dfs, ignore_index=True)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        return self.run(batch_df)


class TextChunkActor(ArchetypeOperator):
    _cpu_variant_class = TextChunkCPUActor

    def __init__(self, params: TextChunkParams | None = None) -> None:
        super().__init__(params=params)
        self._params = params


class TxtSplitActor(ArchetypeOperator):
    _cpu_variant_class = TxtSplitCPUActor

    def __init__(self, params: TextChunkParams | None = None) -> None:
        super().__init__(params=params)
        self._params = params
