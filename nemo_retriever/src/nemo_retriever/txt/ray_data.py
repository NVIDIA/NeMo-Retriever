# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ray Data adapter for .txt: TxtSplitActor turns bytes+path batches into chunk rows.
"""

from __future__ import annotations

from typing import Any, Dict, List  # noqa: F401

import pandas as pd

from nemo_retriever.params import TextChunkParams
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.graph.operator_archetype import ArchetypeOperator

from .split import txt_bytes_to_chunks_df


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

    def __init__(self, params: TextChunkParams | None = None) -> None:
        super().__init__()
        self._params = params or TextChunkParams()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        from .split import split_df

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
    Ray Data map_batches callable: DataFrame with bytes, path -> DataFrame of chunks.

    Each output row has: text, path, page_number, metadata (same shape as txt_file_to_chunks_df).
    """

    def __init__(self, params: TextChunkParams | None = None) -> None:
        super().__init__()
        self._params = params or TextChunkParams()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return pd.DataFrame(columns=["text", "path", "page_number", "metadata"])
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
            if (raw is None and text is None) or path is None:
                continue
            path_str = str(path) if path is not None else ""
            try:
                payload = raw or text.encode("utf-8")
                chunk_df = txt_bytes_to_chunks_df(payload, path_str, params=params)
                if not chunk_df.empty:
                    out_dfs.append(chunk_df)
            except Exception:
                continue
        if not out_dfs:
            return pd.DataFrame(columns=["text", "path", "page_number", "metadata"])
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
