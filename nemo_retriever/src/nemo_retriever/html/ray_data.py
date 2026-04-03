# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ray Data adapter for .html: HtmlSplitActor turns bytes+path batches into chunk rows.
"""

from __future__ import annotations

from typing import Any, List

import pandas as pd

from nemo_retriever.params import HtmlChunkParams
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.operator_archetype import ArchetypeOperator

from .convert import html_bytes_to_chunks_df


class HtmlSplitCPUActor(AbstractOperator, CPUOperator):
    """
    Ray Data map_batches callable: DataFrame with bytes, path -> DataFrame of chunks.

    Each output row has: text, path, page_number, metadata (same shape as html_file_to_chunks_df).
    """

    def __init__(self, params: HtmlChunkParams | None = None) -> None:
        super().__init__()
        self._params = params or HtmlChunkParams()

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
                chunk_df = html_bytes_to_chunks_df(payload, path_str, params=params)
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


class HtmlSplitActor(ArchetypeOperator):
    _cpu_variant_class = HtmlSplitCPUActor

    def __init__(self, params: HtmlChunkParams | None = None) -> None:
        super().__init__(params=params)
        self._params = params
