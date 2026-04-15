# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ray Data adapter for images: ImageLoadActor turns bytes+path batches into page rows.
"""

from __future__ import annotations

from typing import Any, List

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.graph.operator_archetype import ArchetypeOperator

from .load import image_bytes_to_pages_df

# Output columns matching the PDF extraction schema.
_PAGE_COLUMNS = [
    "path",
    "page_number",
    "source_id",
    "text",
    "page_image",
    "images",
    "tables",
    "charts",
    "infographics",
    "metadata",
]


@designer_component(
    name="Image Loader",
    category="Text & Content",
    compute="cpu",
    description="Loads and prepares images for processing",
    category_color="#42d6a4",
)
class ImageLoadCPUActor(AbstractOperator, CPUOperator):
    """
    Ray Data map_batches callable: DataFrame with bytes, path -> DataFrame of page rows.

    Each output row matches the PDF extraction schema so downstream GPU stages
    (page-element detection, OCR, table/chart/infographic extraction) work unchanged.
    """

    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return pd.DataFrame(columns=_PAGE_COLUMNS)
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return data

        out_dfs: List[pd.DataFrame] = []
        for _, row in data.iterrows():
            raw = row.get("bytes")
            path = row.get("path")
            if raw is None or path is None:
                continue
            path_str = str(path) if path is not None else ""
            try:
                page_df = image_bytes_to_pages_df(raw, path_str)
                if not page_df.empty:
                    out_dfs.append(page_df)
            except ImportError:
                raise
            except Exception:
                continue
        if not out_dfs:
            return pd.DataFrame(columns=_PAGE_COLUMNS)
        return pd.concat(out_dfs, ignore_index=True)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        return self.run(batch_df)


class ImageLoadActor(ArchetypeOperator):
    _cpu_variant_class = ImageLoadCPUActor

    def __init__(self) -> None:
        super().__init__()
