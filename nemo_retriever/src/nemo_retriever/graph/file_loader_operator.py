# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operators for loading file paths into DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator


class FileListLoaderOperator(AbstractOperator, CPUOperator):
    """Load a list of files into a DataFrame with ``path`` and ``bytes`` columns."""

    def preprocess(self, data: Any, **kwargs: Any) -> list[str]:
        if isinstance(data, (str, Path)):
            return [str(data)]
        if isinstance(data, list):
            return [str(item) for item in data]
        raise TypeError(f"data must be a file path or list of file paths, got {type(data).__name__}")

    def process(self, data: list[str], **kwargs: Any) -> pd.DataFrame:
        rows = []
        for file_path in data:
            path = Path(file_path)
            if path.is_file():
                rows.append({"path": str(path.resolve()), "bytes": path.read_bytes()})
        if not rows:
            return pd.DataFrame(columns=["path", "bytes"])
        return pd.DataFrame(rows, columns=["path", "bytes"])

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data
