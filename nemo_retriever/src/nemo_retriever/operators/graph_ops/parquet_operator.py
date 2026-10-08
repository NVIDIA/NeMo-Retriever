# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operators that persist pipeline rows to Parquet and load them back."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from nemo_retriever.common.io.dataframe import read_extraction_parquet
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator


@designer_component(
    name="Parquet Writer",
    category="Document Processing",
    compute="cpu",
    description="Writes pipeline rows to a directory of Parquet files",
    category_color="#64b4ff",
)
class ParquetWriterOperator(AbstractOperator, CPUOperator):
    """Write each batch to ``output_dir`` as its own Parquet part file and pass the rows through.

    One part file per batch keeps the writer safe under executors that call it once per batch.
    Existing part files are never removed, so clear ``output_dir`` before a fresh run.

    Parameters
    ----------
    output_dir : str
        Directory that receives the ``part-<uuid>.parquet`` files. It is created on the first
        non-empty batch.
    exclude_columns : Sequence[str], optional
        Columns left out of the saved files, for example large image payloads. Names that are
        not in a batch are ignored. The rows passed downstream keep every column.
    """

    def __init__(self, *, output_dir: str, exclude_columns: Sequence[str] = ()) -> None:
        super().__init__(output_dir=output_dir, exclude_columns=tuple(exclude_columns))
        self._output_dir = Path(output_dir)
        self._exclude_columns = tuple(exclude_columns)

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        """Check that the batch is a DataFrame.

        Parameters
        ----------
        data : Any
            Batch of pipeline rows from the upstream operator.
        **kwargs : Any
            Unused; accepted for the operator interface.

        Returns
        -------
        pandas.DataFrame
            The batch, unchanged.

        Raises
        ------
        TypeError
            If ``data`` is not a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"ParquetWriterOperator expects a pandas DataFrame of pipeline rows, got {type(data).__name__}. "
                "Place it after an operator that emits DataFrame batches, such as ExplodeContentActor."
            )
        return data

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Write the batch to a new part file, without the excluded columns.

        Empty batches write nothing.

        Parameters
        ----------
        data : pandas.DataFrame
            Batch of pipeline rows.
        **kwargs : Any
            Unused; accepted for the operator interface.

        Returns
        -------
        pandas.DataFrame
            The batch, unchanged and with every column, so the operator can sit in the middle
            of a graph.

        Raises
        ------
        OSError
            If ``output_dir`` cannot be created or written to.
        """
        if data.empty:
            return data
        self._output_dir.mkdir(parents=True, exist_ok=True)
        rows = data.drop(columns=[c for c in self._exclude_columns if c in data.columns])
        rows.to_parquet(self._output_dir / f"part-{uuid.uuid4().hex}.parquet", index=False)
        return data

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Return the batch unchanged.

        Parameters
        ----------
        data : pandas.DataFrame
            Batch returned by :meth:`process`.
        **kwargs : Any
            Unused; accepted for the operator interface.

        Returns
        -------
        pandas.DataFrame
            The batch, unchanged.
        """
        return data


@designer_component(
    name="Parquet Reader",
    category="Document Processing",
    compute="cpu",
    description="Loads pipeline rows from Parquet files or directories",
    category_color="#64b4ff",
)
class ParquetReaderOperator(AbstractOperator, CPUOperator):
    """Load rows written by :class:`ParquetWriterOperator` (or any row-per-record Parquet).

    Accepts a path, a list of paths, or a DataFrame with a ``path`` column such as the one
    ``InprocessExecutor.ingest`` builds. A directory contributes every ``*.parquet`` file in it.
    The operator takes no constructor arguments.
    """

    def preprocess(self, data: Any, **kwargs: Any) -> list[Path]:
        """Resolve the input into a list of Parquet files.

        Parameters
        ----------
        data : str, pathlib.Path, list, or pandas.DataFrame
            A file or directory path, a list of them, or a DataFrame whose ``path`` column holds
            them. A directory expands to its ``*.parquet`` files in sorted order; subdirectories
            are not searched.
        **kwargs : Any
            Unused; accepted for the operator interface.

        Returns
        -------
        list[pathlib.Path]
            Parquet files to read, in order.

        Raises
        ------
        TypeError
            If ``data`` is none of the supported input types.
        FileNotFoundError
            If a listed path does not exist.
        """
        if isinstance(data, (str, Path)):
            sources = [data]
        elif isinstance(data, list):
            sources = data
        elif isinstance(data, pd.DataFrame) and "path" in data.columns:
            sources = data["path"].tolist()
        else:
            raise TypeError(
                "data must be a Parquet path, a list of paths, or a DataFrame with a 'path' column, "
                f"got {type(data).__name__}"
            )
        files: list[Path] = []
        for source in sources:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(
                    f"Parquet path {str(path)!r} does not exist. Pass the output_dir that "
                    "ParquetWriterOperator wrote to, or a .parquet file inside it."
                )
            files.extend(sorted(path.glob("*.parquet")) if path.is_dir() else [path])
        return files

    def process(self, data: list[Path], **kwargs: Any) -> pd.DataFrame:
        """Read the files and concatenate their rows.

        Parameters
        ----------
        data : list[pathlib.Path]
            Parquet files from :meth:`preprocess`.
        **kwargs : Any
            Unused; accepted for the operator interface.

        Returns
        -------
        pandas.DataFrame
            All rows in file order with a fresh index, or an empty DataFrame when there are no
            files or every file is empty.
        """
        frames = [read_extraction_parquet(path) for path in data]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Return the rows unchanged.

        Parameters
        ----------
        data : pandas.DataFrame
            Rows returned by :meth:`process`.
        **kwargs : Any
            Unused; accepted for the operator interface.

        Returns
        -------
        pandas.DataFrame
            The rows, unchanged.
        """
        return data
