# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming-friendly accessors for graph-pipeline ingest output.

:class:`IngestResult` wraps the ingest output and exposes only accessors
that can be served by streaming/per-block reads:

- ``row_count()`` — uses ``Dataset.count()`` or ``len(df.index)``
- ``unique_source_count()`` — uses ``Dataset.unique(col)`` or ``Series.nunique``
- ``iter_records()`` — streams rows one at a time
- ``count_uploadable_vdb_records()`` — streams in batches, sums per-batch
- ``detection_summary()`` — feeds streaming rows into ``compute_detection_summary``
- ``write_parquet_dir(out_dir)`` — Ray Data writes per-block files into the dir;
  pandas-backed mode writes a single file inside the same dir for symmetry.

Three constructors map cleanly to the three ingest run modes:

- ``IngestResult.from_dataset(ds)`` — batch mode (Ray Dataset).
- ``IngestResult.from_dataframe(df)`` — inprocess mode (single pandas DataFrame).
- ``IngestResult.from_service(records)`` — service mode (list of records from SSE).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pandas as pd

from nemo_retriever.vdb.operators import VDB_UPLOADABLE_COLUMN

logger = logging.getLogger(__name__)


_VDB_RECORD_SCAN_BATCH = 1024


@dataclass
class IngestResult:
    """Streaming-friendly handle over an ingest result.

    Construct via the ``from_*`` classmethods rather than calling this directly.
    Exactly one of ``_dataframe``, ``_dataset``, ``_service_records`` is set.
    """

    _dataframe: Optional[pd.DataFrame] = None
    _dataset: Optional[Any] = None
    _service_records: Optional[list[dict[str, Any]]] = None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "IngestResult":
        return cls(_dataframe=df)

    @classmethod
    def from_dataset(cls, dataset: Any) -> "IngestResult":
        return cls(_dataset=dataset)

    @classmethod
    def from_service(cls, records: Iterable[dict[str, Any]]) -> "IngestResult":
        return cls(_service_records=list(records))

    # ------------------------------------------------------------------ counts

    def row_count(self) -> int:
        if self._dataset is not None:
            return int(self._dataset.count())
        if self._dataframe is not None:
            return int(len(self._dataframe.index))
        assert self._service_records is not None
        return len(self._service_records)

    def unique_source_count(self) -> int:
        """Number of distinct input units (used by run-summary reporting)."""
        if self._service_records is not None:
            # SSE results don't carry source-side columns today; the row count
            # is the closest available proxy for input units.
            return len(self._service_records)
        if self._dataset is not None:
            for col in ("source_id", "source_path"):
                try:
                    return len(self._dataset.unique(col))
                except (KeyError, ValueError):
                    continue
            return int(self._dataset.count())
        assert self._dataframe is not None
        df = self._dataframe
        if "source_id" in df.columns:
            return int(df["source_id"].nunique())
        if "source_path" in df.columns:
            return int(df["source_path"].nunique())
        return int(len(df.index))

    # ----------------------------------------------------------------- streams

    def iter_records(self) -> Iterator[dict[str, Any]]:
        if self._dataset is not None:
            yield from self._dataset.iter_rows()
            return
        if self._dataframe is not None:
            yield from self._dataframe.to_dict("records")
            return
        assert self._service_records is not None
        yield from self._service_records

    # ------------------------------------------------------- derived accessors

    def count_uploadable_vdb_records(self) -> int:
        """Count rows that produced a client-VDB record.

        Two branches in this method:

        - **Precomputed branch** (batch-mode result): ``IngestVdbOperator``
          already projected the heavy embedding/metadata payload away and
          emitted a ``_vdb_uploadable`` boolean column per row. Summing that
          Series is the cheapest possible answer and keeps Ray object store +
          driver memory bounded.

        - **Re-derive branch** (inprocess result or raw DataFrame without the
          flag): walk the rows and re-run ``to_client_vdb_records`` to decide
          uploadability per row. This is more expensive per row but only runs
          in modes that already materialize the corpus on the driver, so the
          cost is bounded by the on-driver row count.

        Service mode doesn't reach this method — ``pipeline/__main__.py`` uses
        the SSE row count directly for the same reporting field.
        """
        from nemo_retriever.vdb.records import to_client_vdb_records

        total = 0
        # Iterate as pandas batches so the ``_vdb_uploadable`` column check
        # is a single Series sum per batch.
        if self._dataset is not None:
            batches = self._dataset.iter_batches(batch_format="pandas", batch_size=_VDB_RECORD_SCAN_BATCH)
        elif self._dataframe is not None:
            batches = iter([self._dataframe])
        else:
            assert self._service_records is not None
            batches = iter([pd.DataFrame(self._service_records)])

        for batch_df in batches:
            if VDB_UPLOADABLE_COLUMN in batch_df.columns:
                total += int(batch_df[VDB_UPLOADABLE_COLUMN].sum())
            else:
                total += sum(len(group) for group in to_client_vdb_records(batch_df.to_dict("records")))
        return total

    def detection_summary(self) -> dict[str, Any]:
        from nemo_retriever.utils.detection_summary import (
            compute_detection_summary,
            iter_dataframe_rows,
        )

        # compute_detection_summary wants (page_key, meta, row_dict) tuples.
        # For DataFrame backing reuse the existing helper (it handles JSON-
        # string metadata); for dataset/service stream the same shape from
        # individual row dicts.
        if self._dataframe is not None:
            return compute_detection_summary(iter_dataframe_rows(self._dataframe))

        def _streaming_rows() -> Iterator[tuple[Any, dict[str, Any], dict[str, Any]]]:
            for row in self.iter_records():
                path = str(row.get("path") or row.get("source_id") or "")
                try:
                    page_number = int(row.get("page_number", -1))
                except (TypeError, ValueError):
                    page_number = -1
                meta = row.get("metadata")
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError as exc:
                        logger.debug(
                            "Could not parse metadata JSON for row %r: %s; treating as empty.",
                            path,
                            exc,
                        )
                        meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                yield (path, page_number), meta, row

        return compute_detection_summary(_streaming_rows())

    def write_parquet_dir(self, out_dir: Path | str) -> Path:
        """Write the result as a directory of per-block parquet files.

        Returns the output directory. Dataset-backed results use
        ``Dataset.write_parquet``; pandas-backed results write a single
        ``part-00000.parquet`` file inside ``out_dir`` so the on-disk shape
        is the same regardless of run mode.
        """
        out_path = Path(out_dir).expanduser().resolve()
        if self._dataset is not None:
            self._dataset.write_parquet(str(out_path))
            return out_path

        out_path.mkdir(parents=True, exist_ok=True)
        df = self._dataframe if self._dataframe is not None else pd.DataFrame(self._service_records or [])
        df.to_parquet(out_path / "part-00000.parquet", index=False)
        return out_path
