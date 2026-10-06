# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet staging for ``retriever ingest batch --stage-dir``.

The ingest graph runs once and ends in :class:`StagedParquetDatasink`, whose
write tasks publish Parquet parts in the staging directory. :func:`load_stage`
then applies the in-driver sink's upload rules to the write results and loads
the reported parts through one ``LanceDB.ingest_arrow`` call, which validates
the row count and index coverage before it returns.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb.records import VdbConversionTally

logger = logging.getLogger(__name__)


class StagingError(RuntimeError):
    """A staging directory or staged run failed a check before loading."""


def prepare_stage_dir(stage_dir: str) -> str:
    """Create ``stage_dir`` or require it to be empty; return its absolute path."""
    stage_dir = os.path.abspath(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)
    if os.listdir(stage_dir):
        raise StagingError(f"Staging directory {stage_dir!r} is not empty; use a new or empty directory")
    return stage_dir


def _staged_reader(paths: Sequence[str]) -> pa.RecordBatchReader:
    """Read parts one row group at a time; each batch retains at most its decoded row group."""
    with pq.ParquetFile(paths[0]) as first:
        schema = first.schema_arrow

    def batches() -> Iterator[pa.RecordBatch]:
        for path in paths:
            with pq.ParquetFile(path) as parquet:
                for index in range(parquet.num_row_groups):
                    yield from parquet.read_row_group(index).to_batches()

    return pa.RecordBatchReader.from_batches(schema, batches())


def load_stage(
    stage_dir: str, results: Sequence[Mapping[str, Any]], *, lancedb_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace the LanceDB table with the parts that the write tasks reported.

    ``results`` are the datasink's per-task write results. Stage errors and
    missing embeddings fail the run before the table changes.
    """
    tally = VdbConversionTally()
    dropped: Counter[str] = Counter()
    for result in results:
        tally.merge(result["tally"])
        dropped.update(result["dropped"])
    stage_errors = sum(result["stage_errors"] for result in results)
    if stage_errors:
        raise StagingError(f"{stage_errors} staged row(s) have stage errors; the table was not changed")
    tally.raise_for_failures()
    if dropped:
        logger.warning("Parquet staging dropped rows: %s", dict(dropped))

    parts = [result["part"] for result in results if result["part"]]
    rows = sum(part["rows"] for part in parts)
    if not rows:
        raise StagingError("Staging produced no rows to load")
    vdb = LanceDB(**{**lancedb_kwargs, "overwrite": True})
    vdb.ingest_arrow(_staged_reader([os.path.join(stage_dir, part["name"]) for part in parts]), expected_rows=rows)
    return {
        "stage_dir": stage_dir,
        "uri": os.path.abspath(vdb.uri),
        "table_name": vdb.table_name,
        "input_rows": tally.rows,
        "rows": rows,
    }
