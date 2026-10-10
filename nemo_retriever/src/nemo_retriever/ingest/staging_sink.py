# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ray datasink that stages embedded graph rows as Parquet from write tasks.

Write tasks convert rows with the same record and LanceDB row functions as the
in-driver sink and write the cached-vector schema that ``LanceDB.ingest_arrow``
loads. The driver receives only per-task counts and part names.
"""

from __future__ import annotations

import dataclasses
import os
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ray.data.datasource import Datasink

from nemo_retriever.common.vdb.arrow import cached_vector_schema
from nemo_retriever.common.vdb.lancedb import LanceDB, _create_lancedb_result
from nemo_retriever.common.vdb.records import VdbConversionTally, VdbUploadError

_ROW_GROUP_ROWS = 4096
# A Parquet reader batch retains its whole decoded row group, so keep groups
# well below the cached loader's default 256 MiB retained-bytes limit.
_ROW_GROUP_BYTES = 64 << 20


def fsync_path(path: str) -> None:
    """Flush a file's data, or a directory's entries, to disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class StageTarget:
    """Where write tasks publish Parquet parts for one staged run."""

    stage_dir: str
    rows_per_part: int = 8192
    write_concurrency: int = 8
    stage_error_columns: tuple[str, ...] = ()


def _row_group_rows(batch: pa.RecordBatch) -> int:
    widths = 4 * batch.schema.field("vector").type.list_size + sum(
        pc.binary_length(batch.column(name)).to_numpy(zero_copy_only=False)
        for name in ("id", "text", "source", "metadata")
    )
    return max(1, min(_ROW_GROUP_ROWS, _ROW_GROUP_BYTES // max(1, int(widths.max()))))


class StagedParquetDatasink(Datasink[dict]):
    """Write each Ray write task's rows as one atomically published Parquet part."""

    def __init__(self, target: StageTarget, vdb: LanceDB) -> None:
        # The cached loader needs dense, finite vectors, so a null-vector policy cannot be staged.
        if vdb.sparse or vdb.on_bad_vectors == "null":
            raise ValueError("Parquet staging needs dense vectors and on_bad_vectors 'drop', 'fill', or 'error'")
        self.target = target
        self.results: list[dict[str, Any]] = []
        self._vector_dim = vdb.vector_dim
        self._expected_dim = vdb.vector_dim if vdb.validate_vector_length and vdb.on_bad_vectors != "error" else None
        self._on_bad_vectors = vdb.on_bad_vectors
        self._fill_value = float(vdb.fill_value)

    @property
    def min_rows_per_write(self) -> int:
        return self.target.rows_per_part

    def get_name(self) -> str:
        return "StageParquet"

    def write(self, blocks: Iterable[Any], ctx: Any) -> dict[str, Any]:
        from nemo_retriever.graph.executor import arrow_table_to_pandas

        tally = VdbConversionTally()
        dropped: Counter[str] = Counter()
        columns: dict[str, list[Any]] = {"vector": [], "id": [], "text": [], "source": [], "metadata": []}
        stage_errors = 0
        for block in blocks:
            frame = arrow_table_to_pandas(block)
            if self.target.stage_error_columns:
                from nemo_retriever.ingestor.graph_ingestor import GraphIngestor

                stage_errors += len(GraphIngestor._stage_error_records(frame, columns=self.target.stage_error_columns))
            for row in frame.to_dict(orient="records"):
                record = tally.convert(row)
                if record is None:
                    continue
                lance_row, reason = _create_lancedb_result(record, expected_dim=self._expected_dim)
                vector = None if lance_row is None else self._vector(lance_row["vector"])
                if vector is None:
                    dropped[reason or "dropped_bad_vector"] += 1
                    continue
                columns["vector"].append(vector)
                for name in ("id", "text", "source", "metadata"):
                    columns[name].append(lance_row[name])
        return {
            "tally": dataclasses.asdict(tally),
            "dropped": dict(dropped),
            "stage_errors": stage_errors,
            "part": self._publish(columns) if columns["id"] else None,
        }

    def on_write_complete(self, write_result: Any) -> None:
        self.results = list(write_result.write_returns)

    def _vector(self, value: Any) -> np.ndarray | None:
        """Apply ``on_bad_vectors``; unlike the in-driver writer, null and infinite values count as bad."""
        try:
            vector = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise VdbUploadError("vdb_upload received an embedding that cannot be converted to float32") from exc
        if self._vector_dim is None and vector.ndim == 1 and vector.size:
            self._vector_dim = int(vector.size)
        if vector.ndim == 1 and vector.size == self._vector_dim and np.isfinite(vector).all():
            return vector
        if self._on_bad_vectors == "drop":
            return None
        if self._on_bad_vectors == "fill":
            return np.full(self._vector_dim, self._fill_value, dtype=np.float32)
        raise ValueError(
            f"Invalid LanceDB vector: expected {self._vector_dim} finite values. "
            "Set on_bad_vectors to 'drop' or 'fill' to handle it."
        )

    def _publish(self, columns: dict[str, list[Any]]) -> dict[str, Any]:
        """Write a temporary file, fsync it, and rename it into the staging directory."""
        dim = int(self._vector_dim)
        values = pa.array(np.stack(columns["vector"]).reshape(-1), type=pa.float32())
        batch = pa.RecordBatch.from_arrays(
            [pa.FixedSizeListArray.from_arrays(values, dim)]
            + [pa.array(columns[name], type=pa.string()) for name in ("id", "text", "source", "metadata")],
            schema=cached_vector_schema(dim),
        )
        # The loader reads only the parts that successful write tasks report,
        # so a part left behind by a failed task attempt is never loaded.
        name = f"part-{uuid.uuid4().hex}.parquet"
        directory = self.target.stage_dir
        temp_path = os.path.join(directory, f".{name}.tmp")
        try:
            pq.write_table(pa.Table.from_batches([batch]), temp_path, row_group_size=_row_group_rows(batch))
            fsync_path(temp_path)
            os.replace(temp_path, os.path.join(directory, name))
        except BaseException:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
        fsync_path(directory)
        return {"name": name, "rows": batch.num_rows}
