# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage canonical LanceDB rows as Parquet from Ray write tasks.

Each Ray write task converts its embedded graph rows with the record and
LanceDB row functions that the in-driver sink uses, then writes one Parquet
part in the cached-vector schema of :mod:`nemo_retriever.common.vdb.arrow`.
A part is published atomically: temporary file, fsync, rename to a name
derived from its row IDs, directory fsync. The driver receives only each
task's part metadata and counts; row data never passes through it.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ray.data.datasource import Datasink

from nemo_retriever.common.stage_errors import iter_stage_errors_from_value
from nemo_retriever.common.vdb.arrow import cached_vector_schema
from nemo_retriever.common.vdb.lancedb import LanceDB, _create_lancedb_result
from nemo_retriever.common.vdb.records import (
    VdbUploadError,
    _client_record_from_graph_row,
    _row_has_uploadable_content_without_embedding,
    _stage_error_field,
)

SHARDS_DIRNAME = "shards"
ATTEMPT_INFIX = ".attempt-"
PART_SUFFIX = ".parquet"
TEMP_SUFFIX = ".tmp"
_ROW_GROUP_ROWS = 4096
# Bound the diagnostics one write task returns to the driver.
_MAX_REPORTED_ITEMS = 20


def shard_dir(stage_dir: str, shard_id: str) -> str:
    """Return the directory holding one committed shard's parts."""
    return os.path.join(stage_dir, SHARDS_DIRNAME, shard_id)


def attempt_dir(stage_dir: str, shard_id: str, attempt: str) -> str:
    """Return the directory one staging attempt publishes into before its shard commits."""
    return os.path.join(stage_dir, SHARDS_DIRNAME, f"{shard_id}{ATTEMPT_INFIX}{attempt}")


def fsync_dir(path: str) -> None:
    """Make renames and creations inside ``path`` durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class StageTarget:
    """Where one staging attempt's Ray write tasks publish Parquet parts.

    Parts go to a per-attempt directory; the driver renames it to the shard
    directory only after verifying every part, so a write task that outlives
    a killed run can never add files to a later attempt or a committed shard.
    ``rows_per_part`` bundles input blocks so each write task publishes one
    part of at least that many rows, except the last. ``write_concurrency``
    caps concurrent write tasks. ``stage_error_columns`` names graph columns
    whose populated stage errors must fail the shard.
    """

    stage_dir: str
    shard_id: str
    attempt: str
    rows_per_part: int = 8192
    write_concurrency: int = 8
    stage_error_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not os.path.isabs(self.stage_dir):
            raise ValueError("stage_dir must be an absolute path")
        for name, value in (("shard_id", self.shard_id), ("attempt", self.attempt)):
            if not value or os.sep in value or value.startswith(".") or ATTEMPT_INFIX in value:
                raise ValueError(f"Invalid {name} {value!r}")
        if self.rows_per_part <= 0 or self.write_concurrency <= 0:
            raise ValueError("rows_per_part and write_concurrency must be positive")
        object.__setattr__(self, "stage_error_columns", tuple(self.stage_error_columns))

    @property
    def attempt_dir(self) -> str:
        return attempt_dir(self.stage_dir, self.shard_id, self.attempt)


@dataclass(frozen=True)
class StagedRowPolicy:
    """The configured LanceDB row policy that staged rows must satisfy.

    The cached-vector loader requires finite, non-null float32 vectors, so
    ``on_bad_vectors`` also governs null and infinite elements here; the
    in-driver writer passes those to LanceDB unchanged. A ``null`` policy
    cannot be staged.
    """

    vector_dim: int | None
    on_bad_vectors: str
    fill_value: float
    validate_vector_length: bool
    embedding_model_name: str | None
    embedding_model_revision: str | None

    @classmethod
    def from_lancedb(cls, vdb: Any) -> StagedRowPolicy:
        if not isinstance(vdb, LanceDB):
            raise ValueError(f"Parquet staging requires a LanceDB VDB, got {type(vdb).__name__}")
        if vdb.sparse:
            raise ValueError("Parquet staging requires a dense or hybrid LanceDB table")
        if vdb.on_bad_vectors == "null":
            raise ValueError("Parquet staging cannot store null vectors; use on_bad_vectors 'drop', 'fill', or 'error'")
        if vdb.on_bad_vectors == "fill" and not np.isfinite(vdb.fill_value):
            raise ValueError("Parquet staging requires a finite fill_value")
        return cls(
            vector_dim=vdb.vector_dim,
            on_bad_vectors=vdb.on_bad_vectors,
            fill_value=float(vdb.fill_value),
            validate_vector_length=vdb.validate_vector_length,
            embedding_model_name=vdb.embedding_model_name,
            embedding_model_revision=vdb.embedding_model_revision,
        )


class _TaskRows:
    """Accumulate one write task's staged columns and record accounting."""

    def __init__(self, policy: StagedRowPolicy) -> None:
        self.policy = policy
        self.vector_dim = policy.vector_dim
        self.expected_dim = (
            policy.vector_dim if policy.validate_vector_length and policy.on_bad_vectors != "error" else None
        )
        self.vectors: list[np.ndarray] = []
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.sources: list[str] = []
        self.metadata: list[str] = []
        self.graph_rows = 0
        self.records = 0
        self.missing_embeddings = 0
        self.upstream_errors = 0
        self.upstream_error_fields: Counter[str] = Counter()
        self.rejected: Counter[str] = Counter()
        self.dropped: Counter[str] = Counter()

    def add_graph_row(self, row: dict[str, Any]) -> None:
        """Mirror ``_iter_client_vdb_records`` and ``LanceDB._iter_stream_rows`` for one row."""
        self.graph_rows += 1
        record = _client_record_from_graph_row(row)
        if record is None:
            missing_embedding = _row_has_uploadable_content_without_embedding(row)
            self.missing_embeddings += int(missing_embedding)
            upstream_errors = list(iter_stage_errors_from_value(row))
            if upstream_errors:
                self.upstream_errors += len(upstream_errors)
                self.upstream_error_fields.update(_stage_error_field(error.get("path")) for error in upstream_errors)
            else:
                self.rejected[
                    "missing embedding" if missing_embedding else "missing searchable text or image backing"
                ] += 1
            return
        self.records += 1
        lance_row, drop_reason = _create_lancedb_result(record, expected_dim=self.expected_dim)
        if drop_reason is not None or lance_row is None:
            self.dropped[drop_reason or "dropped"] += 1
            return
        vector = self._checked_vector(lance_row["vector"])
        if vector is None:
            return
        self.vectors.append(vector)
        self.ids.append(lance_row["id"])
        self.texts.append(lance_row["text"])
        self.sources.append(lance_row["source"])
        self.metadata.append(lance_row["metadata"])

    def _checked_vector(self, value: Any) -> np.ndarray | None:
        try:
            vector = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise VdbUploadError("vdb_upload received an embedding that cannot be converted to float32") from exc
        if self.vector_dim is None and vector.ndim == 1 and vector.size:
            self.vector_dim = int(vector.size)
        if vector.ndim == 1 and vector.size == self.vector_dim and np.isfinite(vector).all():
            return vector
        if self.policy.on_bad_vectors == "drop":
            self.dropped["dropped_bad_vector"] += 1
            return None
        if self.policy.on_bad_vectors == "fill" and self.vector_dim is not None:
            return np.full(self.vector_dim, self.policy.fill_value, dtype=np.float32)
        raise ValueError(
            f"Invalid LanceDB vector: expected {self.vector_dim} finite values. "
            "Set on_bad_vectors to 'drop' or 'fill' to handle it."
        )

    def record_batch(self) -> pa.RecordBatch:
        dim = int(self.vector_dim or 0)
        schema = cached_vector_schema(dim, self.policy.embedding_model_name, self.policy.embedding_model_revision)
        values = pa.array(np.stack(self.vectors).reshape(-1), type=pa.float32())
        columns = [
            pa.FixedSizeListArray.from_arrays(values, dim),
            pa.array(self.ids, type=pa.string()),
            pa.array(self.texts, type=pa.string()),
            pa.array(self.sources, type=pa.string()),
            pa.array(self.metadata, type=pa.string()),
        ]
        return pa.RecordBatch.from_arrays(columns, schema=schema)

    def summary(self) -> dict[str, Any]:
        duplicate_ids = [row_id for row_id, count in Counter(self.ids).items() if count > 1]
        return {
            "graph_rows": self.graph_rows,
            "records": self.records,
            "rows": len(self.ids),
            "missing_embeddings": self.missing_embeddings,
            "upstream_errors": self.upstream_errors,
            "upstream_error_fields": dict(self.upstream_error_fields),
            "rejected": dict(self.rejected),
            "dropped": dict(self.dropped),
            "duplicate_ids": duplicate_ids[:_MAX_REPORTED_ITEMS],
            "vector_dim": self.vector_dim,
        }


def part_name(row_ids: Sequence[str]) -> str:
    """Name a part by its sorted row IDs, so a retried task republishes the same file."""
    digest = hashlib.sha256("\n".join(sorted(row_ids)).encode("utf-8")).hexdigest()
    return f"part-{digest[:32]}{PART_SUFFIX}"


def publish_part(batch: pa.RecordBatch, *, stage_dir: str, directory: str, name: str) -> dict[str, Any]:
    """Write one Parquet part to a temporary file, fsync it, and rename it into place."""
    os.makedirs(directory, exist_ok=True)
    final_path = os.path.join(directory, name)
    temp_path = os.path.join(directory, f".{name}.{uuid.uuid4().hex}{TEMP_SUFFIX}")
    try:
        pq.write_table(pa.Table.from_batches([batch]), temp_path, row_group_size=_ROW_GROUP_ROWS)
        fd = os.open(temp_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        size = os.path.getsize(temp_path)
        checksum = file_sha256(temp_path)
        os.replace(temp_path, final_path)
    except BaseException:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    fsync_dir(directory)
    return {
        "path": os.path.relpath(final_path, stage_dir),
        "rows": batch.num_rows,
        "bytes": size,
        "sha256": checksum,
    }


def _stage_error_records(frame: Any, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    from nemo_retriever.ingestor.graph_ingestor import GraphIngestor

    return GraphIngestor._stage_error_records(frame, columns=set(columns))


def stage_blocks(blocks: Iterable[Any], *, target: StageTarget, policy: StagedRowPolicy) -> dict[str, Any]:
    """Convert one write task's blocks and publish them as at most one Parquet part."""
    from nemo_retriever.graph.executor import arrow_table_to_pandas

    rows = _TaskRows(policy)
    stage_errors: list[dict[str, Any]] = []
    stage_error_count = 0
    for block in blocks:
        frame = arrow_table_to_pandas(block)
        if target.stage_error_columns:
            errors = _stage_error_records(frame, target.stage_error_columns)
            stage_error_count += len(errors)
            stage_errors.extend(errors[: max(0, _MAX_REPORTED_ITEMS - len(stage_errors))])
        for row in frame.to_dict(orient="records"):
            rows.add_graph_row(row)

    result = rows.summary()
    result["stage_errors"] = stage_errors
    result["stage_error_count"] = stage_error_count
    result["parts"] = []
    if rows.ids and not result["duplicate_ids"]:
        result["parts"].append(
            publish_part(
                rows.record_batch(),
                stage_dir=target.stage_dir,
                directory=target.attempt_dir,
                name=part_name(rows.ids),
            )
        )
    return result


def read_part_ids(stage_dir: str, part_paths: Sequence[str]) -> pa.ChunkedArray:
    """Read only the ``id`` column of the given staged parts."""
    tables = [pq.read_table(os.path.join(stage_dir, path), columns=["id"]) for path in part_paths]
    if not tables:
        return pa.chunked_array([], type=pa.string())
    return pa.concat_tables(tables).column("id")


def find_duplicate_ids(stage_dir: str, part_paths: Sequence[str], *, limit: int = _MAX_REPORTED_ITEMS) -> list[str]:
    """Return up to ``limit`` IDs that occur more than once across the given parts."""
    counts = read_part_ids(stage_dir, part_paths).value_counts()
    repeated = counts.filter(pc.greater(counts.field("counts"), 1))
    return [str(value) for value in repeated.field("values").to_pylist()[:limit]]


class StagedParquetDatasink(Datasink[dict]):
    """Ray datasink that stages one shard's rows as atomically published Parquet parts."""

    def __init__(self, target: StageTarget, policy: StagedRowPolicy) -> None:
        self.target = target
        self.policy = policy
        self.results: list[dict[str, Any]] | None = None

    @classmethod
    def for_vdb(cls, target: StageTarget | Mapping[str, Any], vdb: Any) -> StagedParquetDatasink:
        stage_target = target if isinstance(target, StageTarget) else StageTarget(**dict(target))
        return cls(stage_target, StagedRowPolicy.from_lancedb(vdb))

    @property
    def min_rows_per_write(self) -> int:
        return self.target.rows_per_part

    def get_name(self) -> str:
        return "StageParquet"

    def on_write_start(self, schema: Any = None) -> None:
        os.makedirs(self.target.attempt_dir, exist_ok=True)

    def write(self, blocks: Iterable[Any], ctx: Any) -> dict[str, Any]:
        return stage_blocks(blocks, target=self.target, policy=self.policy)

    def on_write_complete(self, write_result: Any) -> None:
        self.results = list(write_result.write_returns)
