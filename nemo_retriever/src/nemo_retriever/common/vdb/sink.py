# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, single-commit ingestion for the Ray Data VDB boundary.

This module deliberately owns the entire stream-to-Lance lifecycle.  Ray
blocks are inputs, not transactions: one coordinator projects them to the
stored schema, emits byte-bounded Arrow batches, performs one Lance data
mutation, validates it, and only then builds the requested indexes.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import struct
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from nemo_retriever.common.vdb.sink_operation import (
    CommitOutcomeUnknown,
    SinkOperationMarkers,
    VdbOperationConflict,
)

_CREATE_OPERATION_KEY = b"nemo_retriever.sink_create_operation_sha256"
_CREATE_REQUEST_KEY = b"nemo_retriever.sink_create_request_sha256"
_MAX_PENDING_CANONICAL_ROWS = 256
_TIMING_PHASES = (
    "prepare",
    "write",
    "time_to_data_commit",
    "validate",
    "vector_index",
    "fts_index",
    "index",
    "optimize",
    "total",
)


@dataclass(frozen=True, slots=True)
class VdbSinkPolicy:
    """Memory and read-ahead policy for one Ray Data VDB sink operation."""

    max_batch_bytes: int = 256 << 20
    prefetch_batches: int = 1
    optimize: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_batch_bytes, bool) or int(self.max_batch_bytes) <= 0:
            raise ValueError("max_batch_bytes must be a positive integer")
        if isinstance(self.prefetch_batches, bool) or int(self.prefetch_batches) < 0:
            raise ValueError("prefetch_batches must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class VdbWriteReport:
    """Observable result of one completed VDB sink lifecycle."""

    operation_id: str
    outcome: str
    configured_max_batch_bytes: int
    configured_prefetch_batches: int
    input_batches: int
    input_rows: int
    input_bytes: int
    max_input_batch_bytes: int
    output_batches: int
    rows_written: int
    logical_bytes: int
    max_batch_bytes: int
    max_pending_rows: int
    versions_before: int
    versions_after: int
    fragments_before: int
    fragments_after: int
    data_files_before: int
    data_files_after: int
    write_rows_per_second: float
    write_bytes_per_second: float
    data_version: int | None
    final_version: int | None
    timings: dict[str, float]
    terminal_result_bytes: int | None = None


class OversizedVdbRowError(ValueError):
    """One canonical stored row cannot fit in the configured Arrow budget."""


class VdbWriteNotFinalized(RuntimeError):
    """A reader reached a table whose coordinated sink lifecycle is incomplete."""


def assert_lancedb_table_ready(table: Any) -> None:
    """Reject reads while a bounded sink operation is pending finalization."""

    tags = table.tags.list()
    incomplete = sorted(name for name in tags if name.startswith(("nemo_sink_pending_", "nemo_sink_data_")))
    if incomplete:
        raise VdbWriteNotFinalized(
            f"LanceDB table {table.name!r} has a data write that is not finalized; "
            "retry or reconcile the original VDB sink operation before reading."
        )

    if not _bounded_create_is_finalized(table):
        raise VdbWriteNotFinalized(
            f"LanceDB table {table.name!r} was created by a sink operation that is not finalized."
        )


def _bounded_create_is_finalized(table: Any) -> bool:
    """Return whether namespaced bounded-create metadata has a success marker."""

    metadata = table.schema.metadata or {}
    create_operation = metadata.get(_CREATE_OPERATION_KEY)
    create_request = metadata.get(_CREATE_REQUEST_KEY)
    if create_operation is None and create_request is None:
        return True
    if create_operation is None or create_request is None:
        raise VdbWriteNotFinalized(
            f"LanceDB table {table.name!r} has incomplete sink creation metadata and is not finalized."
        )
    success_prefix = (
        f"nemo_sink_success_{create_operation.decode('ascii')[:24]}_" f"{create_request.decode('ascii')[:24]}_"
    )
    return any(name.startswith(success_prefix) for name in table.tags.list())


@dataclass(slots=True)
class _StreamStats:
    input_batches: int = 0
    input_rows: int = 0
    input_bytes: int = 0
    max_input_batch_bytes: int = 0
    client_records: int = 0
    output_batches: int = 0
    rows_written: int = 0
    logical_bytes: int = 0
    max_batch_bytes: int = 0
    max_pending_rows: int = 0
    canonical_hash_sum: int = 0
    canonical_hash_xor: int = 0
    vector_dim: int | None = None

    @property
    def digest(self) -> str:
        width = 32
        modulus = 1 << (width * 8)
        payload = (
            int(self.rows_written).to_bytes(8, "big")
            + int(self.canonical_hash_sum % modulus).to_bytes(width, "big")
            + int(self.canonical_hash_xor).to_bytes(width, "big")
        )
        return hashlib.sha256(payload).hexdigest()


def _estimated_arrow_row_bytes(row: dict[str, Any], schema: pa.Schema) -> int:
    """Return a conservative buffer-size estimate for one canonical row."""

    # Offset/validity/alignment overhead is deliberately over-counted.  The
    # emitted RecordBatch is still measured exactly before it crosses into
    # LanceDB.
    total = 64
    for field in schema:
        value = row.get(field.name)
        if pa.types.is_fixed_size_list(field.type):
            # Arrow materializes the fixed-width child buffer even when the
            # parent list is null.
            total += 32 + int(field.type.list_size) * 4
        elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            total += 32 + (len(str(value).encode("utf-8")) if value is not None else 0)
        elif pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            total += 16
        elif value is None:
            total += 8
        else:
            total += 64 + len(str(value).encode("utf-8"))
    return total


def _exact_single_arrow_row_bytes(row: dict[str, Any], schema: pa.Schema) -> int | None:
    """Compute one canonical row's Arrow buffers without allocating them."""

    total = 0
    for field in schema:
        value = row.get(field.name)
        field_type = field.type
        if pa.types.is_fixed_size_list(field_type) and pa.types.is_float32(field_type.value_type):
            list_size = int(field_type.list_size)
            total += list_size * 4
            if value is None:
                total += 1 + ((list_size + 7) // 8)
            elif any(item is None for item in value):
                total += (list_size + 7) // 8
        elif pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
            total += 16 if pa.types.is_large_string(field_type) else 8
            if value is None:
                total += 1
            else:
                total += len(str(value).encode("utf-8"))
        elif pa.types.is_integer(field_type) or pa.types.is_floating(field_type):
            total += int(field_type.bit_width) // 8
            if value is None:
                total += 1
        else:
            return None
    return total


def _record_batch(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist(rows, schema=schema)


_CANONICAL_ROW_DIGEST_DOMAIN = b"nemo-retriever-vdb-row-v1\0"
_UINT64_BE = struct.Struct(">Q")


def _validity_bit(bitmap: memoryview | None, index: int) -> bool:
    return bitmap is None or bool(bitmap[index >> 3] & (1 << (index & 7)))


def _all_valid_bitmap(size: int) -> bytes:
    bitmap = bytearray(b"\xff" * ((size + 7) // 8))
    if size & 7:
        bitmap[-1] = (1 << (size & 7)) - 1
    return bytes(bitmap)


def _column_digest_writer(array: pa.Array):
    """Build a row writer over one canonical Arrow column's existing buffers."""

    array_type = array.type
    array_offset = int(array.offset)
    buffers = array.buffers()
    validity = memoryview(buffers[0]).cast("B") if buffers[0] is not None else None

    if pa.types.is_string(array_type) or pa.types.is_large_string(array_type):
        offset_width = 8 if pa.types.is_large_string(array_type) else 4
        offset_format = "<q" if offset_width == 8 else "<i"
        offsets = buffers[1]
        data = memoryview(buffers[2]).cast("B") if buffers[2] is not None else memoryview(b"")

        def write_string(hasher: Any, row_index: int) -> None:
            absolute_index = array_offset + row_index
            if not _validity_bit(validity, absolute_index):
                hasher.update(b"s\0")
                return
            start = struct.unpack_from(offset_format, offsets, absolute_index * offset_width)[0]
            stop = struct.unpack_from(offset_format, offsets, (absolute_index + 1) * offset_width)[0]
            hasher.update(b"s\1")
            hasher.update(_UINT64_BE.pack(stop - start))
            hasher.update(data[start:stop])

        return write_string

    if pa.types.is_int32(array_type):
        data = memoryview(buffers[1]).cast("B")

        def write_int32(hasher: Any, row_index: int) -> None:
            absolute_index = array_offset + row_index
            if not _validity_bit(validity, absolute_index):
                hasher.update(b"i\0")
                return
            start = absolute_index * 4
            hasher.update(b"i\1")
            hasher.update(data[start : start + 4])

        return write_int32

    if pa.types.is_fixed_size_list(array_type) and pa.types.is_float32(array_type.value_type):
        list_size = int(array_type.list_size)
        values = array.values
        value_buffers = values.buffers()
        value_validity = memoryview(value_buffers[0]).cast("B") if value_buffers[0] is not None else None
        value_data = memoryview(value_buffers[1]).cast("B") if value_buffers[1] is not None else memoryview(b"")
        value_offset = int(values.offset)
        all_valid = _all_valid_bitmap(list_size)

        def write_float32_list(hasher: Any, row_index: int) -> None:
            absolute_index = array_offset + row_index
            if not _validity_bit(validity, absolute_index):
                hasher.update(b"l\0")
                return

            first_value = value_offset + absolute_index * list_size
            data_start = first_value * 4
            data_stop = data_start + list_size * 4
            hasher.update(b"l\1")
            if value_validity is None:
                hasher.update(all_valid)
                hasher.update(value_data[data_start:data_stop])
                return

            normalized_validity = bytearray(len(all_valid))
            normalized_values = bytearray(value_data[data_start:data_stop])
            for value_index in range(list_size):
                if _validity_bit(value_validity, first_value + value_index):
                    normalized_validity[value_index >> 3] |= 1 << (value_index & 7)
                else:
                    start = value_index * 4
                    normalized_values[start : start + 4] = b"\0\0\0\0"
            hasher.update(normalized_validity)
            hasher.update(normalized_values)

        return write_float32_list

    raise TypeError(f"Unsupported canonical VDB digest field type: {array_type}")


def _record_canonical_batch(stats: _StreamStats, batch: pa.RecordBatch) -> None:
    """Update a partition-order-independent identity from canonical Arrow buffers."""

    writers = [_column_digest_writer(column) for column in batch.columns]
    for row_index in range(batch.num_rows):
        hasher = hashlib.sha256(_CANONICAL_ROW_DIGEST_DOMAIN)
        for write_column in writers:
            write_column(hasher, row_index)
        row_hash = int.from_bytes(hasher.digest(), "big")
        stats.canonical_hash_sum += row_hash
        stats.canonical_hash_xor ^= row_hash
        stats.rows_written += 1


def _checked_batches(
    rows: Iterable[dict[str, Any]],
    *,
    schema: pa.Schema,
    max_batch_bytes: int,
    stats: _StreamStats,
) -> Iterator[pa.RecordBatch]:
    """Pack rows into owned Arrow batches under ``max_batch_bytes``."""

    pending: list[dict[str, Any]] = []
    estimated_bytes = 0

    def emit(candidate: list[dict[str, Any]]) -> Iterator[pa.RecordBatch]:
        batch = _record_batch(candidate, schema)
        retained_bytes = int(batch.get_total_buffer_size())
        if retained_bytes > max_batch_bytes:
            if len(candidate) == 1:
                raise OversizedVdbRowError(
                    "One canonical VDB row requires "
                    f"{retained_bytes} Arrow buffer bytes, exceeding max_batch_bytes={max_batch_bytes}."
                )
            midpoint = len(candidate) // 2
            yield from emit(candidate[:midpoint])
            yield from emit(candidate[midpoint:])
            return
        stats.logical_bytes += retained_bytes
        _record_canonical_batch(stats, batch)
        stats.max_batch_bytes = max(stats.max_batch_bytes, retained_bytes)
        stats.output_batches += 1
        yield batch

    for row in rows:
        exact_row_bytes = _exact_single_arrow_row_bytes(row, schema)
        if exact_row_bytes is not None and exact_row_bytes > max_batch_bytes:
            raise OversizedVdbRowError(
                "One canonical VDB row requires "
                f"{exact_row_bytes} Arrow buffer bytes, exceeding max_batch_bytes={max_batch_bytes}."
            )
        row_estimate = _estimated_arrow_row_bytes(row, schema)
        if pending and (
            len(pending) >= _MAX_PENDING_CANONICAL_ROWS or estimated_bytes + row_estimate > max_batch_bytes
        ):
            yield from emit(pending)
            pending = []
            estimated_bytes = 0
        pending.append(row)
        estimated_bytes += row_estimate
        stats.max_pending_rows = max(stats.max_pending_rows, len(pending))
    if pending:
        yield from emit(pending)


def _iter_rows(batch: Any) -> Iterator[dict[str, Any]]:
    """Yield graph rows without copying a complete Ray block to Python dicts."""

    if isinstance(batch, (pa.Table, pa.RecordBatch)):
        for row_index in range(batch.num_rows):
            # Arrow slices share the source buffers; only the current row is
            # converted to Python-owned values.
            yield batch.slice(row_index, 1).to_pylist()[0]
        return

    columns = getattr(batch, "columns", None)
    itertuples = getattr(batch, "itertuples", None)
    if columns is not None and callable(itertuples):
        names = list(columns)
        for values in itertuples(index=False, name=None):
            yield dict(zip(names, values))
        return

    if isinstance(batch, Mapping):
        yield dict(batch)
        return

    for row in batch or ():
        if isinstance(row, Mapping):
            yield dict(row)


def _input_batch_bytes(batch: Any) -> int:
    """Measure the retained bytes of one incoming Ray-native batch."""

    if isinstance(batch, (pa.Table, pa.RecordBatch)):
        return int(batch.get_total_buffer_size())
    memory_usage = getattr(batch, "memory_usage", None)
    if callable(memory_usage):
        usage = memory_usage(index=True, deep=True)
        return int(usage.sum())
    return 0


def _empty_conversion_error(
    *,
    rows: int,
    upstream_error_fields: Counter[str],
    upstream_error_count: int,
    rejection_reasons: Counter[str],
) -> Exception:
    """Build the existing payload-free VDB conversion error from counters."""

    from nemo_retriever.common.vdb.records import VdbUploadError

    if upstream_error_count:
        summary = ", ".join(f"{field}={count}" for field, count in sorted(upstream_error_fields.items()))
        return VdbUploadError(
            f"vdb_upload received {rows} row(s), but none were uploadable because upstream stages "
            f"reported {upstream_error_count} structured row error(s) ({summary}); "
            "error payloads are omitted because they may contain sensitive data."
        )

    summary = ", ".join(f"{reason}={count}" for reason, count in sorted(rejection_reasons.items()))
    if "missing embedding" in rejection_reasons:
        return VdbUploadError(
            "vdb_upload requires embedded records, but no embeddings were found. "
            f"Received {rows} nonempty row(s); rejection reasons: {summary}. "
            "Add an embed stage or provide a supported embedding column."
        )
    return VdbUploadError(f"vdb_upload received {rows} row(s), but none were uploadable; rejection reasons: {summary}.")


def _canonical_rows(
    batches: Iterable[Any],
    *,
    vdb: Any,
    sidecar_spec: dict[str, Any] | None,
    sidecar_lookup: dict[str, dict[str, Any]] | None,
    stats: _StreamStats,
) -> Iterator[dict[str, Any]]:
    """Project graph batches to the exact rows used by the legacy backend."""

    from nemo_retriever.common.stage_errors import iter_stage_errors_from_value
    from nemo_retriever.common.vdb.lancedb import (
        _create_lancedb_results,
        _create_sparse_lancedb_results,
        _to_service_lancedb_rows,
    )
    from nemo_retriever.common.vdb.records import (
        VdbUploadError,
        _client_record_from_graph_row,
        _row_has_uploadable_content_without_embedding,
        _stage_error_field,
    )
    from nemo_retriever.common.vdb.sidecar_metadata import (
        apply_sidecar_metadata_to_client_batches,
    )

    deferred_error: VdbUploadError | None = None
    for batch in batches:
        stats.input_batches += 1
        input_bytes = _input_batch_bytes(batch)
        stats.input_bytes += input_bytes
        stats.max_input_batch_bytes = max(stats.max_input_batch_bytes, input_bytes)
        batch_rows = 0
        batch_client_records = 0
        upstream_error_fields: Counter[str] = Counter()
        upstream_error_count = 0
        rejection_reasons: Counter[str] = Counter()

        # Preserve the legacy nested-client-record input accepted by
        # ``to_client_vdb_records`` without materializing graph rows.
        if isinstance(batch, list) and all(isinstance(record_batch, list) for record_batch in batch):
            record_iter = (record for record_batch in batch for record in record_batch if isinstance(record, dict))
        else:

            def converted_records(
                input_batch: Any = batch,
                error_fields: Counter[str] = upstream_error_fields,
                rejections: Counter[str] = rejection_reasons,
            ) -> Iterator[dict[str, Any]]:
                nonlocal batch_rows, upstream_error_count
                for graph_row in _iter_rows(input_batch):
                    batch_rows += 1
                    stats.input_rows += 1
                    record = _client_record_from_graph_row(graph_row)
                    if record is not None:
                        yield record
                        continue

                    upstream_errors = list(iter_stage_errors_from_value(graph_row))
                    if upstream_errors:
                        upstream_error_count += len(upstream_errors)
                        error_fields.update(_stage_error_field(error.get("path")) for error in upstream_errors)
                    else:
                        reason = (
                            "missing embedding"
                            if _row_has_uploadable_content_without_embedding(graph_row)
                            else "missing searchable text or image backing"
                        )
                        rejections[reason] += 1

            record_iter = converted_records()

        for record in record_iter:
            batch_client_records += 1
            stats.client_records += 1
            records = [[record]]
            if sidecar_spec is not None and sidecar_lookup is not None:
                records = apply_sidecar_metadata_to_client_batches(
                    records,
                    lookup=sidecar_lookup,
                    meta_fields=sidecar_spec["meta_fields"],
                    join_key=sidecar_spec["meta_join_key"],
                )

            if vdb.sparse:
                canonical, _counts = _create_sparse_lancedb_results(records)
            else:
                enforce_dim = vdb.validate_vector_length and vdb.on_bad_vectors != "error"
                expected_dim = stats.vector_dim if enforce_dim else None
                canonical, _counts = _create_lancedb_results(records, expected_dim=expected_dim)
                if vdb._service_table_schema:
                    canonical = _to_service_lancedb_rows(canonical)
            yield from canonical

        if batch_rows and batch_client_records == 0 and deferred_error is None:
            deferred_error = _empty_conversion_error(
                rows=batch_rows,
                upstream_error_fields=upstream_error_fields,
                upstream_error_count=upstream_error_count,
                rejection_reasons=rejection_reasons,
            )

    # Preserve the global conversion invariant: invalid partitions do not
    # fail an operation that produced uploadable rows elsewhere.
    if stats.client_records == 0 and deferred_error is not None:
        raise deferred_error


def _infer_vector_dim_with_spooled_prefix(
    rows: Iterator[dict[str, Any]],
    *,
    vdb: Any,
) -> tuple[int, Iterator[dict[str, Any]]]:
    """Infer from the first nonempty list without retaining an unbounded prefix.

    ``infer_vector_dim`` historically scans until it finds a nonempty Python
    list. Canonical rows before that point still need to be replayed after the
    width is known (a tuple, for example, can then pass length validation).
    Store that rare lookahead prefix on disk so input order and legacy policy
    stay exact without making driver memory depend on prefix length.
    """

    prefix = None
    first_inferable: dict[str, Any] | None = None
    vector_dim = 0
    try:
        for row in rows:
            vector = row.get("vector")
            if isinstance(vector, list) and vector:
                vector_dim = len(vector)
                first_inferable = row
                break
            if prefix is None:
                # Ownership transfers to ``replay`` so the file stays open
                # until the prefix has been consumed.
                prefix = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
            pickle.dump(row, prefix, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        if prefix is not None:
            prefix.close()
        raise

    if first_inferable is None:
        if prefix is not None:
            prefix.close()
        raise ValueError("Cannot infer LanceDB vector_dim because no non-empty embedding was produced.")

    enforce_dim = vdb.validate_vector_length and vdb.on_bad_vectors != "error"

    def replay() -> Iterator[dict[str, Any]]:
        try:
            if prefix is not None:
                prefix.seek(0)
                while True:
                    try:
                        row = pickle.load(prefix)
                    except EOFError:
                        break
                    vector = row.get("vector")
                    if not enforce_dim or (isinstance(vector, (list, tuple)) and len(vector) == vector_dim):
                        yield row
            yield first_inferable
            yield from rows
        finally:
            if prefix is not None:
                prefix.close()

    return vector_dim, replay()


def _apply_deferred_bad_vector_policy(
    rows: Iterable[dict[str, Any]],
    *,
    vdb: Any,
    vector_dim: int,
) -> Iterator[dict[str, Any]]:
    """Apply the LanceDB writer policy before Arrow fixes the vector width.

    The legacy writer gives LanceDB Python rows, so LanceDB can drop, fill, or
    reject variable-length vectors while converting them to its fixed-width
    schema.  A native RecordBatch must already satisfy that schema.  Reproduce
    the pinned LanceDB policy here only when the legacy wrapper intentionally
    defers validation to the writer.
    """

    if vdb.sparse:
        yield from rows
        return

    for row in rows:
        vector = row.get("vector")
        if isinstance(vector, (list, tuple)):
            vector_values = vector
        elif isinstance(vector, np.ndarray) and vector.ndim == 1:
            # Pinned LanceDB accepts one-dimensional NumPy arrays as nested
            # Python vectors. Other merely iterable containers (for example,
            # range and pandas Series) fail its Arrow conversion, so do not
            # broaden this normalization to arbitrary iterables.
            vector_values = vector
        else:
            vector_values = None
        wrong_dim = vector_values is None or len(vector_values) != vector_dim
        has_nan = False
        normalized_vector: list[float | None] = []
        conversion_failed = False
        if vector_values is not None:
            for index, value in enumerate(vector_values):
                if value is None:
                    normalized_value = None
                else:
                    try:
                        normalized_value = float(value)
                    except (TypeError, ValueError):
                        conversion_failed = True
                        break
                if normalized_value is not None and math.isnan(normalized_value):
                    has_nan = True
                if index < vector_dim:
                    normalized_vector.append(normalized_value)
        if conversion_failed:
            # Let Arrow report non-coercible values exactly as the legacy
            # LanceDB Python-row conversion does. They are not shape/NaN cases
            # governed by ``on_bad_vectors``.
            yield row
            continue
        if not wrong_dim and not has_nan:
            yield {**row, "vector": normalized_vector}
            continue

        if vdb.on_bad_vectors == "drop":
            continue
        if vdb.on_bad_vectors == "fill":
            # LanceDB 0.34 replaces the complete vector when its width is
            # wrong or any element is NaN. Preserve that supported runtime
            # contract instead of retaining a prefix that legacy ingestion
            # discards.
            yield {**row, "vector": [float(vdb.fill_value)] * vector_dim}
            continue
        if vdb.on_bad_vectors == "null":
            yield {**row, "vector": None}
            continue

        if wrong_dim:
            detail = (
                "Vector column 'vector' has variable length vectors. "
                "Set on_bad_vectors='drop' to remove them, "
                "set on_bad_vectors='fill' and fill_value=<value> to replace them, "
                "or set on_bad_vectors='null' to replace them with null."
            )
        else:
            detail = (
                "Vector column 'vector' has NaNs. "
                "Set on_bad_vectors='drop' to remove them, "
                "set on_bad_vectors='fill' and fill_value=<value> to replace them, "
                "or set on_bad_vectors='null' to replace them with null."
            )
        # Match the public exception class and actionable portion of the
        # pinned LanceDB 0.34 writer error.  Arrow cannot represent the bad row
        # in the fixed-width RecordBatch needed by the streaming interface.
        raise RuntimeError(f"Arrow error: C Data interface error: Invalid: {detail}")


def _reject_empty_operation_bypass(table: Any | None, *, operation_id: str) -> None:
    """Keep a true empty no-op from concealing durable operation state."""

    if table is None:
        return
    tags = table.tags.list()
    operation_token = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:24]
    has_operation_state = any(f"_{operation_token}_" in name for name in tags)
    has_incomplete_state = any(name.startswith(("nemo_sink_pending_", "nemo_sink_data_")) for name in tags)
    if has_operation_state or has_incomplete_state:
        raise VdbOperationConflict(
            f"VDB sink operation_id {operation_id!r} has durable state; "
            "empty input cannot reconcile or verify that operation."
        )
    if not _bounded_create_is_finalized(table):
        raise VdbWriteNotFinalized(
            f"LanceDB table {table.name!r} has an unfinished create operation; " "empty input cannot finalize it."
        )


def _schema_for_stream(vdb: Any, *, vector_dim: int | None) -> pa.Schema:
    from nemo_retriever.common.vdb.lancedb import (
        _lancedb_arrow_schema,
        _sparse_lancedb_arrow_schema,
        _with_retrieval_mode_metadata,
    )
    from nemo_retriever.common.vdb.lancedb_schema import lancedb_schema

    if vdb.sparse:
        return _sparse_lancedb_arrow_schema()

    if vector_dim is None:
        raise ValueError("Cannot infer LanceDB vector_dim because no non-empty embedding was produced.")

    retrieval_mode = "hybrid" if vdb.hybrid else "dense"
    if vdb._service_table_schema:
        return _with_retrieval_mode_metadata(
            lancedb_schema(vector_dim),
            retrieval_mode,
            embedding_model_name=vdb.embedding_model_name,
            embedding_model_revision=vdb.embedding_model_revision,
        )
    return _lancedb_arrow_schema(
        vector_dim,
        retrieval_mode=retrieval_mode,
        embedding_model_name=vdb.embedding_model_name,
        embedding_model_revision=vdb.embedding_model_revision,
    )


def _identity_token(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().encode("ascii")


def _with_create_identity(schema: pa.Schema, *, operation_id: str, request_fingerprint: str) -> pa.Schema:
    """Bind a newly created table to the operation in its data commit."""

    metadata = dict(schema.metadata or {})
    metadata[_CREATE_OPERATION_KEY] = _identity_token(operation_id)
    metadata[_CREATE_REQUEST_KEY] = _identity_token(request_fingerprint)
    return schema.with_metadata(metadata)


def _matches_create_identity(table: Any, *, operation_id: str, request_fingerprint: str) -> bool:
    from nemo_retriever.common.vdb.lancedb import _table_schema

    metadata = _table_schema(table).metadata or {}
    return metadata.get(_CREATE_OPERATION_KEY) == _identity_token(operation_id) and metadata.get(
        _CREATE_REQUEST_KEY
    ) == _identity_token(request_fingerprint)


def _table_content_identity(table: Any, schema: pa.Schema) -> tuple[int, str]:
    """Read a table through bounded one-row batches for rare create recovery."""

    stats = _StreamStats()
    reader = table.search().select(schema.names).to_batches(batch_size=1)
    for batch in reader:
        _record_canonical_batch(stats, batch)
    return stats.rows_written, stats.digest


def _schemas_have_same_fields(actual: pa.Schema, expected: pa.Schema) -> bool:
    """Ignore only durable sink identity keys while validating the schema."""

    def product_schema(schema: pa.Schema) -> pa.Schema:
        metadata = {
            key: value
            for key, value in (schema.metadata or {}).items()
            if key not in {_CREATE_OPERATION_KEY, _CREATE_REQUEST_KEY}
        }
        return schema.with_metadata(metadata or None)

    return product_schema(actual).equals(product_schema(expected), check_metadata=True)


def _validate_index_coverage(
    table: Any,
    *,
    rows: int,
    sparse: bool,
    hybrid: bool,
    index_type: str,
) -> None:
    """Require every configured index to cover the finalized table version."""

    from nemo_retriever.common.vdb.lancedb import _is_ivf_vector_index

    vector_index_expected = not sparse and not (_is_ivf_vector_index(index_type) and rows < 2)
    expected_columns = {("vector",)} if vector_index_expected else set()
    if sparse:
        expected_columns.add(("text",))
    if hybrid:
        expected_columns.add(("text",))

    indices = {tuple(index.columns): index for index in table.list_indices()}
    missing = sorted(expected_columns - set(indices))
    if missing:
        raise RuntimeError(f"LanceDB index validation failed; missing index columns: {missing!r}.")

    for columns in sorted(expected_columns):
        index = indices[columns]
        stats = table.index_stats(index.name)
        if stats is None or int(stats.num_indexed_rows) != int(rows) or int(stats.num_unindexed_rows) != 0:
            raise RuntimeError(
                f"LanceDB index validation failed for {columns!r}: "
                f"expected {rows} indexed and 0 unindexed rows, got {stats!r}."
            )


@dataclass(frozen=True, slots=True)
class _TableInventory:
    versions: int = 0
    fragments: int = 0
    data_files: int = 0


def _table_inventory(table: Any | None) -> _TableInventory:
    if table is None:
        return _TableInventory()
    table.checkout_latest()
    versions = table.list_versions()
    current = next(
        (version for version in versions if int(version["version"]) == int(table.version)),
        versions[-1] if versions else {},
    )
    metadata = current.get("metadata") or {}
    return _TableInventory(
        versions=len(versions),
        fragments=int(metadata.get("total_fragments", 0)),
        data_files=int(metadata.get("total_data_files", 0)),
    )


def _write_report(
    *,
    operation_id: str,
    outcome: str,
    policy: VdbSinkPolicy,
    stats: _StreamStats,
    before: _TableInventory,
    after: _TableInventory,
    data_version: int | None,
    final_version: int | None,
    timings: dict[str, float],
) -> VdbWriteReport:
    normalized_timings = {phase: float(timings.get(phase, 0.0)) for phase in _TIMING_PHASES}
    normalized_timings.update(timings)
    write_seconds = normalized_timings["write"]
    return VdbWriteReport(
        operation_id=operation_id,
        outcome=outcome,
        configured_max_batch_bytes=int(policy.max_batch_bytes),
        configured_prefetch_batches=int(policy.prefetch_batches),
        input_batches=stats.input_batches,
        input_rows=stats.input_rows,
        input_bytes=stats.input_bytes,
        max_input_batch_bytes=stats.max_input_batch_bytes,
        output_batches=stats.output_batches,
        rows_written=stats.rows_written,
        logical_bytes=stats.logical_bytes,
        max_batch_bytes=stats.max_batch_bytes,
        max_pending_rows=stats.max_pending_rows,
        versions_before=before.versions,
        versions_after=after.versions,
        fragments_before=before.fragments,
        fragments_after=after.fragments,
        data_files_before=before.data_files,
        data_files_after=after.data_files,
        write_rows_per_second=(stats.rows_written / write_seconds if write_seconds > 0 else 0.0),
        write_bytes_per_second=(stats.logical_bytes / write_seconds if write_seconds > 0 else 0.0),
        data_version=data_version,
        final_version=final_version,
        timings=normalized_timings,
    )


def _rows_at_version(uri: str, table_name: str, version: int | None) -> int:
    if version is None:
        return 0
    import lancedb

    table = lancedb.connect(uri=uri).open_table(table_name)
    table.checkout(int(version))
    return int(table.count_rows())


def _drain_batches(first: pa.RecordBatch, rest: Iterator[pa.RecordBatch]) -> None:
    _ = first
    for _batch in rest:
        pass


def write_lancedb_batches(
    vdb: Any,
    batches: Iterable[Any],
    *,
    operation_id: str,
    policy: VdbSinkPolicy,
    sidecar_spec: dict[str, Any] | None = None,
    sidecar_lookup: dict[str, dict[str, Any]] | None = None,
) -> VdbWriteReport:
    """Consume many graph batches through one coordinated LanceDB mutation."""

    import lancedb

    from nemo_retriever.common.vdb.lancedb import (
        _is_missing_lancedb_table_error,
        _schema_vector_dim,
        _table_schema,
        _validate_append_embedding_model,
        _validate_append_schema,
    )

    if not operation_id or not str(operation_id).strip():
        raise ValueError("operation_id must be a non-empty string")

    operation_id = str(operation_id)
    started = time.perf_counter()
    stats = _StreamStats()
    timings: dict[str, float] = {}

    with vdb._write_lock:
        prepare_started = time.perf_counter()
        db = lancedb.connect(uri=vdb.uri)
        try:
            existing_table = db.open_table(vdb.table_name)
        except ValueError as exc:
            if not _is_missing_lancedb_table_error(exc):
                raise
            existing_table = None
        table_exists = existing_table is not None
        inventory_before = _table_inventory(existing_table)

        stats.vector_dim = vdb.vector_dim
        if stats.vector_dim is None and existing_table is not None and not vdb.overwrite and not vdb.sparse:
            stats.vector_dim = _schema_vector_dim(_table_schema(existing_table))

        canonical_rows = _canonical_rows(
            batches,
            vdb=vdb,
            sidecar_spec=sidecar_spec,
            sidecar_lookup=sidecar_lookup,
            stats=stats,
        )
        try:
            first_canonical_row = next(canonical_rows)
        except StopIteration:
            first_canonical_row = None
            if stats.client_records == 0:
                _reject_empty_operation_bypass(existing_table, operation_id=operation_id)
                empty_version = int(existing_table.version) if existing_table is not None else None
                timings["prepare"] = time.perf_counter() - prepare_started
                timings["total"] = time.perf_counter() - started
                return _write_report(
                    operation_id=operation_id,
                    outcome="empty_noop",
                    policy=policy,
                    stats=stats,
                    before=inventory_before,
                    after=inventory_before,
                    data_version=empty_version,
                    final_version=empty_version,
                    timings=timings,
                )

        def all_canonical_rows() -> Iterator[dict[str, Any]]:
            if first_canonical_row is not None:
                yield first_canonical_row
            yield from canonical_rows

        canonical_row_stream = all_canonical_rows()
        if stats.vector_dim is None and not vdb.sparse:
            stats.vector_dim, canonical_row_stream = _infer_vector_dim_with_spooled_prefix(
                canonical_row_stream,
                vdb=vdb,
            )

        policy_rows = _apply_deferred_bad_vector_policy(
            canonical_row_stream,
            vdb=vdb,
            vector_dim=int(stats.vector_dim or 0),
        )
        try:
            first_row = next(policy_rows)
        except StopIteration:
            first_row = None

        base_schema = _schema_for_stream(vdb, vector_dim=stats.vector_dim)
        mode = "overwrite" if vdb.overwrite else "append"
        request_fingerprint = json.dumps(
            {
                "table": vdb.table_name,
                "mode": mode,
                "schema": base_schema.to_string(show_schema_metadata=True),
                "build_index": bool(vdb.build_index),
                "index_type": str(vdb.index_type),
                "metric": str(vdb.metric),
                "num_partitions": int(vdb.num_partitions),
                "num_sub_vectors": int(vdb.num_sub_vectors),
                "fts_language": str(vdb.fts_language),
                "hybrid": bool(vdb.hybrid),
                "sparse": bool(vdb.sparse),
                "on_bad_vectors": str(vdb.on_bad_vectors),
                "fill_value": float(vdb.fill_value),
                "validate_vector_length": bool(vdb.validate_vector_length),
                "service_table_schema": bool(vdb._service_table_schema),
                "optimize": bool(policy.optimize),
                "max_batch_bytes": int(policy.max_batch_bytes),
                "prefetch_batches": int(policy.prefetch_batches),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        recovered_create = existing_table is not None and _matches_create_identity(
            existing_table,
            operation_id=operation_id,
            request_fingerprint=request_fingerprint,
        )
        if existing_table is not None and not recovered_create and not _bounded_create_is_finalized(existing_table):
            raise VdbWriteNotFinalized(
                f"LanceDB table {vdb.table_name!r} has an unfinished create operation; "
                "retry or reconcile that operation before starting another write."
            )
        schema = (
            _with_create_identity(
                base_schema,
                operation_id=operation_id,
                request_fingerprint=request_fingerprint,
            )
            if existing_table is None
            else base_schema
        )
        expected_schema = base_schema
        if existing_table is not None and not vdb.overwrite:
            _validate_append_schema(existing_table, schema, table_name=vdb.table_name, uri=vdb.uri)
            _validate_append_embedding_model(
                existing_table,
                vdb.embedding_model_name,
                vdb.embedding_model_revision,
                table_name=vdb.table_name,
                uri=vdb.uri,
            )
            expected_schema = _table_schema(existing_table)
        markers = SinkOperationMarkers.prepare(
            existing_table,
            operation_id=operation_id,
            request_fingerprint=request_fingerprint,
            mode=mode,
        )
        timings["prepare"] = time.perf_counter() - prepare_started

        def all_rows() -> Iterator[dict[str, Any]]:
            if first_row is not None:
                yield first_row
            yield from policy_rows

        arrow_batches = _checked_batches(
            all_rows(),
            schema=schema,
            max_batch_bytes=int(policy.max_batch_bytes),
            stats=stats,
        )
        try:
            first_batch = next(arrow_batches)
        except StopIteration:
            first_batch = None
        except Exception:
            markers.abort_if_unchanged(existing_table)
            raise

        if recovered_create and markers.state == "write":
            if first_batch is not None:
                _drain_batches(first_batch, arrow_batches)
            stored_rows, stored_digest = _table_content_identity(existing_table, base_schema)
            if stored_rows != stats.rows_written or stored_digest != stats.digest:
                raise VdbOperationConflict(
                    f"VDB sink operation_id {operation_id!r} found a created table with different canonical content."
                )
            existing_table.checkout_latest()
            markers.mark_data(
                existing_table,
                version=int(existing_table.version),
                rows=stats.rows_written,
                digest=stats.digest,
            )
        elif markers.state in {"data", "success"}:
            if first_batch is not None:
                _drain_batches(first_batch, arrow_batches)
            markers.verify_input(rows=stats.rows_written, digest=stats.digest)

        if markers.state == "success":
            markers.cleanup_after_success(existing_table)
            assert_lancedb_table_ready(existing_table)
            current_inventory = _table_inventory(existing_table)
            timings["total"] = time.perf_counter() - started
            return _write_report(
                operation_id=operation_id,
                outcome="already_succeeded",
                policy=policy,
                stats=stats,
                before=inventory_before,
                after=current_inventory,
                data_version=markers.recorded_version,
                final_version=markers.recorded_version,
                timings=timings,
            )

        def all_batches() -> Iterator[pa.RecordBatch]:
            if first_batch is None:
                # Preserve legacy create/overwrite semantics when nonempty
                # client records were all dropped by the configured policy.
                yield pa.RecordBatch.from_pylist([], schema=schema)
            else:
                yield first_batch
            yield from arrow_batches

        write_kwargs: dict[str, Any] = {"on_bad_vectors": vdb.on_bad_vectors}
        if vdb.on_bad_vectors == "fill":
            write_kwargs["fill_value"] = vdb.fill_value

        if markers.state == "data":
            table = lancedb.connect(uri=vdb.uri).open_table(vdb.table_name)
            table.checkout_latest()
            data_version = int(markers.recorded_version)
            timings["write"] = 0.0
            timings["time_to_data_commit"] = 0.0
        elif existing_table is not None and mode == "append" and stats.rows_written == 0:
            # Legacy append does not create a table version when every client
            # record is dropped, but it still proceeds through finalization.
            table = existing_table
            table.checkout_latest()
            data_version = int(table.version)
            timings["write"] = 0.0
            timings["time_to_data_commit"] = time.perf_counter() - started
            markers.mark_data(
                table,
                version=data_version,
                rows=0,
                digest=stats.digest,
            )
        else:
            reader = pa.RecordBatchReader.from_batches(schema, all_batches())
            write_started = time.perf_counter()
            try:
                if existing_table is None:
                    table = db.create_table(
                        vdb.table_name,
                        data=reader,
                        schema=schema,
                        mode="create",
                        **write_kwargs,
                    )
                    data_version = int(table.version)
                else:
                    add_result = existing_table.add(reader, mode=mode, **write_kwargs)
                    table = existing_table
                    table.checkout_latest()
                    data_version = int(add_result.version)
            except Exception as exc:
                if markers.base_version is not None:
                    latest = lancedb.connect(uri=vdb.uri).open_table(vdb.table_name)
                    latest.checkout_latest()
                    if int(latest.version) == markers.base_version:
                        markers.abort_if_unchanged(latest)
                    elif mode == "append":
                        raise CommitOutcomeUnknown(
                            f"VDB sink operation_id {operation_id!r} failed after the table advanced; "
                            "refusing to replay append because the commit outcome is indeterminate."
                        ) from exc
                raise
            timings["write"] = time.perf_counter() - write_started
            timings["time_to_data_commit"] = time.perf_counter() - started
            # This tag is deliberately after the data mutation. If it fails,
            # the retained base marker makes a later append retry fail closed.
            markers.mark_data(
                table,
                version=data_version,
                rows=stats.rows_written,
                digest=stats.digest,
            )

        validate_started = time.perf_counter()
        fresh_table = lancedb.connect(uri=vdb.uri).open_table(vdb.table_name)
        fresh_schema = _table_schema(fresh_table)
        if not _schemas_have_same_fields(fresh_schema, expected_schema):
            raise RuntimeError(f"LanceDB schema validation failed for table {vdb.table_name!r}")
        base_rows = _rows_at_version(vdb.uri, vdb.table_name, markers.base_version)
        created_from_absent = not table_exists or recovered_create
        expected_rows = stats.rows_written if vdb.overwrite or created_from_absent else base_rows + stats.rows_written
        actual_rows = int(fresh_table.count_rows())
        if actual_rows != expected_rows:
            raise RuntimeError(
                f"LanceDB row-count validation failed for table {vdb.table_name!r}: "
                f"expected {expected_rows}, got {actual_rows}."
            )
        timings["validate"] = time.perf_counter() - validate_started

        index_started = time.perf_counter()
        timings["vector_index"] = 0.0
        timings["fts_index"] = 0.0
        if vdb.build_index:
            vdb.write_to_index(
                None,
                table=fresh_table,
                index_type=vdb.index_type,
                metric=vdb.metric,
                num_partitions=vdb.num_partitions,
                num_sub_vectors=vdb.num_sub_vectors,
                hybrid=vdb.hybrid,
                sparse=vdb.sparse,
                fts_language=vdb.fts_language,
                _phase_timings=timings,
            )
            fresh_table.checkout_latest()
            indexed_rows = int(fresh_table.count_rows())
            if indexed_rows != actual_rows:
                raise RuntimeError(
                    f"LanceDB row count changed during index finalization for table {vdb.table_name!r}: "
                    f"expected {actual_rows}, got {indexed_rows}."
                )
            _validate_index_coverage(
                fresh_table,
                rows=actual_rows,
                sparse=bool(vdb.sparse),
                hybrid=bool(vdb.hybrid),
                index_type=str(vdb.index_type),
            )
        timings["index"] = time.perf_counter() - index_started

        optimize_started = time.perf_counter()
        if policy.optimize:
            fresh_table.optimize()
        timings["optimize"] = time.perf_counter() - optimize_started
        fresh_table.checkout_latest()
        final_validate_started = time.perf_counter()
        final_schema = _table_schema(fresh_table)
        if not _schemas_have_same_fields(final_schema, expected_schema):
            raise RuntimeError(f"LanceDB schema validation failed after finalization for table {vdb.table_name!r}")
        final_rows = int(fresh_table.count_rows())
        if final_rows != expected_rows:
            raise RuntimeError(
                f"LanceDB row-count validation failed after finalization for table {vdb.table_name!r}: "
                f"expected {expected_rows}, got {final_rows}."
            )
        if vdb.build_index:
            _validate_index_coverage(
                fresh_table,
                rows=final_rows,
                sparse=bool(vdb.sparse),
                hybrid=bool(vdb.hybrid),
                index_type=str(vdb.index_type),
            )
        timings["validate"] += time.perf_counter() - final_validate_started
        final_version = int(fresh_table.version)
        versions = {int(item["version"]) for item in fresh_table.list_versions()}
        if data_version not in versions or final_version < data_version:
            raise RuntimeError(
                f"LanceDB version validation failed for table {vdb.table_name!r}: "
                f"data_version={data_version}, final_version={final_version}."
            )
        markers.mark_success(
            fresh_table,
            version=final_version,
            rows=stats.rows_written,
            digest=stats.digest,
        )
        inventory_after = _table_inventory(fresh_table)

    timings["total"] = time.perf_counter() - started
    return _write_report(
        operation_id=operation_id,
        outcome="success",
        policy=policy,
        stats=stats,
        before=inventory_before,
        after=inventory_after,
        data_version=data_version,
        final_version=final_version,
        timings=timings,
    )
