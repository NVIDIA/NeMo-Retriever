# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pyarrow as pa
import pytest

from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.operators.vdb import IngestVdbOperator


def _staged_table() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("document_id", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("url", pa.string(), nullable=False),
            pa.field("file_path", pa.string(), nullable=False),
            pa.field("dump", pa.string(), nullable=False),
            pa.field("source_id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), 2), nullable=False),
        ]
    )
    return pa.Table.from_pylist(
        [
            {
                "id": "row-1",
                "document_id": "doc-1",
                "text": "hello",
                "url": 'https://example.test/a?x="quoted"',
                "file_path": "s3://bucket/path\\name",
                "dump": "CC-MAIN-2026-30",
                "source_id": "source-1",
                "vector": [1.0, 2.0],
            }
        ],
        schema=schema,
    )


def test_partition_controls_default_to_native_target_size(tmp_path) -> None:
    backend = LanceDB(uri=str(tmp_path), vector_dim=2, build_index=False)
    assert backend.num_partitions is None
    assert backend.target_partition_size == 1_048_576


def test_partition_controls_are_mutually_exclusive(tmp_path) -> None:
    with pytest.raises(ValueError, match="only one partition control"):
        LanceDB(
            uri=str(tmp_path),
            vector_dim=2,
            num_partitions=16,
            target_partition_size=1_048_576,
        )


def test_embedding_parquet_projection_preserves_vectors_and_metadata(tmp_path) -> None:
    backend = LanceDB(
        uri=str(tmp_path),
        vector_dim=2,
        input_format="embedding_parquet",
        build_index=False,
        stream_operation_id=None,
    )
    staged = _staged_table()
    prepared = backend.prepare_embedding_parquet_batch(staged)

    assert prepared.column_names == ["vector", "text", "metadata", "source", "id"]
    prepared_values = prepared.column("vector").chunk(0).values
    staged_values = staged.column("vector").chunk(0).values
    assert prepared_values.buffers()[1].address == staged_values.buffers()[1].address
    assert prepared.column("vector").to_pylist() == [[1.0, 2.0]]
    assert json.loads(prepared.column("metadata")[0].as_py()) == {
        "id": "row-1",
        "document_id": "doc-1",
        "type": "text",
    }
    assert json.loads(prepared.column("source")[0].as_py()) == {
        "source_id": "source-1",
        "source_name": 'https://example.test/a?x="quoted"',
        "url": 'https://example.test/a?x="quoted"',
        "file_path": "s3://bucket/path\\name",
        "dump": "CC-MAIN-2026-30",
        "document_id": "doc-1",
    }


def test_ingest_operator_routes_embedding_parquet_without_row_conversion(tmp_path) -> None:
    operator = IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(tmp_path),
            "vector_dim": 2,
            "input_format": "embedding_parquet",
            "build_index": False,
            "stream_operation_id": None,
        },
    )
    prepared = operator._prepare_stream_batch(_staged_table())
    assert prepared.column("id").to_pylist() == ["row-1"]
