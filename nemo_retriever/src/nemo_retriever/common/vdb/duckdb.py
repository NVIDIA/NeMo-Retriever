# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write-only DuckDB backend: Ray workers write Parquet files, the driver loads them in one transaction."""

from __future__ import annotations

import copy
import shutil
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from nemo_retriever.common.vdb.adt_vdb import VDB, UnsupportedVDBOperation
from nemo_retriever.common.vdb.records import stored_row_fields

_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("metadata", pa.string()),
        ("source", pa.string()),
        ("vector", pa.list_(pa.float32())),
    ]
)


class DuckDB(VDB):
    """Store canonical records in one DuckDB table.

    Each write stages its Parquet files in its own folder under ``<path>.staging`` while the workers write; the
    commit replaces (``overwrite=True``) or appends to the table and removes them.
    """

    supports_worker_write = True

    def __init__(
        self,
        path: str,
        table_name: str = "nemo_retriever",
        overwrite: bool = True,
        worker_write_rows: int = 2048,
    ) -> None:
        self.path = str(path)
        self.table_name = table_name
        self.overwrite = overwrite
        self.worker_write_rows = worker_write_rows
        self._staging = Path(f"{self.path}.staging")

    @contextmanager
    def worker_write(self) -> Iterator["DuckDB"]:
        writer = copy.copy(self)
        writer._staging = self._staging / uuid.uuid4().hex
        writer._staging.mkdir(parents=True)
        try:
            yield writer
        finally:
            shutil.rmtree(writer._staging, ignore_errors=True)

    def write_partition_records(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        rows = []
        for record in records:
            fields, _drop_reason = stored_row_fields(record)
            if fields is not None:
                rows.append({**fields, "vector": record["metadata"]["embedding"]})
        if not rows:
            return {"files": [], "rows": 0}
        path = self._staging / f"{uuid.uuid4().hex}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA), path)
        return {"files": [str(path)], "rows": len(rows)}

    def commit_partitions(self, receipts: list[dict[str, Any]]) -> None:
        files = [file for receipt in receipts for file in receipt["files"]]
        if not files:
            return
        table = '"' + self.table_name.replace('"', '""') + '"'
        with duckdb.connect(self.path) as connection:
            connection.begin()
            if self.overwrite:
                connection.execute(f"DROP TABLE IF EXISTS {table}")
            connection.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM read_parquet(?) LIMIT 0", [files])
            connection.execute(f"INSERT INTO {table} BY NAME SELECT * FROM read_parquet(?)", [files])
            connection.commit()

    def run(self, records: Iterable[Iterable[dict[str, Any]]]) -> None:
        with self.worker_write() as writer:
            receipt = writer.write_partition_records(record for batch in records for record in batch)
            writer.commit_partitions([receipt])

    def create_index(self, **kwargs: Any) -> None:
        pass

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        self.run(records)

    def retrieval(self, queries: list, **kwargs: Any):
        raise UnsupportedVDBOperation("DuckDB is write-only.")

    @staticmethod
    def _collections_not_supported(*args: Any, **kwargs: Any):
        raise UnsupportedVDBOperation("DuckDB does not implement the collection API.")

    create_collection = get_collection = list_collections = update_collection = delete_collection = (
        _collections_not_supported
    )
    get_document = list_documents = delete_document = write_collection = retrieve_collection = (
        _collections_not_supported
    )
