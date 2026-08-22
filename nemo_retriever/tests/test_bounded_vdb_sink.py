# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

lancedb = pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.common.vdb.sink import OversizedVdbRowError, VdbSinkPolicy
from nemo_retriever.operators.vdb import IngestVdbOperator

_POLICY = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)


def _frame(start: int, stop: int, *, vector_dim: int = 2, wide: bool = False) -> pd.DataFrame:
    rows = []
    for row_id in range(start, stop):
        row = {
            "text": f"chunk-{row_id}",
            "text_embeddings_1b_v2": {
                "embedding": [
                    float(row_id),
                    *[float(offset) for offset in range(1, vector_dim)],
                ]
            },
            "source_id": f"/tmp/doc-{row_id}.pdf",
            "page_number": row_id,
            "metadata": {"content_metadata": {"type": "text", "id": f"row-{row_id}"}},
        }
        if wide:
            row["wide_result_only"] = "x" * (32 << 10)
        rows.append(row)
    return pd.DataFrame(rows)


def _operator(uri: Path, **overrides: Any) -> IngestVdbOperator:
    kwargs = {
        "uri": str(uri),
        "table_name": "chunks",
        "vector_dim": 2,
        "overwrite": True,
        "build_index": False,
    }
    kwargs.update(overrides)
    return IngestVdbOperator(vdb_op="lancedb", vdb_kwargs=kwargs)


def _product_metadata(schema: pa.Schema) -> dict[bytes, bytes]:
    return {key: value for key, value in (schema.metadata or {}).items() if not key.startswith(b"nemo_retriever.sink_")}


def test_streams_batches_into_one_indexed_lance_lifecycle(tmp_path: Path, monkeypatch) -> None:
    """The bounded path preserves legacy data/query semantics with one finalization."""

    sidecar = pd.DataFrame(
        [
            {
                "source": f"doc-{row_id}.pdf",
                "department": "finance" if row_id < 2 else "legal",
            }
            for row_id in range(4)
        ]
    )
    common = {
        "build_index": True,
        "hybrid": True,
        "index_type": "IVF_FLAT",
        "num_partitions": 2,
        "meta_dataframe": sidecar,
        "meta_source_field": "source",
        "meta_fields": ["department"],
        "meta_join_key": "source_name",
    }
    legacy = _operator(tmp_path / "legacy", **common)
    bounded = _operator(tmp_path / "bounded", **common)
    frame = _frame(0, 4)
    frame["wide_result_only"] = [{"not": "stored"} for _ in range(4)]

    finalized_row_counts: list[int] = []
    original_write_to_index = bounded._vdb.write_to_index

    def observe_finalization(*args, table=None, **kwargs):
        finalized_row_counts.append(int(table.count_rows()))
        return original_write_to_index(*args, table=table, **kwargs)

    monkeypatch.setattr(bounded._vdb, "write_to_index", observe_finalization)
    legacy.process(frame)
    report = bounded.consume_batches(
        iter([frame.iloc[:2], frame.iloc[2:]]),
        operation_id="indexed-overwrite",
        policy=_POLICY,
    )

    legacy_table = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks")
    bounded_table = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks")
    assert legacy_table.schema.remove_metadata() == bounded_table.schema.remove_metadata()
    assert _product_metadata(legacy_table.schema) == _product_metadata(bounded_table.schema)
    assert legacy_table.to_arrow().sort_by("id").to_pylist() == bounded_table.to_arrow().sort_by("id").to_pylist()
    assert {tuple(index.columns) for index in bounded_table.list_indices()} == {
        ("vector",),
        ("text",),
    }
    assert finalized_row_counts == [4]

    vectors = [[1.0, 0.0], [3.0, 1.0]]
    texts = ["chunk-0", "chunk-3"]
    legacy_hits = legacy._vdb.retrieval(vectors, query_texts=texts, hybrid=True, top_k=2)
    bounded_hits = bounded._vdb.retrieval(vectors, query_texts=texts, hybrid=True, top_k=2)
    assert [[hit["id"] for hit in hits] for hits in bounded_hits] == [
        [hit["id"] for hit in hits] for hits in legacy_hits
    ]

    assert report.outcome == "success"
    assert report.input_batches == 2
    assert report.rows_written == 4
    assert report.output_batches >= 2
    assert report.max_batch_bytes <= report.configured_max_batch_bytes == 1024
    assert report.fragments_after == report.data_files_after == 1
    assert report.timings["write"] > 0
    assert report.timings["index"] >= 0


def test_large_wide_block_is_projected_incrementally_into_bounded_arrow_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sink never projects a corpus-sized Python record list before batching."""

    from nemo_retriever.common.vdb import records as records_module

    input_rows = 64
    budget = 4096
    frame = _frame(0, input_rows, wide=True)
    original_project = records_module.to_client_vdb_records
    projected_rows: list[int] = []

    def observe_projection(rows: Any) -> list[list[dict[str, Any]]]:
        projected_rows.append(len(rows))
        assert len(rows) <= 2
        return original_project(rows)

    monkeypatch.setattr(records_module, "to_client_vdb_records", observe_projection)

    observed_rows: list[int] = []
    observed_bytes: list[int] = []

    def observe_batches(data: Any) -> Iterable[pa.RecordBatch]:
        assert not isinstance(data, (list, tuple, pd.DataFrame, pa.Table))
        for batch in data:
            observed_rows.append(batch.num_rows)
            observed_bytes.append(int(batch.get_total_buffer_size()))
            yield batch

    connection_type = type(lancedb.connect(str(tmp_path)))
    original_create_table = connection_type.create_table

    def observed_create_table(self, name, data=None, *args, **kwargs):
        if name == "chunks" and data is not None:
            data = observe_batches(data)
        return original_create_table(self, name, data, *args, **kwargs)

    monkeypatch.setattr(connection_type, "create_table", observed_create_table)
    report = _operator(tmp_path).consume_batches(
        iter([frame]),
        operation_id="bounded-projection",
        policy=VdbSinkPolicy(max_batch_bytes=budget, prefetch_batches=0),
    )

    assert not projected_rows or max(projected_rows) <= 2
    assert len(observed_rows) > 1
    assert sum(observed_rows) == input_rows
    assert max(observed_bytes) <= budget
    assert report.rows_written == input_rows
    table = lancedb.connect(str(tmp_path)).open_table("chunks")
    assert table.schema.names == ["vector", "text", "metadata", "source", "id"]


def test_empty_input_is_a_noop(tmp_path: Path) -> None:
    report = _operator(tmp_path).consume_batches(iter(()), operation_id="empty", policy=_POLICY)

    assert report.outcome == "empty_noop"
    assert report.rows_written == 0
    assert "chunks" not in lancedb.connect(str(tmp_path)).list_tables().tables


def test_oversized_row_fails_before_arrow_allocation_or_table_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_retriever.common.vdb import sink as sink_module

    frame = _frame(0, 1)
    frame.at[0, "text"] = "x" * 4096
    monkeypatch.setattr(
        sink_module,
        "_record_batch",
        lambda *args, **kwargs: pytest.fail("oversized row reached Arrow allocation"),
    )

    with pytest.raises(OversizedVdbRowError, match="max_batch_bytes=256"):
        _operator(tmp_path).consume_batches(
            iter([frame]),
            operation_id="oversized",
            policy=VdbSinkPolicy(max_batch_bytes=256, prefetch_batches=0),
        )

    assert "chunks" not in lancedb.connect(str(tmp_path)).list_tables().tables


@pytest.mark.parametrize(
    ("bad_vector_policy", "expected_vectors"),
    [("drop", []), ("fill", [[-3.5, -3.5]]), ("null", [None])],
)
def test_bad_vector_policies_match_legacy(
    tmp_path: Path,
    bad_vector_policy: str,
    expected_vectors: list[list[float] | None],
) -> None:
    frame = _frame(0, 1)
    frame.at[0, "text_embeddings_1b_v2"] = {"embedding": [1.0]}
    common = {
        "validate_vector_length": False,
        "on_bad_vectors": bad_vector_policy,
        "fill_value": -3.5,
    }
    legacy = _operator(tmp_path / "legacy", **common)
    bounded = _operator(tmp_path / "bounded", **common)

    legacy.process(frame)
    report = bounded.consume_batches(
        iter([frame]),
        operation_id=f"bad-vector-{bad_vector_policy}",
        policy=_POLICY,
    )

    legacy_vectors = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks").to_arrow()["vector"].to_pylist()
    bounded_vectors = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks").to_arrow()["vector"].to_pylist()
    assert bounded_vectors == legacy_vectors == expected_vectors
    assert report.rows_written == len(expected_vectors)


def test_nan_drop_and_deferred_dimension_inference_match_legacy(tmp_path: Path) -> None:
    frame = _frame(0, 3, vector_dim=3)
    frame.at[0, "text_embeddings_1b_v2"] = {"embedding": []}
    frame.at[1, "text_embeddings_1b_v2"] = {"embedding": [float("nan"), 1.0, 2.0]}
    legacy = _operator(tmp_path / "legacy", vector_dim=None, on_bad_vectors="drop")
    bounded = _operator(tmp_path / "bounded", vector_dim=None, on_bad_vectors="drop")

    legacy.process(frame)
    report = bounded.consume_batches(
        iter([frame.iloc[:1], frame.iloc[1:]]),
        operation_id="infer-after-invalid-prefix",
        policy=_POLICY,
    )

    legacy_table = lancedb.connect(str(tmp_path / "legacy")).open_table("chunks")
    bounded_table = lancedb.connect(str(tmp_path / "bounded")).open_table("chunks")
    assert bounded_table.schema.field("vector").type.list_size == 3
    assert bounded_table.to_arrow().to_pylist() == legacy_table.to_arrow().to_pylist()
    assert report.rows_written == 1


def test_append_preserves_compatible_existing_schema(tmp_path: Path) -> None:
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("id", pa.string()),
            pa.field("legacy_extra", pa.string()),
        ],
        metadata={
            b"retrieval_mode": b"dense",
            b"nemo_retriever.retrieval_mode": b"dense",
        },
    )
    db = lancedb.connect(str(tmp_path))
    db.create_table(
        "chunks",
        data=[
            {
                "vector": [1.0, 0.0],
                "text": "seed",
                "metadata": '{"type":"text","id":"seed"}',
                "source": '{"source_id":"/tmp/seed.pdf"}',
                "id": "seed",
                "legacy_extra": "preserved",
            }
        ],
        schema=schema,
    )

    report = _operator(tmp_path, overwrite=False).consume_batches(
        iter([_frame(1, 2)]),
        operation_id="append-schema-superset",
        policy=_POLICY,
    )

    table = db.open_table("chunks")
    rows = sorted(table.to_arrow().to_pylist(), key=lambda row: row["id"])
    assert report.rows_written == 1
    assert table.schema == schema
    assert [(row["id"], row["legacy_extra"]) for row in rows] == [
        ("row-1", None),
        ("seed", "preserved"),
    ]
