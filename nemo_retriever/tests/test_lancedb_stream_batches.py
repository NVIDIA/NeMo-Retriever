# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The worker write's columnar projection stores exactly what the record stream stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

lancedb = pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.common.vdb._lancedb_stream import OversizedVdbRowError
from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb import records as records_module
from nemo_retriever.common.vdb._lancedb_stream import _StreamStats
from nemo_retriever.common.vdb.records import VdbUploadError, iter_canonical_record_batches
from nemo_retriever.graph.executor import arrow_table_to_pandas
from nemo_retriever.operators.vdb import IngestVdbOperator

DIM = 4


def _vector(seed: int, *, dim: int = DIM) -> list[float]:
    return [float(seed) + 0.25 * index for index in range(dim)]


def _row(
    row_id: int,
    *,
    text: str | None = None,
    embedding: list[float] | None = ...,  # type: ignore[assignment]
    document_type: str = "text",
    metadata: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    content_metadata = {"type": document_type, "id": f"row-{row_id}", "page_number": row_id}
    row: dict[str, Any] = {
        "text": f"chunk {row_id}" if text is None else text,
        "embedding": _vector(row_id) if embedding is ... else embedding,
        "document_type": document_type,
        "source_id": f"/tmp/doc-{row_id % 3}.pdf",
        "metadata": {
            "content_metadata": content_metadata,
            "source_metadata": {"source_id": f"/tmp/doc-{row_id % 3}.pdf", "source_name": f"doc-{row_id % 3}.pdf"},
            **(metadata or {}),
        },
    }
    row.update(extra)
    return row


def _frames(rows: list[dict[str, Any]], *, batch_size: int = 3) -> list[pd.DataFrame]:
    """Arrow-backed pandas batches, converted the way ``RayDataExecutor`` converts Ray blocks."""
    columns = dict.fromkeys(key for row in rows for key in row)
    metadata_keys = dict.fromkeys(key for row in rows for key in row["metadata"])
    table = pa.Table.from_pylist(
        [
            {
                **{key: row.get(key) for key in columns},
                "metadata": {key: row["metadata"].get(key) for key in metadata_keys},
            }
            for row in rows
        ]
    )
    return [arrow_table_to_pandas(table.slice(start, batch_size)) for start in range(0, table.num_rows, batch_size)]


def _backend(uri: Path, **overrides: Any) -> LanceDB:
    kwargs: dict[str, Any] = {
        "uri": str(uri),
        "table_name": "chunks",
        "vector_dim": DIM,
        "overwrite": True,
        "build_index": False,
    }
    kwargs.update(overrides)
    return LanceDB(**kwargs)


def _table(uri: Path):
    table = lancedb.connect(str(uri)).open_table("chunks")
    table.checkout_latest()
    return table


def _ingest_both(
    tmp_path: Path,
    frames: list[pd.DataFrame],
    *,
    backend_kwargs: dict[str, Any] | None = None,
    operator_kwargs: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Write the same batches through the record stream and the worker write's columnar projection."""
    submitted = []
    for name, columnar in (("records", False), ("columnar", True)):
        operator = IngestVdbOperator(vdb=_backend(tmp_path / name, **(backend_kwargs or {})), **(operator_kwargs or {}))
        submitted.append(_write(operator, frames, columnar=columnar))
    return submitted[0], submitted[1]


def _write(operator: IngestVdbOperator, frames: list[pd.DataFrame], *, columnar: bool) -> int:
    """Record stream on the driver, or the worker write, whose dense tables take the columnar projection."""
    if not columnar:
        return operator._stream_ingest(iter(frames))
    with operator._worker_write() as writer:
        return writer._commit_partitions([writer._write_partition(frame) for frame in frames])


def _assert_tables_equal(tmp_path: Path) -> pa.Table:
    expected = _table(tmp_path / "records").to_arrow()
    actual = _table(tmp_path / "columnar").to_arrow()
    assert actual.schema.equals(expected.schema, check_metadata=True)
    assert actual.equals(expected), f"{actual.to_pylist()} != {expected.to_pylist()}"
    assert actual.schema.field("vector").type == pa.list_(pa.float32(), DIM)
    return actual


def test_mixed_rows_store_identically(tmp_path: Path) -> None:
    rows = [
        _row(0),
        _row(1, text="", embedding=_vector(1)),  # dropped: no text
        _row(2, document_type="structured", metadata={"table_metadata": {"table_content": "a | b"}}),
        _row(
            3,
            text="",
            document_type="image",
            _image_b64="aGk=",
            _stored_image_uri="s3://bucket/page-3.png",
            metadata={"image_metadata": {"caption": ""}},
        ),
        _row(4, metadata={"embedding_split": {"chunk_id": "row-4-child", "content": "exact child text"}}),
        _row(5, metadata={"source_path": "/data/doc-5.pdf"}),
        _row(6),
    ]
    frames = _frames(rows, batch_size=2)

    records_submitted, columnar_submitted = _ingest_both(tmp_path, frames)

    assert records_submitted == columnar_submitted
    table = _assert_tables_equal(tmp_path)
    assert table.num_rows == 6
    assert "exact child text" in table.column("text").to_pylist()
    assert "" in table.column("text").to_pylist()  # the canonical image row keeps an empty text


@pytest.mark.parametrize("policy", ["drop", "fill", "null"])
@pytest.mark.parametrize("validate_vector_length", [True, False])
def test_bad_vector_policies_store_identically(tmp_path: Path, policy: str, validate_vector_length: bool) -> None:
    rows = [
        _row(0),
        _row(1, embedding=[1.0, 2.0]),  # wrong width
        _row(2, embedding=[float("nan"), 1.0, 2.0, 3.0]),
        _row(3),
        _row(4, embedding=[7.0]),
    ]
    kwargs = {"on_bad_vectors": policy, "fill_value": -3.5, "validate_vector_length": validate_vector_length}

    _ingest_both(tmp_path, _frames(rows), backend_kwargs=kwargs)

    table = _assert_tables_equal(tmp_path)
    vectors = table.column("vector").to_pylist()
    if policy == "drop":
        assert len(vectors) == 2
    else:
        assert len(vectors) == (3 if validate_vector_length else 5)
        assert (None if policy == "null" else [-3.5] * DIM) in vectors


@pytest.mark.parametrize(
    ("embedding", "message"),
    [
        pytest.param([1.0], "expected dimension 4", id="wrong-width"),
        pytest.param([1.0, float("nan"), 2.0, 3.0], "contains NaN", id="nan"),
    ],
)
def test_error_policy_rejects_identically(tmp_path: Path, embedding: list[float], message: str) -> None:
    frames = _frames([_row(0), _row(1, embedding=embedding)])
    for name, columnar in (("records", False), ("columnar", True)):
        backend = _backend(tmp_path / name, on_bad_vectors="error")
        with pytest.raises(ValueError, match=message):
            _write(IngestVdbOperator(vdb=backend), frames, columnar=columnar)
        with pytest.raises(ValueError):
            _table(tmp_path / name)


def test_object_embedding_column_matches_arrow_column(tmp_path: Path) -> None:
    rows = [_row(0), _row(1), _row(2, text="", embedding=None), _row(3)]
    object_frames = [pd.DataFrame(rows)]
    assert object_frames[0]["embedding"].dtype == object

    _ingest_both(tmp_path, object_frames)

    table = _assert_tables_equal(tmp_path)
    assert table.num_rows == 3


def test_inferred_vector_dim_matches(tmp_path: Path) -> None:
    rows = [_row(0, text="", embedding=_vector(0)), _row(1), _row(2, embedding=[1.0, 2.0]), _row(3)]

    _ingest_both(tmp_path, _frames(rows, batch_size=4), backend_kwargs={"vector_dim": None})

    table = _assert_tables_equal(tmp_path)
    assert table.column("id").to_pylist() == ["row-1", "row-3"]


def test_custom_embedding_column(tmp_path: Path) -> None:
    rows = [{**_row(index), "vec": _vector(index + 10)} for index in range(4)]
    for row in rows:
        del row["embedding"]

    _ingest_both(tmp_path, _frames(rows), operator_kwargs={"embedding_column": "vec"})

    table = _assert_tables_equal(tmp_path)
    assert table.column("vector").to_pylist()[0] == pytest.approx(_vector(10))


def test_sidecar_metadata_is_applied_identically(tmp_path: Path) -> None:
    sidecar = tmp_path / "sidecar.csv"
    pd.DataFrame(
        {"source": ["/tmp/doc-0.pdf", "/tmp/doc-1.pdf"], "team": ["alpha", "beta"], "year": [2024, 2025]}
    ).to_csv(sidecar, index=False)
    rows = [_row(index) for index in range(5)]
    submitted = []
    for name, columnar in (("records", False), ("columnar", True)):
        operator = IngestVdbOperator(
            vdb_op="lancedb",
            vdb_kwargs={
                "uri": str(tmp_path / name),
                "table_name": "chunks",
                "vector_dim": DIM,
                "overwrite": True,
                "build_index": False,
                "meta_dataframe": str(sidecar),
                "meta_source_field": "source",
                "meta_fields": ["team", "year"],
            },
        )
        submitted.append(_write(operator, _frames(rows), columnar=columnar))

    assert submitted[0] == submitted[1] == 5
    table = _assert_tables_equal(tmp_path)
    assert all('"team":' in value for value in table.column("metadata").to_pylist() if '"id":"row-0"' in value)


def test_byte_bound_splits_large_batches_and_rejects_oversized_rows(tmp_path: Path) -> None:
    rows = [_row(index, text="x" * 200) for index in range(40)]

    _ingest_both(tmp_path, _frames(rows, batch_size=40), backend_kwargs={"stream_batch_bytes": 2048})
    _assert_tables_equal(tmp_path)

    oversized = _frames([_row(0, text="y" * 4096)])
    for name, columnar in (("records-big", False), ("columnar-big", True)):
        backend = _backend(tmp_path / name, stream_batch_bytes=2048)
        # Lance's fragment writer re-raises the projection's error as OSError.
        with pytest.raises((OversizedVdbRowError, OSError), match="exceeding max_batch_bytes"):
            _write(IngestVdbOperator(vdb=backend), oversized, columnar=columnar)


def test_missing_embeddings_fail_closed_before_any_table_exists(tmp_path: Path) -> None:
    partial = _frames([_row(0), _row(1, embedding=None, embedding_error="embed failed")])
    none_at_all = _frames([_row(0, embedding=None), _row(1, embedding=None)])
    for name, columnar in (("records", False), ("columnar", True)):
        backend = _backend(tmp_path / name)
        operator = IngestVdbOperator(vdb=backend)
        with pytest.raises(VdbUploadError, match="refusing a partial write"):
            _write(operator, partial, columnar=columnar)
        with pytest.raises(VdbUploadError, match="no embeddings were found"):
            _write(operator, none_at_all, columnar=columnar)
        with pytest.raises(ValueError):
            _table(tmp_path / name)


def test_empty_and_header_only_batches_create_no_table(tmp_path: Path) -> None:
    empty = [pd.DataFrame(columns=["text", "embedding", "metadata"])]
    records_submitted, columnar_submitted = _ingest_both(tmp_path, empty)
    assert records_submitted == columnar_submitted == 0
    for name in ("records", "columnar"):
        with pytest.raises(ValueError):
            _table(tmp_path / name)


def test_arrow_backed_embedding_column_is_never_expanded_per_row(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _frames([_row(0), _row(1)])
    assert isinstance(frames[0]["embedding"].dtype, pd.ArrowDtype)

    def forbid(value: Any) -> None:
        raise AssertionError("embedding_vector() must not run for an Arrow-backed column")

    monkeypatch.setattr(records_module, "embedding_vector", forbid)
    column = records_module.embedding_arrow_column(frames[0]["embedding"], length=2)
    assert pa.types.is_list(column.type) and pa.types.is_floating(column.type.value_type)
    assert column.to_pylist() == [pytest.approx(_vector(0)), pytest.approx(_vector(1))]


def test_numeric_string_vectors_match_record_path(tmp_path: Path) -> None:
    rows = [_row(0, embedding=["0.0", "0.25", "0.5", "0.75"]), _row(1)]

    _ingest_both(tmp_path, [pd.DataFrame(rows)])

    table = _assert_tables_equal(tmp_path)
    assert table.column("vector").to_pylist()[0] == pytest.approx(_vector(0))


def test_prepared_chunks_respect_stream_batch_bytes_before_arrow_allocation(tmp_path: Path) -> None:
    limit = 4096
    backend = _backend(tmp_path, stream_batch_bytes=limit)
    frames = _frames([_row(index, text="x" * 300) for index in range(60)], batch_size=60)
    rows = frames[0].drop(columns=["embedding"]).to_dict("records")
    vectors = records_module.embedding_arrow_column(frames[0]["embedding"], length=60)
    (batch,) = iter_canonical_record_batches([(rows, vectors)])

    chunks = list(backend._iter_stream_columns([batch], _StreamStats(vector_dim=DIM)))

    assert len(chunks) > 1
    schema = backend._stream_schema(DIM)
    assert sum(len(chunk.vector) for chunk in chunks) == 60
    assert all(chunk.record_batch(schema).get_total_buffer_size() <= limit for chunk in chunks)


def test_arrow_backed_numeric_string_vectors_match_record_path(tmp_path: Path) -> None:
    rows = [_row(0, embedding=["0.0", "0.25", "0.5", "0.75"]), _row(1, embedding=["1.0", "1.25", "1.5", "1.75"])]
    frame = pd.DataFrame(rows)
    frame["embedding"] = pd.Series(
        pa.array(frame["embedding"].tolist(), type=pa.list_(pa.string())),
        dtype=pd.ArrowDtype(pa.list_(pa.string())),
    )

    _ingest_both(tmp_path, [frame])

    table = _assert_tables_equal(tmp_path)
    assert table.column("vector").to_pylist() == [pytest.approx(_vector(0)), pytest.approx(_vector(1))]


def test_wrong_width_uncoercible_vector_is_dropped_like_record_path(tmp_path: Path) -> None:
    rows = [_row(0), _row(1, embedding=["bad"]), _row(2)]

    _ingest_both(tmp_path, [pd.DataFrame(rows)])

    table = _assert_tables_equal(tmp_path)
    assert table.column("id").to_pylist() == ["row-0", "row-2"]


def test_correct_width_uncoercible_vector_fails_on_both_paths(tmp_path: Path) -> None:
    rows = [_row(0), _row(1, embedding=["a", "b", "c", "d"])]
    frames = [pd.DataFrame(rows)]

    with pytest.raises(Exception):
        IngestVdbOperator(vdb=_backend(tmp_path / "records"))._stream_ingest(iter(frames))
    with pytest.raises(Exception):
        _write(IngestVdbOperator(vdb=_backend(tmp_path / "arrow")), frames, columnar=True)


def test_mixed_string_and_numeric_vector_elements_match_record_path(tmp_path: Path) -> None:
    rows = [_row(0, embedding=["0.0", True, 0.5, 0.75]), _row(1, embedding=[1.0, "1.25", 1.5, 1.75])]

    _ingest_both(tmp_path, [pd.DataFrame(rows)])

    table = _assert_tables_equal(tmp_path)
    assert table.column("vector").to_pylist() == [pytest.approx([0.0, 1.0, 0.5, 0.75]), pytest.approx(_vector(1))]


def test_python_float_syntax_in_string_vectors_matches_record_path(tmp_path: Path) -> None:
    rows = [_row(0, embedding=[" 0.0 ", "0.25", "5e-1", "0.75"]), _row(1, embedding=["1_000.0", "1.25", "1.5", "1.75"])]
    frame = pd.DataFrame(rows)
    frame["embedding"] = pd.Series(
        pa.array(frame["embedding"].tolist(), type=pa.list_(pa.string())),
        dtype=pd.ArrowDtype(pa.list_(pa.string())),
    )

    _ingest_both(tmp_path, [frame])

    table = _assert_tables_equal(tmp_path)
    assert table.column("vector").to_pylist()[1] == pytest.approx([1000.0, 1.25, 1.5, 1.75])


def test_input_error_after_the_first_batch_keeps_its_type(tmp_path: Path) -> None:
    rows = [_row(index, text="x" * 2000) for index in range(400)]
    rows.append(_row(400, embedding=None, embedding_error="embed failed"))
    operator = IngestVdbOperator(vdb=_backend(tmp_path, overwrite=False, stream_batch_bytes=64 * 1024))

    with pytest.raises(VdbUploadError, match="refusing a partial write"):
        operator._stream_ingest(iter(_frames(rows, batch_size=50)))
    with pytest.raises(ValueError):
        _table(tmp_path)
