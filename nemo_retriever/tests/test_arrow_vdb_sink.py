# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

lancedb = pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.common.vdb.adt_vdb import UnsupportedVDBOperation, VDB, VDBInvalidRequest
from nemo_retriever.common.vdb._lancedb_stream import DataCommittedFinalizationError
from nemo_retriever.common.vdb._lancedb_stream_state import VdbOperationConflict
from nemo_retriever.common.vdb.arrow import cached_vector_schema
from nemo_retriever.common.vdb.lancedb import LanceDB


def _cached(start: int = 0, rows: int = 16) -> pa.Table:
    schema = cached_vector_schema(2).append(
        pa.field("partition", pa.int64(), metadata={b"meaning": b"source partition"})
    )
    schema = schema.with_metadata({b"corpus": b"frozen-corpus"})
    return pa.Table.from_pylist(
        [
            {
                "vector": [float(i), -0.0 if i % 2 else 1.0],
                "id": f"row-{i}",
                "text": f"cached passage {i}",
                "source": f"source-{i % 2}",
                "metadata": f'{{"row": {i}}}',
                "partition": i % 2,
            }
            for i in range(start, start + rows)
        ],
        schema=schema,
    )


def _backend(path: Path, **kwargs) -> LanceDB:
    kwargs = {"embedding_model_name": "cached-model", "embedding_model_revision": "frozen-revision", **kwargs}
    return LanceDB(uri=str(path), table_name="cached", vector_dim=2, build_index=False, **kwargs)


def _reader(table: pa.Table, batch_size: int = 4) -> pa.RecordBatchReader:
    return pa.RecordBatchReader.from_batches(table.schema, table.to_batches(max_chunksize=batch_size))


def _table(path: Path):
    return lancedb.connect(str(path)).open_table("cached")


@pytest.mark.parametrize(
    "model,revision", [(None, None), ("cached-model", None), (None, "revision-only"), ("modèle", "révision")]
)
def test_exported_schema_stages_parquet_accepted_by_ingest_arrow(tmp_path, model, revision):
    schema = cached_vector_schema(2)
    assert schema.names == ["vector", "id", "text", "source", "metadata"]
    assert schema.field("vector").type == pa.list_(pa.float32(), 2)
    assert all(schema.field(name).type == pa.string() for name in ("id", "text", "source", "metadata"))
    assert schema.metadata is None
    source = _cached(rows=4)
    staged = pa.Table.from_arrays([source[name] for name in schema.names], schema=schema)
    metadata = {}
    if model:
        metadata[b"nemo_retriever.embedding_model_name"] = model.encode("utf-8")
    if revision:
        metadata[b"nemo_retriever.embedding_model_revision"] = revision.encode("utf-8")
    staged = staged.replace_schema_metadata(metadata)
    path = tmp_path / "staged.parquet"
    pq.write_table(staged, path)
    parquet = pq.ParquetFile(path)
    backend = _backend(tmp_path / "sink", embedding_model_name=model, embedding_model_revision=revision)
    backend.ingest_arrow(
        pa.RecordBatchReader.from_batches(parquet.schema_arrow, parquet.iter_batches(batch_size=2)), expected_rows=4
    )
    actual = _table(tmp_path / "sink").to_arrow()
    assert actual.equals(staged, check_metadata=False)
    assert backend.get_index_metadata("embedding_model_name") == model
    assert backend.get_index_metadata("embedding_model_revision") == revision


@pytest.mark.parametrize("dim", [0, -1, True, 2.5])
def test_exported_schema_rejects_invalid_dimension(dim):
    with pytest.raises(VDBInvalidRequest, match="positive integer"):
        cached_vector_schema(dim)


def test_public_parquet_load_is_native_lazy_and_preserves_readback(tmp_path, monkeypatch):
    expected = _cached().append_column(
        "additional_embedding", pa.array([[1.0] * 16] * 16, type=pa.list_(pa.float32(), 16))
    )
    for name in ("id", "text", "source", "metadata"):
        index = expected.schema.get_field_index(name)
        expected = expected.set_column(index, name, expected[name].cast(pa.large_string()))
    parquet_path = tmp_path / "cached.parquet"
    pq.write_table(expected, parquet_path)
    parquet = pq.ParquetFile(parquet_path)
    pulled = []
    entered = []

    def batches():
        for batch in parquet.iter_batches(batch_size=4):
            pulled.append(batch)
            yield batch

    connection_type = type(lancedb.connect(str(tmp_path / "sink")))
    original_create = connection_type.create_table

    def observed_create(self, *args, **kwargs):
        reader = kwargs["data"]
        assert isinstance(reader, pa.RecordBatchReader)
        assert len(pulled) < 4
        entered.append(reader)
        return original_create(self, *args, **kwargs)

    def forbidden_records(*args, **kwargs):
        raise AssertionError("cached Arrow went through the Python-record converter")

    monkeypatch.setattr(connection_type, "create_table", observed_create)
    monkeypatch.setattr(LanceDB, "_iter_stream_rows", forbidden_records)
    backend = _backend(tmp_path / "sink")
    backend.ingest_arrow(pa.RecordBatchReader.from_batches(parquet.schema_arrow, batches()), expected_rows=16)

    stored_table = _table(tmp_path / "sink")
    stored = stored_table.to_arrow()
    assert len(entered) == 1 and len(pulled) == 4
    assert stored_table.version == 1
    assert stored.equals(expected, check_metadata=False)
    actual_bits = stored["vector"].combine_chunks().values.to_numpy().view(np.uint32)
    expected_bits = expected["vector"].combine_chunks().values.to_numpy().view(np.uint32)
    np.testing.assert_array_equal(actual_bits, expected_bits)
    assert stored.schema.metadata[b"corpus"] == b"frozen-corpus"
    assert stored.schema.metadata[b"nemo_retriever.embedding_model_name"] == b"cached-model"
    assert stored.schema.metadata[b"nemo_retriever.embedding_model_revision"] == b"frozen-revision"
    assert stored.schema.field("partition").metadata == expected.schema.field("partition").metadata
    assert backend.retrieval([[0.0, 1.0]], top_k=1)[0][0]["id"] == "row-0"


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("failure", ["nan", "inf", "null-child", "null-vector", "short", "long", "producer"])
def test_later_invalid_batch_never_commits_a_prefix(tmp_path, overwrite, failure):
    _backend(tmp_path).ingest_arrow(_reader(_cached(0, 2)))
    before = _table(tmp_path).to_arrow()
    version = _table(tmp_path).version
    incoming = _cached(10, 4)
    if failure in {"nan", "inf", "null-child", "null-vector"}:
        bad_vector = {
            "nan": [float("nan"), 1.0],
            "inf": [float("inf"), 1.0],
            "null-child": [None, 1.0],
            "null-vector": None,
        }[failure]
        vectors = incoming["vector"].to_pylist()
        vectors[-1] = bad_vector
        incoming = incoming.set_column(
            0, incoming.schema.field(0), pa.array(vectors, type=incoming.schema.field(0).type)
        )

    def batches():
        yield incoming.to_batches()[0].slice(0, 2)
        if failure == "producer":
            raise RuntimeError("injected Parquet read failure")
        yield incoming.to_batches()[0].slice(2)

    reader = pa.RecordBatchReader.from_batches(incoming.schema, batches())
    expected_rows = {"short": 5, "long": 3}.get(failure, 4)
    error_pattern = {
        "short": "row count differs",
        "long": "row count exceeds",
        "producer": "injected Parquet read failure",
    }.get(failure, "finite, non-null float32")
    with pytest.raises((VDBInvalidRequest, pa.ArrowException, RuntimeError, OSError, ValueError), match=error_pattern):
        _backend(tmp_path, overwrite=overwrite).ingest_arrow(reader, expected_rows=expected_rows)
    assert _table(tmp_path).version == version
    assert _table(tmp_path).to_arrow().equals(before, check_metadata=True)


def test_retained_parent_buffers_are_bounded_before_create(tmp_path):
    table = _cached(rows=100)
    reader = pa.RecordBatchReader.from_batches(table.schema, [table.to_batches()[0].slice(0, 1)])
    with pytest.raises(VDBInvalidRequest, match="read smaller source batches"):
        _backend(tmp_path, stream_batch_bytes=512).ingest_arrow(reader)
    assert "cached" not in lancedb.connect(str(tmp_path)).list_tables().tables


def test_native_bad_vector_policy_cannot_silently_drop_cached_rows(tmp_path):
    table = _cached(rows=4)
    extra_vectors = pa.array([[1.0] * 16] * 3 + [[float("nan")] * 16], type=pa.list_(pa.float32(), 16))
    table = table.append_column("additional_embedding", extra_vectors)
    # Lance also recognizes additional wide float-list columns as vectors.
    # Even the default record policy (drop) must fail this cached transaction.
    with pytest.raises((ValueError, pa.ArrowException, RuntimeError), match="additional_embedding"):
        _backend(tmp_path).ingest_arrow(_reader(table), expected_rows=4)
    assert "cached" not in lancedb.connect(str(tmp_path)).list_tables().tables


@pytest.mark.parametrize("vector_type", [pa.list_(pa.float32()), pa.list_(pa.float64(), 2), pa.list_(pa.float32(), 3)])
def test_incompatible_cached_schema_fails_without_consuming_input(tmp_path, vector_type):
    schema = _cached().schema.set(0, pa.field("vector", vector_type))
    pulled = []

    def batches():
        pulled.append(True)
        yield _cached().to_batches()[0]

    reader = pa.RecordBatchReader.from_batches(schema, batches())
    with pytest.raises(VDBInvalidRequest):
        _backend(tmp_path).ingest_arrow(reader)
    assert not pulled


def test_nested_fixed_lists_fail_before_consumption_or_mutation(tmp_path):
    sink = tmp_path / "sink"
    backend = _backend(sink)
    backend.ingest_arrow(_reader(_cached(rows=2)))
    before = _table(sink).to_arrow()
    version = _table(sink).version
    extra_type = pa.struct([pa.field("nested", pa.list_(pa.float32(), 2))])
    incoming = _cached(rows=2).append_column("extra", pa.array([{"nested": [1.0, 2.0]}] * 2, type=extra_type))
    path = tmp_path / "nested.parquet"
    pq.write_table(incoming, path)
    parquet = pq.ParquetFile(path)
    pulled = []

    def batches():
        for batch in parquet.iter_batches(batch_size=1):
            pulled.append(batch)
            yield batch

    with pytest.raises(VDBInvalidRequest, match="nested fixed-size lists"):
        backend.ingest_arrow(pa.RecordBatchReader.from_batches(parquet.schema_arrow, batches()), expected_rows=2)
    assert not pulled
    assert _table(sink).version == version
    assert _table(sink).to_arrow().equals(before, check_metadata=True)


def test_retry_with_changed_long_schema_metadata_conflicts(tmp_path):
    original = _cached(rows=4)
    original = original.replace_schema_metadata({**original.schema.metadata, b"corpus": b"x" * 100 + b"first"})
    backend = _backend(tmp_path, stream_operation_id="metadata-retry")
    backend.ingest_arrow(_reader(original), expected_rows=4)
    before = _table(tmp_path).to_arrow()
    assert before.schema.metadata[b"corpus"] == original.schema.metadata[b"corpus"]
    version = _table(tmp_path).version
    changed = original.replace_schema_metadata({**original.schema.metadata, b"corpus": b"x" * 100 + b"other"})
    with pytest.raises(VdbOperationConflict, match="different write request"):
        backend.ingest_arrow(_reader(changed), expected_rows=4)
    with pytest.raises(VdbOperationConflict, match="different write request"):
        _backend(tmp_path, stream_operation_id="metadata-retry", embedding_model_revision="new-revision").ingest_arrow(
            _reader(original), expected_rows=4
        )
    backend.ingest_arrow(_reader(original, batch_size=3), expected_rows=4)
    assert _table(tmp_path).version == version
    assert _table(tmp_path).to_arrow().equals(before, check_metadata=True)


@pytest.mark.parametrize("stored_model", [None, "cached-model"])
def test_revision_only_append_checks_known_revisions(tmp_path, stored_model):
    _backend(tmp_path, embedding_model_name=stored_model).ingest_arrow(_reader(_cached(rows=4)))
    before = _table(tmp_path).to_arrow()
    version = _table(tmp_path).version
    incoming = _cached(4, 4)
    backend = _backend(
        tmp_path, overwrite=False, embedding_model_name=None, embedding_model_revision="different-revision"
    )
    with pytest.raises(ValueError, match="cannot append vectors from revision"):
        backend.ingest_arrow(_reader(incoming), expected_rows=4)
    assert _table(tmp_path).version == version
    assert _table(tmp_path).to_arrow().equals(before, check_metadata=True)
    _backend(tmp_path, overwrite=False, embedding_model_name=None).ingest_arrow(_reader(incoming), expected_rows=4)
    assert _table(tmp_path).to_arrow()["id"].to_pylist() == [f"row-{i}" for i in range(8)]
    assert backend.get_index_metadata("embedding_model_name") == stored_model
    assert backend.get_index_metadata("embedding_model_revision") == "frozen-revision"


def test_append_and_retry_are_exact_once_across_batch_boundaries(tmp_path):
    _backend(tmp_path).ingest_arrow(_reader(_cached(0, 4)))
    backend = _backend(tmp_path, overwrite=False, stream_operation_id="cached-append")
    backend.ingest_arrow(_reader(_cached(4, 8), batch_size=2), expected_rows=8)
    version = _table(tmp_path).version
    backend.ingest_arrow(_reader(_cached(4, 8), batch_size=3), expected_rows=8)
    assert _table(tmp_path).version == version
    assert _table(tmp_path).to_arrow()["id"].to_pylist() == [f"row-{i}" for i in range(12)]
    with pytest.raises(VdbOperationConflict):
        backend.ingest_arrow(_reader(_cached(5, 8)), expected_rows=8)
    assert _table(tmp_path).version == version


def test_finalization_failure_is_recoverable_without_duplicate_vectors(tmp_path, monkeypatch):
    backend = _backend(tmp_path, stream_operation_id="cached-create")
    original = LanceDB._finalize_stream_write

    def fail(*args, **kwargs):
        raise RuntimeError("injected finalization failure")

    monkeypatch.setattr(LanceDB, "_finalize_stream_write", fail)
    with pytest.raises(DataCommittedFinalizationError):
        backend.ingest_arrow(_reader(_cached()), expected_rows=16)
    version = _table(tmp_path).version
    monkeypatch.setattr(LanceDB, "_finalize_stream_write", original)
    backend.ingest_arrow(_reader(_cached(), batch_size=7), expected_rows=16)
    assert _table(tmp_path).version == version
    assert _table(tmp_path).count_rows() == 16


def test_cached_model_conflicts_fail_before_mutation(tmp_path):
    _backend(tmp_path).ingest_arrow(_reader(_cached()))
    before = _table(tmp_path).to_arrow()
    version = _table(tmp_path).version
    with pytest.raises(ValueError, match="embedding model"):
        _backend(tmp_path, overwrite=False, embedding_model_name="different-model").ingest_arrow(_reader(_cached()))
    assert _table(tmp_path).version == version
    assert _table(tmp_path).to_arrow().equals(before, check_metadata=True)


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize(
    "option,configured",
    [
        ("embedding_model_name", "cached-model"),
        ("embedding_model_revision", "frozen-revision"),
        ("embedding_model_name", None),
        ("embedding_model_revision", None),
    ],
)
def test_conflicting_source_identity_fails_before_consumption_or_mutation(tmp_path, overwrite, option, configured):
    _backend(tmp_path).ingest_arrow(_reader(_cached(rows=2)))
    before = _table(tmp_path).to_arrow()
    version = _table(tmp_path).version
    incoming = _cached(10, 4)
    metadata = {**incoming.schema.metadata, f"nemo_retriever.{option}".encode("utf-8"): b"different-source"}
    incoming = incoming.replace_schema_metadata(metadata)
    path = tmp_path / "conflicting.parquet"
    pq.write_table(incoming, path)
    parquet = pq.ParquetFile(path)
    pulled = []

    def batches():
        for batch in parquet.iter_batches(batch_size=2):
            pulled.append(True)
            yield batch

    backend = _backend(tmp_path, overwrite=overwrite, stream_operation_id="conflicting-source", **{option: configured})
    with pytest.raises(VDBInvalidRequest, match=f"Cached Arrow {option}.*Set {option}"):
        backend.ingest_arrow(pa.RecordBatchReader.from_batches(parquet.schema_arrow, batches()), expected_rows=4)
    assert not pulled
    assert _table(tmp_path).version == version
    assert _table(tmp_path).to_arrow().equals(before, check_metadata=True)


def test_append_retains_existing_schema_metadata(tmp_path):
    _backend(tmp_path).ingest_arrow(_reader(_cached(0, 4)))
    incoming = _cached(4, 4)
    metadata = {**incoming.schema.metadata, b"corpus": b"incoming-corpus"}
    incoming = incoming.replace_schema_metadata(metadata)
    _backend(tmp_path, overwrite=False).ingest_arrow(_reader(incoming), expected_rows=4)
    stored = _table(tmp_path).to_arrow()
    assert stored["id"].to_pylist() == [f"row-{i}" for i in range(8)]
    assert stored.schema.metadata[b"corpus"] == b"frozen-corpus"


def test_unsupported_adapter_does_not_consume_arrow(tmp_path):
    backend = _backend(tmp_path)
    pulled = []

    def batches():
        pulled.append(True)
        yield _cached().to_batches()[0]

    reader = pa.RecordBatchReader.from_batches(_cached().schema, batches())
    with pytest.raises(UnsupportedVDBOperation):
        VDB.ingest_arrow(backend, reader)
    assert not pulled


def test_empty_and_indexed_cached_loads(tmp_path):
    _backend(tmp_path / "empty").ingest_arrow(_reader(_cached(rows=0)), expected_rows=0)
    assert "cached" not in lancedb.connect(str(tmp_path / "empty")).list_tables().tables
    backend = LanceDB(
        uri=str(tmp_path / "indexed"),
        table_name="cached",
        vector_dim=2,
        index_type="IVF_FLAT",
        num_partitions=2,
        hybrid=True,
    )
    backend.ingest_arrow(_reader(_cached()), expected_rows=16)
    table = _table(tmp_path / "indexed")
    indices = {tuple(index.columns): index for index in table.list_indices()}
    assert set(indices) == {("vector",), ("text",)}
    for index in indices.values():
        stats = table.index_stats(index.name)
        assert stats.num_indexed_rows == 16 and stats.num_unindexed_rows == 0
    assert backend.retrieval([[0.0, 1.0]], top_k=1, query_texts=["cached passage"])
