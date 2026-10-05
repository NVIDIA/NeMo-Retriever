# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet staging parts, manifest, and shard commit without a Ray cluster."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_retriever.common.vdb.arrow import cached_vector_dimension, cached_vector_schema
from nemo_retriever.common.vdb.lancedb import LanceDB, _create_lancedb_result
from nemo_retriever.common.vdb.records import VdbUploadError, _client_record_from_graph_row, graph_row_id
from nemo_retriever.common.vdb.staged_parquet import (
    StagedRowPolicy,
    StageTarget,
    find_duplicate_ids,
    part_name,
    stage_blocks,
)
from nemo_retriever.ingest import staging
from nemo_retriever.ingest.staging import StagingError, StagingManifest, plan_shards
from nemo_retriever.operators.vdb import STAGE_PARQUET_VDB_KWARG, IngestVdbOperator

_DIM = 4
_MODEL = "nvidia/test-embed"


def _policy(**overrides: object) -> StagedRowPolicy:
    values = {
        "vector_dim": _DIM,
        "on_bad_vectors": "drop",
        "fill_value": 0.0,
        "validate_vector_length": True,
        "embedding_model_name": _MODEL,
        "embedding_model_revision": None,
    }
    values.update(overrides)
    return StagedRowPolicy(**values)


def _graph_row(index: int, *, source: str = "/docs/a.pdf", vector: object = None) -> dict:
    return {
        "text": f"chunk {index}",
        "text_embeddings_1b_v2": {"embedding": [float(index), 1.0, 0.5, 0.25] if vector is None else vector},
        "path": source,
        "page_number": index + 1,
        "_content_type": "text",
        "metadata": {"source_path": source},
    }


def _target(tmp_path, shard_id: str = "000000-shard", attempt: str = "a1") -> StageTarget:
    return StageTarget(stage_dir=str(tmp_path), shard_id=shard_id, attempt=attempt)


def _shard(tmp_path, count: int = 2) -> staging.Shard:
    files = []
    for index in range(count):
        path = tmp_path / "inputs" / f"doc-{index}.pdf"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"%PDF" + bytes([index]))
        files.append(str(path))
    return plan_shards(files, count)[0]


def _stage(tmp_path) -> tuple[str, StagingManifest]:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    return str(stage_dir), StagingManifest.open(str(stage_dir), {"version": 1})


def test_staged_rows_match_the_in_driver_rows_and_cached_schema(tmp_path) -> None:
    rows = [_graph_row(index) for index in range(3)]
    result = stage_blocks([pd.DataFrame(rows)], target=_target(tmp_path), policy=_policy())

    assert result["graph_rows"] == result["records"] == result["rows"] == 3
    [part] = result["parts"]
    path = tmp_path / part["path"]
    assert part["path"].startswith(os.path.join("shards", "000000-shard.attempt-a1", "part-"))
    assert os.path.basename(part["path"]) == part_name([graph_row_id(row) for row in rows])
    assert not [name for name in os.listdir(path.parent) if name.endswith(".tmp")]

    staged = pq.read_table(path)
    assert cached_vector_dimension(staged.schema) == _DIM
    in_memory = pa.Table.from_batches(
        [pa.RecordBatch.from_pylist([], schema=cached_vector_schema(_DIM, _MODEL))]
    ).schema
    # Parquet names the fixed-list child "element"; every other field and the metadata are identical.
    assert staged.schema.remove(0).equals(in_memory.remove(0), check_metadata=True)
    expected = [_create_lancedb_result(_client_record_from_graph_row(row), expected_dim=_DIM)[0] for row in rows]
    assert staged.drop_columns(["vector"]).to_pylist() == [
        {key: row[key] for key in ("id", "text", "source", "metadata")} for row in expected
    ]
    assert np.array_equal(
        np.asarray(staged.column("vector").to_pylist(), dtype=np.float32),
        np.asarray([row["vector"] for row in expected], dtype=np.float32),
    )


def test_staged_part_loads_through_ingest_arrow(tmp_path) -> None:
    result = stage_blocks(
        [pd.DataFrame([_graph_row(index) for index in range(3)])], target=_target(tmp_path), policy=_policy()
    )
    parquet = pq.ParquetFile(tmp_path / result["parts"][0]["path"])
    reader = pa.RecordBatchReader.from_batches(parquet.schema_arrow, parquet.iter_batches(batch_size=2))
    backend = LanceDB(
        uri=str(tmp_path / "lance"), table_name="t", vector_dim=_DIM, embedding_model_name=_MODEL, build_index=False
    )

    backend.ingest_arrow(reader, expected_rows=3)

    import lancedb

    stored = lancedb.connect(str(tmp_path / "lance")).open_table("t").to_arrow()
    assert sorted(stored.column("id").to_pylist()) == sorted(graph_row_id(_graph_row(index)) for index in range(3))


@pytest.mark.parametrize(
    ("policy", "kept", "dropped"),
    [("drop", 1, 3), ("fill", 4, 0)],
)
def test_bad_vectors_follow_the_configured_policy(tmp_path, policy, kept, dropped) -> None:
    rows = [
        _graph_row(0),
        _graph_row(1, vector=[1.0, float("nan"), 0.0, 0.0]),
        _graph_row(2, vector=[1.0, float("inf"), 0.0, 0.0]),
        _graph_row(3, vector=[1.0, None, 0.0, 0.0]),
    ]
    result = stage_blocks([pd.DataFrame(rows)], target=_target(tmp_path), policy=_policy(on_bad_vectors=policy))

    assert result["rows"] == kept
    assert sum(result["dropped"].values()) == dropped
    vectors = pq.read_table(tmp_path / result["parts"][0]["path"]).column("vector").to_pylist()
    assert all(np.isfinite(vector).all() for vector in vectors)


def test_error_policy_and_null_policy_fail_closed(tmp_path) -> None:
    with pytest.raises(ValueError, match="finite values"):
        stage_blocks(
            [pd.DataFrame([_graph_row(0, vector=[1.0, float("inf"), 0.0, 0.0])])],
            target=_target(tmp_path),
            policy=_policy(on_bad_vectors="error"),
        )
    backend = LanceDB(uri=str(tmp_path / "lance"), table_name="t", on_bad_vectors="null")
    with pytest.raises(ValueError, match="null vectors"):
        StagedRowPolicy.from_lancedb(backend)


def test_duplicate_rows_in_a_task_publish_nothing(tmp_path) -> None:
    result = stage_blocks([pd.DataFrame([_graph_row(0), _graph_row(0)])], target=_target(tmp_path), policy=_policy())

    assert result["duplicate_ids"] == [graph_row_id(_graph_row(0))]
    assert result["parts"] == []


def test_shards_are_deterministic_and_reject_bad_inputs(tmp_path) -> None:
    files = []
    for name in ("c.pdf", "a.pdf", "b.pdf"):
        (tmp_path / name).write_bytes(name.encode())
        files.append(str(tmp_path / name))

    first = plan_shards(files, 2)
    second = plan_shards(list(reversed(files)), 2)

    assert [shard.shard_id for shard in first] == [shard.shard_id for shard in second]
    assert [os.path.basename(path) for shard in first for path in shard.documents] == ["a.pdf", "b.pdf", "c.pdf"]
    assert [len(shard.inputs) for shard in first] == [2, 1]
    (tmp_path / "a.pdf").write_bytes(b"changed content")
    assert plan_shards(files, 2)[0].shard_id != first[0].shard_id
    with pytest.raises(StagingError, match="more than once"):
        plan_shards([files[0], files[0]], 2)
    with pytest.raises(StagingError, match="not a file"):
        plan_shards([str(tmp_path / "missing.pdf")], 2)


def test_manifest_truncates_a_torn_append_and_binds_its_inputs(tmp_path) -> None:
    header = {"version": 1, "shard_files": 2, "inputs_sha256": "i", "settings_sha256": "s"}
    manifest = StagingManifest.open(str(tmp_path), header)
    manifest.append({"type": "shard", "shard_id": "000000-x", "parts": []})
    with open(tmp_path / staging.MANIFEST_NAME, "ab") as handle:
        handle.write(b'{"type": "shard", "shard_id": "000001-y"')

    reopened = StagingManifest.open(str(tmp_path), header)

    assert list(reopened.committed_shards) == ["000000-x"]
    assert (tmp_path / staging.MANIFEST_NAME).read_bytes().endswith(b"\n")
    with pytest.raises(StagingError, match="changed: settings_sha256"):
        StagingManifest.open(str(tmp_path), {**header, "settings_sha256": "other"})


def test_new_manifest_requires_an_empty_directory(tmp_path) -> None:
    (tmp_path / "unrelated.txt").write_text("data")

    with pytest.raises(StagingError, match="not empty"):
        StagingManifest.open(str(tmp_path), {"version": 1})


def test_lock_rejects_a_concurrent_run(tmp_path) -> None:
    with staging._exclusive_lock(str(tmp_path)):
        with pytest.raises(StagingError, match="Another process"):
            with staging._exclusive_lock(str(tmp_path)):
                pass


def test_commit_verifies_parts_and_publishes_the_attempt(tmp_path) -> None:
    shard = _shard(tmp_path)
    stage_dir, manifest = _stage(tmp_path)
    target = StageTarget(stage_dir=stage_dir, shard_id=shard.shard_id, attempt="a1")
    first = stage_blocks([pd.DataFrame([_graph_row(0), _graph_row(1)])], target=target, policy=_policy())
    second = stage_blocks([pd.DataFrame([_graph_row(2)])], target=target, policy=_policy())

    record = staging.commit_shard(manifest, shard, target, [first, second])

    assert record["rows"] == 3 and record["vector_dim"] == _DIM and record["embedding_model_name"] == _MODEL
    assert not os.path.exists(target.attempt_dir)
    assert all(part["path"].startswith(os.path.join("shards", shard.shard_id) + os.sep) for part in record["parts"])
    assert (
        staging.verify_shard_parts(
            stage_dir, os.path.join(stage_dir, "shards", shard.shard_id), record["parts"], checksums=True
        )
        == []
    )
    with open(os.path.join(stage_dir, staging.MANIFEST_NAME)) as handle:
        assert json.loads(handle.read().splitlines()[-1])["shard_id"] == shard.shard_id


def test_commit_fails_closed_on_cross_part_duplicates_and_bad_parts(tmp_path) -> None:
    shard = _shard(tmp_path)
    stage_dir, manifest = _stage(tmp_path)
    target = StageTarget(stage_dir=stage_dir, shard_id=shard.shard_id, attempt="a1")
    first = stage_blocks([pd.DataFrame([_graph_row(0), _graph_row(1)])], target=target, policy=_policy())
    second = stage_blocks([pd.DataFrame([_graph_row(1), _graph_row(2)])], target=target, policy=_policy())

    with pytest.raises(StagingError, match="duplicate row IDs across parts") as error:
        staging.commit_shard(manifest, shard, target, [first, second])
    assert graph_row_id(_graph_row(1)) in str(error.value)
    assert find_duplicate_ids(stage_dir, [part["path"] for part in first["parts"] + second["parts"]]) == [
        graph_row_id(_graph_row(1))
    ]

    truncated = dict(first, parts=[dict(first["parts"][0], rows=99)], rows=99)
    with pytest.raises(StagingError, match="failed verification"):
        staging.commit_shard(manifest, shard, target, [truncated])
    assert manifest.committed_shards == {}


def test_commit_refuses_missing_embeddings_and_stage_errors(tmp_path) -> None:
    shard = _shard(tmp_path)
    stage_dir, manifest = _stage(tmp_path)
    target = StageTarget(stage_dir=stage_dir, shard_id=shard.shard_id, attempt="a1")
    missing = {**_graph_row(0)}
    missing.pop("text_embeddings_1b_v2")
    result = stage_blocks([pd.DataFrame([_graph_row(1), missing])], target=target, policy=_policy())

    with pytest.raises(VdbUploadError, match="missing embedding=1"):
        staging.commit_shard(manifest, shard, target, [result])
    with pytest.raises(StagingError, match="stage error"):
        staging.commit_shard(
            manifest,
            shard,
            target,
            [{"stage_error_count": 1, "stage_errors": [{"source_identifier": "/docs/a.pdf"}]}],
        )


def test_reconcile_removes_uncommitted_directories_and_names_damaged_shards(tmp_path) -> None:
    shard = _shard(tmp_path)
    stage_dir, manifest = _stage(tmp_path)
    target = StageTarget(stage_dir=stage_dir, shard_id=shard.shard_id, attempt="a1")
    record = staging.commit_shard(
        manifest, shard, target, [stage_blocks([pd.DataFrame([_graph_row(0)])], target=target, policy=_policy())]
    )
    orphan = os.path.join(stage_dir, "shards", f"{shard.shard_id}.attempt-zz")
    os.makedirs(orphan)
    with open(os.path.join(orphan, "part-x.parquet"), "wb") as handle:
        handle.write(b"stale")

    stray = os.path.join(stage_dir, "shards", "notes.txt")
    with open(stray, "w") as handle:
        handle.write("not a shard")

    assert staging.reconcile_stage_dir(stage_dir, manifest, [shard.shard_id]) == sorted(
        [os.path.basename(orphan), "notes.txt"]
    )
    assert not os.path.exists(orphan) and not os.path.exists(stray)

    with open(os.path.join(stage_dir, record["parts"][0]["path"]), "wb") as handle:
        handle.write(b"damaged")
    with pytest.raises(StagingError, match=shard.shard_id):
        staging.reconcile_stage_dir(stage_dir, manifest, [shard.shard_id])


def test_operator_rejects_unsupported_staging_and_fallback_paths(tmp_path) -> None:
    target = {"stage_dir": str(tmp_path), "shard_id": "000000-x", "attempt": "a1"}
    with pytest.raises(ValueError, match="local filesystem"):
        IngestVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": "s3://bucket/db", STAGE_PARQUET_VDB_KWARG: target})
    with pytest.raises(ValueError, match="dense or hybrid"):
        IngestVdbOperator(
            vdb_op="lancedb", vdb_kwargs={"uri": str(tmp_path), "sparse": True, STAGE_PARQUET_VDB_KWARG: target}
        )
    operator = IngestVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": str(tmp_path), STAGE_PARQUET_VDB_KWARG: target})
    with pytest.raises(RuntimeError, match="terminal VDB upload of a Ray batch graph"):
        operator.process(pd.DataFrame([_graph_row(0)]))


def test_long_rows_get_smaller_row_groups(tmp_path, monkeypatch) -> None:
    from nemo_retriever.common.vdb import staged_parquet

    monkeypatch.setattr(staged_parquet, "_ROW_GROUP_BYTES", 1 << 20)
    rows = [{**_graph_row(index), "text": f"chunk {index} " + "filler " * 20_000} for index in range(30)]

    result = stage_blocks([pd.DataFrame(rows)], target=_target(tmp_path), policy=_policy())

    metadata = pq.read_metadata(tmp_path / result["parts"][0]["path"])
    assert metadata.num_row_groups > 1 and metadata.num_rows == 30
    assert max(metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)) <= 7
