# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

lancedb = pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.common.vdb.sink import (
    CommitOutcomeUnknown,
    VdbOperationConflict,
    VdbSinkPolicy,
    VdbWriteNotFinalized,
)
from nemo_retriever.operators.vdb import IngestVdbOperator

_POLICY = VdbSinkPolicy(max_batch_bytes=1024, prefetch_batches=1)


def _frame(start: int, stop: int, *, text_prefix: str = "chunk") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": f"{text_prefix}-{row_id}",
                "text_embeddings_1b_v2": {"embedding": [float(row_id), 1.0]},
                "source_id": f"/tmp/doc-{row_id}.pdf",
                "page_number": row_id,
                "metadata": {"content_metadata": {"type": "text", "id": f"row-{row_id}"}},
            }
            for row_id in range(start, stop)
        ]
    )


def _operator(uri: Path, *, overwrite: bool, build_index: bool = False) -> IngestVdbOperator:
    return IngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={
            "uri": str(uri),
            "table_name": "chunks",
            "vector_dim": 2,
            "overwrite": overwrite,
            "build_index": build_index,
            "index_type": "IVF_FLAT",
            "num_partitions": 2,
        },
    )


def _state(uri: Path) -> tuple[list[str], list[int]]:
    table = lancedb.connect(str(uri)).open_table("chunks")
    ids = table.to_arrow().sort_by("id").column("id").to_pylist()
    versions = [int(version["version"]) for version in table.list_versions()]
    return ids, versions


def _fail_after_first_batch():
    yield _frame(10, 12)
    raise RuntimeError("injected source failure")


@pytest.mark.parametrize("overwrite", [True, False], ids=["overwrite", "append"])
def test_midstream_failure_preserves_target_and_retry_is_exactly_once(tmp_path: Path, overwrite: bool) -> None:
    _operator(tmp_path, overwrite=True).consume_batches(
        iter([_frame(0, 2)]),
        operation_id="seed",
        policy=_POLICY,
    )
    before = _state(tmp_path)
    operator = _operator(tmp_path, overwrite=overwrite)

    with pytest.raises(RuntimeError, match="injected source failure"):
        operator.consume_batches(
            _fail_after_first_batch(),
            operation_id=f"write-{overwrite}",
            policy=_POLICY,
        )

    assert _state(tmp_path) == before
    report = operator.consume_batches(
        iter([_frame(10, 12), _frame(12, 14)]),
        operation_id=f"write-{overwrite}",
        policy=_POLICY,
    )

    ids, versions = _state(tmp_path)
    expected = [f"row-{row_id}" for row_id in range(10, 14)]
    if not overwrite:
        expected = ["row-0", "row-1", *expected]
    assert ids == expected
    assert len(ids) == len(set(ids))
    assert len(versions) == len(before[1]) + 1
    assert report.rows_written == 4


def test_acknowledged_append_retry_is_a_noop_and_changed_content_conflicts(
    tmp_path: Path,
) -> None:
    _operator(tmp_path, overwrite=True).consume_batches(
        iter([_frame(0, 2)]),
        operation_id="seed",
        policy=_POLICY,
    )
    operator = _operator(tmp_path, overwrite=False)
    first = operator.consume_batches(
        iter([_frame(10, 14)]),
        operation_id="append-10-14",
        policy=_POLICY,
    )
    after_first = _state(tmp_path)

    retry = operator.consume_batches(
        iter([_frame(10, 12), _frame(12, 14)]),
        operation_id="append-10-14",
        policy=_POLICY,
    )
    assert _state(tmp_path) == after_first
    assert retry.outcome == "already_succeeded"
    assert retry.rows_written == first.rows_written == 4

    with pytest.raises(VdbOperationConflict, match="different canonical content"):
        operator.consume_batches(
            iter([_frame(10, 14, text_prefix="changed")]),
            operation_id="append-10-14",
            policy=_POLICY,
        )
    assert _state(tmp_path) == after_first


def test_finalization_failure_blocks_reads_and_resumes_without_readding_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _operator(tmp_path, overwrite=True).consume_batches(
        iter([_frame(0, 2)]),
        operation_id="seed",
        policy=_POLICY,
    )
    operator = _operator(tmp_path, overwrite=False, build_index=True)
    original_write_to_index = operator._vdb.write_to_index

    def fail_index(*args, **kwargs):
        raise RuntimeError("injected index failure")

    monkeypatch.setattr(operator._vdb, "write_to_index", fail_index)
    with pytest.raises(RuntimeError, match="injected index failure"):
        operator.consume_batches(
            iter([_frame(10, 14)]),
            operation_id="append-and-finalize",
            policy=_POLICY,
        )

    table = lancedb.connect(str(tmp_path)).open_table("chunks")
    data_version = int(table.version)
    assert table.count_rows() == 6
    with pytest.raises(VdbWriteNotFinalized, match="not finalized"):
        operator._vdb.retrieval([[0.0, 1.0]], top_k=10)
    with pytest.raises(VdbOperationConflict, match="unfinished bounded-sink operation"):
        _operator(tmp_path, overwrite=False).consume_batches(
            iter([_frame(20, 22)]),
            operation_id="different-append",
            policy=_POLICY,
        )

    monkeypatch.setattr(operator._vdb, "write_to_index", original_write_to_index)
    report = operator.consume_batches(
        iter([_frame(10, 12), _frame(12, 14)]),
        operation_id="append-and-finalize",
        policy=_POLICY,
    )

    assert report.data_version == data_version
    assert _state(tmp_path)[0] == [
        "row-0",
        "row-1",
        "row-10",
        "row-11",
        "row-12",
        "row-13",
    ]
    assert operator._vdb.retrieval([[0.0, 1.0]], top_k=10)


def test_lost_append_commit_acknowledgement_fails_closed(tmp_path: Path, monkeypatch) -> None:
    _operator(tmp_path, overwrite=True).consume_batches(
        iter([_frame(0, 2)]),
        operation_id="seed",
        policy=_POLICY,
    )
    operator = _operator(tmp_path, overwrite=False)
    tags_type = type(lancedb.connect(str(tmp_path)).open_table("chunks").tags)
    original_create = tags_type.create

    def fail_data_marker(self, tag: str, version: int) -> None:
        if tag.startswith("nemo_sink_data_"):
            raise RuntimeError("injected lost acknowledgement")
        original_create(self, tag, version)

    monkeypatch.setattr(tags_type, "create", fail_data_marker)
    with pytest.raises(RuntimeError, match="injected lost acknowledgement"):
        operator.consume_batches(
            iter([_frame(10, 14)]),
            operation_id="append-unknown",
            policy=_POLICY,
        )
    after_commit = _state(tmp_path)
    monkeypatch.setattr(tags_type, "create", original_create)

    with pytest.raises(CommitOutcomeUnknown, match="refusing to replay append"):
        operator.consume_batches(
            iter([_frame(10, 14)]),
            operation_id="append-unknown",
            policy=_POLICY,
        )
    assert _state(tmp_path) == after_commit
