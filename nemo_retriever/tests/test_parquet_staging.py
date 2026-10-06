# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet staging for batch ingest: worker sink, load, and CLI wiring."""

from __future__ import annotations

import importlib
import json
import os
from typing import Any

import lancedb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import nemo_retriever.ingest.execution as ingest_execution
import nemo_retriever.ingest.plan as ingest_plan
from nemo_retriever.common.vdb.arrow import cached_vector_schema
from nemo_retriever.common.vdb.lancedb import LanceDB, _create_lancedb_result
from nemo_retriever.common.vdb.records import VdbUploadError, _client_record_from_graph_row
from nemo_retriever.ingest import staging, staging_sink
from nemo_retriever.ingest.staging import StagingError, load_stage, prepare_stage_dir
from nemo_retriever.ingest.staging_sink import StagedParquetDatasink, StageTarget
from nemo_retriever.operators.vdb import STAGE_PARQUET_VDB_KWARG, IngestVdbOperator

_DIM = 4


def _rows(path: str, count: int = 3, *, vectors: list | None = None) -> list[dict]:
    rows = []
    for index in range(count):
        vector = vectors[index] if vectors else np.random.default_rng(index).standard_normal(_DIM).tolist()
        rows.append(
            {
                "text": f"{os.path.basename(path)} lexical row {index}",
                "text_embeddings_1b_v2": {"embedding": vector},
                "path": path,
                "page_number": index + 1,
                "metadata": {"source_path": path},
            }
        )
    return rows


def _write(stage_dir: str, rows: list[dict], **vdb_kwargs: Any) -> dict:
    """Run one write task of the datasink over ``rows``."""
    sink = StagedParquetDatasink(StageTarget(stage_dir), LanceDB(vector_dim=_DIM, **vdb_kwargs))
    return sink.write([pd.DataFrame(rows)], None)


def _lancedb_kwargs(tmp_path) -> dict:
    return {"uri": str(tmp_path / "lance"), "table_name": "chunks", "vector_dim": _DIM, "hybrid": True}


def test_staged_rows_match_the_in_driver_rows_and_cached_schema(tmp_path) -> None:
    rows = _rows("/docs/a.pdf")

    result = _write(str(tmp_path), rows)

    staged = pq.read_table(tmp_path / result["part"]["name"])
    expected = [_create_lancedb_result(_client_record_from_graph_row(row), expected_dim=_DIM)[0] for row in rows]
    # Parquet names the fixed-list child "element"; every other field matches the cached contract.
    assert staged.schema.remove(0).equals(cached_vector_schema(_DIM).remove(0), check_metadata=True)
    assert staged.drop_columns(["vector"]).to_pylist() == [
        {key: row[key] for key in ("id", "text", "source", "metadata")} for row in expected
    ]
    assert np.array_equal(
        np.asarray(staged.column("vector").to_pylist(), dtype=np.float32),
        np.asarray([row["vector"] for row in expected], dtype=np.float32),
    )


@pytest.mark.parametrize(("policy", "staged"), [("drop", 1), ("fill", 3)])
def test_bad_vectors_follow_on_bad_vectors(tmp_path, policy: str, staged: int) -> None:
    vectors = [[1.0, 0.0, 0.0, 0.0], [1.0, float("inf"), 0.0, 0.0], [1.0, None, 0.0, 0.0]]
    rows = _rows("/docs/a.pdf", vectors=vectors)

    result = _write(str(tmp_path), rows, on_bad_vectors=policy)

    assert result["part"]["rows"] == staged
    with pytest.raises(ValueError, match="finite values"):
        _write(str(tmp_path), rows, on_bad_vectors="error")


def test_reader_batches_retain_at_most_one_bounded_row_group(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(staging_sink, "_ROW_GROUP_BYTES", 1 << 20)
    rows = [{**row, "text": f"{row['text']} " + "filler " * 20_000} for row in _rows("/docs/a.pdf", count=30)]
    path = str(tmp_path / _write(str(tmp_path), rows)["part"]["name"])

    batches = list(staging._staged_reader([path]))

    assert pq.read_metadata(path).num_row_groups > 1
    assert sum(batch.num_rows for batch in batches) == 30
    assert max(batch.get_total_buffer_size() for batch in batches) <= 2 << 20


def test_load_replaces_the_table_with_only_the_reported_parts(tmp_path) -> None:
    stage_dir = prepare_stage_dir(str(tmp_path / "stage"))
    results = [_write(stage_dir, _rows(f"/docs/{name}.pdf")) for name in ("a", "b")]
    # A failed task attempt can leave a published part that no successful task reports.
    _write(stage_dir, _rows("/docs/orphan.pdf"))

    loaded = load_stage(stage_dir, results, lancedb_kwargs=_lancedb_kwargs(tmp_path))

    table = lancedb.connect(str(tmp_path / "lance")).open_table("chunks")
    assert (loaded["rows"], table.count_rows()) == (6, 6)
    assert len(table.search("lexical", query_type="fts").limit(20).to_list()) == 6
    assert not table.search("orphan", query_type="fts").limit(20).to_list()


def test_staging_fails_closed_before_the_table_changes(tmp_path) -> None:
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "notes.txt").write_text("x")
    with pytest.raises(StagingError, match="not empty"):
        prepare_stage_dir(str(tmp_path / "used"))

    stage_dir = prepare_stage_dir(str(tmp_path / "stage"))
    rows = _rows("/docs/a.pdf")
    rows[1].pop("text_embeddings_1b_v2")
    with pytest.raises(VdbUploadError, match="missing embedding=1"):
        load_stage(stage_dir, [_write(stage_dir, rows)], lancedb_kwargs=_lancedb_kwargs(tmp_path))
    with pytest.raises(StagingError, match="stage errors"):
        failed = {**_write(stage_dir, _rows("/docs/b.pdf")), "stage_errors": 1}
        load_stage(stage_dir, [failed], lancedb_kwargs=_lancedb_kwargs(tmp_path))
    with pytest.raises(StagingError, match="no rows"):
        load_stage(stage_dir, [], lancedb_kwargs=_lancedb_kwargs(tmp_path))
    assert not (tmp_path / "lance").exists()


def _request(tmp_path, *, run_mode: str = "batch", overwrite: bool = True, stage_dir: str | None = None):
    document = tmp_path / "doc.txt"
    document.write_text("hello")
    return ingest_plan.IngestPlanRequest(
        source=ingest_plan.IngestSourceOptions(documents=[str(document)]),
        runtime=ingest_plan.IngestRuntimeOptions(run_mode=run_mode),
        storage=ingest_plan.IngestStorageOptions(lancedb_uri=str(tmp_path / "lance"), overwrite=overwrite),
        staging=ingest_plan.IngestStagingOptions(stage_dir=stage_dir),
    )


def test_plan_validates_staging_options(tmp_path) -> None:
    stage_dir = str(tmp_path / "stage")

    plan = ingest_plan.resolve_ingest_plan(_request(tmp_path, stage_dir=stage_dir))

    assert plan.staging == ingest_plan.IngestStagingOptions(stage_dir=stage_dir)
    for request, message in [
        (_request(tmp_path, run_mode="inprocess", stage_dir=stage_dir), "retriever ingest batch"),
        (_request(tmp_path, overwrite=False, stage_dir=stage_dir), "append"),
        (_request(tmp_path, stage_dir="s3://bucket/stage"), "local filesystem"),
    ]:
        with pytest.raises(ValueError, match=message):
            ingest_plan.resolve_ingest_plan(request)


def test_cli_stage_dir_is_batch_only_and_shown_in_dry_run(tmp_path) -> None:
    cli = importlib.import_module("nemo_retriever.cli.main").app
    document = tmp_path / "doc.txt"
    document.write_text("hello")
    stage_dir = str(tmp_path / "stage")

    local = CliRunner().invoke(cli, ["ingest", "local", str(document), "--stage-dir", stage_dir])
    dry_run = CliRunner().invoke(cli, ["ingest", "batch", str(document), "--stage-dir", stage_dir, "--dry-run"])

    assert local.exit_code == 1 and "--stage-dir" in local.output
    assert json.loads(dry_run.output)["staging"] == {"stage_dir": stage_dir}
    assert not os.path.exists(stage_dir)


def test_staged_execution_runs_the_graph_once_with_the_target(tmp_path, monkeypatch) -> None:
    plan = ingest_plan.resolve_ingest_plan(_request(tmp_path, stage_dir=str(tmp_path / "stage")))
    run_plans = []

    class FakeIngestor:
        def __init__(self, run_plan: Any) -> None:
            self.run_plan = run_plan

        def _remote_stage_diagnostics(self) -> dict:
            return {}

        def ingest(self) -> list:
            run_plans.append(self.run_plan)
            vdb_kwargs = dict(self.run_plan.vdb_params.vdb_kwargs)
            target = vdb_kwargs.pop(STAGE_PARQUET_VDB_KWARG)
            rows = _rows(self.run_plan.documents[0], vectors=[[0.5] * 2048] * 3)
            # The upload rules skip a row without searchable content, as on the default path.
            rows.append({"path": self.run_plan.documents[0], "metadata": {}})
            return [StagedParquetDatasink(target, LanceDB(**vdb_kwargs)).write([pd.DataFrame(rows)], None)]

    monkeypatch.setattr(ingest_execution, "build_ingest_pipeline", FakeIngestor)

    summary = ingest_execution.execute_ingest_plan(plan).to_summary_dict()

    [run_plan] = run_plans
    assert run_plan.staging is None and run_plan.documents == plan.documents
    assert STAGE_PARQUET_VDB_KWARG not in plan.vdb_params.vdb_kwargs
    assert (summary["n_rows"], summary["result_n_rows"]) == (3, 4)


def test_operator_never_falls_back_to_a_direct_upload(tmp_path) -> None:
    target = StageTarget(str(tmp_path))
    operator = IngestVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": str(tmp_path), STAGE_PARQUET_VDB_KWARG: target})
    sparse = IngestVdbOperator(vdb_op="lancedb", vdb_kwargs={"sparse": True, STAGE_PARQUET_VDB_KWARG: target})

    with pytest.raises(RuntimeError, match="terminal VDB upload"):
        operator.process(pd.DataFrame(_rows("/docs/a.pdf")))
    with pytest.raises(ValueError, match="dense vectors"):
        sparse.staging_datasink()
