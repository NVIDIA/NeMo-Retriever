# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Ray coverage for durable Parquet staging: commit, resume, and crash recovery."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ray = pytest.importorskip("ray", minversion="2.56.1")

from nemo_retriever.graph.executor import RayDataExecutor
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.ingest import staging
from nemo_retriever.ingest.staging import StagingError, stage_documents
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.vdb import STAGE_PARQUET_VDB_KWARG, IngestVdbOperator

_DIM = 4
_ROWS_PER_FILE = 6
_SETTINGS = {"pipeline": "file-rows", "dim": _DIM}


class FileRows(AbstractOperator):
    """Turn each input file into deterministic embedded rows and log each file it processes."""

    def __init__(self, log_dir: str, block_marker: str = "", duplicate: bool = False) -> None:
        super().__init__(log_dir=log_dir, block_marker=block_marker, duplicate=duplicate)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        rows = []
        for path in data["path"]:
            if self.block_marker and self.block_marker in os.path.basename(path):
                time.sleep(600)
            with open(os.path.join(self.log_dir, f"{os.getpid()}.log"), "a") as log:
                log.write(path + "\n")
            for index in range(_ROWS_PER_FILE):
                text = f"{os.path.basename(path)} row {index}"
                seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
                metadata: dict[str, Any] = {"source_path": path}
                if self.duplicate:
                    # Explicit IDs win, so every file repeats the same six IDs.
                    metadata["content_metadata"] = {"id": f"row-{index}"}
                rows.append(
                    {
                        "text": text,
                        "text_embeddings_1b_v2": {
                            "embedding": np.random.default_rng(seed).standard_normal(_DIM).astype(np.float32)
                        },
                        "path": path,
                        "page_number": index + 1,
                        "metadata": metadata,
                    }
                )
        return pd.DataFrame(rows)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def make_run_shard(work: str, *, block_marker: str = "", duplicate: bool = False):
    def run_shard(shard: staging.Shard, target: Any) -> Any:
        graph = (
            Graph()
            >> FileRows(log_dir=os.path.join(work, "log"), block_marker=block_marker, duplicate=duplicate)
            >> IngestVdbOperator(
                vdb_op="lancedb",
                vdb_kwargs={
                    "uri": os.path.join(work, "lance"),
                    "table_name": "chunks",
                    "vector_dim": _DIM,
                    "hybrid": True,
                    STAGE_PARQUET_VDB_KWARG: dataclasses.asdict(target),
                },
            )
        )
        executor = RayDataExecutor(graph, node_overrides={"FileRows": {"concurrency": 2, "batch_size": 1}})
        return executor.ingest(shard.documents)

    return run_shard


def input_paths(work: str, count: int) -> list[str]:
    return [os.path.join(work, "inputs", f"doc-{index:03d}.pdf") for index in range(count)]


def make_inputs(work: str, count: int) -> list[str]:
    os.makedirs(os.path.join(work, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(work, "log"), exist_ok=True)
    paths = input_paths(work, count)
    for index, path in enumerate(paths):
        with open(path, "wb") as handle:
            handle.write(f"document {index}".encode())
    return paths


def run_staging(work: str, documents: list[str], **kwargs: Any) -> staging.StagingSummary:
    return stage_documents(
        documents,
        stage_dir=os.path.join(work, "stage"),
        shard_files=3,
        settings=_SETTINGS,
        run_shard=make_run_shard(work, **kwargs),
        rows_per_part=4,
        write_concurrency=2,
    )


def processed_files(work: str) -> list[str]:
    log_dir = Path(work, "log")
    return sorted(line for log in log_dir.glob("*.log") for line in log.read_text().splitlines())


def staged_ids(work: str) -> list[str]:
    manifest = staging.StagingManifest._read(os.path.join(work, "stage", staging.MANIFEST_NAME))
    stage_dir = os.path.join(work, "stage")
    ids = []
    for record in manifest:
        if record.get("type") == "shard":
            assert (
                staging.verify_shard_parts(
                    stage_dir, os.path.join(stage_dir, "shards", record["shard_id"]), record["parts"], checksums=True
                )
                == []
            )
            for part in record["parts"]:
                ids.extend(
                    pq.read_table(os.path.join(stage_dir, part["path"]), columns=["id"]).column("id").to_pylist()
                )
    return ids


@pytest.fixture
def local_ray(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    if ray.is_initialized():
        ray.shutdown()
    temp_dir = tmp_path_factory.mktemp("r")
    ray.init(
        address="local", num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False, _temp_dir=str(temp_dir)
    )
    yield
    ray.shutdown()


@pytest.mark.integration
def test_stage_commits_every_shard_and_rerun_skips_them(tmp_path, local_ray) -> None:
    work = str(tmp_path)
    documents = make_inputs(work, 7)

    first = run_staging(work, documents)
    second = run_staging(work, documents)

    assert (first.shards, first.staged_shards, first.rows) == (3, 3, 7 * _ROWS_PER_FILE)
    assert (second.staged_shards, second.skipped_shards, second.rows) == (0, 3, 7 * _ROWS_PER_FILE)
    assert processed_files(work) == sorted(documents)
    ids = staged_ids(work)
    assert len(ids) == len(set(ids)) == 7 * _ROWS_PER_FILE
    assert not os.path.exists(os.path.join(work, "lance"))


@pytest.mark.integration
def test_crash_after_publish_before_manifest_record_restages_the_shard(tmp_path, local_ray, monkeypatch) -> None:
    work = str(tmp_path)
    documents = make_inputs(work, 7)
    append = staging.StagingManifest.append

    def crash_on_second_shard(self, record):
        if record.get("index") == 1:
            raise RuntimeError("simulated crash after the shard directory was published")
        return append(self, record)

    monkeypatch.setattr(staging.StagingManifest, "append", crash_on_second_shard)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_staging(work, documents)
    published = sorted(os.listdir(os.path.join(work, "stage", "shards")))
    monkeypatch.setattr(staging.StagingManifest, "append", append)

    resumed = run_staging(work, documents)

    assert len(published) == 2 and resumed.removed_uncommitted == (published[1],)
    assert (resumed.staged_shards, resumed.skipped_shards, resumed.rows) == (2, 1, 7 * _ROWS_PER_FILE)
    ids = staged_ids(work)
    assert len(ids) == len(set(ids)) == 7 * _ROWS_PER_FILE


@pytest.mark.integration
def test_duplicate_row_ids_fail_closed_before_commit(tmp_path, local_ray) -> None:
    work = str(tmp_path)
    documents = make_inputs(work, 3)

    with pytest.raises(StagingError, match="duplicate row IDs"):
        run_staging(work, documents, duplicate=True)

    manifest = staging.StagingManifest._read(os.path.join(work, "stage", staging.MANIFEST_NAME))
    assert [record["type"] for record in manifest] == ["header"]


_KILL_SCRIPT = """
import json, sys
import ray
from tests.test_parquet_staging_integration import make_inputs, run_staging
work, temp_dir = sys.argv[1], sys.argv[2]
ray.init(address="local", num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False, _temp_dir=temp_dir)
run_staging(work, make_inputs(work, 7), block_marker="doc-004")
"""


def _kill_session(process: subprocess.Popen, temp_dir: str) -> None:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=60)
    # Ray daemons of the killed driver can outlive its process group; stop only
    # the processes of this test's private Ray session.
    leftovers = subprocess.run(["pgrep", "-f", temp_dir], capture_output=True, text=True).stdout.split()
    for pid in leftovers:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.integration
def test_kill_mid_shard_then_resume_stages_exactly_once(tmp_path, tmp_path_factory, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    work = str(tmp_path)
    temp_dir = str(tmp_path_factory.mktemp("k"))
    stage_dir = os.path.join(work, "stage")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).parents[1]), *sys.path])}
    process = subprocess.Popen(
        [sys.executable, "-c", _KILL_SCRIPT, work, temp_dir],
        env=env,
        cwd=str(Path(__file__).parents[1]),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 240
        attempt_parts: list[str] = []
        while time.monotonic() < deadline:
            manifest_path = os.path.join(stage_dir, staging.MANIFEST_NAME)
            committed = os.path.exists(manifest_path) and '"index": 0' in Path(manifest_path).read_text()
            attempts = list(Path(stage_dir, "shards").glob("*.attempt-*/part-*.parquet")) if committed else []
            if attempts:
                attempt_parts = [str(path) for path in attempts]
                break
            assert process.poll() is None, "staging subprocess exited before it could be killed"
            time.sleep(0.5)
        assert attempt_parts, "the second shard never published a part before the kill"
    finally:
        _kill_session(process, temp_dir)

    ray.init(address="local", num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False, _temp_dir=temp_dir)
    try:
        resumed = run_staging(work, input_paths(work, 7))
    finally:
        ray.shutdown()

    assert resumed.skipped_shards == 1 and resumed.staged_shards == 2
    assert any(".attempt-" in name for name in resumed.removed_uncommitted)
    ids = staged_ids(work)
    assert len(ids) == len(set(ids)) == 7 * _ROWS_PER_FILE
    # The first shard's three files ran once; the killed shard's files may have run twice.
    first_shard = sorted(input_paths(work, 7))[:3]
    assert all(processed_files(work).count(path) == 1 for path in first_shard)
    assert json.loads(Path(stage_dir, staging.MANIFEST_NAME).read_text().splitlines()[0])["n_shards"] == 3
