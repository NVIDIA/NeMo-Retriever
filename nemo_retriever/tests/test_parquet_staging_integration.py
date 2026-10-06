# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Ray and LanceDB coverage for Parquet staging."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

ray = pytest.importorskip("ray", minversion="2.56.1")
lancedb = pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.graph.executor import RayDataExecutor
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.ingest.staging import load_stage, prepare_stage_dir
from nemo_retriever.ingest.staging_sink import StageTarget
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.vdb import STAGE_PARQUET_VDB_KWARG, IngestVdbOperator

_DIM = 4


class FileRows(AbstractOperator):
    """Turn each input file into three embedded rows and log each file it processes."""

    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir=log_dir)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        rows = []
        for path in data["path"]:
            with Path(self.log_dir, f"{os.getpid()}.log").open("a") as log:
                log.write(path + "\n")
            for index in range(3):
                text = f"{os.path.basename(path)} row {index}"
                seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
                vector = np.random.default_rng(seed).standard_normal(_DIM).astype(np.float32)
                rows.append(
                    {
                        "text": text,
                        "text_embeddings_1b_v2": {"embedding": vector},
                        "path": path,
                        "page_number": index + 1,
                        "metadata": {"source_path": path},
                    }
                )
        return pd.DataFrame(rows)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


@pytest.mark.integration
def test_stage_and_load_through_real_ray(tmp_path, tmp_path_factory, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    documents = []
    for index in range(7):
        path = tmp_path / "inputs" / f"doc-{index}.pdf"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"document {index}".encode())
        documents.append(str(path))
    (tmp_path / "log").mkdir()
    lancedb_kwargs = {"uri": str(tmp_path / "lance"), "table_name": "chunks", "vector_dim": _DIM, "hybrid": True}
    target = StageTarget(prepare_stage_dir(str(tmp_path / "stage")), rows_per_part=4)
    graph = (
        Graph()
        >> FileRows(log_dir=str(tmp_path / "log"))
        >> IngestVdbOperator(vdb_op="lancedb", vdb_kwargs={**lancedb_kwargs, STAGE_PARQUET_VDB_KWARG: target})
    )

    if ray.is_initialized():
        ray.shutdown()
    temp_dir = str(tmp_path_factory.mktemp("r"))
    ray.init(address="local", num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False, _temp_dir=temp_dir)
    try:
        executor = RayDataExecutor(graph, node_overrides={"FileRows": {"concurrency": 2, "batch_size": 1}})
        results = executor.ingest(documents)
    finally:
        ray.shutdown()
    loaded = load_stage(target.stage_dir, results, lancedb_kwargs=lancedb_kwargs)

    table = lancedb.connect(str(tmp_path / "lance")).open_table("chunks")
    processed = sorted(line for log in (tmp_path / "log").glob("*.log") for line in log.read_text().splitlines())
    assert processed == sorted(documents)
    assert sum(1 for result in results if result["part"]) > 1
    assert (loaded["rows"], table.count_rows()) == (21, 21)
    assert len(table.search("row", query_type="fts").limit(100).to_list()) == 21
