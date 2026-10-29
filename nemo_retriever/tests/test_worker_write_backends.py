# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The worker-native write contract on backends other than LanceDB.

Elasticsearch cases run only when ``NRL_TEST_ELASTICSEARCH_URL`` points at a server.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from nemo_retriever.common.vdb.adt_vdb import VDB
from nemo_retriever.common.vdb.duckdb import DuckDB
from nemo_retriever.common.vdb.elasticsearch import Elasticsearch
from nemo_retriever.operators.vdb import IngestVdbOperator

DIM = 4
ES_URL = os.environ.get("NRL_TEST_ELASTICSEARCH_URL")


def _row(row_id: int) -> dict[str, Any]:
    return {
        "text": f"chunk {row_id}",
        "embedding": [float(row_id) + 0.25 * index for index in range(DIM)],
        "document_type": "text",
        "metadata": {
            "content_metadata": {"type": "text", "id": f"row-{row_id:02d}", "page_number": row_id},
            "source_metadata": {"source_id": f"/tmp/doc-{row_id % 3}.pdf"},
        },
    }


def _frames(ids: range, batch_size: int = 2) -> list[pd.DataFrame]:
    rows = [_row(row_id) for row_id in ids]
    return [pd.DataFrame(rows[start : start + batch_size]) for start in range(0, len(rows), batch_size)]


def _write(backend: VDB, frames: list[pd.DataFrame]) -> int:
    """Drive the two halves the executor runs: one receipt per Ray block, then one driver commit."""
    with IngestVdbOperator(vdb=backend)._worker_write() as writer:
        receipts = [writer._write_partition(frame) for frame in frames]
        return writer._commit_partitions(receipts)


class _DuckDBCase:
    def __init__(self, tmp_path: Path) -> None:
        self.path = tmp_path / "chunks.duckdb"

    def backend(self, **kwargs: Any) -> DuckDB:
        return DuckDB(path=str(self.path), table_name="chunks", **kwargs)

    def rows(self) -> list[tuple[str, list[float]]]:
        import duckdb

        with duckdb.connect(str(self.path)) as connection:
            return [
                (row_id, list(vector))
                for row_id, vector in connection.sql("SELECT id, vector FROM chunks ORDER BY id").fetchall()
            ]

    def leftovers(self) -> list[Any]:
        return list(Path(f"{self.path}.staging").glob("*"))


class _ElasticsearchCase:
    def __init__(self, tmp_path: Path) -> None:
        self.index = f"nrl-test-{uuid.uuid4().hex}"

    def backend(self, **kwargs: Any) -> Elasticsearch:
        return Elasticsearch(url=ES_URL, index_name=self.index, **kwargs)

    def rows(self) -> list[tuple[str, list[float]]]:
        hits = requests.get(f"{ES_URL}/{self.index}/_search", params={"size": 100}, timeout=30).json()["hits"]["hits"]
        return sorted((hit["_source"]["id"], hit["_source"]["vector"]) for hit in hits)

    def leftovers(self) -> list[Any]:
        indices = requests.get(f"{ES_URL}/{self.index}-*", timeout=30).json()
        return [name for name, index in indices.items() if self.index not in index["aliases"]]

    def close(self) -> None:
        requests.delete(f"{ES_URL}/{self.index}-*", timeout=30)


@pytest.fixture(params=["duckdb", "elasticsearch"])
def case(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "duckdb":
        yield _DuckDBCase(tmp_path)
        return
    if ES_URL is None:
        pytest.skip("NRL_TEST_ELASTICSEARCH_URL is not set")
    es_case = _ElasticsearchCase(tmp_path)
    yield es_case
    es_case.close()


def _expected(ids: range) -> list[tuple[str, list[float]]]:
    return [(f"row-{row_id:02d}", _row(row_id)["embedding"]) for row_id in ids]


def test_overwrite_then_append_then_overwrite(case) -> None:
    assert _write(case.backend(), _frames(range(0, 5))) == 5
    assert case.rows() == _expected(range(0, 5))

    assert _write(case.backend(overwrite=False), _frames(range(5, 8))) == 3
    assert case.rows() == _expected(range(0, 8))

    assert _write(case.backend(), _frames(range(8, 9))) == 1
    assert case.rows() == _expected(range(8, 9))
    assert case.leftovers() == []


def test_append_creates_a_missing_table(case) -> None:
    assert _write(case.backend(overwrite=False), _frames(range(0, 3))) == 3
    assert case.rows() == _expected(range(0, 3))


def test_failed_write_keeps_the_committed_rows_and_discards_its_own(case) -> None:
    _write(case.backend(), _frames(range(0, 3)))

    with pytest.raises(RuntimeError, match="write actor failed"):
        with IngestVdbOperator(vdb=case.backend(overwrite=False))._worker_write() as writer:
            writer._write_partition(_frames(range(3, 5))[0])
            raise RuntimeError("write actor failed")

    assert case.rows() == _expected(range(0, 3))
    assert case.leftovers() == []


def test_elasticsearch_overwrite_keeps_an_index_another_alias_uses(case) -> None:
    if not isinstance(case, _ElasticsearchCase):
        pytest.skip("Elasticsearch alias semantics")
    _write(case.backend(), _frames(range(0, 2)))
    (first,) = requests.get(f"{ES_URL}/_alias/{case.index}", timeout=30).json()
    shared = f"{case.index}-shared"
    requests.put(f"{ES_URL}/{first}/_alias/{shared}", timeout=30).raise_for_status()

    _write(case.backend(), _frames(range(2, 3)))

    assert case.rows() == _expected(range(2, 3))
    assert requests.get(f"{ES_URL}/{shared}/_count", timeout=30).json()["count"] == 2


def test_ray_write_actors_write_and_the_driver_commits_once(case, monkeypatch: pytest.MonkeyPatch) -> None:
    ray = pytest.importorskip("ray")
    from nemo_retriever.graph.executor import RayDataExecutor
    from nemo_retriever.graph.pipeline_graph import Node

    if not ray.is_initialized():
        ray.init(num_cpus=2, include_dashboard=False)
    dataset = ray.data.from_pandas(_frames(range(0, 10), batch_size=10))
    node = Node(IngestVdbOperator(vdb=case.backend(worker_write_rows=3)))

    summary = RayDataExecutor.__new__(RayDataExecutor)._worker_write_ingest(dataset, node)

    assert summary["submitted_records"].tolist() == [10]
    assert case.rows() == _expected(range(0, 10))
    assert case.leftovers() == []
