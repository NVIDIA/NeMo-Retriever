# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from nv_ingest_client.util.vdb import VDB
from nemo_retriever.graph import nv_ingest_vdb_operator as vdb_operator_module
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.graph.nv_ingest_vdb_operator import NvIngestVdbOperator


class FakeVDB(VDB):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_calls: list[Any] = []
        self.retrieval_calls: list[tuple[Any, dict[str, Any]]] = []

    def create_index(self, **kwargs: Any) -> None:
        return None

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        return None

    def retrieval(self, queries: list, **kwargs: Any) -> dict[str, Any]:
        self.retrieval_calls.append((queries, kwargs))
        return {"queries": queries, "kwargs": kwargs}

    def run(self, records: Any) -> dict[str, Any]:
        self.run_calls.append(records)
        return {"records": records}


def _client_vdb_records() -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.1] * 2048,
                    "content": "first chunk",
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_id": "doc-a", "source_name": "doc-a.pdf"},
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.2] * 2048,
                    "content": "second chunk",
                    "content_metadata": {"page_number": 2},
                    "source_metadata": {"source_id": "doc-a", "source_name": "doc-a.pdf"},
                },
            },
        ]
    ]


def test_prebuilt_vdb_passthrough_and_process_delegates_to_run() -> None:
    data = [[{"metadata": {"embedding": [0.1]}, "document_type": "text"}]]
    vdb = FakeVDB()
    operator = NvIngestVdbOperator(vdb=vdb)

    assert operator.preprocess(data) is data
    assert operator.process(data) == {"records": data}
    assert operator.postprocess(data) is data
    assert vdb.run_calls == [data]


def test_vdb_op_constructs_client_vdb(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed_kwargs: dict[str, Any] = {}

    class ConstructedFakeVDB(FakeVDB):
        def __init__(self, **kwargs: Any) -> None:
            constructed_kwargs.update(kwargs)
            super().__init__(**kwargs)

    def fake_get_vdb_op_cls(vdb_op: str) -> type[ConstructedFakeVDB]:
        assert vdb_op == "fake"
        return ConstructedFakeVDB

    monkeypatch.setattr(vdb_operator_module, "get_vdb_op_cls", fake_get_vdb_op_cls)

    operator = NvIngestVdbOperator(vdb_op="fake", vdb_kwargs={"answer": 42})

    assert constructed_kwargs == {"answer": 42}
    assert operator.process(["records"]) == {"records": ["records"]}


def test_retrieve_phase_delegates_to_retrieval() -> None:
    vdb = FakeVDB()
    operator = NvIngestVdbOperator(vdb=vdb, phase="retrieve")

    result = operator.process(["query"], top_k=3)

    assert result == {"queries": ["query"], "kwargs": {"top_k": 3}}
    assert vdb.retrieval_calls == [(["query"], {"top_k": 3})]


def test_constructor_requires_exactly_one_vdb_source() -> None:
    with pytest.raises(ValueError, match="Either vdb or vdb_op is required"):
        NvIngestVdbOperator()

    with pytest.raises(ValueError, match="Pass either vdb or vdb_op"):
        NvIngestVdbOperator(vdb=FakeVDB(), vdb_op="lancedb")


def test_lancedb_vdb_writes_records_through_operator(tmp_path) -> None:
    lancedb = pytest.importorskip("lancedb")
    from nv_ingest_client.util.vdb.lancedb import LanceDB

    table_name = "nv_ingest_operator_test"
    records = _client_vdb_records()

    vdb = LanceDB(uri=str(tmp_path), table_name=table_name, num_partitions=1)
    operator = NvIngestVdbOperator(vdb=vdb)

    assert operator(records) is records

    table = lancedb.connect(str(tmp_path)).open_table(table_name)
    assert table.count_rows() == 2


@pytest.mark.integration
def test_milvus_vdb_writes_records_through_graph_with_milvus_lite(tmp_path) -> None:
    pymilvus = pytest.importorskip("pymilvus")

    collection_name = "nv_ingest_graph_operator_test"
    milvus_uri = str(tmp_path / "milvus.db")
    records = _client_vdb_records()
    graph = Graph() >> NvIngestVdbOperator(
        vdb_op="milvus",
        vdb_kwargs={
            "collection_name": collection_name,
            "milvus_uri": milvus_uri,
            "sparse": False,
            "gpu_index": False,
            "gpu_search": False,
            "stream": True,
            "threshold": 1000,
            "dense_dim": 2048,
        },
    )

    assert graph.execute(records) == [records]

    client = pymilvus.MilvusClient(milvus_uri)
    assert client.get_collection_stats(collection_name)["row_count"] == 2


def test_lancedb_vdb_writes_records_through_graph(tmp_path) -> None:
    lancedb = pytest.importorskip("lancedb")

    table_name = "nv_ingest_graph_operator_test"
    records = _client_vdb_records()
    graph = Graph() >> NvIngestVdbOperator(
        vdb_op="lancedb",
        vdb_kwargs={"uri": str(tmp_path), "table_name": table_name, "num_partitions": 1},
    )

    assert graph.execute(records) == [records]

    table = lancedb.connect(str(tmp_path)).open_table(table_name)
    assert table.count_rows() == 2
