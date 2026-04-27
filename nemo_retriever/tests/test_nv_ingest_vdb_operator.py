# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from nv_ingest_client.util.vdb import VDB
from nemo_retriever.vdb import IngestVdbOperator, RetrieveVdbOperator
from nemo_retriever.vdb import operators as vdb_operator_module


class FakeVDB(VDB):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_calls: list[Any] = []
        self.retrieval_calls: list[tuple[Any, dict[str, Any]]] = []

    def create_index(self, **kwargs: Any) -> None:
        return None

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        return None

    def retrieval(self, vectors: list, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.retrieval_calls.append((vectors, kwargs))
        return [
            [
                {
                    "entity": {
                        "text": "retrieved chunk",
                        "source": {"source_id": "doc-a.pdf"},
                        "content_metadata": {"page_number": 1},
                    }
                }
            ]
        ]

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
    operator = IngestVdbOperator(vdb=vdb)

    assert operator.preprocess(data) is data
    assert operator.process(data) is data
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

    data = [[{"metadata": {"embedding": [0.1]}, "document_type": "text"}]]
    operator = IngestVdbOperator(vdb_op="fake", vdb_kwargs={"answer": 42})

    assert constructed_kwargs == {"answer": 42}
    assert operator.process(data) is data


def test_ingest_operator_converts_graph_rows_to_client_vdb_records() -> None:
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)
    data = [
        {
            "text": "graph chunk",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "source_id": "/tmp/doc-a.pdf",
            "page_number": 7,
        }
    ]

    assert operator(data) is data

    assert vdb.run_calls == [
        [
            [
                {
                    "document_type": "text",
                    "metadata": {
                        "embedding": [0.1] * 2048,
                        "content": "graph chunk",
                        "content_metadata": {"page_number": 7},
                        "source_metadata": {
                            "source_id": "/tmp/doc-a.pdf",
                            "source_name": "doc-a.pdf",
                        },
                    },
                }
            ]
        ]
    ]


def test_retrieve_operator_delegates_vectors_to_retrieval() -> None:
    vdb = FakeVDB()
    operator = RetrieveVdbOperator(vdb=vdb, vdb_kwargs={"collection_name": "docs", "model_name": "embedder"})

    result = operator.process([[0.1, 0.2]], top_k=3)

    assert result == [
        [
            {
                "text": "retrieved chunk",
                "metadata": '{"page_number": 1}',
                "source": "doc-a.pdf",
                "source_id": "doc-a.pdf",
                "path": "doc-a.pdf",
                "page_number": 1,
                "pdf_basename": "doc-a",
                "pdf_page": "doc-a_1",
            }
        ]
    ]
    assert vdb.retrieval_calls == [
        ([[0.1, 0.2]], {"collection_name": "docs", "model_name": "embedder", "top_k": 3})
    ]


def test_constructor_requires_exactly_one_vdb_source() -> None:
    with pytest.raises(ValueError, match="Either vdb or vdb_op is required"):
        IngestVdbOperator()

    with pytest.raises(ValueError, match="Pass either vdb or vdb_op"):
        IngestVdbOperator(vdb=FakeVDB(), vdb_op="lancedb")


def test_lancedb_vdb_writes_records_through_operator(tmp_path) -> None:
    lancedb = pytest.importorskip("lancedb")
    from nv_ingest_client.util.vdb.lancedb import LanceDB

    table_name = "nv_ingest_operator_test"
    records = _client_vdb_records()

    vdb = LanceDB(uri=str(tmp_path), table_name=table_name, num_partitions=1)
    operator = IngestVdbOperator(vdb=vdb)

    assert operator(records) is records

    table = lancedb.connect(str(tmp_path)).open_table(table_name)
    assert table.count_rows() == 2

    hits = vdb.retrieval([[0.1] * 2048], top_k=1, result_fields=["text", "metadata", "source"])
    assert len(hits) == 1
    assert len(hits[0]) == 1
    assert hits[0][0]["text"] == "first chunk"


@pytest.mark.integration
def test_milvus_vdb_writes_records_through_operator_with_milvus_lite(tmp_path) -> None:
    pymilvus = pytest.importorskip("pymilvus")

    collection_name = "nv_ingest_operator_test"
    milvus_uri = str(tmp_path / "milvus.db")
    records = _client_vdb_records()
    operator = IngestVdbOperator(
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

    assert operator(records) is records

    client = pymilvus.MilvusClient(milvus_uri)
    assert client.get_collection_stats(collection_name)["row_count"] == 2
