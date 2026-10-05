# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from nemo_retriever.common.vdb.elasticsearch import Elasticsearch
from nemo_retriever.common.vdb.factory import get_vdb_op_cls


class _FakeXpack:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def usage(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "gpu_vector_indexing": {
                "available": self.enabled,
                "enabled": self.enabled,
                "nodes_with_gpu": 1 if self.enabled else 0,
            }
        }


class _FakeIndices:
    def __init__(self) -> None:
        self.exists_value = False
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.refreshed: list[str] = []

    def exists(self, *, index: str) -> bool:
        return self.exists_value

    def create(self, *, index: str, **kwargs: Any) -> dict[str, bool]:
        self.created.append({"index": index, **kwargs})
        self.exists_value = True
        return {"acknowledged": True}

    def delete(self, *, index: str) -> dict[str, bool]:
        self.deleted.append(index)
        self.exists_value = False
        return {"acknowledged": True}

    def refresh(self, *, index: str) -> None:
        self.refreshed.append(index)

    def get_mapping(self, *, index: str) -> dict[str, Any]:
        created = self.created[-1]
        return {index: {"mappings": created["mappings"]}}


class _FakeClient:
    def __init__(self, *, gpu_enabled: bool = True) -> None:
        self.xpack = _FakeXpack(enabled=gpu_enabled)
        self.indices = _FakeIndices()
        self.searches: list[dict[str, Any]] = []

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.searches.append({"index": index, "body": body})
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "doc-1",
                        "_score": 0.75,
                        "_source": {"text": "matching text", "metadata": {"url": "https://example.test"}},
                    }
                ]
            }
        }


def _bulk_helper(client: Any, actions: Any, **kwargs: Any):
    for action in actions:
        yield True, {"index": {"_id": action.get("_id")}}


def test_default_mapping_uses_requested_int8_hnsw_parameters() -> None:
    client = _FakeClient()
    backend = Elasticsearch(client=client, vector_dim=2048)

    backend.create_index()

    vector = client.indices.created[0]["mappings"]["properties"]["vector"]
    assert vector["element_type"] == "float"
    assert vector["index_options"] == {
        "type": "int8_hnsw",
        "m": 64,
        "ef_construction": 2000,
    }
    assert client.indices.created[0]["mappings"]["_meta"]["num_candidates"] == 10_000


def test_retrieval_uses_ten_thousand_candidates_and_excludes_vectors() -> None:
    client = _FakeClient()
    backend = Elasticsearch(client=client, vector_dim=2)

    results = backend.retrieval([[0.1, 0.2]], top_k=10)

    request = client.searches[0]["body"]
    assert request["knn"] == {
        "field": "vector",
        "query_vector": [0.1, 0.2],
        "k": 10,
        "num_candidates": 10_000,
    }
    assert request["_source"]["excludes"] == ["vector"]
    assert results == [
        [
            {
                "id": "doc-1",
                "text": "matching text",
                "metadata": {"url": "https://example.test"},
                "_score": 0.75,
            }
        ]
    ]


def test_concurrent_retrieval_preserves_query_order() -> None:
    class EchoClient(_FakeClient):
        def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
            vector = body["knn"]["query_vector"]
            return {"hits": {"hits": [{"_id": str(vector[0]), "_source": {"text": "hit"}}]}}

    backend = Elasticsearch(client=EchoClient(), vector_dim=2)
    results = backend.retrieval(
        [[0.1, 0.2], [0.3, 0.4]],
        top_k=1,
        retrieval_workers=2,
    )

    assert [hits[0]["id"] for hits in results] == ["0.1", "0.3"]


def test_retrieval_rejects_nonpositive_worker_count() -> None:
    backend = Elasticsearch(client=_FakeClient(), vector_dim=2)

    with pytest.raises(ValueError, match="retrieval_workers must be positive"):
        backend.retrieval([[0.1, 0.2]], retrieval_workers=0)


def test_stream_ingest_converts_nrl_records_to_bulk_actions() -> None:
    client = _FakeClient()
    captured: list[dict[str, Any]] = []

    def capture_bulk(client: Any, actions: Any, **kwargs: Any):
        for action in actions:
            captured.append(action)
            yield True, {"index": {"_id": action.get("_id")}}

    backend = Elasticsearch(client=client, bulk_helper=capture_bulk, vector_dim=2)
    backend.stream_ingest(
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.1, 0.2],
                    "content": "hello",
                    "content_metadata": {"id": "doc-1", "url": "https://example.test"},
                    "source_metadata": {"source_name": "source.parquet"},
                },
            }
        ]
    )

    assert captured == [
        {
            "_index": "nemo-retriever",
            "_id": "doc-1",
            "_source": {
                "vector": "PczMzT5MzM0=",
                "text": "hello",
                "metadata": {"id": "doc-1", "url": "https://example.test"},
                "source": {"source_name": "source.parquet"},
                "id": "doc-1",
            },
        }
    ]
    assert client.indices.refreshed == ["nemo-retriever"]


def test_create_index_fails_closed_when_cuvs_gpu_is_unavailable() -> None:
    backend = Elasticsearch(client=_FakeClient(gpu_enabled=False))

    with pytest.raises(RuntimeError, match="cuVS indexing is required"):
        backend.create_index()


def test_parallel_bulk_omits_streaming_only_retry_settings() -> None:
    client = _FakeClient()
    captured_kwargs: dict[str, Any] = {}

    def capture_bulk(client: Any, actions: Any, **kwargs: Any):
        captured_kwargs.update(kwargs)
        for action in actions:
            yield True, {"index": {"_id": action.get("_id")}}

    backend = Elasticsearch(
        client=client,
        bulk_helper=capture_bulk,
        vector_dim=2,
        request_timeout=600,
        bulk_max_retries=8,
        bulk_initial_backoff=2,
        bulk_max_backoff=600,
    )
    assert backend._streaming_bulk([{"_index": "nemo-retriever", "_id": "doc-1", "_source": {}}]) == 1

    assert captured_kwargs["request_timeout"] == 600
    assert "max_retries" not in captured_kwargs
    assert "initial_backoff" not in captured_kwargs
    assert "max_backoff" not in captured_kwargs
    assert captured_kwargs["thread_count"] == 16
    assert captured_kwargs["queue_size"] == 16


def test_serial_bulk_forwards_retry_settings() -> None:
    client = _FakeClient()
    captured_kwargs: dict[str, Any] = {}

    def capture_bulk(client: Any, actions: Any, **kwargs: Any):
        captured_kwargs.update(kwargs)
        for action in actions:
            yield True, {"index": {"_id": action.get("_id")}}

    backend = Elasticsearch(
        client=client,
        bulk_helper=capture_bulk,
        vector_dim=2,
        bulk_workers=1,
        bulk_max_retries=8,
        bulk_initial_backoff=2,
        bulk_max_backoff=600,
    )
    assert backend._streaming_bulk([{"_index": "nemo-retriever", "_id": "doc-1", "_source": {}}]) == 1

    assert captured_kwargs["max_retries"] == 8
    assert captured_kwargs["initial_backoff"] == 2
    assert captured_kwargs["max_backoff"] == 600


def test_stream_failure_can_preserve_partial_index_for_resume() -> None:
    client = _FakeClient()

    def failing_bulk(client: Any, actions: Any, **kwargs: Any):
        for _action in actions:
            raise TimeoutError("transient bulk timeout")
            yield

    backend = Elasticsearch(
        client=client,
        bulk_helper=failing_bulk,
        vector_dim=2,
        delete_index_on_failure=False,
    )

    with pytest.raises(TimeoutError, match="transient bulk timeout"):
        backend.stream_ingest(
            [
                {
                    "document_type": "text",
                    "metadata": {
                        "embedding": [0.1, 0.2],
                        "content": "hello",
                        "content_metadata": {"id": "doc-1"},
                        "source_metadata": {},
                    },
                }
            ]
        )

    assert client.indices.exists_value is True
    assert client.indices.deleted == []


def test_prepared_append_uses_deterministic_ids() -> None:
    pa = pytest.importorskip("pyarrow")
    client = _FakeClient()
    client.indices.exists_value = True
    captured: list[dict[str, Any]] = []

    def capture_bulk(client: Any, actions: Any, **kwargs: Any):
        for action in actions:
            captured.append(action)
            yield True, {"index": {"_id": action.get("_id")}}

    backend = Elasticsearch(
        client=client,
        bulk_helper=capture_bulk,
        vector_dim=2,
        overwrite=False,
        input_format="embedding_parquet",
        use_auto_ids=False,
    )
    table = pa.table(
        {
            "id": ["doc-1"],
            "text": ["hello"],
            "vector": pa.array([[0.1, 0.2]], type=pa.list_(pa.float32(), 2)),
        }
    )

    backend.stream_ingest_prepared_batches([backend.prepare_embedding_parquet_batch(table)])

    assert backend.supports_prepared_stream_ingest is True
    assert captured[0]["_id"] == "doc-1"
    assert client.indices.created == []


def test_factory_resolves_elasticsearch_backend() -> None:
    assert get_vdb_op_cls("elasticsearch") is Elasticsearch
