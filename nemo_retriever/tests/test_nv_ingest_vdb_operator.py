# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from nemo_retriever.common.vdb.adt_vdb import VDB
from nemo_retriever.operators.vdb import IngestVdbOperator, RetrieveVdbOperator
from nemo_retriever.operators import vdb as vdb_operator_module
from nemo_retriever.operators.vdb import PutVdbOperator


class FakeVDB(VDB):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_calls: list[Any] = []
        self.retrieval_calls: list[tuple[Any, dict[str, Any]]] = []
        self.put_calls: list[tuple[Any, dict[str, Any]]] = []

    def create_index(self, **kwargs: Any) -> None:
        return None

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        return None

    def retrieval(self, vectors: list, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.retrieval_calls.append((vectors, kwargs))
        return [
            [
                {
                    "_distance": 0.12,
                    "bbox_xyxy_norm": "[0, 0, 1, 1]",
                    "content_type": "table",
                    "entity": {
                        "text": "retrieved chunk",
                        "source": {"source_id": "doc-a.pdf"},
                        "content_metadata": {"page_number": 1},
                    },
                    "stored_image_uri": "file:///tmp/page.png",
                }
            ]
        ]

    def run(self, records: Any) -> dict[str, Any]:
        self.run_calls.append(records)
        return {"records": records}

    def put(self, records: list, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append((records, dict(kwargs)))
        return {"put": sum(len(b) for b in records)}


def _graph_rows() -> list[dict[str, Any]]:
    return [
        {
            "text": "first chunk",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "path": "/tmp/doc-a.pdf",
            "page_number": 1,
            "metadata": {"content_metadata": {"type": "text"}},
        },
        {
            "text": "second chunk",
            "text_embeddings_1b_v2": {"embedding": [0.2] * 2048},
            "path": "/tmp/doc-a.pdf",
            "page_number": 2,
            "metadata": {"content_metadata": {"type": "text"}},
        },
    ]


def test_process_returns_original_graph_rows_and_delegates_converted_records_to_run() -> None:
    data = _graph_rows()
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)

    assert operator.preprocess(data) is data
    assert operator.process(data) is data
    assert operator.postprocess(data) is data
    assert vdb.run_calls[0][0][0]["document_type"] == "text"
    assert vdb.run_calls[0][0][0]["metadata"]["content"] == "first chunk"
    assert vdb.run_calls[0][0][0]["metadata"]["embedding"] == [0.1] * 2048


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

    operator = IngestVdbOperator(vdb_op="fake", vdb_kwargs={"answer": 42})

    assert constructed_kwargs == {"answer": 42}
    assert operator.process(_graph_rows()) is not None


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
                "metadata": {"page_number": 1},
                "source": "doc-a.pdf",
                "source_id": "doc-a.pdf",
                "path": "doc-a.pdf",
                "page_number": 1,
                "pdf_basename": "doc-a",
                "pdf_page": "doc-a_1",
                "_distance": 0.12,
                "stored_image_uri": "file:///tmp/page.png",
                "content_type": "table",
                "bbox_xyxy_norm": "[0, 0, 1, 1]",
            }
        ]
    ]
    assert vdb.retrieval_calls == [([[0.1, 0.2]], {"collection_name": "docs", "model_name": "embedder", "top_k": 3})]


def test_retrieve_operator_forwards_runtime_query_texts() -> None:
    vdb = FakeVDB()
    operator = RetrieveVdbOperator(
        vdb=vdb,
        vdb_kwargs={"collection_name": "docs", "model_name": "embedder", "hybrid": True, "query_texts": ["stale"]},
    )

    operator.process([[0.1, 0.2]], top_k=3, query_texts=["current"])

    assert vdb.retrieval_calls == [
        (
            [[0.1, 0.2]],
            {
                "collection_name": "docs",
                "model_name": "embedder",
                "hybrid": True,
                "top_k": 3,
                "query_texts": ["current"],
            },
        )
    ]


def test_retrieve_operator_forwards_query_texts_for_hybrid_vdb_instance() -> None:
    vdb = FakeVDB(hybrid=True)
    operator = RetrieveVdbOperator(vdb=vdb)

    operator.process([[0.1, 0.2]], top_k=3, query_texts=["current"])

    assert vdb.retrieval_calls == [([[0.1, 0.2]], {"top_k": 3, "query_texts": ["current"]})]


def test_retrieve_operator_respects_dense_override_for_hybrid_vdb_instance() -> None:
    vdb = FakeVDB(hybrid=True)
    operator = RetrieveVdbOperator(vdb=vdb)

    operator.process([[0.1, 0.2]], top_k=3, hybrid=False, query_texts=["current"])

    assert vdb.retrieval_calls == [([[0.1, 0.2]], {"top_k": 3, "hybrid": False})]


def test_retrieve_operator_does_not_forward_query_texts_for_dense_retrieval() -> None:
    vdb = FakeVDB()
    operator = RetrieveVdbOperator(vdb=vdb, vdb_kwargs={"collection_name": "docs", "model_name": "embedder"})

    operator.process([[0.1, 0.2]], top_k=3, query_texts=["current"])

    assert vdb.retrieval_calls == [([[0.1, 0.2]], {"collection_name": "docs", "model_name": "embedder", "top_k": 3})]


def test_constructor_requires_exactly_one_vdb_source() -> None:
    with pytest.raises(ValueError, match="Either vdb or vdb_op is required"):
        IngestVdbOperator()

    with pytest.raises(ValueError, match="Pass either vdb or vdb_op"):
        IngestVdbOperator(vdb=FakeVDB(), vdb_op="lancedb")


# ──────────────────────────────────────────────────────────────────────────────
# PutVdbOperator
# ──────────────────────────────────────────────────────────────────────────────


class _StubPutVDB(VDB):
    """VDB subclass that intentionally does NOT override ``put``.

    Used to exercise the construction-time guard in
    :class:`PutVdbOperator.__init__`, which compares
    ``type(self._vdb).put is VDB.put`` to detect backends that
    inherit the base-class ``NotImplementedError`` stub.

    Note: this class being instantiable at all is itself a regression
    check — :meth:`VDB.put` must NOT be decorated with
    ``@abstractmethod``; otherwise ABC machinery would reject this class
    before the operator-level guard could run, making the guard dead code.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def create_index(self, **kwargs: Any) -> None:
        return None

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        return None

    def retrieval(self, queries: list, **kwargs: Any) -> list[list[dict[str, Any]]]:
        return []

    def run(self, records: Any) -> None:
        return None


def test_put_operator_rejects_vdb_without_put_override() -> None:
    """Backends inheriting the ``VDB.put`` stub fail fast at construction."""
    stub = _StubPutVDB()
    # Sanity-check the precondition the guard relies on: the subclass really
    # is using the inherited stub, not its own implementation. If this ever
    # fails, the guard's identity comparison would silently never fire.
    assert type(stub).put is VDB.put

    with pytest.raises(NotImplementedError, match=r"does not implement put"):
        PutVdbOperator(vdb=stub)


def test_put_operator_delegates_records_with_configured_key_and_table_name() -> None:
    """Happy path: nv-ingest-converted records reach ``vdb.put`` with the configured key/table."""
    vdb = FakeVDB()
    operator = PutVdbOperator(vdb=vdb, key="entity_id", table_name="entities")

    data = [
        {
            "text": "graph chunk",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "source_id": "/tmp/doc-a.pdf",
            "page_number": 7,
        }
    ]

    assert operator(data) is data

    assert vdb.run_calls == []
    assert len(vdb.put_calls) == 1
    call_records, call_kwargs = vdb.put_calls[0]
    assert call_kwargs == {"table_name": "entities", "key": "entity_id"}
    # The records that reach the backend must already be in nv-ingest-client
    # shape (same conversion IngestVdbOperator performs), not the flat graph rows.
    assert call_records == [
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


def test_put_operator_merges_sidecar_metadata_into_records_before_put() -> None:
    """Sidecar kwargs are split out from ``vdb_kwargs`` and applied before delegation."""
    vdb = FakeVDB()
    meta_df = pd.DataFrame(
        {
            "source_id": ["/tmp/doc-a.pdf"],
            "category": ["legal"],
        }
    )
    operator = PutVdbOperator(
        vdb=vdb,
        vdb_kwargs={
            "meta_dataframe": meta_df,
            "meta_source_field": "source_id",
            "meta_fields": ["category"],
            "meta_join_key": "source_id",
        },
        key="id",
        table_name="my_table",
    )

    data = [
        {
            "text": "graph chunk",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "source_id": "/tmp/doc-a.pdf",
            "page_number": 7,
        }
    ]

    assert operator.process(data) is data

    assert len(vdb.put_calls) == 1
    call_records, call_kwargs = vdb.put_calls[0]
    assert call_kwargs == {"table_name": "my_table", "key": "id"}
    merged_content_meta = call_records[0][0]["metadata"]["content_metadata"]
    # Sidecar column merged in alongside the per-row ``page_number``.
    assert merged_content_meta["category"] == "legal"
    assert merged_content_meta["page_number"] == 7
