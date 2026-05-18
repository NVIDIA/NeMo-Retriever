# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from nemo_retriever.vdb.adt_vdb import VDB
from nemo_retriever.vdb import IngestVdbOperator, RetrieveVdbOperator
from nemo_retriever.vdb import operators as vdb_operator_module


class FakeVDB(VDB):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_calls: list[Any] = []
        self.append_calls: list[tuple[Any, bool]] = []
        self.build_index_calls: int = 0
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

    def append(self, records: Any, *, overwrite: bool) -> bool:
        self.append_calls.append((records, overwrite))
        return True

    def build_index(self) -> None:
        self.build_index_calls += 1


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

    assert vdb.append_calls == [
        (
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
            ],
            True,
        )
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
                "_distance": 0.12,
                "stored_image_uri": "file:///tmp/page.png",
                "content_type": "table",
                "bbox_xyxy_norm": "[0, 0, 1, 1]",
            }
        ]
    ]
    assert vdb.retrieval_calls == [([[0.1, 0.2]], {"collection_name": "docs", "model_name": "embedder", "top_k": 3})]


def test_constructor_requires_exactly_one_vdb_source() -> None:
    with pytest.raises(ValueError, match="Either vdb or vdb_op is required"):
        IngestVdbOperator()

    with pytest.raises(ValueError, match="Pass either vdb or vdb_op"):
        IngestVdbOperator(vdb=FakeVDB(), vdb_op="lancedb")


def test_ingest_operator_streams_with_overwrite_flag_flip() -> None:
    """First batch overwrites the table; subsequent batches append. This handshake
    is the entire streaming contract — without concurrency=1 it'd be impossible to
    keep coherent, and without the flag flip we'd either rewrite per batch or
    never overwrite at all."""
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)

    operator.process(_graph_rows())
    operator.process(_graph_rows())
    operator.process(_graph_rows())

    overwrites = [overwrite for _records, overwrite in vdb.append_calls]
    assert overwrites == [True, False, False]
    assert vdb.run_calls == []  # legacy global-batch path must not be invoked


def test_finalize_vdb_upload_builds_index_when_params_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GraphIngestor`` finalizes a configured VDB upload by constructing the
    backend once and calling ``build_index()`` — replacing the old downstream
    ``VdbBuildIndexOperator`` Ray-Data barrier with a driver-side hook so the
    graph itself can stay fully streaming end-to-end."""
    from nemo_retriever.graph_ingestor import GraphIngestor
    from nemo_retriever.params import VdbUploadParams

    constructed: list[dict[str, Any]] = []
    indexed: list[FakeVDB] = []

    class _CountingVDB(FakeVDB):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            constructed.append(kwargs)

        def build_index(self) -> None:
            super().build_index()
            indexed.append(self)

    monkeypatch.setattr(vdb_operator_module, "get_vdb_op_cls", lambda _name: _CountingVDB)

    ingestor = GraphIngestor(run_mode="inprocess", documents=[])
    ingestor.vdb_upload(VdbUploadParams(vdb_op="fake", vdb_kwargs={"uri": "/tmp/x", "table_name": "t"}))
    ingestor._finalize_vdb_upload()

    assert len(constructed) == 1
    assert constructed[0] == {"uri": "/tmp/x", "table_name": "t"}
    assert len(indexed) == 1 and indexed[0].build_index_calls == 1


def test_dataframe_output_keeps_only_whitelisted_columns_and_marks_uploadable() -> None:
    """When the operator receives a pandas DataFrame (real Ray Data path), the
    returned block must keep ONLY a small whitelist of accounting columns —
    anything else (embeddings, metadata blobs, raw PDF bytes, page images,
    extracted text) accumulates in plasma at the downstream global-batch
    barrier and dominates the driver-side materialization. A whitelist also
    survives future schema additions: any newly-added heavy column gets
    dropped by default."""
    import pandas as pd

    rows = pd.DataFrame(
        [
            # Valid row: text + embedding present.
            {
                "text": "alpha",
                "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
                "source_id": "doc-a.pdf",
                "source_path": "/tmp/doc-a.pdf",
                "path": "/tmp/doc-a.pdf",
                "page_number": 1,
                "metadata": {"content_metadata": {"type": "text"}},
                "bytes": b"x" * 1024,  # heavy column not in any blacklist
                "image_data": "base64..." * 100,  # another heavy column
            },
            # Invalid row: text but no embedding.
            {
                "text": "beta",
                "text_embeddings_1b_v2": None,
                "source_id": "doc-a.pdf",
                "source_path": "/tmp/doc-a.pdf",
                "path": "/tmp/doc-a.pdf",
                "page_number": 2,
                "metadata": {"content_metadata": {"type": "text"}},
                "bytes": b"y" * 1024,
                "image_data": "base64..." * 100,
            },
        ]
    )

    operator = IngestVdbOperator(vdb=FakeVDB())
    out = operator(rows)

    # Only the accounting whitelist + the _vdb_uploadable flag must remain.
    expected = {"source_id", "source_path", "path", "page_number", "_vdb_uploadable"}
    assert set(out.columns) == expected
    # First row uploaded, second row dropped (no embedding).
    assert list(out["_vdb_uploadable"]) == [True, False]
