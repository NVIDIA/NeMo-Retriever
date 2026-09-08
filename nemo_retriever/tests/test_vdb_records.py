# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from nemo_retriever.common.schemas.collections import QueryHit
from nemo_retriever.common.vdb.records import (
    VdbUploadError,
    normalize_retrieval_results,
    to_client_vdb_records,
)
from pydantic import ValidationError
from ray.data.extensions import TensorArray


def _normalize_one(hit: dict) -> dict:
    return normalize_retrieval_results([[hit]])[0][0]


def test_legacy_entity_is_flattened_once_with_top_level_precedence() -> None:
    hit = _normalize_one(
        {
            "entity": {
                "text": "nested text",
                "source": {"source_id": "nested.pdf"},
                "content_metadata": {"page_number": 2},
                "chunk_id": "nested-chunk",
                "document_id": "nested-document",
            },
            "text": "flat text",
            "source": {"source_id": "flat.pdf"},
            "content_metadata": {"page_number": 3},
            "chunk_id": "flat-chunk",
            "document_id": "flat-document",
            "filename": "flat.pdf",
        }
    )

    assert hit["text"] == "flat text"
    assert hit["source_id"] == "flat.pdf"
    assert hit["page_number"] == 3
    assert hit["chunk_id"] == "flat-chunk"
    assert hit["document_id"] == "flat-document"
    assert "entity" not in hit


def test_legacy_entity_accepts_only_pre_collection_fields() -> None:
    hit = _normalize_one(
        {
            "entity": {
                "text": "legacy text",
                "source": {"source_id": "legacy.pdf"},
                "content_metadata": {"page_number": 4},
                "chunk_id": "nested-chunk",
                "document_id": "nested-document",
                "document_version": "nested-version",
            }
        }
    )

    assert hit["text"] == "legacy text"
    assert hit["source_id"] == "legacy.pdf"
    assert hit["page_number"] == 4
    assert "chunk_id" not in hit
    assert "document_id" not in hit
    assert "document_version" not in hit


def test_flat_hit_is_canonicalized_without_entity() -> None:
    hit = _normalize_one(
        {
            "text": "flat",
            "source": {"source_id": "flat.pdf"},
            "content_metadata": {"page_number": "5"},
        }
    )

    assert hit["text"] == "flat"
    assert hit["source_id"] == "flat.pdf"
    assert hit["page_number"] == 5
    assert hit["pdf_page"] == "flat_5"


@pytest.mark.parametrize("content_type", ["audio", "video", "video_frame"])
def test_legacy_media_page_values_remain_unchanged(content_type: str) -> None:
    metadata = {
        "type": content_type,
        "page_number": 3,
        "chunk_index": 3,
        "frame_timestamp_seconds": 9.0,
    }

    hit = _normalize_one({"text": "media", "content_metadata": metadata})

    assert hit["page_number"] == 3
    assert hit["metadata"] == metadata


def test_query_hit_validates_canonical_page_instead_of_repairing_it() -> None:
    payload = {
        "chunk_id": "chunk",
        "document_id": "document",
        "text": "text",
        "distance": 0.2,
        "filename": "document.pdf",
    }

    assert QueryHit(**payload, page_number=None).page_number is None
    with pytest.raises(ValidationError):
        QueryHit(**payload, page_number=0)


def test_canonical_record_batches_pass_through_without_reconversion() -> None:
    records = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.1, 0.2],
                    "content": "already canonical",
                },
            }
        ]
    ]

    assert to_client_vdb_records(records) is records


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param([[]], [], id="single-empty-batch"),
        pytest.param([[], []], [], id="multiple-empty-batches"),
        pytest.param(
            [[], [{"metadata": {"content": "canonical", "embedding": [0.1, 0.2]}}], []],
            [[{"metadata": {"content": "canonical", "embedding": [0.1, 0.2]}}]],
            id="mixed",
        ),
    ],
)
def test_canonical_record_batches_remove_empty_inner_batches(
    records: list[list[dict]],
    expected: list[list[dict]],
) -> None:
    assert to_client_vdb_records(records) == expected


def test_graph_record_conversion_preserves_service_provenance() -> None:
    records = to_client_vdb_records(
        [
            {
                "text": "table content",
                "text_embeddings_1b_v2": {"embedding": [0.1, 0.2]},
                "path": "/tmp/source.pdf",
                "page_number": 1,
                "_page_number": 7,
                "_content_type": "table_caption",
                "_stored_image_uri": "s3://artifacts/table.png",
                "_bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
                "page_elements_v3_num_detections": 3,
                "page_elements_v3_counts_by_label": {"table": 2, "chart": 1},
                "table": [{"content": "a"}, {"content": "b"}],
                "metadata": {
                    "chunk_index": 4,
                    "chunk_count": 9,
                    "embedding_split": {
                        "content": "table content",
                        "parent_id": "parent-1",
                        "chunk_id": "child-2",
                        "chunk_index": 2,
                        "chunk_count": 3,
                        "start_token": 200,
                        "end_token": 400,
                    },
                    "segment_start_seconds": 1.5,
                    "frame_timestamp_seconds": 2.5,
                    "content_metadata": {"page_number": 1},
                },
            }
        ]
    )

    metadata = records[0][0]["metadata"]
    assert metadata["embedding"] == [0.1, 0.2]
    assert metadata["content"] == "table content"
    assert metadata["source_metadata"] == {
        "source_id": "/tmp/source.pdf",
        "source_name": "source.pdf",
    }
    assert metadata["embedding_split"] == {
        "content": "table content",
        "parent_id": "parent-1",
        "chunk_id": "child-2",
        "chunk_index": 2,
        "chunk_count": 3,
        "start_token": 200,
        "end_token": 400,
    }
    assert metadata["content_metadata"] == {
        "page_number": 7,
        "type": "table",
        "fidelity": "ocr",
        "stored_image_uri": "s3://artifacts/table.png",
        "uploaded_image_uri": "s3://artifacts/table.png",
        "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
        "page_elements_v3_num_detections": 3,
        "page_elements_v3_counts_by_label": {"table": 2, "chart": 1},
        "ocr_table_detections": 2,
        "chunk_index": 4,
        "chunk_count": 9,
        "segment_start_seconds": 1.5,
        "frame_timestamp_seconds": 2.5,
    }


@pytest.mark.parametrize("content_type", ["table", "chart_caption", "infographic"])
def test_graph_record_conversion_does_not_publish_inherited_page_uri(content_type: str) -> None:
    records = to_client_vdb_records(
        [
            {
                "text": "structured content",
                "text_embeddings_1b_v2": {"embedding": [0.1, 0.2]},
                "_content_type": content_type,
                "_stored_image_uri": "s3://artifacts/page.png",
                "page_image": {"stored_image_uri": "s3://artifacts/page.png"},
            }
        ]
    )

    content_metadata = records[0][0]["metadata"]["content_metadata"]
    assert content_metadata["stored_image_uri"] == "s3://artifacts/page.png"
    assert "uploaded_image_uri" not in content_metadata


def test_graph_record_conversion_normalizes_arrow_backed_bbox_array() -> None:
    records = to_client_vdb_records(
        [
            {
                "text": "table content",
                "text_embeddings_1b_v2": {"embedding": [0.1, 0.2]},
                "_content_type": "table",
                "_bbox_xyxy_norm": np.array([0.1, 0.2, 0.8, 0.9]),
            }
        ]
    )

    assert records[0][0]["metadata"]["content_metadata"]["bbox_xyxy_norm"] == [0.1, 0.2, 0.8, 0.9]


def test_graph_record_conversion_normalizes_arrow_backed_embedding_array() -> None:
    records = to_client_vdb_records(
        [
            {
                "text": "embedded content",
                "metadata": {"embedding": np.array([0.1, 0.2])},
            }
        ]
    )

    assert records[0][0]["metadata"]["embedding"] == [0.1, 0.2]


def test_graph_record_conversion_normalizes_ray_tensor_embedding() -> None:
    embedding = TensorArray([np.array([0.1, 0.2])])[0]
    records = to_client_vdb_records([{"text": "embedded content", "metadata": {"embedding": embedding}}])

    assert records[0][0]["metadata"]["embedding"] == [0.1, 0.2]


def test_dense_record_conversion_rejects_partial_embedding_coverage() -> None:
    rows = [
        {"text": "embedded", "metadata": {"embedding": [0.1, 0.2]}},
        {
            "text": "failed",
            "metadata": {"embedding": None},
            "text_embeddings_1b_v2": {
                "embedding": None,
                "error": {"stage": "embed", "type": "ValueError", "message": "invalid input"},
            },
        },
    ]

    with pytest.raises(
        VdbUploadError,
        match=r"refusing a partial write.*input rows=2.*uploadable rows=1.*missing embedding=1",
    ):
        to_client_vdb_records(rows)


def test_dense_record_conversion_rejects_runtime_failure_empty_vectors() -> None:
    rows = [
        {
            "text": "failed before",
            "text_embeddings_1b_v2": {
                "embedding": [],
                "error": "RuntimeError: embedding batch failed; inspect embed-stage logs for the cause",
            },
        },
        {
            "text": "failed after",
            "text_embeddings_1b_v2": {
                "embedding": [],
                "error": "RuntimeError: embedding batch failed; inspect embed-stage logs for the cause",
            },
        },
    ]

    with pytest.raises(
        VdbUploadError,
        match=r"none were uploadable",
    ):
        to_client_vdb_records(rows)


def test_dense_record_conversion_rejects_partial_canonical_embedding_coverage() -> None:
    records = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "content": "embedded child",
                    "embedding": [0.1, 0.2],
                    "embedding_split": {"chunk_id": "child-0", "content": "embedded child"},
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "content": "missing child",
                    "embedding": [],
                    "embedding_split": {"chunk_id": "child-1", "content": "missing child"},
                },
            },
        ]
    ]

    with pytest.raises(VdbUploadError, match=r"canonical records.*missing embeddings"):
        to_client_vdb_records(records)


def test_dense_record_conversion_preserves_whitespace_only_embedding_child() -> None:
    rows = [
        {"text": "embedded", "metadata": {"embedding": [0.1, 0.2]}},
        {
            "text": "  ",
            "metadata": {
                "embedding": [0.3, 0.4],
                "embedding_split": {
                    "content": "  ",
                    "parent_id": "parent",
                    "chunk_id": "child-1",
                    "chunk_index": 1,
                    "chunk_count": 2,
                    "start_token": 8187,
                    "end_token": 8188,
                },
            },
        },
    ]

    records = to_client_vdb_records(rows)

    assert len(records[0]) == 2
    assert records[0][1]["metadata"]["content"] == "  "


def test_dense_record_conversion_rejects_missing_whitespace_only_child_embedding() -> None:
    rows = [
        {"text": "embedded", "metadata": {"embedding": [0.1, 0.2]}},
        {
            "text": "  ",
            "metadata": {
                "embedding": None,
                "embedding_split": {
                    "content": "  ",
                    "parent_id": "parent",
                    "chunk_id": "child-1",
                    "chunk_index": 1,
                    "chunk_count": 2,
                    "start_token": 8187,
                    "end_token": 8188,
                },
            },
        },
    ]

    with pytest.raises(
        VdbUploadError,
        match=r"refusing a partial write.*input rows=2.*uploadable rows=1.*missing embedding=1",
    ):
        to_client_vdb_records(rows)


def test_canonical_record_rejects_missing_whitespace_only_child_embedding() -> None:
    records = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "content": "  ",
                    "embedding_split": {
                        "content": "  ",
                        "parent_id": "parent",
                        "chunk_id": "child-1",
                        "chunk_index": 1,
                        "chunk_count": 2,
                        "start_token": 8187,
                        "end_token": 8188,
                    },
                },
            }
        ]
    ]

    with pytest.raises(VdbUploadError, match=r"canonical records.*missing embeddings"):
        to_client_vdb_records(records)


def test_dense_record_conversion_ignores_inherited_page_uri_without_searchable_content() -> None:
    rows = [
        {"text": "embedded", "metadata": {"embedding": [0.1, 0.2]}},
        {
            "_content_type": "table_caption",
            "_stored_image_uri": "s3://artifacts/page.png",
            "page_image": {"stored_image_uri": "s3://artifacts/page.png"},
            "metadata": {},
        },
    ]

    records = to_client_vdb_records(rows)

    assert len(records) == 1
    assert len(records[0]) == 1
    assert records[0][0]["metadata"]["content"] == "embedded"


def test_narrow_lancedb_hit_promotes_canonical_multimodal_metadata() -> None:
    hit = _normalize_one(
        {
            "text": "table content",
            "metadata": {
                "page_number": 7,
                "type": "table_caption",
                "stored_image_uri": "s3://artifacts/table.png",
                "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
            },
            "source": {"source_id": "/tmp/source.pdf"},
        }
    )

    assert hit["content_type"] == "table"
    assert hit["stored_image_uri"] == "s3://artifacts/table.png"
    assert hit["bbox_xyxy_norm"] == [0.1, 0.2, 0.8, 0.9]
    assert hit["page_number"] == 7
    assert hit["source_id"] == "/tmp/source.pdf"
