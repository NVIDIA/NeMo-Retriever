# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ingest DataFrame transport round-trip helpers."""

from __future__ import annotations

import pandas as pd

from nemo_retriever.ingestor.results import (
    concat_ingest_results,
    dataframe_from_transport_records,
    dataframe_to_transport_records,
)
from nemo_retriever.service.services.pipeline_executor import _sanitize_result_data


def test_transport_preserves_column_layout_and_strips_bulky_payloads() -> None:
    df = pd.DataFrame(
        {
            "path": ["/a.pdf"],
            "page_number": [1],
            "text": ["hello"],
            "bytes": [b"pdf-bytes"],
            "_content_type": ["text"],
            "_bbox_xyxy_norm": [[0.1, 0.2, 0.3, 0.4]],
            "_stored_image_uri": ["file:///stored/page.png"],
            "page_image": [{"image_b64": "raw-page", "stored_image_uri": "file:///stored/page.png"}],
            "images": [[{"image_b64": "raw-crop", "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4]}]],
            "page_elements_v3": [[{"label": "Table"}]],
            "text_embeddings_1b_v2": [{"embedding": [0.1, 0.2]}],
            "metadata": [{"embedding": [0.3, 0.4], "dpi": 200, "source_path": "/a.pdf"}],
        }
    )
    records = dataframe_to_transport_records(df)

    record = records[0]
    assert set(record) == set(df.columns)
    assert record["path"] == "/a.pdf"
    assert record["page_number"] == 1
    assert record["text"] == "hello"
    assert record["bytes"] == "<bytes len=9>"
    assert record["_stored_image_uri"] == "file:///stored/page.png"
    assert record["_bbox_xyxy_norm"] == [0.1, 0.2, 0.3, 0.4]
    assert record["text_embeddings_1b_v2"] == {"embedding": None}
    assert record["metadata"] == {"embedding": None, "dpi": 200, "source_path": "/a.pdf"}
    assert record["page_image"] == {"image_b64": None, "stored_image_uri": "file:///stored/page.png"}
    assert record["images"][0] == {"image_b64": None, "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4]}


def test_transport_summarizes_long_lists_after_nested_sanitization() -> None:
    rows = [{"image_b64": f"raw-{idx}", "label": "image"} for idx in range(21)]
    df = pd.DataFrame({"path": ["/a.pdf"], "images": [rows]})

    record = dataframe_to_transport_records(df)[0]

    assert record["images"] == "<list len=21>"


def test_transport_compact_document_rows_drop_legacy_metadata_and_bbox() -> None:
    df = pd.DataFrame(
        {
            "path": ["/a.pdf"],
            "page_number": [1],
            "text": ["hello"],
            "_content_type": ["text"],
            "_bbox_xyxy_norm": [[0.1, 0.2, 0.3, 0.4]],
            "_stored_image_uri": ["file:///stored/page.png"],
            "page_image": [{"image_b64": "raw-page", "stored_image_uri": "file:///stored/page.png"}],
            "images": [[{"image_b64": "raw-crop", "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4]}]],
            "metadata": [{"embedding": [0.3, 0.4], "dpi": 200, "source_path": "/a.pdf"}],
        }
    )

    assert dataframe_to_transport_records(df, result_schema="compact") == [
        {
            "text": "hello",
            "source_id": "/a.pdf",
            "element_type": "text",
            "page_number": 1,
            "stored_image_uri": "file:///stored/page.png",
        }
    ]


def test_transport_compact_media_rows_return_timings() -> None:
    df = pd.DataFrame(
        {
            "path": ["/tmp/chunk-000.wav"],
            "source_path": ["/media/call.wav"],
            "page_number": [3],
            "text": ["hello from audio"],
            "_content_type": ["audio"],
            "metadata": [
                {
                    "duration": 30.0,
                    "segment_start_seconds": 10.5,
                    "segment_end_seconds": 12.0,
                    "source_path": "/media/call.wav",
                    "embedding": [0.1, 0.2],
                }
            ],
        }
    )

    assert dataframe_to_transport_records(df, result_schema="compact") == [
        {
            "text": "hello from audio",
            "source_id": "/media/call.wav",
            "element_type": "audio",
            "start_time_seconds": 10.5,
            "end_time_seconds": 12.0,
            "duration_seconds": 1.5,
        }
    ]


def test_round_trip_matches_inprocess_column_layout() -> None:
    df = pd.DataFrame(
        {
            "path": ["/a.pdf", "/a.pdf"],
            "page_number": [1, 2],
            "text": ["a", "b"],
            "metadata": [{"content_metadata": {"type": "text"}}, {"content_metadata": {"type": "text"}}],
        }
    )
    rebuilt = dataframe_from_transport_records(dataframe_to_transport_records(df))
    assert list(rebuilt.columns) == list(df.columns)
    assert len(rebuilt) == len(df)
    assert rebuilt["text"].tolist() == df["text"].tolist()


def test_sanitize_result_data_delegates_to_shared_helper() -> None:
    df = pd.DataFrame({"path": ["/x.pdf"], "text": ["x"]})
    assert _sanitize_result_data(df) == dataframe_to_transport_records(df)


def test_sanitize_result_data_accepts_compact_schema() -> None:
    df = pd.DataFrame({"path": ["/x.pdf"], "text": ["x"]})
    assert _sanitize_result_data(df, result_schema="compact") == dataframe_to_transport_records(
        df,
        result_schema="compact",
    )


def test_concat_ingest_results_follows_document_order() -> None:
    rows_a = [{"path": "/a.pdf", "page_number": 1, "text": "a"}]
    rows_b = [{"path": "/b.pdf", "page_number": 1, "text": "b"}]
    combined = concat_ingest_results(
        {"doc-b": rows_b, "doc-a": rows_a},
        ["doc-a", "doc-b"],
    )
    assert combined["path"].tolist() == ["/a.pdf", "/b.pdf"]
    assert list(combined.columns) == ["path", "page_number", "text"]
