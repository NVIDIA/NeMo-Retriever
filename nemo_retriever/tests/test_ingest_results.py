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


def test_transport_returns_compact_document_rows() -> None:
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

    assert records == [
        {
            "text": "hello",
            "source_id": "/a.pdf",
            "element_type": "text",
            "page_number": 1,
            "stored_image_uri": "file:///stored/page.png",
        }
    ]


def test_transport_returns_audio_timing_fields_without_metadata() -> None:
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

    assert dataframe_to_transport_records(df) == [
        {
            "text": "hello from audio",
            "source_id": "/media/call.wav",
            "element_type": "audio",
            "start_time_seconds": 10.5,
            "end_time_seconds": 12.0,
            "duration_seconds": 1.5,
        }
    ]


def test_transport_returns_video_timing_fields() -> None:
    df = pd.DataFrame(
        {
            "path": ["/media/demo.mp4"],
            "text": ["visible text"],
            "_content_type": ["video_frame"],
            "metadata": [
                {
                    "source_path": "/media/demo.mp4",
                    "frame_timestamp_seconds": 21.0,
                    "segment_start_seconds": 19.0,
                    "segment_end_seconds": 23.0,
                }
            ],
        }
    )

    assert dataframe_to_transport_records(df) == [
        {
            "text": "visible text",
            "source_id": "/media/demo.mp4",
            "element_type": "video_frame",
            "start_time_seconds": 19.0,
            "end_time_seconds": 23.0,
            "duration_seconds": 4.0,
        }
    ]


def test_transport_returns_compact_error_without_traceback() -> None:
    df = pd.DataFrame(
        {
            "path": ["/broken.pdf"],
            "page_number": [2],
            "text": [""],
            "metadata": [
                {
                    "error": {
                        "stage": "extract",
                        "type": "RuntimeError",
                        "message": "boom",
                        "traceback": "large traceback",
                    }
                }
            ],
        }
    )

    assert dataframe_to_transport_records(df) == [
        {
            "text": "",
            "source_id": "/broken.pdf",
            "element_type": "text",
            "page_number": 2,
            "error": {"stage": "extract", "type": "RuntimeError", "message": "boom"},
        }
    ]


def test_round_trip_rebuilds_compact_column_layout() -> None:
    df = pd.DataFrame(
        {
            "path": ["/a.pdf", "/a.pdf"],
            "page_number": [1, 2],
            "text": ["a", "b"],
            "metadata": [{"content_metadata": {"type": "text"}}, {"content_metadata": {"type": "text"}}],
        }
    )
    rebuilt = dataframe_from_transport_records(dataframe_to_transport_records(df))
    assert list(rebuilt.columns) == ["text", "source_id", "element_type", "page_number"]
    assert len(rebuilt) == len(df)
    assert rebuilt["text"].tolist() == df["text"].tolist()


def test_sanitize_result_data_delegates_to_shared_helper() -> None:
    df = pd.DataFrame({"path": ["/x.pdf"], "text": ["x"]})
    assert _sanitize_result_data(df) == dataframe_to_transport_records(df)


def test_concat_ingest_results_follows_document_order() -> None:
    rows_a = [{"text": "a", "source_id": "/a.pdf", "element_type": "text", "page_number": 1}]
    rows_b = [{"text": "b", "source_id": "/b.pdf", "element_type": "text", "page_number": 1}]
    combined = concat_ingest_results(
        {"doc-b": rows_b, "doc-a": rows_a},
        ["doc-a", "doc-b"],
    )
    assert combined["source_id"].tolist() == ["/a.pdf", "/b.pdf"]
    assert list(combined.columns) == ["text", "source_id", "element_type", "page_number"]
