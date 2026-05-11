# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from nemo_retriever.agent_mcp.evidence import normalize_hit, normalize_hits


def test_normalizes_document_hit_with_json_source_and_metadata() -> None:
    hit = {
        "text": "A policy document excerpt.",
        "_distance": "0.42",
        "source": json.dumps({"source_id": "/data/policy.pdf"}),
        "metadata": json.dumps({"page_number": "7", "content_type": "text"}),
    }

    evidence = normalize_hit(hit)

    assert evidence.text == "A policy document excerpt."
    assert evidence.score == 0.42
    assert evidence.source_path == "/data/policy.pdf"
    assert evidence.media_type == "document"
    assert evidence.content_type == "text"
    assert evidence.locator.page_number == 7


def test_normalizes_video_hit_with_artifact_and_locator_metadata() -> None:
    hit = {
        "text": "Speaker explains the demo.",
        "_rerank_score": 0.91,
        "path": "/data/demo.mp4",
        "content_type": "transcript",
        "stored_image_uri": "file:///artifacts/demo-frame.jpg",
        "metadata": {
            "segment_start_seconds": "12.5",
            "segment_end_seconds": 15,
            "frame_index": "30",
        },
    }

    evidence = normalize_hit(hit)

    assert evidence.score == 0.91
    assert evidence.source_path == "/data/demo.mp4"
    assert evidence.media_type == "video"
    assert evidence.content_type == "transcript"
    assert evidence.locator.timestamp_start_s == 12.5
    assert evidence.locator.timestamp_end_s == 15.0
    assert evidence.locator.frame_index == 30
    assert evidence.artifacts.stored_image_uri == "file:///artifacts/demo-frame.jpg"


def test_normalize_hits_preserves_order() -> None:
    hits = [
        {"text": "first", "source_path": "/data/first.pdf"},
        {"text": "second", "source_path": "/data/second.mp4"},
        {"text": "third", "source_path": "/data/third.png"},
    ]

    evidence = normalize_hits(hits)

    assert [hit.text for hit in evidence] == ["first", "second", "third"]


def test_malformed_numeric_values_normalize_to_none() -> None:
    hit = {
        "text": "bad numeric fields",
        "_rerank_score": "nan",
        "score": "0.9",
        "metadata": {
            "timestamp_start_s": "inf",
            "timestamp_end_s": "1e309",
            "page_number": float("inf"),
            "frame_index": "nan",
        },
    }

    evidence = normalize_hit(hit)

    assert evidence.score is None
    assert evidence.locator.timestamp_start_s is None
    assert evidence.locator.timestamp_end_s is None
    assert evidence.locator.page_number is None
    assert evidence.locator.frame_index is None


def test_string_bbox_normalizes_to_floats() -> None:
    hit = {
        "text": "bbox",
        "metadata": {"bbox_xyxy_norm": "[0, 0.25, \"0.75\", 1]"},
    }

    evidence = normalize_hit(hit)

    assert evidence.locator.bbox_xyxy_norm == [0.0, 0.25, 0.75, 1.0]


def test_malformed_bbox_normalizes_to_none() -> None:
    hit = {"text": "bad bbox", "metadata": {"bbox_xyxy_norm": ["bad"]}}

    evidence = normalize_hit(hit)

    assert evidence.locator.bbox_xyxy_norm is None


def test_wrong_arity_bbox_normalizes_to_none() -> None:
    short_bbox = normalize_hit({"text": "short bbox", "metadata": {"bbox_xyxy_norm": [0, 1]}})
    long_string_bbox = normalize_hit({"text": "long bbox", "metadata": {"bbox_xyxy_norm": "[0, 0.25, 0.75, 1, 2]"}})

    assert short_bbox.locator.bbox_xyxy_norm is None
    assert long_string_bbox.locator.bbox_xyxy_norm is None
