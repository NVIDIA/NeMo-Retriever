# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify the existing ``audio_segment`` recall matcher works for video rows.

For each row type our video pipeline emits (``audio``, ``video_frame``,
``audio_visual``), we synthesize a hit dict in the shape produced by
LanceDB lookups and confirm the matcher accepts it for a GT window from
the video_retrieval ground truth.
"""

from __future__ import annotations

import pandas as pd

from nemo_retriever.recall.core import (
    _hit_to_audio_segment_key,
    _normalize_audio_query_df,
    is_hit_at_k,
)


def _make_hit(source: str, start: float, end: float) -> dict:
    """Mirror the LanceDB hit shape that ``_hit_to_audio_segment_key`` reads."""
    import json

    return {
        "source": json.dumps({"source_id": source}),
        "metadata": json.dumps({"segment_start_seconds": start, "segment_end_seconds": end}),
    }


def test_normalize_video_gt_columns_via_audio_loader() -> None:
    """The eval GT csv has ``name``/``question``/``start_time``/``end_time``;
    the audio loader should map these to the canonical query columns."""
    df = pd.DataFrame(
        [
            {
                "name": "2024_04_SAP_Datasphere_Top_Features_1_36zflqjo_",
                "question": "What is Confluent Kafka used for?",
                "start_time": 29.85,
                "end_time": 41.56,
            }
        ]
    )
    out = _normalize_audio_query_df(df)
    assert "query" in out.columns
    assert "expected_media_id" in out.columns
    assert "golden_answer" in out.columns
    assert out.iloc[0]["query"] == "What is Confluent Kafka used for?"
    assert out.iloc[0]["expected_media_id"] == "2024_04_SAP_Datasphere_Top_Features_1_36zflqjo_"
    assert out.iloc[0]["expected_start_time"] == 29.85
    assert out.iloc[0]["expected_end_time"] == 41.56


def test_audio_frame_and_scene_rows_all_hit_under_audio_segment_mode() -> None:
    """Synthesize one hit per modality whose midpoint falls inside the GT window."""
    media = "video_a"
    gold_start, gold_end = 30.0, 40.0
    gold_key = f"{media}\t{gold_start}\t{gold_end}"

    audio_hit = _make_hit(media, 32.0, 34.0)
    frame_hit = _make_hit(media, 34.5, 35.5)
    scene_hit = _make_hit(media, 33.0, 38.0)

    for hit in (audio_hit, frame_hit, scene_hit):
        encoded = _hit_to_audio_segment_key(hit)
        assert encoded is not None
        assert is_hit_at_k(
            golden_key=gold_key,
            retrieved=[encoded],
            k=1,
            match_mode="audio_segment",
        )


def test_hit_outside_tolerance_is_a_miss() -> None:
    """Frames whose midpoint sits outside ``[gold_start - tol, gold_end + tol]`` should miss."""
    media = "video_a"
    gold_start, gold_end = 30.0, 40.0
    gold_key = f"{media}\t{gold_start}\t{gold_end}"

    far_hit = _make_hit(media, 50.0, 51.0)  # midpoint 50.5 -> outside +tol=42
    encoded = _hit_to_audio_segment_key(far_hit)
    assert not is_hit_at_k(
        golden_key=gold_key,
        retrieved=[encoded],
        k=5,
        match_mode="audio_segment",
        audio_match_tolerance_secs=2.0,
    )


def test_hit_with_different_media_id_is_a_miss() -> None:
    media = "video_a"
    gold_key = f"{media}\t30.0\t40.0"
    other_hit = _make_hit("video_b", 30.0, 40.0)
    encoded = _hit_to_audio_segment_key(other_hit)
    assert not is_hit_at_k(
        golden_key=gold_key,
        retrieved=[encoded],
        k=5,
        match_mode="audio_segment",
    )
