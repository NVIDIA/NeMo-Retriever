# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.video.audio_visual_fuser.AudioVisualFuser."""

from __future__ import annotations

import pandas as pd

from nemo_retriever.params import AudioVisualFuseParams
from nemo_retriever.video.audio_visual_fuser import AudioVisualFuser


def _audio_row(source: str, text: str, start: float, end: float) -> dict:
    return {
        "source_path": source,
        "text": text,
        "metadata": {
            "segment_start_seconds": start,
            "segment_end_seconds": end,
            "_content_type": "audio",
        },
    }


def _frame_row(source: str, text: str, ts: float) -> dict:
    return {
        "source_path": source,
        "text": text,
        "metadata": {
            "segment_start_seconds": max(0.0, ts - 0.5),
            "segment_end_seconds": ts + 0.5,
            "frame_timestamp_seconds": ts,
            "_content_type": "video_frame",
        },
    }


def test_fuser_emits_one_fused_row_per_overlapping_utterance() -> None:
    rows = [
        _audio_row("/v.mp4", "hello world", 0.0, 3.0),
        _audio_row("/v.mp4", "second utt", 5.0, 8.0),
        _frame_row("/v.mp4", "TITLE SLIDE", 1.0),
        _frame_row("/v.mp4", "next slide", 2.0),
        _frame_row("/v.mp4", "orphan", 4.0),  # no overlapping audio
    ]
    df = pd.DataFrame(rows)
    out = AudioVisualFuser(AudioVisualFuseParams()).run(df)
    fused = [r for _, r in out.iterrows() if r["metadata"]["_content_type"] == "audio_visual"]
    assert len(fused) == 1
    text = fused[0]["text"]
    assert "[AUDIO] hello world" in text
    assert "[VISUAL] TITLE SLIDE | next slide" in text
    # Top-level _content_type is set so the LanceDB sink picks it up.
    assert fused[0]["_content_type"] == "audio_visual"
    # Original rows preserved (additive).
    assert len(out) == len(df) + 1


def test_fuser_disabled_returns_input_unchanged() -> None:
    df = pd.DataFrame([_audio_row("/v.mp4", "x", 0.0, 1.0), _frame_row("/v.mp4", "y", 0.5)])
    out = AudioVisualFuser(AudioVisualFuseParams(enabled=False)).run(df)
    pd.testing.assert_frame_equal(out, df)


def test_fuser_skips_when_no_concurrent_frame_text() -> None:
    rows = [
        _audio_row("/v.mp4", "lonely audio", 0.0, 3.0),
        _frame_row("/v.mp4", "outside the window", 5.0),
    ]
    df = pd.DataFrame(rows)
    out = AudioVisualFuser(AudioVisualFuseParams()).run(df)
    assert len(out) == 2  # no fused row appended
    assert all(r["metadata"]["_content_type"] != "audio_visual" for _, r in out.iterrows())


def test_fuser_groups_per_source_path() -> None:
    rows = [
        _audio_row("/v1.mp4", "video1 audio", 0.0, 3.0),
        _audio_row("/v2.mp4", "video2 audio", 0.0, 3.0),
        _frame_row("/v1.mp4", "video1 visual", 1.0),
        _frame_row("/v2.mp4", "video2 visual", 1.0),
    ]
    df = pd.DataFrame(rows)
    out = AudioVisualFuser(AudioVisualFuseParams()).run(df)
    fused = [r for _, r in out.iterrows() if r["metadata"]["_content_type"] == "audio_visual"]
    assert len(fused) == 2
    # Each fused row's text combines its own video's modalities, not the other's.
    pairs = {(r["source_path"], r["text"]) for r in fused}
    assert ("/v1.mp4", "[AUDIO] video1 audio\n[VISUAL] video1 visual") in pairs
    assert ("/v2.mp4", "[AUDIO] video2 audio\n[VISUAL] video2 visual") in pairs


def test_fuser_preserves_segment_window_from_audio_row() -> None:
    rows = [
        _audio_row("/v.mp4", "speech", 7.5, 12.25),
        _frame_row("/v.mp4", "slide", 9.0),
    ]
    out = AudioVisualFuser(AudioVisualFuseParams()).run(pd.DataFrame(rows))
    fused = [r for _, r in out.iterrows() if r["metadata"]["_content_type"] == "audio_visual"][0]
    assert fused["metadata"]["segment_start_seconds"] == 7.5
    assert fused["metadata"]["segment_end_seconds"] == 12.25
    assert fused["metadata"]["fused_frame_count"] == 1
