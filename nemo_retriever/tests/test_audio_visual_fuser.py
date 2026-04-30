# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.video.audio_visual_fuser.AudioVisualFuser.

The fuser's behaviour is fixed (no per-call knobs beyond ``enabled``):
  * one fused row per audio utterance with at least one concurrent frame,
  * visual side = the single most-centred concurrent frame (tiebreak: longer OCR),
  * visual capped at ``FRAME_TEXT_MAX_CHARS`` characters,
  * fused text rendered as ``"<audio>. <visual>"`` (no labels),
  * source audio rows whose window produced a fused row are dropped.
"""

from __future__ import annotations

import pandas as pd

from nemo_retriever.params import AudioVisualFuseParams
from nemo_retriever.video.audio_visual_fuser import AudioVisualFuser, FRAME_TEXT_MAX_CHARS


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
    assert text.startswith("hello world.")
    # Top-level _content_type set so the LanceDB sink picks it up.
    assert fused[0]["_content_type"] == "audio_visual"
    # Audio row whose window produced the fused row is dropped; the orphan
    # audio utterance (no concurrent frame) and the orphan frame are preserved.
    kinds = [r["metadata"]["_content_type"] for _, r in out.iterrows()]
    assert kinds.count("audio") == 1  # only the second utterance
    assert kinds.count("video_frame") == 3
    assert kinds.count("audio_visual") == 1


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
    assert len(out) == 2  # no fused row appended, audio row preserved
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
    pairs = {(r["source_path"], r["text"]) for r in fused}
    assert ("/v1.mp4", "video1 audio. video1 visual") in pairs
    assert ("/v2.mp4", "video2 audio. video2 visual") in pairs


def test_fuser_preserves_segment_window_from_audio_row() -> None:
    rows = [
        _audio_row("/v.mp4", "speech", 7.5, 12.25),
        _frame_row("/v.mp4", "slide", 9.0),
    ]
    out = AudioVisualFuser(AudioVisualFuseParams()).run(pd.DataFrame(rows))
    fused = [r for _, r in out.iterrows() if r["metadata"]["_content_type"] == "audio_visual"][0]
    assert fused["metadata"]["segment_start_seconds"] == 7.5
    assert fused["metadata"]["segment_end_seconds"] == 12.25


def test_fuser_picks_centred_concurrent_frame() -> None:
    """When multiple frames overlap the utterance, the centred frame wins."""
    rows = [
        _audio_row("/v.mp4", "spoken sentence", 10.0, 14.0),  # midpoint 12.0
        _frame_row("/v.mp4", "EARLY SLIDE", 10.5),  # offset 1.5
        _frame_row("/v.mp4", "CENTRED SLIDE", 12.0),  # offset 0.0 -> winner
        _frame_row("/v.mp4", "LATER SLIDE", 13.4),  # offset 1.4
    ]
    out = AudioVisualFuser(AudioVisualFuseParams()).run(pd.DataFrame(rows))
    fused = [r for _, r in out.iterrows() if r["metadata"]["_content_type"] == "audio_visual"]
    assert len(fused) == 1
    text = fused[0]["text"]
    assert "CENTRED SLIDE" in text
    assert "EARLY SLIDE" not in text
    assert "LATER SLIDE" not in text
    assert fused[0]["metadata"]["fused_concurrent_total"] == 3


def test_fuser_tiebreaks_on_longer_ocr() -> None:
    """Frames equidistant from the utterance midpoint -> longer OCR wins."""
    rows = [
        _audio_row("/v.mp4", "spoken sentence", 10.0, 14.0),
        _frame_row("/v.mp4", "SHORT", 11.0),  # offset 1.0
        _frame_row("/v.mp4", "MUCH LONGER OCR TEXT WITH DETAIL", 13.0),  # offset 1.0
    ]
    out = AudioVisualFuser(AudioVisualFuseParams()).run(pd.DataFrame(rows))
    fused = [r for _, r in out.iterrows() if r["metadata"]["_content_type"] == "audio_visual"]
    assert len(fused) == 1
    assert "MUCH LONGER" in fused[0]["text"]
    assert "SHORT" not in fused[0]["text"]


def test_fuser_truncates_long_visual() -> None:
    long_visual = "X" * (FRAME_TEXT_MAX_CHARS * 4)
    rows = [
        _audio_row("/v.mp4", "speech", 0.0, 2.0),
        _frame_row("/v.mp4", long_visual, 1.0),
    ]
    out = AudioVisualFuser(AudioVisualFuseParams()).run(pd.DataFrame(rows))
    fused = [r for _, r in out.iterrows() if r["metadata"]["_content_type"] == "audio_visual"]
    assert len(fused) == 1
    text = fused[0]["text"]
    visual = text.split(". ", 1)[1]
    assert len(visual) <= FRAME_TEXT_MAX_CHARS


def test_fuser_drops_redundant_audio_only_for_fused_windows() -> None:
    """Audio rows whose window matches a fused row are dropped; orphans stay."""
    rows = [
        _audio_row("/v.mp4", "fused utterance", 0.0, 3.0),  # has frame -> dropped
        _audio_row("/v.mp4", "lonely utterance", 10.0, 13.0),  # no frame -> kept
        _frame_row("/v.mp4", "slide", 1.0),
    ]
    out = AudioVisualFuser(AudioVisualFuseParams()).run(pd.DataFrame(rows))
    audio = [r for _, r in out.iterrows() if r["metadata"]["_content_type"] == "audio"]
    fused = [r for _, r in out.iterrows() if r["metadata"]["_content_type"] == "audio_visual"]
    assert len(fused) == 1
    assert len(audio) == 1
    assert audio[0]["text"] == "lonely utterance"


def test_fuser_replace_audio_window_match_is_exact() -> None:
    """Duplicate-text audio at a different window must be preserved."""
    rows = [
        _audio_row("/v.mp4", "yes", 0.0, 1.0),
        _audio_row("/v.mp4", "yes", 100.0, 101.0),
        _frame_row("/v.mp4", "slide", 0.5),
    ]
    out = AudioVisualFuser(AudioVisualFuseParams()).run(pd.DataFrame(rows))
    audio = [r for _, r in out.iterrows() if r["metadata"]["_content_type"] == "audio"]
    assert len(audio) == 1
    assert audio[0]["metadata"]["segment_start_seconds"] == 100.0
