# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.video.text_dedup.VideoFrameTextDedup."""

from __future__ import annotations

import pandas as pd

from nemo_retriever.params import VideoFrameTextDedupParams
from nemo_retriever.video.text_dedup import VideoFrameTextDedup


def _frame_row(source: str, text: str, start: float, end: float, fps: float = 0.5) -> dict:
    return {
        "source_path": source,
        "text": text,
        "_content_type": "video_frame",
        "metadata": {
            "segment_start_seconds": start,
            "segment_end_seconds": end,
            "frame_timestamp_seconds": (start + end) / 2.0,
            "fps": fps,
            "_content_type": "video_frame",
        },
    }


def _audio_row(source: str, text: str, start: float, end: float) -> dict:
    return {
        "source_path": source,
        "text": text,
        "_content_type": "audio",
        "metadata": {
            "segment_start_seconds": start,
            "segment_end_seconds": end,
            "_content_type": "audio",
        },
    }


def test_merges_consecutive_identical_text_frames() -> None:
    rows = [
        _frame_row("/v.mp4", "SLIDE A", 100.0, 102.0),
        _frame_row("/v.mp4", "SLIDE A", 102.0, 104.0),
        _frame_row("/v.mp4", "SLIDE A", 104.0, 106.0),
    ]
    out = VideoFrameTextDedup(VideoFrameTextDedupParams()).run(pd.DataFrame(rows))
    assert len(out) == 1
    md = out.iloc[0]["metadata"]
    assert md["segment_start_seconds"] == 100.0
    assert md["segment_end_seconds"] == 106.0
    assert md["merged_frame_count"] == 3
    # midpoint of the merged window
    assert md["frame_timestamp_seconds"] == 103.0


def test_user_example_keynote_slides() -> None:
    """Reproduce the user's GTC keynote example: 388s..420s should merge to one row.

    fps=0.5 means each frame is 2s wide; max_dropped_frames=2 → max gap = 4s,
    which bridges all the 0s and 2s gaps in the user's data.
    """
    text = "NVIDIA ADA LOVELACE ..."
    starts = [388, 390, 392, 396, 400, 402, 406, 408, 410, 414, 416, 418]
    rows = [_frame_row("/k.mp4", text, float(s), float(s + 2), fps=0.5) for s in starts]
    out = VideoFrameTextDedup(VideoFrameTextDedupParams(max_dropped_frames=2)).run(pd.DataFrame(rows))
    merged = out[out["text"] == text].reset_index(drop=True)
    assert len(merged) == 1
    md = merged.iloc[0]["metadata"]
    assert md["segment_start_seconds"] == 388.0
    assert md["segment_end_seconds"] == 420.0
    assert md["merged_frame_count"] == 12


def test_keeps_separate_runs_across_large_gaps() -> None:
    """Two same-text occurrences far apart in the video stay as separate rows."""
    text = "SLIDE A"
    rows = [
        _frame_row("/v.mp4", text, 10.0, 12.0),
        _frame_row("/v.mp4", text, 12.0, 14.0),
        _frame_row("/v.mp4", text, 100.0, 102.0),  # large gap (86s) → new run
        _frame_row("/v.mp4", text, 102.0, 104.0),
    ]
    out = VideoFrameTextDedup(VideoFrameTextDedupParams(max_dropped_frames=2)).run(pd.DataFrame(rows))
    merged = out.sort_values(by="metadata", key=lambda s: s.apply(lambda m: m["segment_start_seconds"]))
    assert len(merged) == 2
    md0 = merged.iloc[0]["metadata"]
    md1 = merged.iloc[1]["metadata"]
    assert (md0["segment_start_seconds"], md0["segment_end_seconds"]) == (10.0, 14.0)
    assert (md1["segment_start_seconds"], md1["segment_end_seconds"]) == (100.0, 104.0)


def test_does_not_merge_different_text() -> None:
    rows = [
        _frame_row("/v.mp4", "SLIDE A", 0.0, 2.0),
        _frame_row("/v.mp4", "SLIDE B", 2.0, 4.0),
    ]
    out = VideoFrameTextDedup(VideoFrameTextDedupParams()).run(pd.DataFrame(rows))
    assert len(out) == 2


def test_does_not_merge_across_sources() -> None:
    rows = [
        _frame_row("/v1.mp4", "X", 0.0, 2.0),
        _frame_row("/v2.mp4", "X", 0.0, 2.0),
    ]
    out = VideoFrameTextDedup(VideoFrameTextDedupParams()).run(pd.DataFrame(rows))
    assert len(out) == 2


def test_passes_through_audio_rows_unchanged() -> None:
    rows = [
        _audio_row("/v.mp4", "speech", 0.0, 3.0),
        _frame_row("/v.mp4", "SLIDE A", 1.0, 3.0),
        _frame_row("/v.mp4", "SLIDE A", 3.0, 5.0),
    ]
    out = VideoFrameTextDedup(VideoFrameTextDedupParams()).run(pd.DataFrame(rows))
    audio_rows = out[out["_content_type"] == "audio"]
    frame_rows = out[out["_content_type"] == "video_frame"]
    assert len(audio_rows) == 1
    assert audio_rows.iloc[0]["text"] == "speech"
    assert len(frame_rows) == 1  # two frames merged
    assert frame_rows.iloc[0]["metadata"]["segment_end_seconds"] == 5.0


def test_max_gap_scales_with_fps() -> None:
    """A 2-second gap between same-text frames bridges at fps=1.0
    (max_gap = 2/1.0 = 2s) but breaks at fps=0.5 with max_dropped_frames=0
    (max_gap = 0/0.5 = 0s)."""
    text = "SLIDE A"
    # Same gap pattern, different fps annotations.
    rows_fps_1 = [
        _frame_row("/v.mp4", text, 0.0, 1.0, fps=1.0),
        _frame_row("/v.mp4", text, 3.0, 4.0, fps=1.0),  # gap = 2s
    ]
    rows_fps_0p5 = [
        _frame_row("/v.mp4", text, 0.0, 2.0, fps=0.5),
        _frame_row("/v.mp4", text, 6.0, 8.0, fps=0.5),  # gap = 4s
    ]

    # max_dropped_frames=2 at fps=1.0 → max_gap=2s → bridges 2s gap.
    out1 = VideoFrameTextDedup(VideoFrameTextDedupParams(max_dropped_frames=2)).run(pd.DataFrame(rows_fps_1))
    assert len(out1) == 1
    assert out1.iloc[0]["metadata"]["segment_end_seconds"] == 4.0

    # max_dropped_frames=2 at fps=0.5 → max_gap=4s → bridges 4s gap exactly.
    out2 = VideoFrameTextDedup(VideoFrameTextDedupParams(max_dropped_frames=2)).run(pd.DataFrame(rows_fps_0p5))
    assert len(out2) == 1
    assert out2.iloc[0]["metadata"]["segment_end_seconds"] == 8.0

    # max_dropped_frames=0 → only consecutive frames merge → never merges these.
    out3 = VideoFrameTextDedup(VideoFrameTextDedupParams(max_dropped_frames=0)).run(pd.DataFrame(rows_fps_1))
    assert len(out3) == 2


def test_disabled_returns_input_unchanged() -> None:
    rows = [
        _frame_row("/v.mp4", "SLIDE A", 0.0, 2.0),
        _frame_row("/v.mp4", "SLIDE A", 2.0, 4.0),
    ]
    df = pd.DataFrame(rows)
    out = VideoFrameTextDedup(VideoFrameTextDedupParams(enabled=False)).run(df)
    pd.testing.assert_frame_equal(out, df)
