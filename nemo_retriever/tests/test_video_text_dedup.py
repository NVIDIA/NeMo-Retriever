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
