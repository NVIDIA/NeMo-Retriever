# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.video.frame_actor."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from nemo_retriever.audio.media_interface import is_media_available
from nemo_retriever.params import VideoFrameParams
from nemo_retriever.video.frame_actor import (
    FRAME_COLUMNS,
    VideoFrameActor,
    dedup_video_frames,
    video_path_to_frames_df,
)


def _have_ffmpeg_binary() -> bool:
    return is_media_available() and shutil.which("ffmpeg") is not None


def _make_test_mp4(path: Path, duration_sec: int = 5, size: str = "320x240", fps: int = 30) -> None:
    """Generate a synthetic test mp4 via ffmpeg lavfi testsrc."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_sec}:size={size}:rate={fps}",
        "-c:v",
        "libx264",
        str(path),
    ]
    subprocess.run(cmd, check=True)


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_video_path_to_frames_df_basic_count_and_timestamps(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.mp4"
    _make_test_mp4(fixture, duration_sec=5)

    df = video_path_to_frames_df(str(fixture), VideoFrameParams(fps=1.0))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    for col in FRAME_COLUMNS:
        assert col in df.columns

    timestamps = [row["metadata"]["frame_timestamp_seconds"] for _, row in df.iterrows()]
    # Midpoints at fps=1.0 are 0.5, 1.5, 2.5, 3.5, 4.5
    assert timestamps == sorted(timestamps)
    assert pytest.approx(timestamps[0], rel=1e-3) == 0.5
    assert pytest.approx(timestamps[-1], rel=1e-3) == 4.5

    md0 = df.iloc[0]["metadata"]
    assert md0["_content_type"] == "video_frame"
    assert md0["modality"] == "video_frame"
    assert md0["fps"] == 1.0
    assert md0["segment_start_seconds"] == 0.0
    assert md0["segment_end_seconds"] == 1.0
    assert df.iloc[0]["source_path"] == str(fixture)
    # _content_type is also a top-level row column — the LanceDB sink reads it
    # from the row, not from metadata.
    assert "_content_type" in df.columns
    assert df.iloc[0]["_content_type"] == "video_frame"


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_video_frame_actor_handles_empty_batch() -> None:
    actor = VideoFrameActor(VideoFrameParams(fps=1.0))
    empty = pd.DataFrame(columns=["path", "bytes"])
    out = actor(empty)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == FRAME_COLUMNS
    assert len(out) == 0


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_video_frame_actor_runs_on_dataframe(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.mp4"
    _make_test_mp4(fixture, duration_sec=3)
    batch = pd.DataFrame([{"path": str(fixture)}])

    actor = VideoFrameActor(VideoFrameParams(fps=1.0))
    out = actor(batch)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3
    assert all(isinstance(b, str) and b for b in out["image_b64"])


def test_dedup_video_frames_drops_identical_image_b64() -> None:
    rows = [
        {"image_b64": "AAA", "source_path": "/v.mp4", "metadata": {"frame_timestamp_seconds": 0.5}},
        {"image_b64": "AAA", "source_path": "/v.mp4", "metadata": {"frame_timestamp_seconds": 1.5}},
        {"image_b64": "BBB", "source_path": "/v.mp4", "metadata": {"frame_timestamp_seconds": 2.5}},
        {"image_b64": "AAA", "source_path": "/other.mp4", "metadata": {"frame_timestamp_seconds": 0.5}},
    ]
    df = pd.DataFrame(rows)
    out = dedup_video_frames(df)
    # Two AAA rows from /v.mp4 collapse to one; BBB and the /other.mp4 AAA both kept.
    assert len(out) == 3
    timestamps = sorted(r["metadata"]["frame_timestamp_seconds"] for _, r in out.iterrows())
    assert timestamps == [0.5, 0.5, 2.5]


def test_dedup_video_frames_passthrough_for_empty_df() -> None:
    empty = pd.DataFrame()
    assert dedup_video_frames(empty).empty
