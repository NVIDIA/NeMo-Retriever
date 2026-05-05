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
def test_video_frame_actor_runs_on_dataframe(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.mp4"
    _make_test_mp4(fixture, duration_sec=3)
    batch = pd.DataFrame([{"path": str(fixture)}])

    actor = VideoFrameActor(VideoFrameParams(fps=1.0))
    out = actor(batch)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3
    assert all(isinstance(b, str) and b for b in out["image_b64"])


def _solid_png_b64(rgb: tuple[int, int, int], size: int = 16) -> str:
    """Tiny solid-color PNG, base64-encoded — used to exercise dhash dedup."""
    import base64
    import io
    from PIL import Image

    img = Image.new("RGB", (size, size), rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_dedup_video_frames_drops_perceptually_identical_frames() -> None:
    # Two frames with the same flat colour have an identical dhash; a frame
    # with a very different brightness pattern stays distinct.
    flat = _solid_png_b64((128, 128, 128))
    other = _solid_png_b64((128, 128, 128))  # same colour → same dhash
    distinct = _checkerboard_png_b64()  # very different perceptual hash
    rows = [
        {"image_b64": flat, "source_path": "/v.mp4", "metadata": {"frame_timestamp_seconds": 0.5}},
        {"image_b64": other, "source_path": "/v.mp4", "metadata": {"frame_timestamp_seconds": 1.5}},
        {"image_b64": distinct, "source_path": "/v.mp4", "metadata": {"frame_timestamp_seconds": 2.5}},
        {"image_b64": flat, "source_path": "/other.mp4", "metadata": {"frame_timestamp_seconds": 0.5}},
    ]
    out = dedup_video_frames(pd.DataFrame(rows), max_hamming_distance=5)
    # Two flat rows from /v.mp4 collapse to one; distinct and the /other.mp4 flat survive.
    assert len(out) == 3
    timestamps = sorted(r["metadata"]["frame_timestamp_seconds"] for _, r in out.iterrows())
    assert timestamps == [0.5, 0.5, 2.5]


def test_dedup_video_frames_extends_time_window_across_run() -> None:
    """A run of perceptually-similar adjacent frames collapses to one row whose
    ``segment_end_seconds`` covers the whole run — so retrieval can still
    match utterances anywhere in the slide's visible span."""
    flat = _solid_png_b64((128, 128, 128))
    rows = []
    for i in range(5):
        start = float(i * 2)
        end = start + 2.0
        rows.append(
            {
                "image_b64": flat,
                "source_path": "/v.mp4",
                "metadata": {
                    "segment_start_seconds": start,
                    "segment_end_seconds": end,
                    "frame_timestamp_seconds": (start + end) / 2.0,
                    "fps": 0.5,
                },
            }
        )
    out = dedup_video_frames(pd.DataFrame(rows), max_hamming_distance=5, max_dropped_frames=2)
    assert len(out) == 1
    md = out.iloc[0]["metadata"]
    assert md["segment_start_seconds"] == 0.0
    assert md["segment_end_seconds"] == 10.0  # all 5 frames merged
    assert md["frame_timestamp_seconds"] == 5.0
    assert md["dedup_merged_count"] == 5


def test_dedup_video_frames_starts_new_run_after_long_gap() -> None:
    """Same content reappearing after a gap larger than max_dropped_frames/fps
    seconds opens a new run rather than merging across the gap."""
    flat = _solid_png_b64((128, 128, 128))
    rows = []
    for start in (0.0, 2.0, 4.0, 200.0, 202.0):  # 196s gap between groups
        rows.append(
            {
                "image_b64": flat,
                "source_path": "/v.mp4",
                "metadata": {
                    "segment_start_seconds": start,
                    "segment_end_seconds": start + 2.0,
                    "frame_timestamp_seconds": start + 1.0,
                    "fps": 0.5,
                },
            }
        )
    out = dedup_video_frames(pd.DataFrame(rows), max_hamming_distance=5, max_dropped_frames=2)
    assert len(out) == 2
    starts = sorted(r["metadata"]["segment_start_seconds"] for _, r in out.iterrows())
    ends = sorted(r["metadata"]["segment_end_seconds"] for _, r in out.iterrows())
    assert starts == [0.0, 200.0]
    assert ends == [6.0, 204.0]


def _checkerboard_png_b64(size: int = 16) -> str:
    """High-contrast checkerboard PNG with a very different dhash from solids."""
    import base64
    import io
    from PIL import Image

    img = Image.new("RGB", (size, size))
    pixels = [(255, 255, 255) if (x + y) % 2 == 0 else (0, 0, 0) for y in range(size) for x in range(size)]
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
