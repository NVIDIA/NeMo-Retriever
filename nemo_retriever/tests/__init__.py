# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retriever tests: helpers for ``pytest.mark.skipif`` around ffmpeg availability."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from nemo_retriever.audio.media_interface import is_ffmpeg_available
from nemo_retriever.audio.media_interface import is_media_available

__all__ = [
    "is_ffmpeg_cli_available",
    "is_media_extract_available",
    "_have_ffmpeg_binary",
    "is_ffmpeg_jpeg_encoder_available",
    "_have_ffmpeg_binary_for_jpeg_frames",
    "_have_media_dependencies_for_jpeg_video_pipeline",
    "_make_test_mp4_with_av",
    "_ffprobe_first_stream_type",
    "_assert_jpeg_bytes",
]


def is_ffmpeg_cli_available() -> bool:
    """True if the ``ffmpeg`` executable is on PATH (required for extract/chunk)."""
    return shutil.which("ffmpeg") is not None


def is_media_extract_available() -> bool:
    """True when probing and ffmpeg CLI are both usable (audio/video extract and chunking)."""
    return is_media_available() and is_ffmpeg_cli_available()


def _have_ffmpeg_binary() -> bool:
    """Same as :func:`is_media_extract_available`; for ``pytest.mark.skipif`` on extract tests."""
    return is_media_extract_available()


def is_ffmpeg_jpeg_encoder_available() -> bool:
    """True if ffmpeg can encode JPEG stills for ``MediaInterface.extract_frames``.

    Minimal ffmpeg builds may omit encoders; probe the default mjpeg/JPEG frame path.
    """
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    with tempfile.TemporaryDirectory(prefix="retriever_jpeg_enc_probe_") as tmp:
        out_path = Path(tmp) / "probe.jpg"
        cmd = [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=0.1:size=16x16:rate=1",
            "-frames:v",
            "1",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "2",
            str(out_path),
        ]
        try:
            r = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return r.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0


def _have_ffmpeg_binary_for_jpeg_frames() -> bool:
    """For pytest skips on default JPEG frame extraction paths."""
    return is_ffmpeg_available() and is_ffmpeg_jpeg_encoder_available()


def _have_media_dependencies_for_jpeg_video_pipeline() -> bool:
    """For pytest skips on video-pipeline paths needing ffprobe plus JPEG frames."""
    return is_media_available() and is_ffmpeg_jpeg_encoder_available()


def _make_test_mp4_with_av(path: Path, duration_sec: int = 5) -> None:
    """Synthetic MP4 with video+audio; ``mpeg4`` avoids requiring ``libx264``."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_sec}:size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration_sec}",
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True)


def _ffprobe_first_stream_type(path: Path) -> str:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    return lines[0].strip() if lines else ""


def _assert_jpeg_bytes(raw: bytes) -> None:
    import io
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        assert image.format == "JPEG"
