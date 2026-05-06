# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retriever tests: helpers for ``pytest.mark.skipif`` around ffmpeg availability."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from nemo_retriever.audio.media_interface import is_media_available

__all__ = [
    "is_ffmpeg_cli_available",
    "is_media_extract_available",
    "_have_ffmpeg_binary",
    "is_ffmpeg_png_encoder_available",
    "_have_ffmpeg_binary_for_png_frames",
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


def is_ffmpeg_png_encoder_available() -> bool:
    """True if ffmpeg can encode PNG stills (``image2`` / ``MediaInterface.extract_frames``).

    Minimal ffmpeg builds may omit the PNG encoder; probe with a one-frame lavfi encode.
    """
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    with tempfile.TemporaryDirectory(prefix="retriever_png_enc_probe_") as tmp:
        out_path = Path(tmp) / "probe.png"
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


def _have_ffmpeg_binary_for_png_frames() -> bool:
    """For pytest skips on paths that call ``MediaInterface.extract_frames`` (PNG output)."""
    return is_media_extract_available() and is_ffmpeg_png_encoder_available()
