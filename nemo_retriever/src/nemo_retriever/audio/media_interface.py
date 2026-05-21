# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared media (audio/video) chunking logic for the retriever.

Minimal copy of ffmpeg/ffprobe and MediaInterface semantics from
`nemo_retriever.api` dataloader so the retriever stays self-contained.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import ffmpeg
except ImportError:
    ffmpeg = None  # type: ignore[assignment]

MANUAL_FFMPEG_INSTALL_COMMAND = "apt-get update && apt-get install -y --no-install-recommends ffmpeg"
CONTAINER_FFMPEG_INSTALL_ENV = "-e INSTALL_FFMPEG=true"
HELM_FFMPEG_INSTALL_VALUE = "service.installFfmpeg=true"
MEDIA_DEPENDENCIES: Tuple[str, ...] = ("ffmpeg-python", "ffmpeg", "ffprobe")
FFMPEG_DEPENDENCIES: Tuple[str, ...] = ("ffmpeg-python", "ffmpeg")
FFPROBE_DEPENDENCIES: Tuple[str, ...] = ("ffmpeg-python", "ffprobe")


def is_ffmpeg_python_available() -> bool:
    """True when the ``ffmpeg-python`` wrapper package can be imported."""
    return ffmpeg is not None


def is_ffmpeg_cli_available() -> bool:
    """True when the ``ffmpeg`` executable is on PATH."""
    return shutil.which("ffmpeg") is not None


def is_ffprobe_cli_available() -> bool:
    """True when the ``ffprobe`` executable is on PATH."""
    return shutil.which("ffprobe") is not None


def is_ffmpeg_available() -> bool:
    """True when the ``ffmpeg-python`` wrapper and ``ffmpeg`` executable are available."""
    return is_ffmpeg_python_available() and is_ffmpeg_cli_available()


def is_ffprobe_available() -> bool:
    """True when the ``ffmpeg-python`` wrapper and ``ffprobe`` executable are available."""
    return is_ffmpeg_python_available() and is_ffprobe_cli_available()


def missing_media_dependencies(required: Tuple[str, ...] = MEDIA_DEPENDENCIES) -> List[str]:
    """Return missing media dependencies in user-facing install order."""
    checks = {
        "ffmpeg-python": is_ffmpeg_python_available,
        "ffmpeg": is_ffmpeg_cli_available,
        "ffprobe": is_ffprobe_cli_available,
    }
    missing: List[str] = []
    for dependency in required:
        check = checks.get(dependency)
        if check is None or not check():
            missing.append(dependency)
    return missing


def media_dependency_error_message(
    component: str = "Media processing",
    required: Tuple[str, ...] = MEDIA_DEPENDENCIES,
) -> str:
    """Build an actionable error for missing audio/video dependencies."""
    missing = missing_media_dependencies(required)
    if not missing:
        return f"{component} media dependencies are available."

    missing_text = ", ".join(missing)
    install_hints = []
    if "ffmpeg-python" in missing:
        install_hints.append("Install the Python wrapper with `pip install ffmpeg-python`.")
    if "ffmpeg" in missing or "ffprobe" in missing:
        install_hints.append(
            "Install system FFmpeg with "
            f"`{MANUAL_FFMPEG_INSTALL_COMMAND}`. "
            "For the bundled service container, run with "
            f"`docker run {CONTAINER_FFMPEG_INSTALL_ENV} ...`. "
            f"For Helm deployments, set `{HELM_FFMPEG_INSTALL_VALUE}`."
        )
    hints_str = (" " + " ".join(install_hints)) if install_hints else ""
    return f"{component} requires media dependencies; missing: {missing_text}.{hints_str}"


class SplitType:
    """Split strategy for media; values match `nemo_retriever.api`."""

    SIZE = "size"
    TIME = "time"
    FRAME = "frame"


def _probe(
    filename: str,
    format: Optional[str] = None,
    file_handle: Any = None,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    if not is_ffprobe_available():
        raise RuntimeError(media_dependency_error_message("Media probing", required=FFPROBE_DEPENDENCIES))
    args = ["ffprobe", "-show_format", "-show_streams", "-of", "json"]
    args += ffmpeg._utils.convert_kwargs_to_cmd_line_args(kwargs)
    if file_handle:
        args += ["pipe:"]
    else:
        args += [filename]
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    communicate_kwargs: dict = {}
    if timeout is not None:
        communicate_kwargs["timeout"] = timeout
    if file_handle:
        communicate_kwargs["input"] = file_handle
    out, err = p.communicate(**communicate_kwargs)
    if p.returncode != 0:
        raise ffmpeg._run.Error("ffprobe", out, err)
    return json.loads(out.decode("utf-8"))


def _run_ffmpeg(stream: Any, *, label: str, input_path: str) -> None:
    """Invoke an ``ffmpeg-python`` stream without the pipe-buffer deadlock.

    ``stream.run(capture_stderr=True)`` reads ffmpeg's stderr through a 64 KB OS
    pipe drained by a Python thread inside ``subprocess.communicate()``. In a
    Ray Data worker that drain thread can starve under GIL contention; the pipe
    fills, ffmpeg's I/O thread blocks on ``write(2)``, and its ``-threads N``
    workers spin (=> 99% CPU, no progress, indefinite hang). Send stderr to a
    tempfile instead — file writes never block, so ffmpeg always makes progress
    and the call returns. We only read stderr when ``returncode != 0``.
    """
    if not is_ffmpeg_available():
        raise RuntimeError(media_dependency_error_message(f"FFmpeg operation '{label}'", required=FFMPEG_DEPENDENCIES))
    args = ffmpeg.compile(stream)
    with tempfile.TemporaryFile(mode="w+b") as stderr_buf:
        result = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=stderr_buf)
        if result.returncode != 0:
            stderr_buf.seek(0)
            err = stderr_buf.read()
            raise ffmpeg.Error(label, b"", err)


def _get_audio_from_video(input_path: str, output_file: str, cache_path: Optional[str] = None) -> Optional[Path]:
    """Extract audio from a video file. Returns output Path or None on failure."""
    if not is_ffmpeg_available():
        raise RuntimeError(media_dependency_error_message("Audio extraction", required=FFMPEG_DEPENDENCIES))
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = (
            ffmpeg.input(str(input_path)).output(str(output_path), acodec="libmp3lame", map="0:a").overwrite_output()
        )
        _run_ffmpeg(stream, label="extract_audio", input_path=str(input_path))
        return output_path
    except ffmpeg.Error as e:
        logger.error("FFmpeg error for file %s: %s", input_path, e.stderr.decode())
        return None


def _resolve_duration(probe: Any, stream0: Any, file_size: int) -> Optional[float]:
    """Best-effort duration in seconds from an ffprobe payload.

    Some encodings expose duration only at the format level, others only at
    the stream level, and some segmented files (e.g. fragmented MP4 chunks
    produced by ``-f segment``) don't carry either reliably — for those we
    fall back to ``file_size * 8 / bit_rate``. Returns ``None`` when nothing
    in the probe can be coerced to a float.
    """
    fmt = probe.get("format") if isinstance(probe, dict) else None
    if isinstance(fmt, dict):
        try:
            d = fmt.get("duration")
            if d is not None:
                return float(d)
        except (TypeError, ValueError):
            pass
    if isinstance(stream0, dict):
        try:
            d = stream0.get("duration")
            if d is not None:
                return float(d)
        except (TypeError, ValueError):
            pass
    if isinstance(fmt, dict):
        bitrate = fmt.get("bit_rate")
        try:
            if bitrate is not None and float(bitrate) > 0:
                return (file_size * 8) / float(bitrate)
        except (TypeError, ValueError):
            pass
    if isinstance(stream0, dict):
        bitrate = stream0.get("bit_rate")
        try:
            if bitrate is not None and float(bitrate) > 0:
                return (file_size * 8) / float(bitrate)
        except (TypeError, ValueError):
            pass
    return None


def _effective_cores() -> int:
    return int(max((os.cpu_count() or 4) * 0.2, 4))


class _LoaderInterface(ABC):
    @abstractmethod
    def split(self, input_path: str, output_dir: str, split_interval: int = 0) -> Any:
        pass

    @abstractmethod
    def _get_path_metadata(self, path: Optional[str] = None) -> Any:
        pass


class MediaInterface(_LoaderInterface):
    """Split and probe media files (audio/video) using ffmpeg/ffprobe."""

    def __init__(self) -> None:
        self.path_metadata: dict = {}

    def probe_media(
        self,
        path_file: Path,
        split_interval: int,
        split_type: str,
        file_handle: Any = None,
    ) -> Tuple[Optional[Any], Optional[float], Optional[float]]:
        """Return (probe, num_splits, duration)."""
        num_splits = None
        duration = None
        probe = None
        try:
            file_size = path_file.stat().st_size
            if file_handle:
                probe = _probe("pipe:", format=path_file.suffix, file_handle=file_handle)
            else:
                probe = _probe(str(path_file), format=path_file.suffix)
            stream0 = probe["streams"][0]
            codec_type = stream0.get("codec_type")
            if codec_type == "video":
                sample_rate = float(stream0["avg_frame_rate"].split("/")[0])
                duration = _resolve_duration(probe, stream0, file_size)
            elif codec_type == "audio":
                sample_rate = float(stream0["sample_rate"])
                duration = _resolve_duration(probe, stream0, file_size)
            else:
                raise ValueError(f"Unknown codec_type: {stream0}")
            if duration is None:
                raise ValueError(f"Could not determine duration for {path_file}")
            num_splits = self.find_num_splits(file_size, sample_rate, duration, split_interval, split_type)
        except RuntimeError:
            raise
        except OSError as e:
            logger.error("OS error accessing file %s: %s", path_file, e)
        except ffmpeg.Error as e:
            logger.error("FFmpeg error for file %s: %s", path_file, e.stderr.decode())
        except (KeyError, ValueError) as e:
            logger.error("Error probing media for file %s: %s", path_file, e)
        return (probe, num_splits, duration)

    def get_audio_from_video(
        self,
        input_path: str,
        output_file: str,
        cache_path: Optional[str] = None,
    ) -> Optional[Path]:
        return _get_audio_from_video(input_path, output_file, cache_path)

    def split(
        self,
        input_path: str,
        output_dir: str,
        split_interval: int = 0,
        split_type: str = SplitType.SIZE,
        cache_path: Optional[str] = None,
        video_audio_separate: bool = False,
        audio_only: bool = False,
    ) -> List[str]:
        """Split media into chunk files. Returns list of chunk file paths."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        original_input_path = input_path
        path_input = Path(input_path)
        if audio_only and path_input.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
            out_mp3 = output_dir / f"{path_input.stem}.mp3"
            result = self.get_audio_from_video(str(input_path), str(out_mp3), cache_path)
            if result is None:
                return []
            input_path = str(result)
        path_file = Path(input_path)
        file_name = path_file.stem
        suffix = path_file.suffix
        output_pattern = output_dir / f"{file_name}_chunk_%04d{suffix}"
        num_splits = 0
        cache_path = cache_path or output_dir
        try:
            probe, num_splits, duration = self.probe_media(path_file, split_interval, split_type)
            if num_splits is None or duration is None or num_splits <= 0:
                return []
            segment_time = math.ceil(duration / num_splits)
            output_kwargs = {
                "f": "segment",
                "segment_time": segment_time,
                "c": "copy",
                "map": "0",
                "threads": _effective_cores(),
            }
            if suffix.lower() == ".mp4":
                output_kwargs.update(
                    {
                        "force_key_frames": f"expr:gte(t,n_forced*{segment_time})",
                        "crf": 22,
                        "g": 50,
                        "sc_threshold": 0,
                    }
                )
            stream = ffmpeg.input(str(input_path)).output(str(output_pattern), **output_kwargs)
            _run_ffmpeg(stream, label="split", input_path=str(input_path))
            self.path_metadata[str(input_path)] = probe
        except RuntimeError:
            raise
        except OSError as e:
            logger.error("OS error accessing file %s: %s", original_input_path, e)
            return []
        except ffmpeg.Error as e:
            logger.error("FFmpeg error for file %s: %s", original_input_path, e.stderr.decode())
            return []
        # Use actual chunk files produced by ffmpeg (may differ from num_splits)
        files = sorted(str(p) for p in output_dir.glob(f"{file_name}_chunk_*{suffix}") if p.is_file())
        if video_audio_separate and suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
            for f in files:
                fp = Path(f)
                audio_path = self.get_audio_from_video(f, str(fp.with_suffix(".mp3")), str(cache_path))
                if audio_path is not None:
                    files.append(str(audio_path))
        return files

    def extract_frames(
        self,
        input_path: str,
        output_dir: str,
        fps: float = 1.0,
        max_frames: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Extract frames at ``fps`` frames/second; return ``[(png_path, timestamp_s), ...]``.

        Each timestamp is the wall-clock midpoint of the frame's window in the
        original video: ``frame_index / fps + 0.5 / fps``. This matches the
        canonical ``segment_start_seconds`` / ``segment_end_seconds`` convention
        used downstream by the recall scorer.

        Returns an empty list when ffmpeg fails or no frames are produced.
        """
        if not is_ffmpeg_available():
            raise RuntimeError(media_dependency_error_message("Frame extraction", required=FFMPEG_DEPENDENCIES))
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path_file = Path(input_path)
        file_name = path_file.stem
        output_pattern = str(out_dir / f"{file_name}_frame_%06d.png")

        try:
            output_kwargs: dict = {"vf": f"fps={fps}", "q:v": 2}
            if max_frames is not None and int(max_frames) > 0:
                output_kwargs["frames:v"] = int(max_frames)
            stream = ffmpeg.input(str(input_path)).output(output_pattern, **output_kwargs).overwrite_output()
            _run_ffmpeg(stream, label="extract_frames", input_path=str(input_path))
        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if getattr(e, "stderr", None) else ""
            logger.error("FFmpeg frame extraction error for file %s: %s", input_path, stderr)
            return []

        produced = sorted(p for p in out_dir.glob(f"{file_name}_frame_*.png") if p.is_file())
        results: List[Tuple[str, float]] = []
        midpoint_offset = 0.5 / float(fps)
        for idx, frame_path in enumerate(produced):
            timestamp = idx / float(fps) + midpoint_offset
            results.append((str(frame_path), float(timestamp)))
        return results

    def find_num_splits(
        self,
        file_size: int,
        sample_rate: float,
        duration: float,
        split_interval: int,
        split_type: str,
    ) -> float:
        if split_type == SplitType.SIZE:
            return math.ceil(file_size / split_interval)
        if split_type == SplitType.TIME:
            return math.ceil(duration / split_interval)
        if split_type == SplitType.FRAME:
            seconds_cap = split_interval / sample_rate
            return math.ceil(duration / seconds_cap)
        raise ValueError(f"Invalid split type: {split_type}")

    def _get_path_metadata(self, path: Optional[str] = None) -> dict:
        return self.path_metadata


def is_media_available() -> bool:
    """True if the full audio/video media pipeline can run."""
    return is_ffmpeg_available() and is_ffprobe_cli_available()
