# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for nemo_retriever.audio: MediaChunkActor and audio_path_to_chunks_df.
"""

import logging
import wave
from pathlib import Path

import pandas as pd
import pytest

from nemo_retriever.operators.extract.audio.chunk_actor import CHUNK_COLUMNS
from nemo_retriever.operators.extract.audio.chunk_actor import MediaChunkActor
from nemo_retriever.operators.extract.audio.chunk_actor import audio_path_to_chunks_df
from nemo_retriever.common.modality.audio.media_interface import is_media_available
from tests import _have_ffmpeg_binary
from tests import _ffprobe_first_stream_type
from tests import _make_test_mp4_with_av
from nemo_retriever.common.params import AudioChunkParams


def _make_small_wav(path: Path, duration_sec: float = 0.5, sample_rate: int = 8000) -> None:
    """Write a minimal WAV file (e.g. 0.5s mono 8kHz) for tests."""
    n_frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_frames)


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_media_chunk_actor_empty_batch():
    from nemo_retriever.operators.extract.audio.chunk_actor import MediaChunkActor

    params = AudioChunkParams(split_type="size", split_interval=1000)
    actor = MediaChunkActor(params=params)
    empty = pd.DataFrame(columns=["path", "bytes"])
    out = actor(empty)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == CHUNK_COLUMNS
    assert len(out) == 0


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_media_chunk_actor_single_small_file(tmp_path: Path):
    from nemo_retriever.operators.extract.audio.chunk_actor import MediaChunkActor

    wav = tmp_path / "tiny.wav"
    _make_small_wav(wav, duration_sec=0.3)
    with open(wav, "rb") as f:
        body = f.read()

    params = AudioChunkParams(split_type="size", split_interval=1_000_000)
    actor = MediaChunkActor(params=params)
    batch = pd.DataFrame([{"path": str(wav.resolve()), "bytes": body}])
    out = actor(batch)

    assert isinstance(out, pd.DataFrame)
    for col in ["path", "source_path", "duration", "chunk_index", "metadata", "page_number", "bytes"]:
        assert col in out.columns
    assert len(out) >= 1
    assert out["source_path"].iloc[0] == str(wav.resolve())
    assert out["chunk_index"].iloc[0] == 0
    assert isinstance(out["bytes"].iloc[0], bytes)


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_video_audio_separate_true_on_video_warns_and_outputs_audio_chunks(tmp_path: Path, caplog) -> None:
    fixture = tmp_path / "fixture.mp4"
    _make_test_mp4_with_av(fixture, duration_sec=2)

    caplog.set_level(logging.WARNING, logger="nemo_retriever.common.modality.audio.media_interface")
    params = AudioChunkParams(
        split_type="time",
        split_interval=10,
        video_audio_separate=True,
    )
    actor = MediaChunkActor(params=params)

    out = actor(pd.DataFrame([{"path": str(fixture), "bytes": fixture.read_bytes()}]))

    assert isinstance(out, pd.DataFrame)
    assert not out.empty
    chunk_paths = [Path(path) for path in out["path"].tolist()]
    assert all(path.suffix == ".mp3" for path in chunk_paths)
    assert not any(path.suffix == ".mp4" for path in chunk_paths)
    assert all(metadata["source_path"] == str(fixture) for metadata in out["metadata"])
    for idx, raw in enumerate(out["bytes"]):
        assert isinstance(raw, bytes) and raw
        chunk_copy = tmp_path / f"chunk_{idx}.mp3"
        chunk_copy.write_bytes(raw)
        assert _ffprobe_first_stream_type(chunk_copy) == "audio"
    assert "video_audio_separate is ignored" in caplog.text
    assert "ASR-safe audio chunks" in caplog.text
    assert "VideoSplitActor" in caplog.text
    assert "video pipeline" in caplog.text


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_audio_path_to_chunks_df(tmp_path: Path):
    wav = tmp_path / "small.wav"
    _make_small_wav(wav, duration_sec=0.4)
    params = AudioChunkParams(split_type="size", split_interval=500_000)
    df = audio_path_to_chunks_df(str(wav), params=params)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1
    assert "path" in df.columns and "source_path" in df.columns
    assert "bytes" in df.columns
    assert df["source_path"].iloc[0] == str(wav.resolve())


def test_media_chunk_actor_requires_ffmpeg():
    """Without ffmpeg, MediaChunkActor.__init__ raises."""
    pytest.importorskip("ffmpeg")
    # If is_media_available() is True, __init__ succeeds; we only exercise the raise path otherwise.
    if is_media_available():
        pytest.skip("ffmpeg available; cannot test missing-ffmpeg path here")
    with pytest.raises(RuntimeError, match="ffmpeg"):
        MediaChunkActor(params=AudioChunkParams())
