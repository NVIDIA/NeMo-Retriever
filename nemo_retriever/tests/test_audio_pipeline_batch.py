# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration-style tests for the audio pipeline using GraphIngestor.

Uses a small generated WAV and mocked ASR so no Parakeet endpoint or mp3/ dir is required.
Skip if Ray or ffmpeg is not available.
"""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from nemo_retriever.audio.chunk_actor import _chunk_one
from nemo_retriever.audio.media_interface import MediaInterface
from nemo_retriever.audio.media_interface import is_media_available
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.params import ASRParams
from nemo_retriever.params import AudioChunkParams


def _make_small_wav(path: Path, duration_sec: float = 0.5, sample_rate: int = 8000) -> None:
    import wave

    n_frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_frames)


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_audio_chunk_then_mock_asr_flow(tmp_path: Path):
    """Chunk one small WAV and verify chunk rows have expected shape (no Ray)."""
    wav = tmp_path / "tiny.wav"
    _make_small_wav(wav, duration_sec=0.4)
    params = AudioChunkParams(split_type="size", split_interval=500_000)
    interface = MediaInterface()
    rows = _chunk_one(str(wav), params, interface)
    assert len(rows) >= 1
    row = rows[0]
    assert "path" in row and "source_path" in row and "duration" in row
    assert "chunk_index" in row and "bytes" in row
    assert row["source_path"] == str(wav.resolve())


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_inprocess_audio_pipeline_with_mocked_asr(tmp_path: Path):
    """Inprocess: files -> extract_audio (chunk + mocked ASR) -> ingest(); assert result DataFrame has text."""
    wav = tmp_path / "small.wav"
    _make_small_wav(wav, duration_sec=0.5)

    mock_client = MagicMock()
    mock_client.infer.return_value = ([], "inprocess mock transcript")

    with patch("nemo_retriever.audio.asr_actor._get_client", return_value=mock_client):
        ingestor = (
            GraphIngestor(run_mode="inprocess", documents=[])
            .files([str(wav)])
            .extract_audio(
                params=AudioChunkParams(split_type="size", split_interval=500_000),
                asr_params=ASRParams(audio_endpoints=("localhost:50051", None)),
            )
        )
        results = ingestor.ingest()

    assert results is not None
    assert isinstance(results, pd.DataFrame)
    assert "text" in results.columns
    assert len(results) >= 1
    assert (results["text"] == "inprocess mock transcript").all()


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_inprocess_audio_pipeline_with_mocked_segmented_asr(tmp_path: Path):
    """Inprocess audio pipeline can fan out punctuation-delimited Parakeet segments into multiple rows."""
    wav = tmp_path / "small.wav"
    _make_small_wav(wav, duration_sec=0.5)

    mock_client = MagicMock()
    mock_client.infer.return_value = (
        [
            {"start": 0.0, "end": 0.2, "text": "First sentence."},
            {"start": 0.2, "end": 0.5, "text": "Second sentence!"},
        ],
        "First sentence. Second sentence!",
    )

    with patch("nemo_retriever.audio.asr_actor._get_client", return_value=mock_client):
        ingestor = (
            GraphIngestor(run_mode="inprocess", documents=[])
            .files([str(wav)])
            .extract_audio(
                params=AudioChunkParams(split_type="size", split_interval=500_000),
                asr_params=ASRParams(audio_endpoints=("localhost:50051", None), segment_audio=True),
            )
        )
        results = ingestor.ingest()

    assert results is not None
    assert isinstance(results, pd.DataFrame)
    assert results["text"].tolist() == ["First sentence.", "Second sentence!"]
    assert results["metadata"].iloc[0]["segment_index"] == 0
    assert results["metadata"].iloc[0]["segment_count"] == 2
    assert results["metadata"].iloc[1]["segment_index"] == 1
    assert results["metadata"].iloc[1]["segment_start"] == 0.2
    assert results["metadata"].iloc[1]["segment_end"] == 0.5


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_inprocess_audio_pipeline_local_asr_mocked(tmp_path: Path):
    """Inprocess with audio_endpoints=(None, None) uses local ASR; mock ParakeetCTC1B1ASR so no real model."""
    wav = tmp_path / "small.wav"
    _make_small_wav(wav, duration_sec=0.5)

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ["local asr mock transcript"]

    with patch("nemo_retriever.audio.asr_actor._get_client") as mock_get_client:
        with patch("nemo_retriever.model.local.ParakeetCTC1B1ASR", return_value=mock_model):
            ingestor = (
                GraphIngestor(run_mode="inprocess", documents=[])
                .files([str(wav)])
                .extract_audio(
                    params=AudioChunkParams(split_type="size", split_interval=500_000),
                    asr_params=ASRParams(audio_endpoints=(None, None)),
                )
            )
            results = ingestor.ingest()

    mock_get_client.assert_not_called()
    assert results is not None
    assert isinstance(results, pd.DataFrame)
    assert "text" in results.columns
    assert len(results) >= 1
    assert (results["text"] == "local asr mock transcript").all()
