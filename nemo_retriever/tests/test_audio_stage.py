# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the audio extraction-only stage (retriever audio extract).
"""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from nemo_retriever.operators.extract.audio.asr_actor import DEFAULT_NGC_ASR_FUNCTION_ID
from tests import _have_ffmpeg_binary
from nemo_retriever.cli.audio.stage import _audio_extraction_json_path
from nemo_retriever.cli.audio.stage import _run_extract_one
from nemo_retriever.cli.audio.stage import extract
from nemo_retriever.common.params import ASRParams
from nemo_retriever.common.params import AudioChunkParams


def _make_small_wav(path: Path, duration_sec: float = 0.5, sample_rate: int = 8000) -> None:
    import wave

    n_frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_frames)


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_audio_stage_extract_one_mocked_asr(tmp_path: Path):
    """Run extraction (chunk + ASR) on one small WAV with mocked ASR; assert output shape."""
    wav = tmp_path / "tiny.wav"
    _make_small_wav(wav, duration_sec=0.4)
    chunk_params = AudioChunkParams(split_type="size", split_interval=500_000)
    asr_params = ASRParams(audio_endpoints=("localhost:50051", None))

    mock_client = MagicMock()
    mock_client.infer.return_value = ([], "mock transcript for stage test")

    with patch("nemo_retriever.operators.extract.audio.asr_actor._get_client", return_value=mock_client):
        df = _run_extract_one(str(wav), chunk_params, asr_params)

    assert not df.empty
    assert "text" in df.columns
    assert "source_path" in df.columns
    assert "chunk_index" in df.columns
    assert "duration" in df.columns
    assert df["text"].iloc[0] == "mock transcript for stage test"


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_audio_stage_extract_cli_writes_sidecar(tmp_path: Path):
    """Run audio extract CLI; assert sidecar JSON exists and has expected shape."""
    wav = tmp_path / "sample.wav"
    _make_small_wav(wav, duration_sec=0.3)
    mock_client = MagicMock()
    mock_client.infer.return_value = ([], "cli mock transcript")

    with patch("nemo_retriever.operators.extract.audio.asr_actor._get_client", return_value=mock_client):
        extract(
            input_dir=tmp_path,
            glob="*.wav",
            output_dir=None,
            split_type="size",
            split_interval=500_000,
            video_audio_separate=False,
            use_env_asr=False,
            audio_grpc_endpoint="localhost:50051",
            auth_token=None,
            limit=None,
            write_json=True,
        )

    sidecar = tmp_path / "sample.wav.audio_extraction.json"
    assert sidecar.exists()
    import json

    data = json.loads(sidecar.read_text())
    assert data.get("schema_version") == 1
    assert data.get("stage") == "audio_extract"
    assert "chunks" in data
    assert data.get("num_chunks") == len(data["chunks"])
    if data["chunks"]:
        chunk = data["chunks"][0]
        assert "text" in chunk
        assert "source_path" in chunk
        assert chunk["text"] == "cli mock transcript"


def test_audio_stage_extract_cli_grpc_endpoint_preserves_env_auth(monkeypatch, tmp_path: Path):
    """CLI endpoint defaults must resolve together with env auth/function_id."""
    endpoint = "grpc.nvcf.nvidia.com:443"
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"")
    captured: dict[str, ASRParams] = {}

    def fake_run_extract_one(path: str, chunk_params: AudioChunkParams, asr_params: ASRParams) -> pd.DataFrame:
        captured["asr_params"] = asr_params
        return pd.DataFrame(
            [
                {
                    "path": path,
                    "source_path": path,
                    "duration": 0.0,
                    "chunk_index": 0,
                    "text": "ok",
                    "metadata": {},
                }
            ]
        )

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("AUDIO_GRPC_ENDPOINT", raising=False)
    monkeypatch.delenv("AUDIO_FUNCTION_ID", raising=False)

    with (
        patch("nemo_retriever.cli.audio.stage.is_media_available", return_value=True),
        patch("nemo_retriever.cli.audio.stage._run_extract_one", side_effect=fake_run_extract_one),
    ):
        extract(
            input_dir=tmp_path,
            glob="*.wav",
            output_dir=None,
            split_type="size",
            split_interval=500_000,
            video_audio_separate=False,
            use_env_asr=True,
            audio_grpc_endpoint=endpoint,
            auth_token=None,
            limit=None,
            write_json=False,
        )

    asr_params = captured["asr_params"]
    assert asr_params.audio_endpoints[0] == endpoint
    assert asr_params.auth_token == "nvapi-test"
    assert asr_params.function_id == DEFAULT_NGC_ASR_FUNCTION_ID


def test_audio_extraction_json_path():
    """Sidecar path is next to source or under output_dir."""
    p = Path("/foo/bar/file.wav")
    assert _audio_extraction_json_path(p, None) == Path("/foo/bar/file.wav.audio_extraction.json")
    assert _audio_extraction_json_path(p, Path("/out")) == Path("/out/file.wav.audio_extraction.json")
