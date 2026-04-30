# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the video branch of MultiTypeExtractOperator."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_retriever.audio.media_interface import is_media_available
from nemo_retriever.params import (
    ASRParams,
    AudioChunkParams,
    AudioVisualFuseParams,
    VideoFrameParams,
    VideoOCRParams,
)


def _have_ffmpeg_binary() -> bool:
    return is_media_available() and shutil.which("ffmpeg") is not None


def _make_test_mp4(path: Path, duration_sec: int = 5) -> None:
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
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True)


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_run_video_pipeline_emits_audio_frame_and_scene_rows(tmp_path: Path) -> None:
    """End-to-end through MultiTypeExtractOperator._run_video_pipeline.

    Mocks the OCR + ASR backends so the test runs offline.
    """
    fixture = tmp_path / "fixture.mp4"
    _make_test_mp4(fixture, duration_sec=5)

    from nemo_retriever.graph.multi_type_extract_operator import _MultiTypeExtractBase

    # Build a base operator with a remote-ocr URL so VideoFrameOCRActor
    # resolves to its CPU (NIM) variant.
    op = _MultiTypeExtractBase(
        extraction_mode="auto",
        audio_chunk_params=AudioChunkParams(split_type="time", split_interval=10),
        asr_params=ASRParams(),
        video_frame_params=VideoFrameParams(fps=1.0, dedup=False),
        video_ocr_params=VideoOCRParams(ocr_invoke_url="https://example/ocr"),
        av_fuse_params=AudioVisualFuseParams(enabled=True),
    )

    # Mock the ASRActor's behaviour: emit per-utterance rows for the input video.
    fake_asr_rows = [
        {
            "source_path": str(fixture),
            "text": "first utterance",
            "metadata": {
                "segment_start_seconds": 0.0,
                "segment_end_seconds": 2.5,
                "_content_type": "audio",
                "modality": "audio_segment",
            },
        },
        {
            "source_path": str(fixture),
            "text": "second utterance",
            "metadata": {
                "segment_start_seconds": 2.5,
                "segment_end_seconds": 5.0,
                "_content_type": "audio",
                "modality": "audio_segment",
            },
        },
    ]
    fake_asr_df = pd.DataFrame(fake_asr_rows)

    # Mock the NIM client's batched call so all five frames return distinct OCR text.
    fake_responses = [[{"text_prediction": {"text": f"frame_text_{i}"}}] for i in range(5)]

    with patch("nemo_retriever.graph.multi_type_extract_operator.MediaChunkActor") as MockChunk, patch(
        "nemo_retriever.graph.multi_type_extract_operator.ASRActor"
    ) as MockASR, patch("nemo_retriever.video.ocr_actor.NIMClient") as MockNIM:
        MockChunk.return_value.run.return_value = pd.DataFrame()
        MockASR.return_value.run.return_value = fake_asr_df
        nim_instance = MagicMock()
        nim_instance.invoke_image_inference_batches.return_value = fake_responses
        MockNIM.return_value = nim_instance

        batch = pd.DataFrame([{"path": str(fixture)}])
        out = op._run_video_pipeline(batch)

    assert isinstance(out, pd.DataFrame)
    content_types = out["metadata"].apply(lambda md: md.get("_content_type")).tolist()
    assert "audio" in content_types
    assert "video_frame" in content_types
    assert "audio_visual" in content_types

    # All frame OCR rows have non-empty text
    frame_rows = out[out["metadata"].apply(lambda md: md.get("_content_type") == "video_frame")]
    assert len(frame_rows) == 5
    assert all(t.startswith("frame_text_") for t in frame_rows["text"])

    # Each scene row covers an audio utterance and joins concurrent OCR text
    scene_rows = out[out["metadata"].apply(lambda md: md.get("_content_type") == "audio_visual")]
    assert len(scene_rows) == 2
    for _, row in scene_rows.iterrows():
        assert "[AUDIO]" in row["text"]
        assert "[VISUAL]" in row["text"]
        md = row["metadata"]
        assert md["segment_end_seconds"] > md["segment_start_seconds"]
