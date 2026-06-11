# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the video branch of MultiTypeExtractOperator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests import _have_media_dependencies_for_jpeg_video_pipeline
from tests import _make_test_mp4_with_av
from nemo_retriever.common.params import (
    ASRParams,
    AudioChunkParams,
    AudioVisualFuseParams,
    ExtractParams,
    VideoFrameParams,
)


def test_run_video_pipeline_forces_audio_demux_chunk_params_without_ffmpeg() -> None:
    from nemo_retriever.operators.graph_ops.multi_type_extract_operator import _MultiTypeExtractBase

    op = _MultiTypeExtractBase(
        extraction_mode="auto",
        audio_chunk_params=AudioChunkParams(
            split_type="time",
            split_interval=10,
            audio_only=False,
            video_audio_separate=True,
        ),
        asr_params=ASRParams(),
        video_frame_params=VideoFrameParams(enabled=False),
        av_fuse_params=AudioVisualFuseParams(enabled=False),
    )

    with patch("nemo_retriever.operators.graph_ops.multi_type_extract_operator.MediaChunkActor") as MockChunk, patch(
        "nemo_retriever.operators.graph_ops.multi_type_extract_operator.ASRActor"
    ) as MockASR, patch("nemo_retriever.operators.graph_ops.multi_type_extract_operator.VideoFrameActor") as MockFrames:
        MockChunk.return_value.run.return_value = pd.DataFrame([{"path": "audio_chunk.mp3"}])
        MockASR.return_value.run.return_value = pd.DataFrame(
            [{"source_path": "/tmp/video.mp4", "text": "speech", "metadata": {"_content_type": "audio"}}]
        )
        MockFrames.return_value.run.return_value = pd.DataFrame()

        out = op._run_video_pipeline(pd.DataFrame([{"path": "/tmp/video.mp4"}]))

    chunk_params = MockChunk.call_args.kwargs["params"]
    # video_asr_audio_chunk_params no longer overrides audio_only; it now
    # only forces video_audio_separate=False. The caller's audio_only=False
    # must pass through unchanged.
    assert chunk_params.audio_only is False
    assert chunk_params.video_audio_separate is False
    assert chunk_params.split_type == "time"
    assert chunk_params.split_interval == 10
    assert not out.empty


@pytest.mark.skipif(
    not _have_media_dependencies_for_jpeg_video_pipeline(),
    reason="ffmpeg/ffprobe with JPEG encoder required for video pipeline frame extraction",
)
def test_run_video_pipeline_emits_audio_frame_and_scene_rows(tmp_path: Path) -> None:
    """End-to-end through MultiTypeExtractOperator._run_video_pipeline.

    Mocks the OCR + ASR backends so the test runs offline.
    """
    fixture = tmp_path / "fixture.mp4"
    _make_test_mp4_with_av(fixture, duration_sec=5)

    from nemo_retriever.operators.graph_ops.multi_type_extract_operator import _MultiTypeExtractBase

    # Build a base operator with a remote-ocr URL so VideoFrameOCRActor
    # resolves to its CPU (NIM) variant.
    op = _MultiTypeExtractBase(
        extraction_mode="auto",
        extract_params=ExtractParams(ocr_invoke_url="https://example/ocr"),
        audio_chunk_params=AudioChunkParams(split_type="time", split_interval=10),
        asr_params=ASRParams(),
        video_frame_params=VideoFrameParams(fps=1.0, dedup=False),
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

    with patch("nemo_retriever.operators.graph_ops.multi_type_extract_operator.MediaChunkActor") as MockChunk, patch(
        "nemo_retriever.operators.graph_ops.multi_type_extract_operator.ASRActor"
    ) as MockASR, patch("nemo_retriever.operators.extract.video.ocr_actor.NIMClient") as MockNIM:
        MockChunk.return_value.run.return_value = pd.DataFrame()
        MockASR.return_value.run.return_value = fake_asr_df
        nim_instance = MagicMock()
        nim_instance.invoke_image_inference_batches.return_value = fake_responses
        MockNIM.return_value = nim_instance

        batch = pd.DataFrame([{"path": str(fixture)}])
        out = op._run_video_pipeline(batch)

    chunk_params = MockChunk.call_args.kwargs["params"]
    # audio_only is now caller-controlled (default False here); only
    # video_audio_separate is forced by video_asr_audio_chunk_params.
    assert chunk_params.audio_only is False
    assert chunk_params.video_audio_separate is False

    assert isinstance(out, pd.DataFrame)
    content_types = out["metadata"].apply(lambda md: md.get("_content_type")).tolist()
    # The baked-in fuser drops audio rows whose windows match a fused row
    # AND drops every ``video_frame`` row from the output (paired frames
    # are folded into the audio_visual rows; orphan frames are net-noise).
    # Both utterances here have concurrent frames so only audio_visual
    # rows appear.
    assert "video_frame" not in content_types
    assert "audio_visual" in content_types

    # Each scene row covers an audio utterance and pairs it with the most
    # representative concurrent frame in labelled "[AUDIO] <a> | [VISUAL] <v>"
    # format.
    scene_rows = out[out["metadata"].apply(lambda md: md.get("_content_type") == "audio_visual")]
    assert len(scene_rows) == 2
    for _, row in scene_rows.iterrows():
        text = row["text"]
        assert text.startswith(("[AUDIO] first utterance |", "[AUDIO] second utterance |"))
        assert " | [VISUAL] " in text
        md = row["metadata"]
        assert md["segment_end_seconds"] > md["segment_start_seconds"]
