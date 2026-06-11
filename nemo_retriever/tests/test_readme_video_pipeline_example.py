# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests aligned with the README "Video (MP4): speech + frames" example.

Validates the documented ``GraphIngestor.extract_video`` / ``build_graph`` wiring
and that ``VideoSplitActor`` emits both audio and frame rows on a synthetic MP4.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import pytest

from tests import _have_ffmpeg_binary
from tests import _have_media_dependencies_for_jpeg_video_pipeline
from tests import _assert_jpeg_bytes
from tests import _ffprobe_first_stream_type
from tests import _make_test_mp4_with_av
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.common.params import (
    ASRParams,
    AudioChunkParams,
    AudioVisualFuseParams,
    EmbedParams,
    ExtractParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
)
from nemo_retriever.operators.extract.video.split import VideoSplitActor
from nemo_retriever.common.modality.video import _content_types as _CT


def _collect_node_names(graph: Graph) -> list[str]:
    names: list[str] = []

    def walk(node) -> None:
        names.append(node.name)
        for child in node.children:
            walk(child)

    for root in graph.roots:
        walk(root)
    return names


def test_video_asr_chunk_params_force_audio_demux() -> None:
    params = AudioChunkParams(
        enabled=True,
        split_type="time",
        split_interval=60,
        audio_only=False,
        video_audio_separate=True,
    )

    from nemo_retriever.operators.extract.video.split import video_asr_audio_chunk_params

    normalized = video_asr_audio_chunk_params(params)

    # ``video_asr_audio_chunk_params`` only forces video_audio_separate=False;
    # it must not overwrite the caller's audio_only flag (which now controls
    # the user-facing OCR-disable semantic).
    assert normalized.audio_only is False
    assert normalized.video_audio_separate is False
    assert normalized.split_type == "time"
    assert normalized.split_interval == 60
    assert params.audio_only is False
    assert params.video_audio_separate is True


def test_video_asr_chunk_params_disabled_passthrough() -> None:
    """Disabled params must pass through unchanged."""
    from nemo_retriever.operators.extract.video.split import video_asr_audio_chunk_params

    disabled = AudioChunkParams(enabled=False, audio_only=False)
    result = video_asr_audio_chunk_params(disabled)

    assert result.enabled is False
    assert result.audio_only is False
    assert result is disabled


@pytest.mark.skipif(
    not _have_ffmpeg_binary(),
    reason="ffmpeg/ffprobe required for VideoSplitActor construction",
)
def test_readme_video_pipeline_build_graph_chain() -> None:
    """``build_graph`` for the README video params starts with the documented chain."""
    graph = build_graph(
        extraction_mode="auto",
        extract_params=ExtractParams(
            ocr_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1",
        ),
        audio_chunk_params=AudioChunkParams(
            enabled=True,
            split_type="time",
            split_interval=60,
        ),
        asr_params=ASRParams(),
        video_frame_params=VideoFrameParams(enabled=True, fps=1.0, dedup=True),
        video_text_dedup_params=VideoFrameTextDedupParams(enabled=True),
        av_fuse_params=AudioVisualFuseParams(enabled=True),
        embed_params=EmbedParams(),
        stage_order=("embed",),
    )
    names = _collect_node_names(graph)
    expected_prefix = [
        "VideoSplitActor",
        "ASRActor",
        "VideoFrameOCRActor",
        "VideoFrameTextDedup",
        "AudioVisualFuser",
    ]
    assert names[: len(expected_prefix)] == expected_prefix, names
    assert "_BatchEmbedActor" in names


@pytest.mark.skipif(
    not _have_ffmpeg_binary(),
    reason="ffmpeg/ffprobe required for VideoSplitActor construction",
)
def test_audio_only_excludes_visual_branch_from_graph() -> None:
    """``audio_only=True`` must strip VideoFrameOCRActor, VideoFrameTextDedup,
    and AudioVisualFuser from the graph — only the audio (ASR) branch runs.

    Graph-topology check that still instantiates ``VideoSplitActor``, whose
    constructor probes ffmpeg/ffprobe.
    """
    graph = build_graph(
        extraction_mode="auto",
        extract_params=ExtractParams(
            ocr_invoke_url="https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1",
        ),
        audio_chunk_params=AudioChunkParams(
            enabled=True,
            split_type="time",
            split_interval=60,
            audio_only=True,
        ),
        asr_params=ASRParams(),
        video_frame_params=VideoFrameParams(enabled=True, fps=1.0, dedup=True),
        video_text_dedup_params=VideoFrameTextDedupParams(enabled=True),
        av_fuse_params=AudioVisualFuseParams(enabled=True),
        embed_params=EmbedParams(),
        stage_order=("embed",),
    )
    names = _collect_node_names(graph)
    assert "VideoSplitActor" in names
    assert "ASRActor" in names
    assert "VideoFrameOCRActor" not in names
    assert "VideoFrameTextDedup" not in names
    assert "AudioVisualFuser" not in names
    assert "_BatchEmbedActor" in names


@pytest.mark.skipif(
    not _have_media_dependencies_for_jpeg_video_pipeline(),
    reason="ffmpeg/ffprobe with JPEG encoder required for video pipeline frame extraction",
)
def test_readme_video_split_actor_emits_audio_and_frame_rows(tmp_path: Path) -> None:
    """Mirror README ``AudioChunkParams`` / ``VideoFrameParams`` on a synthetic MP4."""
    fixture = tmp_path / "readme_fixture.mp4"
    _make_test_mp4_with_av(fixture, duration_sec=5)

    actor = VideoSplitActor(
        audio_chunk_params=AudioChunkParams(
            enabled=True,
            split_type="time",
            split_interval=60,
        ),
        video_frame_params=VideoFrameParams(enabled=True, fps=1.0, dedup=True),
    )
    batch = pd.DataFrame([{"path": str(fixture)}])
    out = actor.process(batch)

    assert isinstance(out, pd.DataFrame)
    assert not out.empty
    assert "_content_type" in out.columns
    types = set(out["_content_type"].unique().tolist())
    assert _CT.AUDIO in types
    assert _CT.VIDEO_FRAME in types

    audio_rows = out[out["_content_type"] == _CT.AUDIO]
    assert set(audio_rows["path"].apply(lambda p: Path(str(p)).suffix)) == {".mp3"}
    for idx, row in audio_rows.iterrows():
        audio_chunk = tmp_path / f"audio_chunk_{idx}.mp3"
        audio_chunk.write_bytes(row["bytes"])
        assert _ffprobe_first_stream_type(audio_chunk) == "audio"

    frame_rows = out[out["_content_type"] == _CT.VIDEO_FRAME]
    assert not frame_rows.empty
    for image_b64 in frame_rows["image_b64"]:
        _assert_jpeg_bytes(base64.b64decode(image_b64))
