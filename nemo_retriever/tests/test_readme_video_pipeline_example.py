# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests aligned with the README "Video (MP4): speech + frames" example.

Validates the documented ``GraphIngestor.extract_video`` / ``build_graph`` wiring
and that ``VideoSplitActor`` emits both audio and frame rows on a synthetic MP4.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from tests import _have_ffmpeg_binary_for_png_frames
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.params import (
    ASRParams,
    AudioChunkParams,
    AudioVisualFuseParams,
    EmbedParams,
    ExtractParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
)
from nemo_retriever.video import VideoSplitActor
from nemo_retriever.video import _content_types as _CT


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


def _collect_node_names(graph: Graph) -> list[str]:
    names: list[str] = []

    def walk(node) -> None:
        names.append(node.name)
        for child in node.children:
            walk(child)

    for root in graph.roots:
        walk(root)
    return names


@pytest.mark.skipif(
    not _have_ffmpeg_binary_for_png_frames(),
    reason="ffmpeg with PNG encoder required for frame extraction",
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
    not _have_ffmpeg_binary_for_png_frames(),
    reason="ffmpeg with PNG encoder required for frame extraction",
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
