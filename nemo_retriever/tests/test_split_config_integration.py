# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for split_config flowing through build_graph and the ingestor."""

from __future__ import annotations

import pytest

from nemo_retriever.audio.media_interface import is_media_available
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.graph.multi_type_extract_operator import MultiTypeExtractCPUActor
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.params import EmbedParams, ExtractParams
from nemo_retriever.params import resolve_split_params


def _node_names(graph) -> list[str]:
    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            return names
        node = node.children[0]


def test_build_graph_pdf_branch_appends_text_chunk_actor_when_pdf_split_set():
    split_config = resolve_split_params({"pdf": {"max_tokens": 256}})
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
        ),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            embed_invoke_url="http://embed.example/v1",
        ),
        split_config=split_config,
    )
    names = _node_names(graph)
    assert "TextChunkActor" in names
    assert names.index("TextChunkActor") < names.index("_BatchEmbedActor")


def test_multi_type_extract_operator_uses_text_split_from_split_config():
    cfg = resolve_split_params({"text": {"max_tokens": 64}})
    op = MultiTypeExtractCPUActor(extraction_mode="auto", split_config=cfg)
    text_chunk_params = op._effective_chunk_params("text")
    assert text_chunk_params is not None
    assert text_chunk_params.max_tokens == 64


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_build_graph_audio_only_path_appends_text_chunk_actor_when_audio_split_set():
    from nemo_retriever.params import ASRParams

    cfg = resolve_split_params({"audio": {"max_tokens": 256}})
    graph = build_graph(
        extraction_mode="audio",
        extract_params=ExtractParams(method="audio"),
        asr_params=ASRParams(),
        split_config=cfg,
    )
    names = _node_names(graph)
    assert names[:2] == ["MediaChunkActor", "ASRActor"]
    assert "TextChunkActor" in names


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_build_graph_video_branch_appends_text_chunk_actor_when_video_split_set():
    """The dedicated has_video_branch path appends TextChunkActor when video split is set."""
    from nemo_retriever.params import (
        ASRParams,
        AudioChunkParams,
        AudioVisualFuseParams,
        VideoFrameParams,
        VideoFrameTextDedupParams,
    )

    cfg = resolve_split_params({"video": {"max_tokens": 200}})
    graph = build_graph(
        extraction_mode="auto",
        extract_params=ExtractParams(),
        audio_chunk_params=AudioChunkParams(),
        asr_params=ASRParams(),
        video_frame_params=VideoFrameParams(),
        video_text_dedup_params=VideoFrameTextDedupParams(),
        av_fuse_params=AudioVisualFuseParams(),
        split_config=cfg,
    )
    names = _node_names(graph)
    assert "VideoSplitActor" in names
    assert "TextChunkActor" in names
    assert names.index("TextChunkActor") > names.index("VideoSplitActor")


def test_split_method_removed_from_graph_ingestor():
    """Tombstone — calling ingestor.split should no longer resolve."""
    ingestor = GraphIngestor(run_mode="inprocess")
    assert not hasattr(ingestor, "split")
