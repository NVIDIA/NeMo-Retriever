# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the README.md video (MP4) Python snippet in sync with the graph builder."""

from __future__ import annotations

from nemo_retriever import create_ingestor
from nemo_retriever.audio.asr_actor import ASRActor
from nemo_retriever.audio.chunk_actor import MediaChunkActor
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.params import ASRParams, AudioChunkParams, EmbedParams
from nemo_retriever.text_embed.operators import _BatchEmbedActor


def _readme_mp4_video_ingestor(video_paths: list[str], *, run_mode: str = "batch") -> GraphIngestor:
    """Must match README.md (inline snippet: MP4 inputs, chunk + ASR + embed)."""
    return create_ingestor(run_mode=run_mode).files(video_paths).extract_audio().embed()


def _linear_node_names(graph) -> list[str]:
    node = graph.roots[0]
    names: list[str] = []
    while True:
        names.append(node.name)
        if not node.children:
            return names
        node = node.children[0]


def test_readme_mp4_video_example_matches_readme() -> None:
    """Same construction as README.md \"Video (MP4 files)\" (ffmpeg chunk + ASR)."""
    video_paths = ["/data/example.mp4"]
    ingestor = _readme_mp4_video_ingestor(video_paths, run_mode="inprocess")

    assert ingestor._extraction_mode == "audio"
    assert ingestor._documents == video_paths
    assert isinstance(ingestor._audio_chunk_params, AudioChunkParams)
    assert isinstance(ingestor._asr_params, ASRParams)
    assert isinstance(ingestor._embed_params, EmbedParams)

    post_extract = tuple(s for s in ingestor._stage_order if s != "extract")
    graph = build_graph(
        extraction_mode=ingestor._extraction_mode,
        extract_params=ingestor._extract_params,
        audio_chunk_params=ingestor._audio_chunk_params,
        asr_params=ingestor._asr_params,
        embed_params=ingestor._embed_params,
        stage_order=post_extract,
    )
    names = _linear_node_names(graph)
    assert names[0] == MediaChunkActor.__name__
    assert names[1] == ASRActor.__name__
    assert names[2] == _BatchEmbedActor.__name__
