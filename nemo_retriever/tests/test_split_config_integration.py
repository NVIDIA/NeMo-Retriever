# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for split_config flowing through build_graph and the ingestor."""

from __future__ import annotations

import pytest

from nemo_retriever.common.modality.audio.media_interface import is_media_available
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractCPUActor
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor
from nemo_retriever.common.params import EmbedParams, ExtractParams
from nemo_retriever.common.params import resolve_split_params


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


@pytest.mark.parametrize("method", ["pdfium_hybrid", "ocr"])
def test_pdf_page_embedding_collapses_after_configured_text_split(method):
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(method=method),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            embed_invoke_url="http://embed.example/v1",
            embed_granularity="page",
        ),
        split_config=resolve_split_params({"pdf": {"max_tokens": 1}}),
    )
    names = _node_names(graph)
    assert names.index("TextChunkActor") < names.index("CollapseContentToPageRows")
    assert names.index("CollapseContentToPageRows") < names.index("_BatchEmbedActor")


@pytest.mark.parametrize("with_table", [False, True])
def test_pdf_split_page_embeddings_keep_chunk_text_without_restoring_whole_page(monkeypatch, with_table):
    import pandas as pd

    from nemo_retriever.common.modality.content_transforms import collapse_content_to_page_rows
    from nemo_retriever.common.modality.page_postprocessing import finalize_page_regions
    from nemo_retriever.operators.extract.txt.ray_data import TextChunkCPUActor

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids)

    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda model_id, cache_dir=None: Tokenizer()
    )
    table = [{"text": "TABLE", "caption": "table caption", "bbox_xyxy_norm": [0, 0.4, 1, 0.5]}] if with_table else []
    pages = finalize_page_regions(
        pd.DataFrame(
            [
                {
                    "text": "first last",
                    "page_number": 1,
                    "metadata": {},
                    "ocr_text_regions": [
                        {"text": "first", "bbox_xyxy_norm": [0, 0.1, 1, 0.2]},
                        {"text": "last", "bbox_xyxy_norm": [0, 0.8, 1, 0.9]},
                    ],
                    "table": table,
                },
                {"text": "unsplit", "page_number": 2, "metadata": {}, "table": []},
            ]
        )
    )
    split_config = resolve_split_params({"pdf": {"max_tokens": 1}})
    chunks = TextChunkCPUActor(split_config["pdf"])(pages)
    assert chunks.text.tolist() == ["first", "last", "unsplit"]

    embedded = collapse_content_to_page_rows(chunks)
    first = "first\n\nTABLE\n\ntable caption" if with_table else "first"
    assert embedded.text.tolist() == [first, "last", "unsplit"]
    assert embedded.page_number.tolist() == [1, 1, 2]
    assert embedded.iloc[0].metadata["chunk_count"] == 2
    assert embedded.iloc[1].metadata["chunk_index"] == 1
    assert embedded.iloc[1].table == []
    assert embedded.iloc[2].ordered_page_text == "unsplit"
    assert pages.iloc[0].ordered_page_text == ("first\nTABLE\nlast" if with_table else "first\nlast")


def test_multi_type_extract_operator_uses_text_split_from_split_config():
    cfg = resolve_split_params({"text": {"max_tokens": 64}})
    op = MultiTypeExtractCPUActor(extraction_mode="auto", split_config=cfg)
    text_chunk_params = op._effective_chunk_params("text")
    assert text_chunk_params is not None
    assert text_chunk_params.max_tokens == 64


@pytest.mark.skipif(not is_media_available(), reason="ffmpeg not available")
def test_build_graph_audio_only_path_appends_text_chunk_actor_when_audio_split_set():
    from nemo_retriever.common.params import ASRParams

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
    from nemo_retriever.common.params import (
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
