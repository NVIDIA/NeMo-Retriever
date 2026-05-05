# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""README.md — Video with GraphIngestor (extract_video, ASR, OCR frames, embed, VDB)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_retriever.audio import asr_params_from_env, AudioChunkParams
from nemo_retriever.audio.asr_actor import ASRActor
from nemo_retriever.audio.chunk_actor import MediaChunkActor
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.model import VL_EMBED_MODEL
from nemo_retriever.params import EmbedParams, ExtractParams
from nemo_retriever.text_embed.operators import _BatchEmbedActor
from nemo_retriever.vdb import IngestVdbOperator


def _linear_node_names(graph) -> list[str]:
    node = graph.roots[0]
    names: list[str] = []
    while True:
        names.append(node.name)
        if not node.children:
            return names
        node = node.children[0]


def _extract_video_or_audio(ingestor: GraphIngestor):
    """README compatibility: prefer extract_video when the class defines it."""
    return getattr(ingestor, "extract_video", ingestor.extract_audio)


def test_readme_video_extract_media_then_embed_graph(tmp_path: Path) -> None:
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"")
    embed_params = EmbedParams(model_name=VL_EMBED_MODEL)
    ing = GraphIngestor(run_mode="inprocess").files([str(mp4)])
    extract_fn = _extract_video_or_audio(ing)
    ingestor = extract_fn(
        params=AudioChunkParams(split_type="size", split_interval=500_000),
        asr_params=asr_params_from_env(),
    ).embed(embed_params)

    assert ingestor._extraction_mode in {"audio", "video"}
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


def test_readme_frames_ocr_then_embed_graph() -> None:
    embed_params = EmbedParams(
        model_name=VL_EMBED_MODEL,
        embed_invoke_url="http://embed.example/v1",
    )
    extract_params = ExtractParams(
        method="ocr",
        dpi=300,
        extract_text=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
        page_elements_invoke_url="http://page.example/v1",
        ocr_invoke_url="http://ocr.example/v1",
    )
    graph = build_graph(
        extraction_mode="image",
        extract_params=extract_params,
        embed_params=embed_params,
        stage_order=("embed",),
    )
    names = _linear_node_names(graph)
    assert names[0] == "MultiTypeExtractOperator"
    assert names[-1] == _BatchEmbedActor.__name__


def test_readme_vdb_upload_invokes_backend_like_main() -> None:
    mock_vdb = MagicMock()
    records: list[dict[str, object]] = []

    def _fake_get_vdb_op_cls(_op: str):
        def _ctor(**_kwargs: object) -> MagicMock:
            return mock_vdb

        return _ctor

    with patch("nemo_retriever.vdb.operators.get_vdb_op_cls", side_effect=_fake_get_vdb_op_cls):
        op = IngestVdbOperator(vdb_op="lancedb", vdb_kwargs={"uri": "lancedb", "table_name": "nv-ingest"})
        op(records)

    mock_vdb.run.assert_called_once()


def test_readme_extract_video_method_exists_or_documented_fallback() -> None:
    assert hasattr(GraphIngestor, "extract_audio")
    if not hasattr(GraphIngestor, "extract_video"):
        pytest.skip("GraphIngestor.extract_video not defined; README documents extract_audio fallback")


def test_readme_video_example_speech_embed_then_vdb_matches_readme_block(tmp_path: Path) -> None:
    """Simplified README block: media ingest -> DataFrame -> records -> IngestVdbOperator.

    Mirrors the active lines in README.md (``extract_video`` / ``extract_audio`` fallback,
    ``embed``, ``ingest``, ``to_dict``, VDB) without requiring ffmpeg: executor ingest is mocked.
    """
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"")
    embed_params = EmbedParams(model_name=VL_EMBED_MODEL)

    ingestor_video = GraphIngestor(run_mode="inprocess").files([str(mp4)])
    extract_fn = _extract_video_or_audio(ingestor_video)
    ingestor_video = extract_fn(
        params=AudioChunkParams(split_type="size", split_interval=500_000),
        asr_params=asr_params_from_env(),
    ).embed(embed_params)

    sample_df = pd.DataFrame([{"col": "row"}])
    with patch("nemo_retriever.graph_ingestor.InprocessExecutor") as mock_exec_cls:
        mock_exec_cls.return_value.ingest.return_value = sample_df
        df_speech = ingestor_video.ingest()

    assert isinstance(df_speech, pd.DataFrame)
    records = df_speech.to_dict("records")
    assert records == [{"col": "row"}]

    mock_vdb = MagicMock()

    def _fake_get_vdb_op_cls(_op: str):
        def _ctor(**_kwargs: object) -> MagicMock:
            return mock_vdb

        return _ctor

    with patch("nemo_retriever.vdb.operators.get_vdb_op_cls", side_effect=_fake_get_vdb_op_cls):
        IngestVdbOperator(
            vdb_op="lancedb",
            vdb_kwargs={"uri": "lancedb", "table_name": "nv-ingest"},
        )(records)

    mock_vdb.run.assert_called_once()
