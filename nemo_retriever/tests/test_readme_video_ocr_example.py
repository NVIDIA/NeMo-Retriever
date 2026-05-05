# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the README.md \"Video (visual OCR on frames)\" Python snippet in sync with the graph builder."""

from __future__ import annotations

from nemo_retriever import create_ingestor
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.params import EmbedParams, ExtractParams


def _readme_video_ocr_ingestor(frame_globs: list[str], *, run_mode: str = "batch") -> GraphIngestor:
    """Must match README.md (inline snippet after ffmpeg frame extraction)."""
    extract_params = ExtractParams(
        method="ocr",
        dpi=300,
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )
    return create_ingestor(run_mode=run_mode).files(frame_globs).extract_image_files(extract_params).embed()


def test_readme_video_ocr_example_matches_readme_and_pipeline_image_ocr() -> None:
    """Same construction as README.md \"Video (visual OCR on frames)\" and ``--input-type image --method ocr``."""
    frame_globs = ["/tmp/video_frames/*.png"]
    ingestor = _readme_video_ocr_ingestor(frame_globs, run_mode="inprocess")

    assert ingestor._extraction_mode == "image"
    assert ingestor._documents == frame_globs
    ep = ingestor._extract_params
    assert isinstance(ep, ExtractParams)
    assert ep.method == "ocr"
    assert ep.dpi == 300
    assert ep.extract_text is True
    assert ep.extract_tables is True
    assert ep.extract_charts is True
    assert ep.extract_infographics is True
    assert isinstance(ingestor._embed_params, EmbedParams)

    post_extract = tuple(s for s in ingestor._stage_order if s != "extract")
    graph = build_graph(
        extraction_mode=ingestor._extraction_mode,
        extract_params=ingestor._extract_params,
        embed_params=ingestor._embed_params,
        stage_order=post_extract,
    )
    assert graph.roots[0].name == "MultiTypeExtractOperator"
