# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video (frame) OCR ingestor — README and tests share this entry point.

This mirrors ``nemo_retriever.pipeline.__main__._build_ingestor`` when the
pipeline CLI is used with ``--input-type image`` and ``--method ocr``: raster
images go through page-element detection and Nemotron OCR, then optional embed
stages. For video, decode frames with ffmpeg and pass glob patterns to
``build_video_ocr_ingestor``.
"""

from __future__ import annotations

from typing import List, Union

from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.ingestor import create_ingestor
from nemo_retriever.params import ExtractParams


def build_video_ocr_ingestor(
    frame_globs: Union[str, List[str]],
    *,
    run_mode: str = "batch",
) -> GraphIngestor:
    """Build a graph ingestor that runs the image OCR pipeline on frame paths.

    Parameters
    ----------
    frame_globs
        One or more filesystem globs (e.g. ``\"/tmp/frames/*.png\"``) for raster
        images produced from a video or other source.
    run_mode
        ``\"batch\"`` (Ray Data) or ``\"inprocess\"`` (pandas), same as
        ``nemo_retriever.ingestor.create_ingestor``.

    Returns
    -------
    GraphIngestor
        Configured with ``extract_image_files`` (``method=\"ocr\"``) and
        ``embed()`` using default ``EmbedParams``, matching the quick-start PDF
        example style.
    """
    patterns = [frame_globs] if isinstance(frame_globs, str) else list(frame_globs)

    extract_params = ExtractParams(
        method="ocr",
        dpi=300,
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )
    return create_ingestor(run_mode=run_mode).files(patterns).extract_image_files(extract_params).embed()
