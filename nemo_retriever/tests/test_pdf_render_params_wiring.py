# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for forwarding public PDF rendering parameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nemo_retriever import create_ingestor
from nemo_retriever.common.params import EmbedParams, ExtractParams
from nemo_retriever.graph.graph_pipeline_registry import get_node_kwargs
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.operators.graph_ops import multi_type_extract_operator as multi_type_module
from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractCPUActor


def _render_params() -> ExtractParams:
    return ExtractParams(
        extract_tables=False,
        extract_charts=False,
        render_mode="full_dpi",
        image_format="png",
        jpeg_quality=73,
    )


def test_pdf_graph_forwards_rendering_params() -> None:
    graph = build_graph(extraction_mode="pdf", extract_params=_render_params())

    extract_kwargs = get_node_kwargs(graph, "PDFExtractionActor")

    assert extract_kwargs["render_mode"] == "full_dpi"
    assert extract_kwargs["image_format"] == "png"
    assert extract_kwargs["jpeg_quality"] == 73


class _IdentityStage:
    def __init__(self, **_kwargs) -> None:
        pass

    def run(self, data):
        return data


def test_multitype_pdf_graph_forwards_rendering_params(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _CapturingPDFExtractionActor(_IdentityStage):
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(multi_type_module, "DocToPdfConversionActor", _IdentityStage)
    monkeypatch.setattr(multi_type_module, "PDFSplitActor", _IdentityStage)
    monkeypatch.setattr(multi_type_module, "PDFExtractionActor", _CapturingPDFExtractionActor)

    operator = MultiTypeExtractCPUActor(extraction_mode="pdf", extract_params=_render_params())
    monkeypatch.setattr(operator, "_run_detection_pipeline", lambda data: data)
    monkeypatch.setattr(operator, "_maybe_chunk", lambda data, _mode: data)

    operator._run_pdf_pipeline(pd.DataFrame())

    assert captured["render_mode"] == "full_dpi"
    assert captured["image_format"] == "png"
    assert captured["jpeg_quality"] == 73


@pytest.mark.parametrize("modality", ["image", "text_image"])
def test_sdk_page_image_embedding_materializes_pdf_rasters_and_vectors(monkeypatch, modality: str) -> None:
    class _FakeVLEmbedder:
        def embed_images(self, images: list[str], batch_size: int = 8) -> np.ndarray:
            assert all(isinstance(image, str) and image for image in images)
            return np.ones((len(images), 2048), dtype=np.float32)

        def embed_text_image(self, texts: list[str], images: list[str], batch_size: int = 8) -> np.ndarray:
            assert all(isinstance(text, str) and text for text in texts)
            assert all(isinstance(image, str) and image for image in images)
            return np.ones((len(images), 2048), dtype=np.float32)

    monkeypatch.setattr(
        "nemo_retriever.models.create_local_embedder",
        lambda *_args, **_kwargs: _FakeVLEmbedder(),
    )
    document = Path(__file__).resolve().parents[2] / "data" / "multimodal_test.pdf"

    result = (
        create_ingestor(run_mode="inprocess", allow_no_gpu=True)
        .files([str(document)])
        .extract(
            ExtractParams(
                extract_images=False,
                extract_tables=False,
                extract_charts=False,
                extract_page_as_image=False,
                use_page_elements=False,
            )
        )
        .embed(
            EmbedParams(
                model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
                embed_model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
                local_ingest_embed_backend="hf",
                embed_modality=modality,
                embed_granularity="page",
            )
        )
        .ingest()
    )

    assert len(result) == 3
    assert (
        result["page_image"]
        .map(lambda page_image: isinstance(page_image, dict) and bool(page_image.get("image_b64")))
        .all()
    )
    assert result["_image_b64"].map(lambda image_b64: isinstance(image_b64, str) and bool(image_b64)).all()
    assert result["_contains_embeddings"].all()
    assert result["text_embeddings_1b_v2_has_embedding"].all()
    assert (result["text_embeddings_1b_v2_dim"] == 2048).all()
