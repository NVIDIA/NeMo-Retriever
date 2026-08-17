# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for forwarding public PDF rendering parameters."""

from __future__ import annotations

import pandas as pd

from nemo_retriever.common.params import ExtractParams
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
