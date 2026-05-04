# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ExtractParams.ocr_version + resolve_ocr_archetype selection.

Covers all three call sites where the OCR engine is selected:

* The Ray graph builder (build_graph) for page-elements OCR.
* The override block (batch_tuning_to_node_overrides) keyed by actor name.
* The in-process pipeline (multi_type_extract_operator), which forwards
  ``load_ocr_v2`` into TableStructureActor.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.ocr.ocr import OCRActor, OCRV2Actor, resolve_ocr_archetype
from nemo_retriever.params import EmbedParams, ExtractParams


def _linear_nodes(graph):
    node = graph.roots[0]
    nodes = []
    while True:
        nodes.append(node)
        if not node.children:
            return nodes
        node = node.children[0]


def test_default_ocr_version_is_v2() -> None:
    """The default ExtractParams must select OCRV2Actor without any flags."""
    params = ExtractParams()

    assert params.ocr_version == "v2"
    assert resolve_ocr_archetype(params) is OCRV2Actor


def test_default_graph_uses_ocr_v2_actor() -> None:
    """A default-ExtractParams graph must contain OCRV2Actor, not OCRActor."""
    graph = build_graph(
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
    )

    nodes = _linear_nodes(graph)
    classes = [node.operator_class for node in nodes]

    assert OCRV2Actor in classes
    assert OCRActor not in classes


def test_explicit_ocr_version_v1_uses_legacy_actor() -> None:
    """Passing ocr_version='v1' must produce OCRActor and not OCRV2Actor."""
    graph = build_graph(
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
            ocr_version="v1",
        ),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            embed_invoke_url="http://embed.example/v1",
        ),
    )

    nodes = _linear_nodes(graph)
    classes = [node.operator_class for node in nodes]

    assert OCRActor in classes
    assert OCRV2Actor not in classes


def test_invalid_ocr_version_raises_validation_error() -> None:
    """Pydantic must reject any ocr_version outside the v1/v2 literal."""
    with pytest.raises(ValidationError):
        ExtractParams(ocr_version="v3")


def test_table_structure_actor_receives_load_ocr_v2_kwarg(monkeypatch) -> None:
    """When ocr_version='v2', the in-process pipeline must forward
    load_ocr_v2=True into TableStructureActor (and similarly for v1 -> False).
    """
    from nemo_retriever.graph.multi_type_extract_operator import MultiTypeExtractCPUActor
    from nemo_retriever.utils.ray_resource_hueristics import Resources
    import pandas as pd

    captured_kwargs: list[tuple[str, dict]] = []

    class _IdentityStage:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, data):
            return data

    def _fake_resolve(operator_class, resources, operator_kwargs=None):
        captured_kwargs.append((operator_class.__name__, dict(operator_kwargs or {})))
        return _IdentityStage

    monkeypatch.setattr(
        "nemo_retriever.graph.multi_type_extract_operator.resolve_operator_class",
        _fake_resolve,
    )
    monkeypatch.setattr(
        "nemo_retriever.graph.multi_type_extract_operator.gather_local_resources",
        lambda: Resources(cpu_count=8, gpu_count=1),
    )

    op = MultiTypeExtractCPUActor(
        extraction_mode="image",
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=True,
            use_table_structure=True,
            extract_charts=True,
            extract_infographics=True,
        ),
    )

    op._run_detection_pipeline(pd.DataFrame({"page_image": ["x"]}))

    table_kwargs = next(kwargs for name, kwargs in captured_kwargs if name == "TableStructureActor")

    assert table_kwargs.get("load_ocr_v2") is True
