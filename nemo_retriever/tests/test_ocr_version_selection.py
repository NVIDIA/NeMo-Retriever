# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ExtractParams OCR selectors.

Covers all three call sites where the OCR engine is selected:

* The Ray graph builder (build_graph) for page-elements OCR.
* The override block (batch_tuning_to_node_overrides) keyed by actor name.
* The in-process pipeline (multi_type_extract_operator), which forwards
  ``ocr_version`` and ``ocr_lang`` into TableStructureActor and
  GraphicElementsActor.
"""

from __future__ import annotations

from typing import get_type_hints

import pytest
from pydantic import ValidationError

from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.ocr import config as ocr_config
from nemo_retriever.ocr.config import OCRLang, OCRVersion, resolve_ocr_v2_lang, resolve_ocr_v2_model_dir
from nemo_retriever.ocr.ocr import OCRActor, resolve_ocr_archetype
from nemo_retriever.params import (
    AudioChunkParams,
    AudioVisualFuseParams,
    EmbedParams,
    ExtractParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
)
from nemo_retriever.utils.ray_resource_hueristics import Resources
from nemo_retriever.video.ocr_actor import VideoFrameOCRActor


def _linear_nodes(graph):
    node = graph.roots[0]
    nodes = []
    while True:
        nodes.append(node)
        if not node.children:
            return nodes
        node = node.children[0]


def test_default_ocr_version_is_v2() -> None:
    """The default ExtractParams must select the unified OCRActor."""
    params = ExtractParams()

    assert params.ocr_version == "v2"
    assert resolve_ocr_archetype(params) is OCRActor
    assert params.ocr_lang is None


def test_v2_accepts_supported_ocr_lang_values() -> None:
    assert ExtractParams(ocr_lang="multi").ocr_lang == "multi"
    assert ExtractParams(ocr_lang="english").ocr_lang == "english"


def test_invalid_ocr_lang_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        ExtractParams(ocr_lang="legacy")


def test_v1_rejects_ocr_lang() -> None:
    with pytest.raises(ValidationError):
        ExtractParams(ocr_version="v1", ocr_lang="english")


def test_ocr_v2_model_dir_ignores_legacy_v1_env_var() -> None:
    assert resolve_ocr_v2_model_dir({"NEMOTRON_OCR_V1_MODEL_DIR": "/models/ocr-v1"}) == ""


def test_ocr_v2_model_dir_accepts_v2_compatible_env_vars() -> None:
    assert (
        resolve_ocr_v2_model_dir(
            {
                "NEMOTRON_OCR_V1_MODEL_DIR": "/models/ocr-v1",
                "NEMOTRON_OCR_V2_MODEL_DIR": "/models/ocr-v2",
            }
        )
        == "/models/ocr-v2"
    )


def test_resolve_ocr_v2_model_dir_is_public_api() -> None:
    assert "resolve_ocr_v2_model_dir" in ocr_config.__all__


def test_resolve_ocr_v2_lang_uses_public_selector_aliases() -> None:
    hints = get_type_hints(resolve_ocr_v2_lang)

    assert hints["ocr_version"] == OCRVersion
    assert hints["ocr_lang"] == OCRLang | None


def test_default_graph_uses_unified_ocr_actor() -> None:
    """A default-ExtractParams graph must contain the unified OCRActor."""
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
    ocr_node = next(node for node in nodes if node.operator_class is OCRActor)

    assert OCRActor in classes
    assert ocr_node.operator_kwargs["ocr_version"] == "v2"
    assert "ocr_lang" not in ocr_node.operator_kwargs


def test_explicit_ocr_version_v1_uses_unified_actor_with_legacy_selector() -> None:
    """Passing ocr_version='v1' must keep OCRActor and pass the legacy selector."""
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
    ocr_node = next(node for node in nodes if node.operator_class is OCRActor)

    assert OCRActor in classes
    assert ocr_node.operator_kwargs["ocr_version"] == "v1"


def test_graph_forwards_v2_ocr_lang_selector() -> None:
    graph = build_graph(
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
            ocr_lang="english",
        ),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            embed_invoke_url="http://embed.example/v1",
        ),
    )

    ocr_node = next(node for node in _linear_nodes(graph) if node.operator_class is OCRActor)

    assert ocr_node.operator_kwargs["ocr_version"] == "v2"
    assert ocr_node.operator_kwargs["ocr_lang"] == "english"


def test_video_graph_forwards_v2_ocr_lang_selector(monkeypatch) -> None:
    from nemo_retriever.graph.abstract_operator import AbstractOperator

    class _GraphOnlyVideoSplitActor(AbstractOperator):
        def preprocess(self, data, **kwargs):
            return data

        def process(self, data, **kwargs):
            return data

        def postprocess(self, data, **kwargs):
            return data

    monkeypatch.setattr(
        "nemo_retriever.graph.ingestor_runtime.VideoSplitActor",
        _GraphOnlyVideoSplitActor,
    )

    graph = build_graph(
        extraction_mode="video",
        extract_params=ExtractParams(ocr_lang="english"),
        audio_chunk_params=AudioChunkParams(enabled=False),
        video_frame_params=VideoFrameParams(enabled=True),
    )

    ocr_node = next(node for node in _linear_nodes(graph) if node.operator_class is VideoFrameOCRActor)

    assert ocr_node.operator_kwargs["ocr_version"] == "v2"
    assert ocr_node.operator_kwargs["ocr_lang"] == "english"


def test_video_inprocess_pipeline_forwards_v2_ocr_lang_selector(monkeypatch) -> None:
    from nemo_retriever.graph import multi_type_extract_operator as multi_type_module
    from nemo_retriever.graph.multi_type_extract_operator import MultiTypeExtractCPUActor
    import pandas as pd

    captured_kwargs: list[tuple[str, dict]] = []

    class _FakeFrameActor:
        def __init__(self, **_kwargs):
            pass

        def run(self, _data):
            return pd.DataFrame(
                {
                    "path": ["/tmp/frame_0.png"],
                    "source_path": ["/tmp/video.mp4"],
                    "image_b64": ["frame"],
                    "_content_type": ["video_frame"],
                }
            )

    class _IdentityStage:
        def run(self, data):
            return data

    def _fake_instantiate_resolved(self, operator_class, **operator_kwargs):
        captured_kwargs.append((operator_class.__name__, dict(operator_kwargs)))
        return _IdentityStage()

    monkeypatch.setattr(multi_type_module, "VideoFrameActor", _FakeFrameActor)
    monkeypatch.setattr(MultiTypeExtractCPUActor, "_instantiate_resolved", _fake_instantiate_resolved)

    op = MultiTypeExtractCPUActor(
        extraction_mode="video",
        extract_params=ExtractParams(ocr_lang="english"),
        audio_chunk_params=AudioChunkParams(enabled=False),
        video_frame_params=VideoFrameParams(enabled=True, dedup=False),
        video_text_dedup_params=VideoFrameTextDedupParams(enabled=False),
        av_fuse_params=AudioVisualFuseParams(enabled=False),
        split_config={},
    )

    op._run_video_pipeline(pd.DataFrame({"path": ["/tmp/video.mp4"]}))

    video_ocr_kwargs = next(kwargs for name, kwargs in captured_kwargs if name == "VideoFrameOCRActor")

    assert video_ocr_kwargs["ocr_version"] == "v2"
    assert video_ocr_kwargs["ocr_lang"] == "english"


def test_resolved_remote_ocr_stages_drop_local_selector_kwargs() -> None:
    graph = build_graph(
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=True,
            use_table_structure=True,
            extract_charts=True,
            use_graphic_elements=True,
            extract_infographics=False,
            page_elements_invoke_url="http://page.example/v1",
            ocr_invoke_url="http://ocr.example/v1",
            table_structure_invoke_url="http://table.example/v1",
            graphic_elements_invoke_url="http://graphic.example/v1",
            ocr_lang="english",
        ),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            embed_invoke_url="http://embed.example/v1",
        ),
    )

    resolved = graph.resolve(Resources(cpu_count=8, gpu_count=4))
    nodes = {node.name: node for node in _linear_nodes(resolved)}

    for name in ("TableStructureActor", "GraphicElementsActor", "OCRActor"):
        assert nodes[name].operator_class.__name__.endswith("CPUActor")
        assert "ocr_version" not in nodes[name].operator_kwargs
        assert "ocr_lang" not in nodes[name].operator_kwargs

    assert nodes["TableStructureActor"].operator_kwargs["ocr_invoke_url"] == "http://ocr.example/v1"
    assert nodes["GraphicElementsActor"].operator_kwargs["ocr_invoke_url"] == "http://ocr.example/v1"
    assert nodes["OCRActor"].operator_kwargs["ocr_invoke_url"] == "http://ocr.example/v1"


def test_invalid_ocr_version_raises_validation_error() -> None:
    """Pydantic must reject any ocr_version outside the v1/v2 literal."""
    with pytest.raises(ValidationError):
        ExtractParams(ocr_version="v3")


def test_table_structure_actor_receives_ocr_selectors(monkeypatch) -> None:
    """The in-process pipeline must forward OCR selectors to OCR-using stages."""
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
            use_graphic_elements=True,
            extract_infographics=True,
            ocr_lang="english",
        ),
    )

    op._run_detection_pipeline(pd.DataFrame({"page_image": ["x"]}))

    table_kwargs = next(kwargs for name, kwargs in captured_kwargs if name == "TableStructureActor")
    graphic_kwargs = next(kwargs for name, kwargs in captured_kwargs if name == "GraphicElementsActor")

    assert table_kwargs.get("ocr_version") == "v2"
    assert graphic_kwargs.get("ocr_version") == "v2"
    assert table_kwargs.get("ocr_lang") == "english"
    assert graphic_kwargs.get("ocr_lang") == "english"
