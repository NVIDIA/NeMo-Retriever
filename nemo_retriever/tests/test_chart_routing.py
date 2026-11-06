# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chart routing preserves OCR content outside detector-selected regions."""

import base64
from copy import deepcopy
import io
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from nemo_retriever.common.modality import chart_routing as routing
from nemo_retriever.common.modality.parse import nemotron_parse35_postprocessing as postprocessing
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.operators.extract.parse.chart_routing import (
    ChartRoutingActor,
    ChartRoutingCPUActor,
    ChartRoutingGPUActor,
)

BOX = [0.2, 0.1, 0.7, 0.6]
URL = "http://localhost:8000/v1/chat/completions"


def frame():
    image = Image.fromarray(np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    regions = [
        {"label_name": "text", "bbox_xyxy_norm": [0, 0, 1, 0.1], "text": "Before"},
        {"label_name": "chart", "bbox_xyxy_norm": BOX, "text": "old chart"},
        {"label_name": "text", "bbox_xyxy_norm": [0, 0.8, 1, 1], "text": "After"},
    ]
    return pd.DataFrame(
        [
            {
                "text": "Before\nAfter",
                "page_image": {"image_b64": base64.b64encode(buffer.getvalue()).decode()},
                "page_elements_v3": {
                    "detections": [
                        {"label_name": "chart", "bbox_xyxy_norm": BOX},
                        {"label_name": "table", "bbox_xyxy_norm": [0, 0, 1, 1]},
                    ]
                },
                "chart": [{"bbox_xyxy_norm": BOX, "text": "old chart"}],
                "table": [{"text": "untouched table"}],
                "infographic": [{"text": "untouched graphic"}],
                "ordered_regions": regions,
                "ordered_page_text": "Before\nold chart\nAfter",
            }
        ],
        index=[7],
    )


@pytest.fixture
def helpers(monkeypatch):
    blocks = [
        {"class": "Text", "bbox": [0, 0, 1, 1], "text": "Title", "content": "Title"},
        {
            "class": "Table",
            "bbox": [0, 0, 1, 1],
            "text": "table",
            "content": '<table><tr><td rowspan="1" colspan="2">Numbers</td></tr></table>',
        },
        {"class": "Chart", "bbox": [0, 0, 1, 1], "text": "chart", "content": '<td colspan="1">Chart</td>'},
    ]
    value = SimpleNamespace(postprocess_document=Mock(side_effect=lambda raw: {"blocks": deepcopy(blocks)}))
    monkeypatch.setattr(postprocessing, "_load_parse35_helpers", lambda: value)
    return value


def test_replaces_only_chart_and_maintains_order_and_crop_pixels(helpers):
    original = frame()
    before = deepcopy(original.iloc[0].to_dict())
    client = Mock()
    client.invoke_chat_completions_images.return_value = ["raw output"]
    out = routing.route_chart_regions(original, invoke_url=URL, nim_client=client)
    row = out.iloc[0]
    assert original.iloc[0].to_dict() == before
    assert out.index.tolist() == [7]
    for key in ["text", "table", "infographic", "page_image", "page_elements_v3"]:
        assert row[key] == before[key]
    text = row["chart"][0]["text"]
    assert text == 'Title\n<table><tr><td colspan="2">Numbers</td></tr></table>\n<td colspan="1">Chart</td>'
    assert row["chart"][0]["bbox_xyxy_norm"] == BOX
    assert row["ordered_page_text"] == "Before\n" + text + "\nAfter"
    args = client.invoke_chat_completions_images.call_args.kwargs
    assert args["temperature"] == 0.1 and args["repetition_penalty"] == 1.0
    assert args["extra_body"] == dict(top_k=50, top_p=1.0, seed=0, max_tokens=8744, skip_special_tokens=False)
    crop = Image.open(io.BytesIO(base64.b64decode(args["image_b64_list"][0])))
    page = Image.open(io.BytesIO(base64.b64decode(before["page_image"]["image_b64"])))
    assert np.array_equal(np.asarray(crop), np.asarray(page.crop((6, 2, 21, 12))))
    assert row["chart_routing"]["regions"][0]["raw_output"] == "raw output"


def test_picture_with_no_text_clears_original_chart(helpers):
    helpers.postprocess_document.side_effect = lambda raw: {
        "blocks": [{"class": "Picture", "bbox": [0, 0, 1, 1], "text": "", "content": ""}]
    }
    client = Mock()
    client.invoke_chat_completions_images.return_value = ["picture"]
    row = routing.route_chart_regions(frame(), invoke_url=URL, nim_client=client).iloc[0]
    assert row["chart"][0]["text"] == ""
    assert row["ordered_page_text"] == "Before\nAfter"
    assert row["chart_routing"]["regions"][0]["error"] is None


@pytest.mark.parametrize("failure", ["transport", "postprocess", "count"])
def test_failures_retain_original_chart(helpers, failure):
    client = Mock()
    client.invoke_chat_completions_images.return_value = ["raw"]
    if failure == "transport":
        client.invoke_chat_completions_images.side_effect = TimeoutError("service timed out")
    elif failure == "postprocess":
        helpers.postprocess_document.side_effect = ValueError("bad table")
    else:
        client.invoke_chat_completions_images.return_value = []
    source = frame()
    out = routing.route_chart_regions(source, invoke_url=URL, nim_client=client)
    assert out.iloc[0]["chart"] == source.iloc[0]["chart"]
    assert out.iloc[0]["ordered_page_text"] == source.iloc[0]["ordered_page_text"]
    assert out.iloc[0]["chart_routing"]["regions"][0]["error"]


def test_no_charts_does_not_invoke_or_require_model():
    source = frame()
    source.iloc[0]["page_elements_v3"]["detections"] = []
    client = Mock()
    out = routing.route_chart_regions(source, nim_client=client)
    client.invoke_chat_completions_images.assert_not_called()
    assert out.iloc[0]["chart"] == source.iloc[0]["chart"]


def test_arrow_style_arrays_and_missing_ocr_entry(helpers):
    source = frame()
    source.at[7, "chart"] = np.array([], dtype=object)
    source.at[7, "page_elements_v3"] = {
        "detections": np.array([{"label_name": "chart", "bbox_xyxy_norm": np.array(BOX)}], dtype=object)
    }
    model = Mock()
    model.invoke_batch.return_value = ["raw"]
    out = routing.route_chart_regions(source, model=model)
    assert out.iloc[0]["chart"][0]["bbox_xyxy_norm"] == BOX
    assert model.invoke_batch.call_args.args[0][0].shape == (10, 15, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "nemotron_parse"},
        {"method": "pdfium"},
        {"use_page_elements": False},
        {"extract_charts": False},
        {"extract_page_as_image": False},
        {"nemotron_parse_model": "nvidia/nemotron-parse-v1.2"},
        {"nemotron_parse_invoke_url": "http://localhost:8000/v1/parse"},
    ],
)
def test_invalid_routing_configuration_fails_early(kwargs):
    with pytest.raises(ValueError, match="chart_routing"):
        ExtractParams(chart_routing=True, **kwargs)


def test_default_and_endpoint_selection():
    assert ExtractParams().chart_routing is False
    params = ExtractParams(method="ocr", chart_routing=True, nemotron_parse_invoke_url=URL)
    assert params.nemotron_parse_invoke_url == URL
    assert ChartRoutingActor.prefers_cpu_variant({"nemotron_parse_invoke_url": URL})
    assert not ChartRoutingActor.prefers_cpu_variant({})
    assert ChartRoutingActor.cpu_variant_class() is ChartRoutingCPUActor
    assert ChartRoutingActor.gpu_variant_class() is ChartRoutingGPUActor


@pytest.mark.parametrize("mode", ["pdf", "image", "auto"])
def test_public_graph_keeps_ocr_and_wires_routing_for_all_page_inputs(mode):
    from nemo_retriever.graph.ingestor_runtime import build_graph
    from nemo_retriever.graph.graph_pipeline_registry import get_node_kwargs
    from nemo_retriever.operators.graph_ops.multi_type_extract_operator import _parse_mode_enabled

    params = ExtractParams(method="ocr", chart_routing=True, nemotron_parse_invoke_url=URL)
    assert not _parse_mode_enabled(params)
    graph = build_graph(extraction_mode=mode, extract_params=params)
    if mode == "pdf":
        kwargs = get_node_kwargs(graph, "ChartRoutingActor")
        assert kwargs["nemotron_parse_invoke_url"] == URL
        assert get_node_kwargs(graph, "OCRActor")["extract_charts"] is True
    else:
        assert get_node_kwargs(graph, "MultiTypeExtractOperator")["extract_params"].chart_routing is True


def test_multitype_stage_runs_after_ocr(monkeypatch, helpers):
    from nemo_retriever.operators.graph_ops import multi_type_extract_operator as module

    operator = module.MultiTypeExtractCPUActor(
        extraction_mode="image",
        extract_params=ExtractParams(method="ocr", chart_routing=True, nemotron_parse_invoke_url=URL),
    )
    stages = []

    class Stage:
        def __init__(self, cls):
            self.cls = cls

        def run(self, data):
            stages.append(self.cls.__name__)
            return data

    monkeypatch.setattr(operator, "_instantiate_resolved", lambda cls, **kwargs: Stage(cls))
    operator._run_detection_pipeline(frame())
    assert stages[-2:] == ["OCRActor", "ChartRoutingActor"]


@pytest.mark.parametrize("table_format", ["HTML", "simple_html", "markdown", "latex"])
def test_routing_format_and_generation_overrides_reach_actual_request(helpers, table_format):
    helpers.postprocessing = SimpleNamespace(postprocess_text=Mock(return_value="converted table"))
    client = Mock()
    client.invoke_chat_completions_images.return_value = ["raw"]
    out = routing.route_chart_regions(
        frame(),
        invoke_url=URL,
        nim_client=client,
        table_format=table_format,
        sampling_params={
            "temperature": 0,
            "repetition_penalty": 1.2,
            "top_k": -1,
            "top_p": 0.9,
            "seed": 7,
            "max_tokens": 123,
        },
    )
    text = out.iloc[0]["chart"][0]["text"]
    assert out.iloc[0]["chart_routing"]["regions"][0]["error"] is None
    if table_format == "HTML":
        assert '<td rowspan="1" colspan="2">Numbers</td>' in text
    elif table_format == "simple_html":
        assert '<td colspan="2">Numbers</td>' in text
    else:
        assert "converted table" in text
        helpers.postprocessing.postprocess_text.assert_called_once_with("table", "Table", table_format=table_format)
    assert '<td colspan="1">Chart</td>' in text
    kwargs = client.invoke_chat_completions_images.call_args.kwargs
    assert kwargs["temperature"] == 0 and kwargs["repetition_penalty"] == 1.2
    assert kwargs["extra_body"] == dict(top_k=-1, top_p=0.9, seed=7, max_tokens=123, skip_special_tokens=False)
