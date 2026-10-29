# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for extraction geometry and reading order."""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from nemo_retriever.common.modality import page_postprocessing
from nemo_retriever.common.modality import table_and_chart
from nemo_retriever.common.modality.ocr import shared


@pytest.fixture
def modules():
    return SimpleNamespace(table=table_and_chart, compat=page_postprocessing, ocr=shared)


def _page():
    image = Image.new("RGB", (100, 200), color=(25, 50, 75))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "page_image": {"image_b64": base64.b64encode(buf.getvalue()).decode()},
        "metadata": {"needs_ocr_for_text": True},
        "text": "",
        "page_elements_v3": {
            "detections": [
                {"label_name": "text", "bbox_xyxy_norm": [0, 0.7, 1, 0.9]},
                {"label_name": "title", "bbox_xyxy_norm": [0, 0.1, 1, 0.3]},
            ]
        },
    }


class _Model:
    def __init__(self):
        self.shapes = []

    def invoke(self, crops, *, merge_level):
        assert merge_level == "paragraph"
        self.shapes.extend(crop.shape for crop in crops)
        return [
            [{"left": 0, "right": 1, "upper": 0.1, "lower": 0.2, "text": text}]
            for text in ("BOTTOM", "TOP")[: len(crops)]
        ]


def test_local_padding_and_page_order_are_applied_by_default(modules):
    frame = pd.DataFrame([_page()])
    model = _Model()
    fixed = modules.ocr.ocr_page_elements(frame, model=model, extract_text=True)
    assert fixed.iloc[0].text == "TOP\nBOTTOM"
    assert fixed.iloc[0].ordered_page_text == "TOP\nBOTTOM"
    assert model.shapes == [(100, 100, 3), (100, 100, 3)]
    assert frame.iloc[0].text == ""


def test_remote_padding_and_crop_mapping(modules):
    class Client:
        def invoke_image_inference_batches(self, **kwargs):
            images = kwargs["image_b64_list"]
            for encoded in images:
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                    assert image.size == (100, 100)
                    assert image.getpixel((99, 99)) == (255, 255, 255)
            return [[{"left": 0, "right": 1, "upper": 0, "lower": 0.1, "text": t}] for t in ("BOTTOM", "TOP")]

    out = modules.ocr.ocr_page_elements(
        pd.DataFrame([_page()]),
        invoke_url="https://example.invalid/ocr",
        nim_client=Client(),
        extract_text=True,
    )
    assert out.iloc[0].ordered_page_text == "TOP\nBOTTOM"


def test_global_order_interleaves_all_content_types(modules):
    frame = pd.DataFrame(
        [
            {
                "text": "first\nlast",
                "ocr_text_regions": [
                    {"bbox_xyxy_norm": [0, 0.9, 1, 1], "text": "last"},
                    {"bbox_xyxy_norm": [0, 0.1, 1, 0.2], "text": "first"},
                ],
                "table": [{"bbox_xyxy_norm": [0, 0.3, 1, 0.4], "text": "TABLE"}],
                "chart": [{"bbox_xyxy_norm": [0, 0.5, 1, 0.6], "text": "CHART"}],
                "infographic": [{"bbox_xyxy_norm": [0, 0.7, 1, 0.8], "text": "INFO"}],
            }
        ]
    )
    out = modules.compat.finalize_page_regions(frame)
    assert out.iloc[0].ordered_page_text == "first\nTABLE\nCHART\nINFO\nlast"
    # The output must preserve geometric ordering across content types.
    regions = pd.DataFrame(out.iloc[0].ordered_regions)
    regions["upper"] = regions.bbox_xyxy_norm.map(lambda b: b[1])
    regions["left"] = regions.bbox_xyxy_norm.map(lambda b: b[0])
    assert out.iloc[0].ordered_page_text == "\n".join(regions.sort_values(["upper", "left"]).text)


def _table_inputs():
    boxes = {
        "cell": np.array([[0, 0, 100, 40], [100, 0, 200, 40], [0, 40, 100, 80], [100, 40, 200, 80]]),
        "row": np.array([[0, 0, 200, 40], [0, 40, 200, 80]]),
        "column": np.array([[0, 0, 100, 80], [100, 0, 200, 80]]),
    }
    rects = np.array([[5, 5, 90, 25], [105, 5, 190, 25], [5, 45, 90, 65], [105, 45, 190, 65]])
    quads = [[[x, y], [r, y], [r, b], [x, b]] for x, y, r, b in rects]
    return boxes, rects, quads, ["A", "B", "10", "20"]


def test_table_padding_coordinates_without_inventing_header(modules):
    boxes, rects, _, texts = _table_inputs()
    dets = [
        {"label_name": label, "bbox_xyxy_norm": (box / [200, 80, 200, 80]).tolist()}
        for label, group in boxes.items()
        for box in group
    ]
    ocr = [
        {
            "left": r[0] / 200,
            "upper": r[1] / 200,
            "right": r[2] / 200,
            "lower": r[3] / 200,
            "text": t,
        }
        for r, t in zip(rects, texts)
    ]
    text = modules.table.join_table_structure_and_ocr_output(dets, ocr, (80, 200), ocr_crop_hw=(200, 200))
    assert text == "| A | B |\n| 10 | 20 |"
    assert "---" not in text


@pytest.mark.parametrize("shape", [(40, 100, 3), (100, 40, 3), (40, 40, 3)])
def test_square_padding_preserves_original_pixels_and_geometry(shape):
    crop = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    padded, original_hw, padded_hw = shared._prepare_ocr_crop(crop, "bottom_right")
    assert original_hw == shape[:2]
    assert padded_hw == (max(shape[:2]),) * 2
    np.testing.assert_array_equal(padded[: shape[0], : shape[1]], crop)
    assert np.all(padded[shape[0] :] == 255)
    assert np.all(padded[:, shape[1] :] == 255)
    unchanged, old_hw, new_hw = shared._prepare_ocr_crop(crop)
    assert unchanged is crop and old_hw == new_hw == shape[:2]


def test_detector_input_uses_rgb_and_bottom_right_padding():
    image = np.full((2, 4, 3), (20, 70, 150), dtype=np.uint8)
    actual = page_postprocessing.prepare_detector_input(image, (8, 8))
    assert actual.shape == (3, 8, 8)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual[:, :4, :], np.broadcast_to(np.array([20, 70, 150])[:, None, None], (3, 4, 8)))
    assert np.all(actual[:, 4:, :] == 114)


def test_nim_schema_uses_padded_ocr_coordinates_for_table_join():
    prediction = [
        {
            "text_prediction": {"text": "cell"},
            "bounding_box": {
                "points": [{"x": 0.1, "y": 0.2}, {"x": 0.4, "y": 0.2}, {"x": 0.4, "y": 0.3}, {"x": 0.1, "y": 0.3}]
            },
        }
    ]
    items = table_and_chart._normalize_ocr_items(prediction)
    boxes, text = table_and_chart._ocr_items_to_pixel_quad_boxes(items, (200, 200))
    assert text == ["cell"]
    assert boxes == [[[20, 40], [80, 40], [80, 60], [20, 60]]]


def test_native_exports_and_embedding_text_use_one_ordered_page():
    from nemo_retriever.common.io.markdown import to_markdown_by_page
    from nemo_retriever.common.modality.content_transforms import _combine_text_with_content

    record = dict(
        page_number=1,
        text="legacy",
        table=[{"text": "duplicate"}],
        ordered_page_text="top\nTABLE\nbottom",
    )
    markdown = to_markdown_by_page([record])[1]
    assert "top\nTABLE\nbottom" in markdown
    assert "legacy" not in markdown and "duplicate" not in markdown
    assert _combine_text_with_content(record, "text", ["table"]) == "top\nTABLE\nbottom"
    record.pop("ordered_page_text")
    assert _combine_text_with_content(record, "text", ["table"]) == "legacy\n\nduplicate"


def test_empty_table_is_explicit():
    result = shared._OCRRowResult()
    shared._append_ocr_prediction(
        result,
        row=SimpleNamespace(),
        label_name="table",
        bbox=[0, 0, 1, 1],
        preds=[],
        crop_hw=(100, 100),
        use_table_structure=False,
    )
    assert result.table_items[0]["text"] == "EMPTY TABLE"
