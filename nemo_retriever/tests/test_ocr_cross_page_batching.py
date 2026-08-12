# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for local OCR crop batching across page rows."""

from __future__ import annotations

import base64
import io
from typing import Any

import pandas as pd
from PIL import Image

from nemo_retriever.common.params import RemoteRetryParams
from nemo_retriever.operators.extract.ocr.gpu_ocr import OCRActor


def _page_png_b64(crop_id: int) -> str:
    image = Image.new("RGB", (16, 16), color=(crop_id, crop_id, crop_id))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _detection(detection_id: str, label_name: str, bbox: list[float]) -> dict[str, Any]:
    return {
        "detection_id": detection_id,
        "label_name": label_name,
        "bbox_xyxy_norm": bbox,
    }


def _page(page_id: str, crop_id: int, detections: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "page_id": page_id,
        "metadata": {},
        "page_image": {"image_b64": _page_png_b64(crop_id)},
        "page_elements_v3": {"detections": detections},
    }


def _ocr_prediction(crop_id: int) -> list[dict[str, Any]]:
    return [
        {
            "left": 0.0,
            "right": 1.0,
            "upper": 0.0,
            "lower": 1.0,
            "text": f"crop-{crop_id}",
        }
    ]


class _RecordingListModel:
    """Record the list-input contract used by the persistent local model."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, crops: Any, *, merge_level: str) -> list[list[dict[str, Any]]]:
        assert isinstance(crops, list)
        crop_ids = [int(crop[0, 0, 0]) for crop in crops]
        self.calls.append(
            {
                "crop_count": len(crops),
                "crop_ids": crop_ids,
                "merge_level": merge_level,
            }
        )
        return [_ocr_prediction(crop_id) for crop_id in crop_ids]


class _FallbackIsolationModel:
    """Force batched failure, then fail one crop during per-item isolation."""

    def __init__(self, failing_crop_id: int) -> None:
        self.failing_crop_id = failing_crop_id
        self.calls: list[dict[str, Any]] = []

    def invoke(self, crops: Any, *, merge_level: str) -> Any:
        is_batch = isinstance(crops, list)
        crop_arrays = crops if is_batch else [crops]
        crop_ids = [int(crop[0, 0, 0]) for crop in crop_arrays]
        self.calls.append(
            {
                "input_kind": "batch" if is_batch else "single",
                "crop_ids": crop_ids,
                "merge_level": merge_level,
            }
        )
        if is_batch:
            raise RuntimeError("force per-crop fallback")
        if crop_ids[0] == self.failing_crop_id:
            raise ValueError(f"bad crop {self.failing_crop_id}")
        return _ocr_prediction(crop_ids[0])


class _WrongCountFallbackModel:
    """Return too few batched results, then succeed during per-item fallback."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, crops: Any, *, merge_level: str) -> Any:
        is_batch = isinstance(crops, list)
        crop_arrays = crops if is_batch else [crops]
        crop_ids = [int(crop[0, 0, 0]) for crop in crop_arrays]
        self.calls.append(
            {
                "input_kind": "batch" if is_batch else "single",
                "crop_ids": crop_ids,
                "merge_level": merge_level,
            }
        )
        if is_batch:
            return [_ocr_prediction(crop_ids[0])]
        return _ocr_prediction(crop_ids[0])


def _local_actor(
    model: Any,
    *,
    inference_batch_size: int,
    extract_text: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
) -> OCRActor:
    actor = object.__new__(OCRActor)
    actor._graph_init_kwargs = {}
    actor.ocr_kwargs = {
        "extract_text": extract_text,
        "extract_tables": extract_tables,
        "extract_charts": extract_charts,
        "extract_infographics": extract_infographics,
        "use_table_structure": False,
        "request_timeout_s": 120.0,
        "inference_batch_size": inference_batch_size,
    }
    actor._remote_retry = RemoteRetryParams()
    actor._model = model
    actor._nim_client = None
    return actor


def test_local_actor_batches_compatible_crops_across_page_rows() -> None:
    model = _RecordingListModel()
    actor = _local_actor(model, inference_batch_size=2, extract_charts=True)
    chart_bboxes = [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]
    batch = pd.DataFrame(
        [
            _page("page-A", 11, [_detection("chart-A", "chart", chart_bboxes[0])]),
            _page("page-B", 22, [_detection("chart-B", "chart", chart_bboxes[1])]),
        ]
    )

    result = actor(batch)

    assert {
        "model_calls": model.calls,
        "rows": [{"page_id": row.page_id, "chart": row.chart} for row in result.itertuples(index=False)],
    } == {
        "model_calls": [
            {
                "crop_count": 2,
                "crop_ids": [11, 22],
                "merge_level": "paragraph",
            }
        ],
        "rows": [
            {
                "page_id": "page-A",
                "chart": [{"bbox_xyxy_norm": chart_bboxes[0], "text": "crop-11"}],
            },
            {
                "page_id": "page-B",
                "chart": [{"bbox_xyxy_norm": chart_bboxes[1], "text": "crop-22"}],
            },
        ],
    }


def test_local_actor_separates_merge_levels_chunks_and_preserves_detection_order() -> None:
    model = _RecordingListModel()
    actor = _local_actor(
        model,
        inference_batch_size=2,
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
    )
    table_a = [0.0, 0.0, 0.4, 0.4]
    chart_a = [0.5, 0.0, 1.0, 0.4]
    title_a = [0.0, 0.5, 1.0, 0.7]
    table_b1 = [0.0, 0.0, 0.3, 0.3]
    table_b2 = [0.35, 0.0, 0.65, 0.3]
    infographic_b = [0.0, 0.5, 1.0, 1.0]
    page_a = _page(
        "page-A",
        11,
        [
            _detection("table-A", "table", table_a),
            _detection("chart-A", "chart", chart_a),
            _detection("title-A", "title", title_a),
        ],
    )
    page_a["metadata"] = {"needs_ocr_for_text": True}
    page_a["text"] = "native-A"
    page_b = _page(
        "page-B",
        22,
        [
            _detection("table-B1", "table", table_b1),
            _detection("table-B2", "table", table_b2),
            _detection("infographic-B", "infographic", infographic_b),
        ],
    )
    page_b["text"] = "native-B"
    empty_page = _page("page-empty", 33, [])
    empty_page["text"] = "native-empty"
    batch = pd.DataFrame(
        [
            page_a,
            page_b,
            empty_page,
            {
                "page_id": "page-malformed",
                "metadata": {"error": {"stage": "page_render", "message": "missing raster"}},
                "page_image": None,
                "page_elements_v3": {"detections": [_detection("chart-malformed", "chart", [0.0, 0.0, 1.0, 1.0])]},
                "text": "native-malformed",
            },
        ]
    )

    result = actor(batch)

    assert {
        "model_calls": model.calls,
        "page_A": {
            "table_bboxes": [entry["bbox_xyxy_norm"] for entry in result.at[0, "table"]],
            "chart": result.at[0, "chart"],
            "text": result.at[0, "text"],
            "metadata": {
                "num_detections": result.at[0, "ocr"]["num_detections"],
                "counts_by_label": result.at[0, "ocr"]["counts_by_label"],
            },
        },
        "page_B": {
            "table_bboxes": [entry["bbox_xyxy_norm"] for entry in result.at[1, "table"]],
            "infographic": result.at[1, "infographic"],
            "text": result.at[1, "text"],
            "metadata": {
                "num_detections": result.at[1, "ocr"]["num_detections"],
                "counts_by_label": result.at[1, "ocr"]["counts_by_label"],
            },
        },
        "empty_page": {
            "table": result.at[2, "table"],
            "chart": result.at[2, "chart"],
            "infographic": result.at[2, "infographic"],
            "text": result.at[2, "text"],
            "num_detections": result.at[2, "ocr"]["num_detections"],
        },
        "malformed_page": {
            "table": result.at[3, "table"],
            "chart": result.at[3, "chart"],
            "infographic": result.at[3, "infographic"],
            "text": result.at[3, "text"],
            "error": result.at[3, "ocr"]["error"],
            "num_detections": result.at[3, "ocr"]["num_detections"],
        },
    } == {
        "model_calls": [
            {"crop_count": 2, "crop_ids": [11, 22], "merge_level": "word"},
            {"crop_count": 1, "crop_ids": [22], "merge_level": "word"},
            {"crop_count": 2, "crop_ids": [11, 11], "merge_level": "paragraph"},
            {"crop_count": 1, "crop_ids": [22], "merge_level": "paragraph"},
        ],
        "page_A": {
            "table_bboxes": [table_a],
            "chart": [{"bbox_xyxy_norm": chart_a, "text": "crop-11"}],
            "text": "crop-11",
            "metadata": {
                "num_detections": 3,
                "counts_by_label": {"table": 1, "chart": 1, "text": 1},
            },
        },
        "page_B": {
            "table_bboxes": [table_b1, table_b2],
            "infographic": [{"bbox_xyxy_norm": infographic_b, "text": "crop-22"}],
            "text": "native-B",
            "metadata": {
                "num_detections": 3,
                "counts_by_label": {"table": 2, "infographic": 1},
            },
        },
        "empty_page": {
            "table": [],
            "chart": [],
            "infographic": [],
            "text": "native-empty",
            "num_detections": 0,
        },
        "malformed_page": {
            "table": [],
            "chart": [],
            "infographic": [],
            "text": "native-malformed",
            "error": {"stage": "page_render", "message": "missing raster"},
            "num_detections": 0,
        },
    }


def test_local_actor_isolates_one_failed_crop_to_its_source_row() -> None:
    model = _FallbackIsolationModel(failing_crop_id=22)
    actor = _local_actor(model, inference_batch_size=3, extract_charts=True)
    bbox = [0.0, 0.0, 1.0, 1.0]
    batch = pd.DataFrame(
        [
            _page("page-A", 11, [_detection("chart-A", "chart", bbox)]),
            _page("page-B", 22, [_detection("chart-B", "chart", bbox)]),
            _page("page-C", 33, [_detection("chart-C", "chart", bbox)]),
        ]
    )

    result = actor(batch)

    assert {
        "model_calls": model.calls,
        "charts": result["chart"].tolist(),
        "errors": [meta["error"] for meta in result["ocr"]],
    } == {
        "model_calls": [
            {
                "input_kind": "batch",
                "crop_ids": [11, 22, 33],
                "merge_level": "paragraph",
            },
            {
                "input_kind": "single",
                "crop_ids": [11],
                "merge_level": "paragraph",
            },
            {
                "input_kind": "single",
                "crop_ids": [22],
                "merge_level": "paragraph",
            },
            {
                "input_kind": "single",
                "crop_ids": [33],
                "merge_level": "paragraph",
            },
        ],
        "charts": [
            [{"bbox_xyxy_norm": bbox, "text": "crop-11"}],
            [],
            [{"bbox_xyxy_norm": bbox, "text": "crop-33"}],
        ],
        "errors": [
            None,
            {
                "stage": "ocr_page_elements",
                "type": "ValueError",
                "message": "bad crop 22",
                "traceback": result.at[1, "ocr"]["error"]["traceback"],
            },
            None,
        ],
    }


def test_local_actor_falls_back_when_batch_result_count_is_wrong() -> None:
    model = _WrongCountFallbackModel()
    actor = _local_actor(model, inference_batch_size=2, extract_charts=True)
    bbox = [0.0, 0.0, 1.0, 1.0]
    batch = pd.DataFrame(
        [
            _page("page-A", 11, [_detection("chart-A", "chart", bbox)]),
            _page("page-B", 22, [_detection("chart-B", "chart", bbox)]),
        ]
    )

    result = actor(batch)

    assert {
        "model_calls": model.calls,
        "charts": result["chart"].tolist(),
        "errors": [meta["error"] for meta in result["ocr"]],
    } == {
        "model_calls": [
            {
                "input_kind": "batch",
                "crop_ids": [11, 22],
                "merge_level": "paragraph",
            },
            {
                "input_kind": "single",
                "crop_ids": [11],
                "merge_level": "paragraph",
            },
            {
                "input_kind": "single",
                "crop_ids": [22],
                "merge_level": "paragraph",
            },
        ],
        "charts": [
            [{"bbox_xyxy_norm": bbox, "text": "crop-11"}],
            [{"bbox_xyxy_norm": bbox, "text": "crop-22"}],
        ],
        "errors": [None, None],
    }
