# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.common.modality.page_elements.shared import (
    _apply_final_score_filter,
    _apply_page_elements_v3_postprocess,
)


def test_rejected_structured_box_cannot_suppress_surviving_title() -> None:
    raw_detections = [
        {
            "bbox_xyxy_norm": [0.081876, 0.077799, 0.738124, 0.111654],
            "label": 2,
            "label_name": "title",
            "score": 0.912088,
        },
        {
            "bbox_xyxy_norm": [0.056368, 0.004097, 0.965700, 0.972466],
            "label": 0,
            "label_name": "table",
            "score": 0.052635,
        },
    ]

    postprocessed = _apply_page_elements_v3_postprocess(raw_detections)
    final_detections = _apply_final_score_filter(postprocessed)

    assert [detection["label_name"] for detection in final_detections] == ["title"]


def test_surviving_structured_box_can_still_absorb_title() -> None:
    raw_detections = [
        {
            "bbox_xyxy_norm": [0.081876, 0.077799, 0.738124, 0.111654],
            "label": 2,
            "label_name": "title",
            "score": 0.912088,
        },
        {
            "bbox_xyxy_norm": [0.056368, 0.004097, 0.965700, 0.972466],
            "label": 0,
            "label_name": "table",
            "score": 0.9,
        },
    ]

    postprocessed = _apply_page_elements_v3_postprocess(raw_detections)
    final_detections = _apply_final_score_filter(postprocessed)

    assert [detection["label_name"] for detection in final_detections] == ["table"]
