# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text inside infographic boxes is kept unless infographic content is extracted."""

import pytest

from nemo_retriever.common.modality.page_elements.shared import (
    _apply_final_score_filter,
    _apply_page_elements_v3_postprocess,
)
from nemo_retriever.models.nim.primitives.model_interface.yolox import postprocess_page_elements_v3

# Raw Page Elements v3 detections (before postprocessing) for a scanned one-page menu whose text
# blocks all lie inside one infographic box.
_MENU_PAGE = [
    ("text", (0.3761, 0.4244, 0.9642, 0.5818), 0.7944),
    ("text", (0.3802, 0.6713, 0.8256, 0.7127), 0.7410),
    ("title", (0.4988, 0.0503, 0.9525, 0.0836), 0.6560),
    ("text", (0.8184, 0.1410, 0.9117, 0.1596), 0.5705),
    ("text", (0.0485, 0.1947, 0.3091, 0.3831), 0.5666),
    ("text", (0.4860, 0.1442, 0.5707, 0.1618), 0.5587),
    ("text", (0.2840, 0.9223, 0.9376, 0.9690), 0.5521),
    ("text", (0.0383, 0.7330, 0.1900, 0.7521), 0.4975),
    ("text", (0.6766, 0.1396, 0.7814, 0.1581), 0.4721),
    ("text", (0.0428, 0.9203, 0.2446, 0.9700), 0.3514),
    ("infographic", (0.0232, 0.0000, 1.0000, 1.0000), 0.3491),
    ("text", (0.6796, 0.1390, 0.9124, 0.1589), 0.2593),
    ("text", (0.3659, 0.1422, 0.5756, 0.1616), 0.2331),
    ("title", (0.1100, 0.0428, 0.2808, 0.0997), 0.1624),
    ("text", (0.3518, 0.9236, 0.6853, 0.9438), 0.0883),
    ("title", (0.0507, 0.1556, 0.2132, 0.1785), 0.0789),
    ("text", (0.3650, 0.1420, 0.4239, 0.1591), 0.0483),
    ("text", (0.2650, 0.9539, 0.9348, 0.9678), 0.0327),
    ("text", (0.3159, 0.9225, 0.8857, 0.9467), 0.0219),
    ("text", (0.0480, 0.1986, 0.2085, 0.2467), 0.0167),
    ("text", (0.0493, 0.2986, 0.3085, 0.3813), 0.0165),
]

_TEXT_LABELS = ("text", "title", "header_footer")


def _detections():
    return [{"label_name": label, "bbox_xyxy_norm": list(box), "score": score} for label, box, score in _MENU_PAGE]


def _counts(dets):
    counts = {}
    for det in dets:
        counts[det["label_name"]] = counts.get(det["label_name"], 0) + 1
    return counts


@pytest.mark.parametrize("extract_infographics", [False, True])
def test_infographic_contained_text_depends_on_infographic_extraction(extract_infographics):
    dets = _apply_page_elements_v3_postprocess(_detections(), extract_infographics=extract_infographics)
    counts = _counts(_apply_final_score_filter(dets))
    assert counts.get("infographic") == 1
    text_boxes = sum(counts.get(label, 0) for label in _TEXT_LABELS)
    if extract_infographics:
        assert text_boxes == 0
    else:
        assert text_boxes > 0


@pytest.mark.parametrize("suppress_text_in_infographics", [False, True])
def test_table_contained_paragraph_is_removed_either_way(suppress_text_in_infographics):
    labels = ["table", "chart", "title", "infographic", "paragraph", "header_footer"]
    annotations = {label: [] for label in labels}
    annotations["table"] = [[0.2, 0.2, 0.8, 0.6, 0.87]]
    annotations["infographic"] = [[0.1, 0.65, 0.9, 0.95, 0.9]]
    annotations["paragraph"] = [[0.3, 0.3, 0.6, 0.4, 0.93], [0.2, 0.7, 0.8, 0.9, 0.95]]
    result = postprocess_page_elements_v3(
        annotations, labels, suppress_text_in_infographics=suppress_text_in_infographics
    )
    assert len(result["table"]) == 1
    assert len(result["paragraph"]) == (0 if suppress_text_in_infographics else 1)
