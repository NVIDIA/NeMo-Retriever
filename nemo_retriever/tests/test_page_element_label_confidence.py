# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for labels and confidences during overlap removal."""

import pytest

from nemo_retriever.models.nim.primitives.model_interface.yolox import postprocess_page_elements_v3


@pytest.mark.parametrize("contained", [True, False])
def test_contained_paragraph_removal_preserves_detection_scores(contained):
    labels = ["table", "chart", "title", "infographic", "paragraph", "header_footer"]
    annotations = {label: [] for label in labels}
    annotations["table"] = [[0.2, 0.2, 0.8, 0.6, 0.87]]
    paragraph = [0.3, 0.3, 0.6, 0.4] if contained else [0.1, 0.8, 0.9, 0.9]
    annotations["paragraph"] = [[*paragraph, 0.93]]
    result = postprocess_page_elements_v3(annotations, labels)
    assert len(result["table"]) == 1
    assert result["table"][0][-1] == pytest.approx(0.87)
    assert len(result["paragraph"]) == (0 if contained else 1)
    if not contained:
        assert result["paragraph"][0][-1] == pytest.approx(0.93)
    assert all(not result[label] for label in ("chart", "title", "infographic", "header_footer"))
