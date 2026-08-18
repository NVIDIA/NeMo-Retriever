# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_retriever.query.evidence import _evidence_item


@pytest.mark.parametrize("bbox", ["", []])
def test_blank_bbox_without_page_falls_through_to_default_locator(bbox) -> None:
    # LanceDB rows default bbox_xyxy_norm to "" (and metadata may be a JSON
    # string, so meta is empty). With no page/segment/frame, a blank bbox must
    # NOT produce a malformed {"kind": "bbox", "value": ""} locator.
    hit = {"text": "t", "source": "a.pdf", "bbox_xyxy_norm": bbox, "metadata": {}}
    assert _evidence_item(hit)["locator"] == {"kind": "page", "value": None}


def test_top_level_bbox_produces_bbox_locator() -> None:
    hit = {"text": "t", "source": "a.pdf", "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4], "metadata": {}}
    assert _evidence_item(hit)["locator"] == {"kind": "bbox", "value": [0.1, 0.2, 0.3, 0.4]}
