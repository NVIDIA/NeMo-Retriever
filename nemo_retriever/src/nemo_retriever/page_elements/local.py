# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional, Sequence

try:
    from nemotron_page_elements_v3.utils import (
        postprocess_preds_page_element as _postprocess_preds_page_element,
    )
except Exception:  # pragma: no cover
    _postprocess_preds_page_element = None

try:
    from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (
        postprocess_page_elements_v3,
        YOLOX_PAGE_V3_CLASS_LABELS,
        YOLOX_PAGE_V3_FINAL_SCORE,
    )
except ImportError:
    postprocess_page_elements_v3 = None  # type: ignore[assignment,misc]
    YOLOX_PAGE_V3_CLASS_LABELS = None  # type: ignore[assignment]
    YOLOX_PAGE_V3_FINAL_SCORE = {}  # type: ignore[assignment]


def postprocess_preds_page_element(
    pred: Any,
    thresholds_per_class: Sequence[float],
    label_names: Optional[Sequence[str]] = None,
) -> Any:
    if _postprocess_preds_page_element is None:  # pragma: no cover
        raise ImportError("nemotron_page_elements_v3 is required for local page-element postprocessing.")
    return _postprocess_preds_page_element(pred, thresholds_per_class, label_names)
