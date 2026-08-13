# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from nemo_retriever.common.api.internal.mutate.deduplicate import calculate_iou
from nemo_retriever.common.params import DedupParams

_STRUCTURED_COLUMNS = ("table", "chart", "infographic")


def _as_element_list(value: Any) -> Optional[List[Any]]:
    """Return an extracted-element cell as a plain list, or ``None`` if unsupported.

    Batch mode delivers structural-element cells as one-dimensional NumPy
    object arrays (Arrow list columns converted to pandas), while inprocess
    mode uses Python lists.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return value.tolist()
    return None


def _as_bbox(value: Any) -> Optional[Tuple[float, ...]]:
    """Return the first four bbox coordinates as a plain tuple."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    return tuple(value[:4])


def _collect_structured_bboxes(row: pd.Series) -> List[Tuple[float, ...]]:
    """Gather all bounding boxes from tables, charts, and infographics columns."""
    bboxes: List[Tuple[float, ...]] = []
    for col in _STRUCTURED_COLUMNS:
        items = _as_element_list(row.get(col))
        if items is None:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            bbox = _as_bbox(item.get("bbox_xyxy_norm"))
            if bbox is not None:
                bboxes.append(bbox)
    return bboxes


def dedup_images(
    batch_df: pd.DataFrame,
    *,
    content_hash: bool = True,
    bbox_iou: bool = True,
    iou_threshold: float = 0.45,
) -> pd.DataFrame:
    """Remove duplicate and overlapping images from the ``images`` column.

    Two passes per row:

    1. **Content-hash dedup** (``content_hash=True``): MD5-hash each
       ``image_b64``; remove exact duplicates (keep first).
    2. **Bbox IoU dedup** (``bbox_iou=True``): Compare each image's
       ``bbox_xyxy_norm`` against all entries in ``table``, ``chart``,
       ``infographic``. If IoU >= ``iou_threshold``, drop the image
       (prefer structured content).
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df
    if "images" not in batch_df.columns:
        return batch_df

    # Arrow-backed columns reject assignment of a filtered Python list.
    if isinstance(batch_df["images"].dtype, pd.ArrowDtype):
        batch_df["images"] = batch_df["images"].astype(object)

    for row_idx, row in batch_df.iterrows():
        images = _as_element_list(row.get("images"))
        if not images:
            continue

        filtered = list(images)

        # Pass 1: content-hash dedup
        if content_hash:
            seen_hashes: set[str] = set()
            deduped: list[dict] = []
            for item in filtered:
                if not isinstance(item, dict):
                    deduped.append(item)
                    continue
                b64 = item.get("image_b64", "")
                if not b64:
                    deduped.append(item)
                    continue
                h = hashlib.md5(b64.encode("utf-8")).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    deduped.append(item)
            filtered = deduped

        # Pass 2: bbox IoU dedup against structured content
        if bbox_iou:
            structured_bboxes = _collect_structured_bboxes(row)
            if structured_bboxes:
                surviving: list[dict] = []
                for item in filtered:
                    if not isinstance(item, dict):
                        surviving.append(item)
                        continue
                    img_bbox_t = _as_bbox(item.get("bbox_xyxy_norm"))
                    if img_bbox_t is None:
                        surviving.append(item)
                        continue
                    overlaps = any(calculate_iou(img_bbox_t, sb) >= iou_threshold for sb in structured_bboxes)
                    if not overlaps:
                        surviving.append(item)
                filtered = surviving

        batch_df.at[row_idx, "images"] = filtered

    return batch_df


class DedupActor:
    """Ray Data actor for batch-mode image deduplication (CPU-only)."""

    def __init__(self, params: DedupParams) -> None:
        self._kwargs = params.model_dump(mode="python")

    def __call__(self, batch_df: Any) -> Any:
        return dedup_images(batch_df, **self._kwargs)
