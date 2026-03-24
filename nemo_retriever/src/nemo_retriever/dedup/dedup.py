# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from typing import Any, List, Tuple

import pandas as pd

from nemo_retriever.params import DedupParams

_STRUCTURED_COLUMNS = ("table", "chart", "infographic")


def calculate_iou(bbox1: Tuple[float, ...], bbox2: Tuple[float, ...]) -> float:
    """Calculate Intersection over Union (IoU) for two bounding boxes.

    Boxes are in format (x1, y1, x2, y2) where (x1, y1) is the top-left corner
    and (x2, y2) is the bottom-right corner.
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
    x1_2, y1_2, x2_2, y2_2 = bbox2[:4]

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def _collect_structured_bboxes(row: pd.Series) -> List[Tuple[float, ...]]:
    """Gather all bounding boxes from tables, charts, and infographics columns."""
    bboxes: List[Tuple[float, ...]] = []
    for col in _STRUCTURED_COLUMNS:
        items = row.get(col)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_xyxy_norm")
            if bbox and len(bbox) >= 4:
                bboxes.append(tuple(bbox[:4]))
    return bboxes


def dedup_images(
    batch_df: pd.DataFrame,
    *,
    content_hash: bool = True,
    bbox_iou: bool = True,
    iou_threshold: float = 0.45,
    **kwargs: Any,
) -> pd.DataFrame:
    """Remove duplicate and overlapping images from the ``images`` column.

    Two passes per row:

    1. **Content-hash dedup** (``content_hash=True``): MD5-hash each
       ``image_b64``; remove exact duplicates (keep first).
    2. **Bbox IoU dedup** (``bbox_iou=True``): Compare each image's
       ``bbox_xyxy_norm`` against all entries in ``tables``, ``charts``,
       ``infographics``. If IoU >= ``iou_threshold``, drop the image
       (prefer structured content).
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df
    if "images" not in batch_df.columns:
        return batch_df

    for row_idx, row in batch_df.iterrows():
        images = row.get("images")
        if not isinstance(images, list) or not images:
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
                    img_bbox = item.get("bbox_xyxy_norm")
                    if not img_bbox or len(img_bbox) < 4:
                        surviving.append(item)
                        continue
                    img_bbox_t = tuple(img_bbox[:4])
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
