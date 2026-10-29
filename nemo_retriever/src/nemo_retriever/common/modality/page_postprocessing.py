# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector preprocessing and page-region serialization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_detector_input(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Match research YOLOX RGB resize/pad without torch interpolation."""
    import cv2

    height, width = image.shape[:2]
    ratio = min(size[0] / height, size[1] / width)
    resized = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_LINEAR).astype(
        np.uint8
    )
    padded = np.pad(
        resized,
        ((0, size[0] - resized.shape[0]), (0, size[1] - resized.shape[1]), (0, 0)),
        mode="constant",
        constant_values=114,
    )
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)


def finalize_page_regions(frame: pd.DataFrame) -> pd.DataFrame:
    """Serialize all page region types in top-to-bottom, left-to-right order."""
    out = frame.copy()
    pages = []
    all_regions = []
    for _, row in out.iterrows():
        regions = []
        text_regions = row.get("ocr_text_regions")
        if isinstance(text_regions, (list, np.ndarray)):
            regions.extend(dict(item) for item in text_regions)
        if not regions:
            native_text = row.get("text")
            if isinstance(native_text, str) and native_text.strip():
                regions.append(
                    {
                        "label_name": "text",
                        "bbox_xyxy_norm": [0, 0, 1, 1],
                        "text": native_text,
                    }
                )
        for label in ("table", "chart", "infographic"):
            items = row.get(label)
            if isinstance(items, (list, np.ndarray)):
                regions.extend({**item, "label_name": label} for item in items if isinstance(item, dict))
        regions = sorted(
            regions,
            key=lambda item: (item["bbox_xyxy_norm"][1], item["bbox_xyxy_norm"][0]),
        )
        all_regions.append(regions)
        pages.append("\n".join(item["text"] for item in regions if isinstance(item.get("text"), str) and item["text"]))
    out["ordered_regions"] = all_regions
    out["ordered_page_text"] = pages
    return out
