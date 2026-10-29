# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector preprocessing and page-region serialization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_detector_input(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Prepare an RGB image for local YOLOX detector inference.

    Resize proportionally with OpenCV ``INTER_LINEAR``, cast the resized
    pixels to ``uint8``, and pad the bottom and right with gray RGB
    ``(114, 114, 114)``. Preserve RGB channel order and the 0–255 pixel scale;
    do not normalize pixel values.

    Parameters
    ----------
    image : numpy.ndarray
        Nonempty RGB image with shape ``(height, width, 3)``, expected dtype
        ``uint8``, and pixel values in the range 0–255.
    size : tuple[int, int]
        Positive target ``(height, width)`` in pixels. Choose dimensions large
        enough that proportional resizing leaves both image dimensions nonzero.

    Returns
    -------
    numpy.ndarray
        Contiguous ``float32`` array with shape ``(3, size[0], size[1])``.
        Image content starts at the top-left corner of the padded array.

    Raises
    ------
    ImportError
        If OpenCV is unavailable. Install the OpenCV runtime dependency.
    ZeroDivisionError
        If an input image dimension is zero. Supply a nonempty image.
    cv2.error
        If OpenCV cannot resize the input. Check its shape and target size.
    ValueError
        If array padding or transposition fails. Supply an HWC RGB image
        and a positive target size.
    """
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
    """Combine page regions and serialize their text in geometric order.

    Sort regions by normalized top coordinate, then left coordinate. This is
    a geometric ordering rule, not a multi-column reading-order model. Retain
    region text as supplied, without deduplication or caption generation.

    Parameters
    ----------
    frame : pandas.DataFrame
        One row per page. Optional ``ocr_text_regions``, ``table``, ``chart``,
        and ``infographic`` columns contain lists or NumPy arrays of region
        dictionaries. Other column value types are ignored. Each region needs
        ``bbox_xyxy_norm`` in normalized ``[left, top, right, bottom]`` form;
        its optional ``text`` value contributes to serialization only when it
        is a nonempty string. Table, chart, and infographic regions receive
        the corresponding ``label_name``. If no OCR text regions are present,
        a nonblank string in ``text`` becomes a whole-page text region with
        bounds ``[0, 0, 1, 1]``. Non-dictionary modality entries are ignored.

    Returns
    -------
    pandas.DataFrame
        A copy retaining the original columns and index, with
        ``ordered_regions`` containing the sorted region dictionaries and
        ``ordered_page_text`` containing their nonempty strings joined by
        newlines. These columns replace existing columns of the same names.
        Pages without regions receive an empty list and an empty string.

    Raises
    ------
    KeyError
        If a region lacks ``bbox_xyxy_norm``. Supply a bounding box for every
        region, including regions without text.
    IndexError
        If a bounding box is too short to access its top or left coordinate.
        Supply all four normalized coordinates.
    TypeError or ValueError
        If OCR region entries cannot be converted to dictionaries or region
        coordinates cannot be compared. Supply dictionaries with numeric boxes.
    """
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
