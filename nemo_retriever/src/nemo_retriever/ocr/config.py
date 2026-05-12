# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

OCRVersion = Literal["v1", "v2"]
OCRLang = Literal["multi", "english"]


def resolve_ocr_v2_lang(ocr_version: str = "v2", ocr_lang: str | None = None) -> str:
    """Resolve public OCR selectors to the ``NemotronOCRV2(lang=...)`` selector.

    Args:
        ocr_version: ``"v1"`` for legacy English-only mode or ``"v2"`` (default)
            for multilingual mode.
        ocr_lang: Optional language selector for v2. ``"multi"`` (default) or
            ``"english"``. Must be ``None`` when *ocr_version* is ``"v1"``.

    Returns:
        The ``lang`` string to pass to ``NemotronOCRV2(lang=...)``.

    Raises:
        ValueError: If *ocr_version* is not one of ``['v1', 'v2']``, or if
            *ocr_lang* is provided together with ``ocr_version='v1'``, or if
            *ocr_lang* is not one of ``['multi', 'english']``.
    """
    if ocr_version == "v1":
        if ocr_lang is not None:
            raise ValueError("ocr_lang is only supported when ocr_version='v2'.")
        return "v1"
    if ocr_version != "v2":
        raise ValueError("ocr_version must be one of ['v1', 'v2'].")
    if ocr_lang is None:
        return "multi"
    if ocr_lang not in {"multi", "english"}:
        raise ValueError("ocr_lang must be one of ['multi', 'english'].")
    return ocr_lang
