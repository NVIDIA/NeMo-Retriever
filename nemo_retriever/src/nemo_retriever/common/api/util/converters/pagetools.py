# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Converters for human-facing document page numbers."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


def normalize_one_based_page_number(value: Any) -> int | None:
    """Return a positive, one-based document page or ``None``.

    This converter is for citation pages only. It must not be used for
    zero-based audio chunks, video frames, or ingestion pipeline indexes.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            numeric = Decimal(value)
        except InvalidOperation:
            return None
        if not numeric.is_finite() or numeric != numeric.to_integral_value():
            return None
        page_number = int(numeric)
    else:
        try:
            page_number = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        try:
            if value != page_number:
                return None
        except (TypeError, ValueError):
            return None
    return page_number if page_number > 0 else None
