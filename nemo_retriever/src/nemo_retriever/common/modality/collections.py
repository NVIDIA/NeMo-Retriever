# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection adapters for multimodal values crossing Ray's pandas boundary."""

from __future__ import annotations

from typing import Any

import numpy as np


def multimodal_collection(value: Any) -> list[Any] | None:
    """Return a mutable list view of a structured multimodal collection.

    Ray may preserve list-valued pandas object cells as one-dimensional NumPy
    object arrays. Restrict ndarray support to one dimension so image tensors
    and other genuine multidimensional model inputs are never reinterpreted as
    collections of extracted objects.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return list(value)
    return None


def bbox_coordinates(value: Any) -> list[Any] | None:
    """Return four-or-more bbox coordinates without array truth testing."""
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            return None
        value = value.tolist()
    elif isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) >= 4:
        return value
    return None
