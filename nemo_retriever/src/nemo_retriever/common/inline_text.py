# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared validation and logical identities for inline text sources."""

from __future__ import annotations

from typing import Sequence

INLINE_TEXT_SOURCE_PREFIX = "inline://"


def normalize_inline_texts(texts: str | Sequence[str]) -> list[str]:
    """Copy and validate raw inline documents."""
    if isinstance(texts, str):
        values = [texts]
    elif isinstance(texts, Sequence):
        values = list(texts)
    else:
        raise TypeError(f"texts must be a string or sequence of strings, got {type(texts).__name__}")

    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"texts[{index}] must be a string, got {type(value).__name__}")
    return values


def inline_text_source_id(index: int) -> str:
    """Return the stable logical identity for an inline document."""
    return f"{INLINE_TEXT_SOURCE_PREFIX}{index:08d}"


def is_inline_text_source(source_id: str) -> bool:
    """Return whether a source identifier denotes inline text."""
    return source_id.startswith(INLINE_TEXT_SOURCE_PREFIX)
