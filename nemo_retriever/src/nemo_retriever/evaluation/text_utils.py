# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared text processing utilities for the evaluation package."""

from __future__ import annotations

import re


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output.

    Handles both closed tags (<think>...</think>) and unclosed tags where the
    model hit the token limit mid-reasoning and never emitted </think>.
    Returns empty string if nothing remains after stripping so callers can
    detect thinking_truncated.
    """
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*", "", stripped, flags=re.DOTALL)
    return stripped.strip()
