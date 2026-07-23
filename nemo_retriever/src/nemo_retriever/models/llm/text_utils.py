# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared text-processing utilities for LLM output hygiene.

Pure-stdlib module. Lives under ``nemo_retriever.models.llm`` so that the
lightweight SDK surface (``LiteLLMClient``, ``Retriever.answer``) does
not pull in ``pandas`` or any evaluation dependencies just to clean
``<think>`` tags out of a model response.
"""

from __future__ import annotations

import re

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class ThinkTagStreamFilter:
    """Incrementally suppress ``<think>`` blocks from a token stream."""

    def __init__(self) -> None:
        self._pending = ""
        self._in_thinking = False

    def feed(self, chunk: str) -> list[str]:
        """Return zero or more visible answer deltas from one streamed chunk."""
        if not chunk:
            return []

        self._pending += chunk
        emitted: list[str] = []

        while self._pending:
            if self._in_thinking:
                close_idx = self._pending.find(_THINK_CLOSE)
                if close_idx == -1:
                    self._pending = ""
                    break
                self._pending = self._pending[close_idx + len(_THINK_CLOSE) :]
                self._in_thinking = False
                continue

            open_idx = self._pending.find(_THINK_OPEN)
            if open_idx == -1:
                safe, self._pending = _split_safe_suffix(self._pending, _THINK_OPEN)
                if safe:
                    emitted.append(safe)
                break

            if open_idx > 0:
                emitted.append(self._pending[:open_idx])
            self._pending = self._pending[open_idx + len(_THINK_OPEN) :]
            self._in_thinking = True

        return emitted


def _split_safe_suffix(text: str, sentinel: str) -> tuple[str, str]:
    """Split *text* into (safe_to_emit, suffix_that_may_prefix_sentinel)."""
    max_prefix = min(len(text), len(sentinel) - 1)
    for prefix_len in range(max_prefix, 0, -1):
        if sentinel.startswith(text[-prefix_len:]):
            return text[:-prefix_len], text[-prefix_len:]
    return text, ""


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from model output.

    Handles both closed tags (``<think>...</think>``) and unclosed tags
    where the model hit the token limit mid-reasoning and never emitted
    ``</think>``.  Returns an empty string if nothing remains after
    stripping so callers can detect ``thinking_truncated``.
    """
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*", "", stripped, flags=re.DOTALL)
    return stripped.strip()
