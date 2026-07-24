# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable, backend-agnostic helpers for LLM backend implementations.

Nothing here is required by the ``BaseLLMBackend`` contract; these exist so
custom backends don't re-invent the fiddly parts (most OpenAI-compatible
backends can use them verbatim).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)


def strip_private_message_keys(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return copies of ``messages`` without ``__``-prefixed top-level keys.

    Agent code stashes non-API metadata on history messages under dunder-style
    keys (e.g. ``"__reasoning__"``). Backends should strip these before the
    wire rather than rely on provider tolerance. Never mutates the input.
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict):
            out.append({k: v for k, v in msg.items() if not (isinstance(k, str) and k.startswith("__"))})
        else:
            out.append(msg)
    return out


def normalize_messages_for_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize message content from list-of-content-blocks to plain strings.

    Some OpenAI-compatible endpoints only accept string content for certain
    roles. This converts text-only ``content`` lists (e.g.
    ``[{"type": "text", "text": "..."}]``) into a plain string. Messages with
    non-text blocks (e.g. ``image_url``) are left as-is. Block-level metadata
    (e.g. ``cache_control``) is discarded when a text-only list is collapsed,
    so run this BEFORE adding block-level markers. Never mutates the input.
    """
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        msg = dict(msg)
        content = msg.get("content")
        if isinstance(content, list):
            text_parts: List[str] = []
            all_text = True
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                else:
                    all_text = False
                    break
            if all_text:
                if len(text_parts) == 0:
                    msg["content"] = None
                elif len(text_parts) == 1:
                    msg["content"] = text_parts[0]
                else:
                    msg["content"] = "\n".join(text_parts)
        normalized.append(msg)
    return normalized


def extract_reasoning_from_message(message: object) -> Optional[str]:
    """Best-effort extraction of the per-turn reasoning trace.

    Accepts either a provider message *object* (attribute access) or a plain
    message *dict* (key access). Handles three exposure shapes observed across
    providers:

    1. ``reasoning_content`` (gpt-oss via NIM, GLM, DeepSeek-R1)
    2. ``thinking_blocks`` (Anthropic extended thinking)
    3. Inline ``<think>...</think>`` blocks in ``content``
       (Tongyi-DeepResearch and similar ReAct-style models)

    Returns ``None`` when no reasoning channel is populated.
    """
    if message is None:
        return None

    reasoning_content = _coerce_str(_get_field(message, "reasoning_content"))
    if reasoning_content:
        return reasoning_content

    thinking_blocks = _get_field(message, "thinking_blocks")
    if thinking_blocks:
        parts: List[str] = []
        for block in thinking_blocks:
            if isinstance(block, dict):
                text = block.get("thinking") or block.get("text") or ""
            else:
                text = getattr(block, "thinking", None) or getattr(block, "text", None) or ""
            if text:
                parts.append(str(text).strip())
        if parts:
            return "\n".join(parts)

    content = _get_field(message, "content")
    if isinstance(content, str) and "<think>" in content and "</think>" in content:
        # Capture the last <think>...</think> block; if multiple are present
        # the final one reflects the model's most recent reasoning state.
        matches = _THINK_BLOCK_RE.findall(content)
        if matches:
            tail = matches[-1].strip()
            if tail:
                return tail

    return None


def _get_field(message: object, name: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(name)
    return getattr(message, name, None)


def _coerce_str(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v if v else None
    try:
        v = str(value).strip()
        return v if v else None
    except Exception:
        return None
