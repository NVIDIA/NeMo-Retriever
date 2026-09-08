# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical metadata contract for lossless embedding-input split children."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

EMBEDDING_SPLIT_METADATA_KEY = "embedding_split"


def embedding_split_metadata(
    *,
    content: str,
    parent_id: str,
    chunk_id: str,
    chunk_index: int,
    chunk_count: int,
    start_token: int,
    end_token: int,
) -> dict[str, Any]:
    """Return canonical metadata for one lossless split child."""
    return {
        EMBEDDING_SPLIT_METADATA_KEY: {
            "content": content,
            "parent_id": parent_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "start_token": start_token,
            "end_token": end_token,
        }
    }


def format_embedding_input(text: Any, prefix: str, *, prefix_if_missing: bool = False) -> str:
    """Apply the exact prefix rule shared by admission and local inference."""
    raw = str(text)
    if prefix_if_missing and prefix and raw.lower().startswith(prefix.lower()):
        return raw
    return f"{prefix}{raw}"


@dataclass(frozen=True)
class SelectedEmbeddingText:
    """The source column and exact text selected for one embedding row."""

    column: str
    content: str
    is_split_child: bool = False


def select_embedding_text(row: Mapping[str, Any], *, text_column: str = "text") -> SelectedEmbeddingText | None:
    """Select embedding text using the one canonical fallback order.

    Split provenance wins because its ``content`` field is the exact reversible
    child text, including whitespace-only children.
    """
    split_content = embedding_split_content(row.get("metadata"))
    if split_content is not None:
        for column in dict.fromkeys((text_column, "text", "content", "chunk", "page_text")):
            value = row.get(column)
            if isinstance(value, str) and value == split_content:
                return SelectedEmbeddingText(column=column, content=split_content, is_split_child=True)
        return SelectedEmbeddingText(column=text_column, content=split_content, is_split_child=True)

    for column in dict.fromkeys((text_column, "text", "content", "chunk", "page_text")):
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return SelectedEmbeddingText(column=column, content=value)
    return None


def embedding_text_input(
    row: Mapping[str, Any],
    *,
    text_column: str = "text",
    default_modality: str = "text",
) -> str | None:
    """Return the exact runtime text, preserving split children byte-for-byte."""
    selected = select_embedding_text(row, text_column=text_column)
    if selected is None:
        return None
    requested_modality = embedding_runtime_modality(row, default_modality=default_modality)
    if selected.is_split_child or requested_modality == "text_image":
        return selected.content
    return selected.content.strip()


def embedding_record_content(row: Mapping[str, Any], *, text_column: str = "text") -> str | None:
    """Return exact searchable text without exposing split metadata layout to callers."""
    selected = select_embedding_text(row, text_column=text_column)
    if selected is not None:
        return selected.content
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        content = metadata.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def embedding_runtime_modality(row: Mapping[str, Any], *, default_modality: str = "text") -> str:
    """Resolve the requested runtime modality, inheriting the default only for nulls."""
    value = row.get("_embed_modality")
    is_missing = value is None or (pd.api.types.is_scalar(value) and bool(pd.isna(value)))
    return str(default_modality if is_missing else value)


def requires_text_admission(row: Mapping[str, Any], *, default_modality: str = "text") -> bool:
    """Return whether the row will reach a text embedder and needs admission."""
    modality = embedding_runtime_modality(row, default_modality=default_modality)
    if modality == "image":
        return False
    image = row.get("_image_b64")
    return modality != "text_image" or not (isinstance(image, str) and image.strip())


def embedding_split_content(metadata: Any) -> str | None:
    """Return the exact child text when *metadata* identifies a split child."""
    if not isinstance(metadata, Mapping):
        return None
    split = metadata.get(EMBEDDING_SPLIT_METADATA_KEY)
    if not isinstance(split, Mapping):
        return None
    chunk_id = split.get("chunk_id")
    content = split.get("content")
    if isinstance(chunk_id, str) and chunk_id.strip() and isinstance(content, str):
        return content
    return None


def embedding_split_id(metadata: Any) -> str | None:
    """Return the canonical child ID when present."""
    if not isinstance(metadata, Mapping):
        return None
    split = metadata.get(EMBEDDING_SPLIT_METADATA_KEY)
    if not isinstance(split, Mapping):
        return None
    chunk_id = split.get("chunk_id")
    return chunk_id if isinstance(chunk_id, str) and chunk_id.strip() else None


__all__ = [
    "EMBEDDING_SPLIT_METADATA_KEY",
    "SelectedEmbeddingText",
    "embedding_record_content",
    "embedding_runtime_modality",
    "embedding_split_metadata",
    "embedding_text_input",
    "embedding_split_content",
    "embedding_split_id",
    "format_embedding_input",
    "requires_text_admission",
    "select_embedding_text",
]
