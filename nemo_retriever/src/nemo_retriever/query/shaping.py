# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Sequence, cast

from nemo_retriever.common.vdb.lancedb_schema import normalize_content_type
from nemo_retriever.common.vdb.records import RetrievalHit
from nemo_retriever.common.vdb.sidecar_metadata import parse_hit_content_metadata


def normalize_query_content_type_allowlist(content_types: str | Sequence[str] | None) -> set[str] | None:
    """Normalize query-time content type filters to stored hit metadata values."""
    if content_types is None:
        return None
    raw_values: list[str]
    if isinstance(content_types, str):
        raw_values = content_types.split(",")
    else:
        raw_values = []
        for value in content_types:
            raw_values.extend(str(value).split(","))

    normalized = {content_type for value in raw_values if (content_type := normalize_content_type(value)) is not None}
    if not normalized:
        raise ValueError("content_types must include at least one non-empty content type.")
    return normalized


def _hit_content_type(hit: dict[str, Any]) -> str | None:
    metadata = parse_hit_content_metadata(hit)
    for value in (
        metadata.get("type"),
        metadata.get("_content_type"),
        hit.get("content_type"),
        hit.get("_content_type"),
    ):
        normalized = normalize_content_type(value)
        if normalized is not None:
            return normalized
    return None


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hit_page_key(hit: dict[str, Any]) -> tuple[str, int] | None:
    metadata = parse_hit_content_metadata(hit)
    page_number = _coerce_int_or_none(hit.get("page_number"))
    if page_number is None:
        page_number = _coerce_int_or_none(metadata.get("page_number"))
    if page_number is None:
        return None

    for value in (hit.get("pdf_basename"), metadata.get("pdf_basename")):
        if isinstance(value, str) and value.strip():
            return (value.strip(), page_number)

    raw_doc = (
        hit.get("source_id")
        or hit.get("source")
        or hit.get("path")
        or metadata.get("source_id")
        or metadata.get("source_name")
    )
    if not isinstance(raw_doc, str) or not raw_doc.strip():
        return None
    return (raw_doc.strip(), page_number)


def shape_query_hits(
    hits: Sequence[dict[str, Any]],
    *,
    top_k: int,
    page_dedup: bool = False,
    content_types: str | Sequence[str] | None = None,
) -> list[RetrievalHit]:
    """Apply query-time filtering, page deduplication, and final truncation."""
    allowed_types = normalize_query_content_type_allowlist(content_types)
    shaped: list[RetrievalHit] = []
    seen_pages: set[tuple[str, int]] = set()

    for hit in hits:
        if allowed_types is not None:
            hit_type = _hit_content_type(hit)
            if hit_type not in allowed_types:
                continue
        if page_dedup:
            page_key = _hit_page_key(hit)
            if page_key is not None:
                if page_key in seen_pages:
                    continue
                seen_pages.add(page_key)
        shaped.append(cast(RetrievalHit, dict(hit)))
        if len(shaped) >= top_k:
            break
    return shaped
