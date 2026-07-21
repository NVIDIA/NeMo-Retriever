# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Sequence, cast

from nemo_retriever.common.vdb.lancedb_schema import normalize_content_type, normalize_content_type_allowlist
from nemo_retriever.common.vdb.records import RetrievalHit
from nemo_retriever.common.vdb.sidecar_metadata import parse_hit_content_metadata


def resolve_hit_content_type(hit: dict[str, Any]) -> str | None:
    """Resolve a hit's canonical modality, preferring persisted metadata."""
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
    allowed_types = normalize_content_type_allowlist(content_types)
    shaped: list[RetrievalHit] = []
    seen_pages: set[tuple[str, int]] = set()

    for hit in hits:
        if allowed_types is not None:
            hit_type = resolve_hit_content_type(hit)
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
