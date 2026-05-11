# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import math
from typing import Any, Iterable

from nemo_retriever.agent_mcp.models import AgentMcpError, EvidenceArtifacts, EvidenceHit, Locator
from nemo_retriever.agent_mcp.paths import media_type_for_path


def _parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        stringified = str(value)
        if stringified:
            return stringified
    return ""


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError):
        return None


def _score(hit: dict[str, Any]) -> float | None:
    for key in ("_rerank_score", "rerank_score", "score", "_distance"):
        if key in hit:
            return _float_or_none(hit.get(key))
    return None


def _media_type(source_path: str) -> str:
    if not source_path:
        return "unknown"
    try:
        return media_type_for_path(source_path)
    except AgentMcpError:
        return "unknown"


def _first_present(hit: dict[str, Any], metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in hit:
            return hit[key]
        if key in metadata:
            return metadata[key]
    return None


def _bbox_or_none(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError, TypeError):
                return None

    if isinstance(value, dict):
        for nested_key in ("bbox_xyxy_norm", "bbox"):
            if nested_key in value:
                return _bbox_or_none(value[nested_key])
        value = [value.get(key) for key in ("x1", "y1", "x2", "y2")]

    if not isinstance(value, (list, tuple)):
        return None

    if len(value) != 4:
        return None

    bbox = [_float_or_none(item) for item in value]
    if any(item is None for item in bbox):
        return None
    return [item for item in bbox if item is not None]


def _locator(hit: dict[str, Any], metadata: dict[str, Any]) -> Locator:
    timestamp_start = _first_present(hit, metadata, "timestamp_start_s", "segment_start_seconds")
    timestamp_end = _first_present(hit, metadata, "timestamp_end_s", "segment_end_seconds")

    return Locator(
        page_number=_int_or_none(_first_present(hit, metadata, "page_number")),
        timestamp_start_s=_float_or_none(timestamp_start),
        timestamp_end_s=_float_or_none(timestamp_end),
        frame_index=_int_or_none(_first_present(hit, metadata, "frame_index")),
        bbox_xyxy_norm=_bbox_or_none(_first_present(hit, metadata, "bbox_xyxy_norm", "bbox")),
        chunk_id=_first_nonempty(_first_present(hit, metadata, "chunk_id", "chunk")) or None,
    )


def normalize_hit(hit: dict[str, Any]) -> EvidenceHit:
    metadata = _parse_mapping(hit.get("metadata"))
    source = _parse_mapping(hit.get("source"))
    source_path = _first_nonempty(
        hit.get("source_path"),
        hit.get("path"),
        hit.get("source_id"),
        source.get("source_id"),
        source.get("source_name"),
    )
    output_metadata = dict(metadata)
    output_metadata["_raw_hit_keys"] = sorted(hit)

    return EvidenceHit(
        text=_first_nonempty(hit.get("text")),
        score=_score(hit),
        source_path=source_path,
        media_type=_media_type(source_path),
        content_type=_first_nonempty(
            hit.get("content_type"),
            metadata.get("content_type"),
            metadata.get("type"),
            "unknown",
        ),
        locator=_locator(hit, metadata),
        artifacts=EvidenceArtifacts(
            stored_image_uri=_first_nonempty(hit.get("stored_image_uri"), metadata.get("stored_image_uri")) or None,
            thumbnail_uri=_first_nonempty(hit.get("thumbnail_uri"), metadata.get("thumbnail_uri")) or None,
        ),
        metadata=output_metadata,
    )


def normalize_hits(hits: Iterable[dict[str, Any]]) -> list[EvidenceHit]:
    return [normalize_hit(hit) for hit in hits]
