# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record adapters for the graph-pipeline VDB upload/retrieval path."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _embedding_from_graph_row(row: dict[str, Any], metadata: dict[str, Any]) -> Any:
    if metadata.get("embedding") is not None:
        return metadata["embedding"]
    payload = row.get("text_embeddings_1b_v2")
    return payload.get("embedding") if isinstance(payload, dict) else None


def _first_str(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _optional_int(value: Any) -> int | None:
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _client_record_from_graph_row(row: dict[str, Any]) -> dict[str, Any] | None:
    metadata = _dict_or_empty(row.get("metadata"))

    embedding = _embedding_from_graph_row(row, metadata)
    text = row.get("text") or row.get("content") or metadata.get("content")
    if embedding is None or not text:
        return None

    content_metadata = _dict_or_empty(metadata.get("content_metadata"))
    page_number = _optional_int(content_metadata.get("page_number"))
    if page_number is None:
        page_number = _optional_int(row.get("page_number"))
    if page_number is not None:
        content_metadata.setdefault("page_number", page_number)

    content_type = row.get("_content_type") or row.get("content_type")
    if content_type:
        content_metadata.setdefault("type", content_type)
    stored_image_uri = row.get("_stored_image_uri") or row.get("stored_image_uri")
    if stored_image_uri:
        content_metadata.setdefault("stored_image_uri", stored_image_uri)
    bbox = row.get("_bbox_xyxy_norm") or row.get("bbox_xyxy_norm")
    if bbox:
        content_metadata.setdefault("bbox_xyxy_norm", bbox)

    source_path = _first_str(
        row.get("path"),
        row.get("source_id"),
        row.get("source"),
        metadata.get("source_path"),
        metadata.get("source_id"),
    )
    source_name = Path(source_path).name if source_path else str(row.get("filename") or row.get("source_id") or "")
    source_metadata = _dict_or_empty(metadata.get("source_metadata"))
    if source_path:
        source_metadata.setdefault("source_id", source_path)
    if source_name:
        source_metadata.setdefault("source_name", source_name)

    record_metadata = dict(metadata)
    record_metadata["embedding"] = embedding
    record_metadata["content"] = str(text)
    record_metadata["content_metadata"] = content_metadata
    record_metadata["source_metadata"] = source_metadata

    return {"document_type": str(row.get("document_type") or "text"), "metadata": record_metadata}


def to_client_vdb_records(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Convert graph-pipeline rows into the nested record shape expected by client VDBs."""
    records: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        record = _client_record_from_graph_row(row)
        if record is not None:
            records.append(record)
    return [records] if records else []


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Adapt LanceDB client hit shapes to Retriever hits."""
    entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else hit

    source = _mapping(entity.get("source") or hit.get("source") or entity.get("source_metadata"))
    if not source and isinstance(entity.get("source"), str):
        source = {"source_id": entity["source"]}
    content_metadata = _mapping(entity.get("content_metadata") or hit.get("content_metadata") or entity.get("metadata"))

    source_id = _first_str(
        source.get("source_id"),
        source.get("source_name"),
        entity.get("source_id"),
        hit.get("source_id"),
        hit.get("path"),
    )
    page_number = content_metadata.get("page_number") if isinstance(content_metadata, dict) else None
    if page_number is None:
        page_number = entity.get("page_number", hit.get("page_number"))
    page_number = _optional_int(page_number)

    path = Path(source_id) if source_id else None
    pdf_basename = path.stem if path is not None else ""
    normalized = {
        "text": _first_str(entity.get("text"), entity.get("content"), hit.get("text")),
        "metadata": json.dumps(content_metadata, default=str),
        "source": source_id,
        "source_id": source_id,
        "path": source_id,
        "page_number": page_number,
        "pdf_basename": pdf_basename,
        "pdf_page": f"{pdf_basename}_{page_number}" if pdf_basename and page_number is not None else "",
    }
    for key in ("stored_image_uri", "content_type", "bbox_xyxy_norm", "_distance", "_score"):
        if key in hit:
            normalized[key] = hit[key]
        elif key in entity:
            normalized[key] = entity[key]
    return normalized


def _hit_to_dict(hit: Any) -> dict[str, Any] | None:
    if isinstance(hit, dict):
        return hit
    if isinstance(hit, Mapping):
        return dict(hit)
    if hasattr(hit, "to_dict"):
        try:
            converted = hit.to_dict()
        except Exception:
            return None
        return converted if isinstance(converted, dict) else None
    return None


def normalize_retrieval_results(results: Any) -> list[list[dict[str, Any]]]:
    if results is None:
        return []
    if isinstance(results, dict):
        results = [[results]]
    normalized: list[list[dict[str, Any]]] = []
    for hits in results:
        if isinstance(hits, dict):
            hits = [hits]
        normalized_hits = []
        for hit in hits:
            hit_dict = _hit_to_dict(hit)
            if hit_dict is not None:
                normalized_hits.append(_normalize_hit(hit_dict))
        normalized.append(normalized_hits)
    return normalized
