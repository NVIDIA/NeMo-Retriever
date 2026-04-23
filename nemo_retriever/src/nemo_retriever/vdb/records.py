# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record conversion helpers for nv-ingest-client VDB adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _as_records(data: Any) -> list[dict[str, Any]]:
    if hasattr(data, "take_all"):
        data = data.take_all()
    if hasattr(data, "to_dict"):
        try:
            return list(data.to_dict(orient="records"))
        except TypeError:
            pass
    if isinstance(data, (str, bytes)):
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return list(data)
    return list(data) if isinstance(data, Iterable) else []


def _already_client_vdb_records(data: Any) -> bool:
    return (
        isinstance(data, list)
        and bool(data)
        and all(isinstance(group, list) for group in data)
        and all(
            isinstance(element, dict) and "document_type" in element and isinstance(element.get("metadata"), dict)
            for group in data
            for element in group
        )
    )


def _extract_embedding(row: dict[str, Any], metadata: dict[str, Any]) -> Any:
    if metadata.get("embedding") is not None:
        return metadata.get("embedding")
    payload = row.get("text_embeddings_1b_v2")
    if isinstance(payload, dict):
        return payload.get("embedding")
    return None


def _source_path(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    for value in (
        row.get("path"),
        row.get("source_id"),
        row.get("source"),
        metadata.get("source_path"),
        metadata.get("source_id"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _page_number(row: dict[str, Any], metadata: dict[str, Any]) -> int | None:
    content_metadata = metadata.get("content_metadata")
    if isinstance(content_metadata, dict) and content_metadata.get("page_number") is not None:
        try:
            return int(content_metadata["page_number"])
        except (TypeError, ValueError):
            return None
    if row.get("page_number") is not None:
        try:
            return int(row["page_number"])
        except (TypeError, ValueError):
            return None
    return None


def _client_record_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    metadata = dict(metadata)

    embedding = _extract_embedding(row, metadata)
    text = row.get("text") or metadata.get("content")
    if embedding is None or not text:
        return None

    page_number = _page_number(row, metadata)
    source_path = _source_path(row, metadata)
    source_name = Path(source_path).name if source_path else str(row.get("filename") or row.get("source_id") or "")

    content_metadata = metadata.get("content_metadata")
    content_metadata = dict(content_metadata) if isinstance(content_metadata, dict) else {}
    if page_number is not None:
        content_metadata.setdefault("page_number", page_number)
    if row.get("_content_type") or row.get("content_type"):
        content_metadata.setdefault("type", row.get("_content_type") or row.get("content_type"))
    if row.get("_stored_image_uri") or row.get("stored_image_uri"):
        content_metadata.setdefault("stored_image_uri", row.get("_stored_image_uri") or row.get("stored_image_uri"))
    if row.get("_bbox_xyxy_norm") or row.get("bbox_xyxy_norm"):
        content_metadata.setdefault("bbox_xyxy_norm", row.get("_bbox_xyxy_norm") or row.get("bbox_xyxy_norm"))

    source_metadata = metadata.get("source_metadata")
    source_metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
    if source_path:
        source_metadata.setdefault("source_id", source_path)
    if source_name:
        source_metadata.setdefault("source_name", source_name)

    record_metadata = dict(metadata)
    record_metadata["embedding"] = embedding
    record_metadata["content"] = str(text)
    record_metadata["content_metadata"] = content_metadata
    record_metadata["source_metadata"] = source_metadata

    return {"document_type": "text", "metadata": record_metadata}


def to_client_vdb_records(data: Any) -> list[list[dict[str, Any]]]:
    """Convert Retriever graph output into the nested record shape expected by client VDBs."""
    if data is None:
        return []
    if _already_client_vdb_records(data):
        return data

    records: list[dict[str, Any]] = []
    for row in _as_records(data):
        if isinstance(row, list):
            for item in row:
                if isinstance(item, dict):
                    record = _client_record_from_row(item)
                    if record is not None:
                        records.append(record)
            continue
        if not isinstance(row, dict):
            continue
        record = _client_record_from_row(row)
        if record is not None:
            records.append(record)
    return [records] if records else []


def _parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_hit(hit: dict[str, Any]) -> dict[str, Any]:
    entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else hit

    source = _parse_mapping(entity.get("source")) or _parse_mapping(hit.get("source"))
    content_metadata = (
        _parse_mapping(entity.get("content_metadata"))
        or _parse_mapping(entity.get("metadata"))
        or _parse_mapping(hit.get("metadata"))
    )
    if not source and isinstance(entity.get("source"), str):
        source = {"source_id": entity.get("source")}
    if isinstance(source.get("source_id"), str):
        parsed_source_id = _parse_mapping(source["source_id"])
        if parsed_source_id.get("source_id"):
            source = parsed_source_id
    if set(content_metadata) == {"page_number"} and isinstance(content_metadata.get("page_number"), str):
        parsed_content_metadata = _parse_mapping(content_metadata["page_number"])
        if parsed_content_metadata:
            content_metadata = parsed_content_metadata

    source_id = str(source.get("source_id") or source.get("source_name") or hit.get("source_id") or "").strip()
    page_number = content_metadata.get("page_number") if isinstance(content_metadata, dict) else None
    try:
        page_number = int(page_number) if page_number is not None else None
    except (TypeError, ValueError):
        page_number = None

    path = Path(source_id) if source_id else None
    pdf_basename = path.stem if path is not None else ""
    normalized = {
        "text": entity.get("text") or hit.get("text") or "",
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


def normalize_retrieval_results(results: Any) -> list[list[dict[str, Any]]]:
    if results is None:
        return []
    if isinstance(results, dict):
        results = [[results]]
    normalized: list[list[dict[str, Any]]] = []
    for hits in results:
        if isinstance(hits, dict):
            hits = [hits]
        normalized.append([_normalize_hit(hit) for hit in hits if isinstance(hit, dict)])
    return normalized
