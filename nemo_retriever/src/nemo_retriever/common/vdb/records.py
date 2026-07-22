# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record adapters for graph VDB upload and retrieval."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

from nemo_retriever.common.api.util.converters.pagetools import (
    normalize_one_based_page_number,
)
from nemo_retriever.common.vdb.lancedb_capabilities import LanceRetrievalMode


class RetrievalContractError(RuntimeError):
    """A backend retrieval result cannot satisfy the canonical hit contract."""


_LEGACY_ENTITY_FIELDS = frozenset(
    {
        "content",
        "content_metadata",
        "metadata",
        "page_number",
        "path",
        "pdf_basename",
        "pdf_page",
        "source",
        "source_id",
        "source_metadata",
        "text",
    }
)

_NATIVE_SCORE_FIELDS = frozenset({"_distance", "_score", "_relevance_score"})


class RetrievalHit(TypedDict, total=False):
    """Shape of a single hit returned by ``Retriever.query`` / ``Retriever.queries``.

    ``metadata`` is a native ``dict`` at this boundary — never a JSON string. The
    LanceDB storage layer JSON-encodes on write and decodes on read; do not let
    a re-encoded string leak back out here. See ``_normalize_hit`` for the
    contract enforcement point.

    ``_distance`` is a backend-native vector distance, ``_score`` is a
    backend-native FTS/BM25 score, and ``_relevance_score`` is a native hybrid
    reranker value. ``score`` is NRL's public query-relative relevance in
    ``[0, 1]`` and is not a probability or comparable across queries.

    ``total=False`` because optional fields (``stored_image_uri``,
    ``content_type``, ``bbox_xyxy_norm``, scores) are only set when present.
    """

    text: str
    metadata: dict[str, Any]
    source: str
    source_id: str
    path: str
    page_number: int | None
    pdf_basename: str
    pdf_page: str
    stored_image_uri: str
    content_type: str
    bbox_xyxy_norm: list[float]
    _distance: float
    _score: float
    _relevance_score: float
    score: float
    chunk_id: str
    document_id: str
    filename: str
    document_version: str
    content_sha256: str


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


def _text_from_graph_row(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Return the first nonblank text field without rewriting its contents."""
    for value in (row.get("text"), row.get("content"), metadata.get("content")):
        if isinstance(value, str) and value.strip():
            return value
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


def _is_image_backed_row(row: dict[str, Any]) -> bool:
    """Return whether a post-embed graph row retains its image or stored URI."""
    return bool(
        _first_str(
            row.get("_image_b64"),
            row.get("_stored_image_uri"),
            row.get("stored_image_uri"),
        )
    )


def _derive_fidelity(content_type: Any, metadata: dict[str, Any], content_metadata: dict[str, Any]) -> str | None:
    """Map a chunk's modality + real provenance signals to a trust tier.

    verbatim (PDF text layer) > ocr (scanned/region OCR) > transcribed (ASR) >
    vlm_caption (chart/image model caption). Returns None for unknown types so
    the field is omitted rather than guessed.
    """
    t = str(content_type or "").lower()
    if t in ("audio", "video", "video_frame"):
        return "transcribed"
    if t == "image":
        return "ocr" if content_metadata.get("subtype") == "page_image" else "vlm_caption"
    if t.startswith(("table", "chart", "infographic")):
        return "ocr"
    if t == "text":
        return "ocr" if metadata.get("needs_ocr_for_text") is True else "verbatim"
    return None


def _client_record_from_graph_row(row: dict[str, Any], *, require_embedding: bool = True) -> dict[str, Any] | None:
    metadata = _dict_or_empty(row.get("metadata"))

    embedding = _embedding_from_graph_row(row, metadata)
    text = _text_from_graph_row(row, metadata)
    if require_embedding and embedding is None:
        return None
    image_only = require_embedding and not text and _is_image_backed_row(row)
    if not text and not image_only:
        return None

    content_metadata = _dict_or_empty(metadata.get("content_metadata"))
    page_number = _optional_int(content_metadata.get("page_number"))
    if page_number is None:
        page_number = _optional_int(row.get("page_number"))
    if page_number is not None:
        content_metadata.setdefault("page_number", page_number)

    if image_only:
        content_type = "image"
        content_metadata["type"] = content_type
        content_metadata.pop("fidelity", None)
    else:
        content_type = row.get("_content_type") or row.get("content_type")
        if content_type:
            content_metadata.setdefault("type", content_type)
        fidelity = _derive_fidelity(content_type, metadata, content_metadata)
        if fidelity:
            content_metadata.setdefault("fidelity", fidelity)
    stored_image_uri = _first_str(row.get("_stored_image_uri"), row.get("stored_image_uri"))
    if stored_image_uri:
        content_metadata.setdefault("stored_image_uri", stored_image_uri)
    bbox = row.get("_bbox_xyxy_norm") or row.get("bbox_xyxy_norm")
    if bbox:
        content_metadata.setdefault("bbox_xyxy_norm", bbox)

    for key in (
        "segment_start_seconds",
        "segment_end_seconds",
        "frame_timestamp_seconds",
    ):
        if key in metadata:
            content_metadata.setdefault(key, metadata[key])

    source_path = _first_str(
        metadata.get("source_path"),
        row.get("path"),
        row.get("source_id"),
        row.get("source"),
        metadata.get("source_id"),
    )
    source_name = Path(source_path).name if source_path else str(row.get("filename") or row.get("source_id") or "")
    source_metadata = _dict_or_empty(metadata.get("source_metadata"))
    if source_path:
        source_metadata.setdefault("source_id", source_path)
    if source_name:
        source_metadata.setdefault("source_name", source_name)

    record_metadata = dict(metadata)
    if embedding is not None:
        record_metadata["embedding"] = embedding
    record_metadata["content"] = text
    record_metadata["content_metadata"] = content_metadata
    record_metadata["source_metadata"] = source_metadata

    document_type = "image" if image_only else row.get("document_type") or "text"
    return {"document_type": str(document_type), "metadata": record_metadata}


def to_client_vdb_records(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Convert graph-ingest rows into the nested record shape expected by client VDBs.

    Dense rows require an embedding and either nonblank text or concrete image backing.
    When no row survives conversion, returns ``[]`` — a falsy value so
    ``if not records`` skips :meth:`~nemo_retriever.vdb.adt_vdb.VDB.run`.
    When at least one row converts, returns ``[batch]`` with a single non-empty inner list
    (never ``[[]]``, which would be truthy and could trip backends on an empty insert).
    """
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    # Walrus: bind conversion once per row — a plain ``if f(row)`` + ``f(row)`` list comp
    # would call _client_record_from_graph_row twice per row on large datasets.
    # isinstance(row, dict): plain lists are not normalized like DataFrame rows; skip None/Series/etc.
    inner = [
        record
        for row in rows or []
        if isinstance(row, dict) and (record := _client_record_from_graph_row(row)) is not None
    ]
    # Preserve legacy contract: no uploadable rows → [], not [[]].
    return [inner] if inner else []


def to_sparse_client_vdb_records(rows: Any) -> list[list[dict[str, Any]]]:
    """Convert graph-ingest rows into text/provenance records for sparse LanceDB ingest."""
    if hasattr(rows, "to_pandas"):
        rows = rows.to_pandas()
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    if isinstance(rows, list) and all(isinstance(batch, list) for batch in rows):
        nested = [[record for record in batch if isinstance(record, dict) and record.get("metadata")] for batch in rows]
        nested = [batch for batch in nested if batch]
        return nested
    inner = [
        record
        for row in rows or []
        if isinstance(row, dict) and (record := _client_record_from_graph_row(row, require_embedding=False)) is not None
    ]
    return [inner] if inner else []


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


def _flatten_legacy_entity_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Flatten the legacy nested ``entity`` shape into canonical fields.

    Only fields that predate collection management are accepted from the
    nested shape. Top-level values are authoritative when both shapes provide
    the same field; collection identity and version fields must be top-level.
    """

    top_level = {key: value for key, value in hit.items() if key != "entity"}
    entity = hit.get("entity")
    if not isinstance(entity, dict):
        return top_level
    legacy = {key: entity[key] for key in _LEGACY_ENTITY_FIELDS if key in entity}
    return {**legacy, **top_level}


def _normalize_hit(hit: dict[str, Any]) -> RetrievalHit:
    """Adapt LanceDB client hit shapes to Retriever hits."""
    hit = _flatten_legacy_entity_hit(hit)

    source = _mapping(hit.get("source") or hit.get("source_metadata"))
    if not source and isinstance(hit.get("source"), str):
        source = {"source_id": hit["source"]}
    content_metadata = _mapping(hit.get("content_metadata") or hit.get("metadata"))

    source_id = _first_str(
        source.get("source_id"),
        source.get("source_name"),
        hit.get("source_id"),
        hit.get("path"),
    )
    page_number = content_metadata.get("page_number") if isinstance(content_metadata, dict) else None
    if page_number is None:
        page_number = hit.get("page_number")
    page_number = normalize_one_based_page_number(page_number)

    path = Path(source_id) if source_id else None
    pdf_basename = path.stem if path is not None else ""
    normalized: RetrievalHit = {
        "text": _first_str(hit.get("text"), hit.get("content")),
        # Keep `metadata` as a native dict on the API boundary. The LanceDB
        # storage layer JSON-encodes it on write (see `_json_str` in
        # `vdb/lancedb.py`); we already parse it back on read in
        # `LanceDB.retrieval`. Re-encoding it here forced every downstream
        # consumer (`Retriever.query()` callers, the CLI, the SKILL.md jq
        # recipe) to do its own `fromjson`/`json.loads` — and most didn't,
        # producing silent `metadata.type == "?"` lookups.
        "metadata": content_metadata,
        "source": source_id,
        "source_id": source_id,
        "path": source_id,
        "page_number": page_number,
        "pdf_basename": pdf_basename,
        "pdf_page": (f"{pdf_basename}_{page_number}" if pdf_basename and page_number is not None else ""),
    }
    chunk_id = hit.get("chunk_id")
    if chunk_id:
        normalized.update(
            {
                "chunk_id": str(chunk_id),
                "document_id": str(hit.get("document_id") or ""),
                "filename": str(hit.get("filename") or (path.name if path else "")),
                "document_version": str(hit.get("document_version") or ""),
                "content_sha256": str(hit.get("content_sha256") or ""),
            }
        )
    for key in (
        "stored_image_uri",
        "content_type",
        "bbox_xyxy_norm",
        "_distance",
        "_score",
        "_relevance_score",
    ):
        if key in hit:
            normalized[key] = hit[key]
    return normalized


def _normalize_query_scores(hits: list[RetrievalHit], retrieval_mode: LanceRetrievalMode) -> None:
    """Attach query-relative public scores using the mode's native field."""

    if not hits:
        return
    if retrieval_mode == "dense":
        field = "_distance"
        lower_is_better = True
    elif retrieval_mode == "hybrid":
        field = "_relevance_score"
        lower_is_better = False
    else:
        raise RetrievalContractError(f"unsupported retrieval mode for public scoring: {retrieval_mode}")

    values: list[float] = []
    for index, hit in enumerate(hits):
        raw = hit.get(field)
        if isinstance(raw, bool):
            raise RetrievalContractError(f"{retrieval_mode} hit {index} is missing a numeric {field}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise RetrievalContractError(f"{retrieval_mode} hit {index} is missing a numeric {field}") from exc
        if not math.isfinite(value):
            raise RetrievalContractError(f"{retrieval_mode} hit {index} has a non-finite {field}")
        values.append(value)

    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    for hit, value in zip(hits, values, strict=True):
        if span == 0.0:
            score = 1.0
        elif lower_is_better:
            score = (maximum - value) / span
        else:
            score = (value - minimum) / span
        hit["score"] = max(0.0, min(1.0, score))


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


def normalize_retrieval_results(
    results: Any,
    *,
    retrieval_mode: LanceRetrievalMode | None = None,
) -> list[list[RetrievalHit]]:
    """Canonicalize backend results and optionally attach public scores."""

    if results is None:
        return []
    if isinstance(results, dict):
        results = [[results]]
    normalized: list[list[RetrievalHit]] = []
    for hits in results:
        if isinstance(hits, dict):
            hits = [hits]
        normalized_hits: list[RetrievalHit] = []
        for hit in hits:
            hit_dict = _hit_to_dict(hit)
            if hit_dict is not None:
                normalized_hits.append(_normalize_hit(hit_dict))
        if retrieval_mode is not None:
            _normalize_query_scores(normalized_hits, retrieval_mode)
        normalized.append(normalized_hits)
    return normalized


def without_native_scores(hit: RetrievalHit) -> RetrievalHit:
    """Return a public collection hit without backend-native score fields."""

    return RetrievalHit({key: value for key, value in hit.items() if key not in _NATIVE_SCORE_FIELDS})
