# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared LanceDB row construction, schema, and table helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def extract_embedding_from_row(
    row: Any,
    *,
    embedding_column: str = "text_embeddings_1b_v2",
    embedding_key: str = "embedding",
) -> Optional[List[float]]:
    """Extract an embedding vector from a row."""
    meta = getattr(row, "metadata", None)
    if isinstance(meta, dict):
        emb = meta.get("embedding")
        if isinstance(emb, list) and emb:
            return emb  # type: ignore[return-value]

    payload = getattr(row, embedding_column, None)
    if isinstance(payload, dict):
        emb = payload.get(embedding_key)
        if isinstance(emb, list) and emb:
            return emb  # type: ignore[return-value]
    return None


def extract_source_path_and_page(row: Any) -> Tuple[str, int]:
    """Best-effort extract of source path and page number from a row."""
    path = ""
    page = -1

    value = getattr(row, "path", None)
    if isinstance(value, str) and value.strip():
        path = value.strip()

    value = getattr(row, "page_number", None)
    try:
        if value is not None:
            page = int(value)
    except Exception:
        pass

    meta = getattr(row, "metadata", None)
    if isinstance(meta, dict):
        source_path = meta.get("source_path")
        if isinstance(source_path, str) and source_path.strip():
            path = source_path.strip()
        content_metadata = meta.get("content_metadata")
        if isinstance(content_metadata, dict) and page == -1:
            hierarchy = content_metadata.get("hierarchy")
            if isinstance(hierarchy, dict) and "page" in hierarchy:
                try:
                    page = int(hierarchy.get("page"))
                except Exception:
                    pass

    return path, page


def _build_detection_metadata(row: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    page_element_count = getattr(row, "page_elements_v3_num_detections", None)
    if page_element_count is not None:
        try:
            out["page_elements_v3_num_detections"] = int(page_element_count)
        except Exception:
            pass

    page_element_counts = getattr(row, "page_elements_v3_counts_by_label", None)
    if isinstance(page_element_counts, dict):
        out["page_elements_v3_counts_by_label"] = {
            str(key): int(value)
            for key, value in page_element_counts.items()
            if isinstance(key, str) and value is not None
        }

    for ocr_column in ("table", "chart", "infographic"):
        entries = getattr(row, ocr_column, None)
        if isinstance(entries, list):
            out[f"ocr_{ocr_column}_detections"] = int(len(entries))

    return out


def build_lancedb_row(
    row: Any,
    *,
    embedding_column: str = "text_embeddings_1b_v2",
    embedding_key: str = "embedding",
    text_column: str = "text",
    include_text: bool = True,
) -> Optional[Dict[str, Any]]:
    """Build a single LanceDB-ready dict from a DataFrame row."""
    embedding = extract_embedding_from_row(row, embedding_column=embedding_column, embedding_key=embedding_key)
    if embedding is None:
        return None

    path, page_number = extract_source_path_and_page(row)
    path_obj = Path(path) if path else None
    filename = path_obj.name if path_obj is not None else ""
    pdf_basename = path_obj.stem if path_obj is not None else ""
    pdf_page = f"{pdf_basename}_{page_number}" if (pdf_basename and page_number >= 0) else ""
    source_id = path or filename or pdf_basename

    metadata_obj: Dict[str, Any] = {"page_number": int(page_number) if page_number is not None else -1}
    if pdf_page:
        metadata_obj["pdf_page"] = pdf_page
    metadata_obj.update(_build_detection_metadata(row))

    orig_meta = getattr(row, "metadata", None)
    if isinstance(orig_meta, dict):
        for key in ("chunk_index", "chunk_count"):
            if key in orig_meta:
                metadata_obj[key] = orig_meta[key]

    source_obj: Dict[str, Any] = {"source_id": str(path)}
    row_out: Dict[str, Any] = {
        "vector": embedding,
        "pdf_page": pdf_page,
        "filename": filename,
        "pdf_basename": pdf_basename,
        "page_number": int(page_number) if page_number is not None else -1,
        "source_id": str(source_id),
        "path": str(path),
        "metadata": json.dumps(metadata_obj, ensure_ascii=False),
        "source": json.dumps(source_obj, ensure_ascii=False),
    }

    if include_text:
        text = getattr(row, text_column, None)
        row_out["text"] = str(text) if isinstance(text, str) else ""
    else:
        row_out["text"] = ""

    stored_uri = getattr(row, "_stored_image_uri", None)
    row_out["stored_image_uri"] = str(stored_uri) if stored_uri else ""

    content_type = getattr(row, "_content_type", None)
    row_out["content_type"] = str(content_type) if content_type else ""

    bbox = getattr(row, "_bbox_xyxy_norm", None)
    row_out["bbox_xyxy_norm"] = json.dumps(bbox) if bbox else ""

    return row_out


def build_lancedb_rows(
    df: Any,
    *,
    embedding_column: str = "text_embeddings_1b_v2",
    embedding_key: str = "embedding",
    text_column: str = "text",
    include_text: bool = True,
) -> List[Dict[str, Any]]:
    """Build LanceDB rows from a pandas DataFrame."""
    rows: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        row_out = build_lancedb_row(
            row,
            embedding_column=embedding_column,
            embedding_key=embedding_key,
            text_column=text_column,
            include_text=include_text,
        )
        if row_out is not None:
            rows.append(row_out)
    return rows


def lancedb_schema(vector_dim: int = 2048) -> Any:
    """Return a PyArrow schema for the standard LanceDB table layout."""
    import pyarrow as pa  # type: ignore

    return pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            pa.field("pdf_page", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("pdf_basename", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("source", pa.string()),
            pa.field("source_id", pa.string()),
            pa.field("path", pa.string()),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("stored_image_uri", pa.string()),
            pa.field("content_type", pa.string()),
            pa.field("bbox_xyxy_norm", pa.string()),
        ]
    )


def infer_vector_dim(rows: List[Dict[str, Any]]) -> int:
    """Return the embedding dimension from the first row that has a vector."""
    for row in rows:
        vector = row.get("vector")
        if isinstance(vector, list) and vector:
            return len(vector)
    return 0


def create_or_append_lancedb_table(
    db: Any,
    table_name: str,
    rows: List[Dict[str, Any]],
    schema: Any,
    overwrite: bool = True,
) -> Any:
    """Create or append to a LanceDB table, returning the table object."""
    if overwrite:
        return db.create_table(str(table_name), data=list(rows), schema=schema, mode="overwrite")

    try:
        table = db.open_table(str(table_name))
        table.add(list(rows))
        return table
    except Exception:
        return db.create_table(str(table_name), data=list(rows), schema=schema, mode="create")
