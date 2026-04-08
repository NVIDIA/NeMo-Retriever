# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-friendly content row transforms used by example pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from nemo_retriever.io.image_store import load_image_b64_from_uri
from nemo_retriever.ocr.ocr import _crop_b64_image_by_norm_bbox
from nemo_retriever.params.models import IMAGE_MODALITIES


def _resolve_image_b64(container: dict) -> Optional[str]:
    """Return image_b64 from a dict, falling back to stored_image_uri if stripped."""
    b64 = container.get("image_b64")
    if isinstance(b64, str) and b64:
        return b64
    uri = container.get("stored_image_uri")
    if isinstance(uri, str) and uri:
        return load_image_b64_from_uri(uri)
    return None

_CONTENT_COLUMNS = ("table", "chart", "infographic")


def _combine_text_with_content(row: Any, text_column: str, content_columns: Sequence[str]) -> str:
    """Combine page text with OCR content text for embedding."""
    parts = []
    base = row.get(text_column)
    if isinstance(base, str) and base.strip():
        parts.append(base.strip())
    for col in content_columns:
        content_list = row.get(col)
        if isinstance(content_list, list):
            for item in content_list:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                    caption = item.get("caption", "")
                    if isinstance(caption, str) and caption.strip():
                        parts.append(caption.strip())
    return "\n\n".join(parts) if parts else ""


def _deep_copy_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Copy row dicts without sharing nested mutable values across exploded rows."""
    import copy

    out: Dict[str, Any] = {}
    for key, value in row_dict.items():
        if isinstance(value, (dict, list)):
            out[key] = copy.deepcopy(value)
        else:
            out[key] = value
    return out


def explode_content_to_rows(
    batch_df: Any,
    *,
    text_column: str = "text",
    content_columns: Sequence[str] = _CONTENT_COLUMNS,
    modality: str = "text",
    text_elements_modality: Optional[str] = None,
    structured_elements_modality: Optional[str] = None,
) -> Any:
    """Expand each page row into multiple rows for per-element embedding."""
    text_mod = text_elements_modality or modality
    struct_mod = structured_elements_modality or modality

    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    any_images = text_mod in IMAGE_MODALITIES or struct_mod in IMAGE_MODALITIES

    if not any(column in batch_df.columns for column in content_columns):
        batch_df = batch_df.copy()
        if text_mod in IMAGE_MODALITIES and "page_image" in batch_df.columns:
            batch_df["_image_b64"] = batch_df["page_image"].apply(
                lambda page_image: _resolve_image_b64(page_image) if isinstance(page_image, dict) else None
            )
        if "page_image" in batch_df.columns:
            batch_df["_stored_image_uri"] = batch_df["page_image"].apply(
                lambda page_image: page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
            )
        batch_df["_embed_modality"] = text_mod
        return batch_df

    new_rows: List[Dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        row_dict = row.to_dict()
        exploded_any = False

        page_image = row_dict.get("page_image")
        page_image_b64: Optional[str] = None
        page_stored_uri: Optional[str] = None
        if isinstance(page_image, dict):
            page_stored_uri = page_image.get("stored_image_uri")
            if any_images:
                page_image_b64 = _resolve_image_b64(page_image)

        page_text = row_dict.get(text_column)
        if isinstance(page_text, str) and page_text.strip():
            page_row = _deep_copy_row(row_dict)
            page_row["_embed_modality"] = text_mod
            page_row["_content_type"] = "text"
            if text_mod in IMAGE_MODALITIES:
                page_row["_image_b64"] = page_image_b64
            page_row["_stored_image_uri"] = page_stored_uri
            new_rows.append(page_row)
            exploded_any = True

        for column in content_columns:
            content_list = row_dict.get(column)
            if not isinstance(content_list, list):
                continue
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                # Emit rows for text and (optionally) caption fields.
                for field, content_type in [("text", column), ("caption", f"{column}_caption")]:
                    value = item.get(field, "")
                    if not isinstance(value, str) or not value.strip():
                        continue
                    content_row = _deep_copy_row(row_dict)
                    content_row[text_column] = value.strip()
                    content_row["_embed_modality"] = struct_mod
                    content_row["_content_type"] = content_type
                    if struct_mod in IMAGE_MODALITIES:
                        item_b64 = _resolve_image_b64(item)
                        if item_b64:
                            content_row["_image_b64"] = item_b64
                        elif page_image_b64:
                            bbox = item.get("bbox_xyxy_norm")
                            if bbox and len(bbox) == 4:
                                cropped_b64, _ = _crop_b64_image_by_norm_bbox(page_image_b64, bbox_xyxy_norm=bbox)
                                content_row["_image_b64"] = cropped_b64
                            else:
                                content_row["_image_b64"] = page_image_b64
                        else:
                            content_row["_image_b64"] = None
                    content_row["_stored_image_uri"] = item.get("stored_image_uri") or page_stored_uri
                    new_rows.append(content_row)
                    exploded_any = True

        if not exploded_any:
            preserved = _deep_copy_row(row_dict)
            preserved["_embed_modality"] = text_mod
            preserved["_content_type"] = "text"
            if text_mod in IMAGE_MODALITIES:
                preserved["_image_b64"] = page_image_b64
            preserved["_stored_image_uri"] = page_stored_uri
            new_rows.append(preserved)

    return pd.DataFrame(new_rows).reset_index(drop=True)


def collapse_content_to_page_rows(
    batch_df: Any,
    *,
    text_column: str = "text",
    content_columns: Sequence[str] = _CONTENT_COLUMNS,
    modality: str = "text",
) -> Any:
    """Collapse each page into a single row for page-level embedding."""
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    batch_df = batch_df.copy()
    batch_df[text_column] = batch_df.apply(
        lambda row: _combine_text_with_content(row, text_column, content_columns),
        axis=1,
    )

    if modality in IMAGE_MODALITIES:
        if "page_image" in batch_df.columns:
            batch_df["_image_b64"] = batch_df["page_image"].apply(
                lambda page_image: _resolve_image_b64(page_image) if isinstance(page_image, dict) else None
            )
        else:
            batch_df["_image_b64"] = None

    if "page_image" in batch_df.columns:
        batch_df["_stored_image_uri"] = batch_df["page_image"].apply(
            lambda page_image: page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
        )

    batch_df["_embed_modality"] = modality
    return batch_df
