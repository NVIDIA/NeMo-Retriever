# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-friendly content row transforms used by example pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from nemo_retriever.common.io.image_store import inline_image_b64
from nemo_retriever.operators.extract.ocr.ocr import _crop_b64_image_by_norm_bbox
from nemo_retriever.common.params.models import IMAGE_MODALITIES

_CONTENT_COLUMNS = ("table", "chart", "infographic")


def _is_content_collection(value: Any) -> bool:
    """Return whether a value is a supported extracted-content collection."""
    return isinstance(value, list) or (isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == object)


def _page_text_state(row: Any, text_column: str = "text") -> tuple[Any, bool]:
    """Return source text or an edited assembled page, with its assembled-state flag."""
    text = row.get(text_column)
    collapsed = row.get("_collapsed_page_text")
    if isinstance(collapsed, dict) and collapsed.get("text_column") == text_column and text_column in row:
        if isinstance(text, str):
            if text != collapsed.get("combined_text") or collapsed.get("assembled_text_edited"):
                return text, True
            return collapsed.get("source_text"), False
        if pd.api.types.is_scalar(text) and pd.isna(text):
            return "", True
    return text, False


def _combine_text_with_content(
    row: Any,
    text_column: str = "text",
    content_columns: Sequence[str] = ("table", "chart", "infographic"),
    _content_spans: list[dict[str, Any]] | None = None,
) -> str:
    """Combine current page content in region order while its source text is valid."""
    regions = row.get("ordered_regions")
    live_text, assembled_text_edited = _page_text_state(row, text_column)
    if assembled_text_edited:
        return _reconcile_assembled_text(row, live_text, content_columns, _content_spans)
    separator = row.get("ordered_page_text_separator")
    paragraphs = isinstance(separator, str) and separator == "\n\n"
    ordered = (
        text_column == "text" and isinstance(row.get("ordered_page_text"), str) and _is_content_collection(regions)
    )
    if ordered and paragraphs:
        source_text = "\n\n".join(
            region["text"]
            for region in regions
            if isinstance(region, dict)
            and region.get("label_name") in ("text", "title", "header_footer")
            and isinstance(region.get("text"), str)
            and region["text"]
        )
        ordered = isinstance(live_text, str) and live_text == source_text

    items_by_box = {}
    for column in content_columns:
        items = row.get(column)
        if _is_content_collection(items):
            for index, item in enumerate(items):
                if isinstance(item, dict):
                    key = (column, tuple(_normalize_bbox(item.get("bbox_xyxy_norm")) or ()))
                    items_by_box.setdefault(key, []).append((index, item))
    parts = []
    consumed = set()

    trace = []

    def append_piece(
        target: list[str], value: str, column: str | None = None, index: int = 0, field: str = "text"
    ) -> None:
        target.append(value)
        if _content_spans is not None:
            trace.append((value, column, index, field))

    def append_item(item: Any, target: list[str], column: str, index: int) -> None:
        for field in ("text", "caption"):
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                append_piece(target, value.strip(), column, index, field)

    if ordered:
        selected = []
        for region in regions:
            if not isinstance(region, dict):
                continue
            label = region.get("label_name")
            if label in ("text", "title", "header_footer"):
                text = region.get("text")
                if isinstance(text, str) and text.strip():
                    append_piece(selected, text.strip() if paragraphs else text)
            elif label in content_columns:
                key = (label, tuple(_normalize_bbox(region.get("bbox_xyxy_norm")) or ()))
                matches = items_by_box.get(key, [])
                match = next(((index, item) for index, item in matches if (label, index) not in consumed), None)
                if match is not None:
                    index, item = match
                    consumed.add((label, index))
                    if paragraphs:
                        append_item(item, selected, label, index)
                    else:
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            append_piece(selected, text, label, index)
        if selected:
            parts.append(("\n\n" if paragraphs else "\n").join(selected))
    else:
        base = live_text
        if isinstance(base, str) and base.strip():
            append_piece(parts, base.strip())
    for column in content_columns:
        items = row.get(column)
        if _is_content_collection(items):
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                if (column, index) not in consumed:
                    append_item(item, parts, column, index)
                    consumed.add((column, index))
                elif not paragraphs:
                    caption = item.get("caption")
                    if isinstance(caption, str) and caption.strip():
                        append_piece(parts, caption.strip(), column, index, "caption")
    combined = "\n\n".join(parts)
    if _content_spans is not None:
        cursor = 0
        for value, column, index, field in trace:
            start = combined.index(value, cursor)
            cursor = start + len(value)
            if column is not None:
                items = row.get(column)
                box = _normalize_bbox(items[index].get("bbox_xyxy_norm")) or []
                occurrence = sum(
                    (_normalize_bbox(item.get("bbox_xyxy_norm")) or []) == box
                    for item in items[:index]
                    if isinstance(item, dict)
                )
                _content_spans.append(
                    {
                        "column": column,
                        "bbox": box,
                        "occurrence": occurrence,
                        "field": field,
                        "source_value": value.strip(),
                        "start": start,
                        "end": cursor,
                    }
                )
    return combined


def _page_edit_opcodes(original: str, edited: str) -> list[tuple[str, int, int, int, int]]:
    """Compare the changed middle and report edit positions in full-page coordinates."""
    from difflib import SequenceMatcher

    if original == edited:
        return [("equal", 0, len(original), 0, len(edited))]
    prefix = 0
    limit = min(len(original), len(edited))
    while prefix < limit and original[prefix] == edited[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and original[-suffix - 1] == edited[-suffix - 1]:
        suffix += 1
    original_end, edited_end = len(original) - suffix, len(edited) - suffix
    opcodes = [("equal", 0, prefix, 0, prefix)] if prefix else []
    matcher = SequenceMatcher(None, original[prefix:original_end], edited[prefix:edited_end], autojunk=False)
    opcodes.extend(
        (tag, left + prefix, right + prefix, edited_left + prefix, edited_right + prefix)
        for tag, left, right, edited_left, edited_right in matcher.get_opcodes()
    )
    if suffix:
        opcodes.append(("equal", original_end, len(original), edited_end, len(edited)))
    return opcodes


def _map_content_span(span: dict[str, Any], opcodes: list) -> tuple[int, int] | None:
    """Map a rendered region through edits that do not cross its boundaries."""
    start, end = int(span["start"]), int(span["end"])
    if start < 0:
        return None
    ranges = []
    for tag, left, right, edited_left, edited_right in opcodes:
        if tag == "equal":
            lo, hi = max(start, left), min(end, right)
            if lo < hi:
                ranges.append((edited_left + lo - left, edited_left + hi - left))
        elif left < end and right > start:
            if left < start or right > end:
                return None
            ranges.append((edited_left, edited_right))
        elif tag == "insert" and start < left < end:
            ranges.append((edited_left, edited_right))
    if not ranges:
        return None
    lo, hi = min(pair[0] for pair in ranges), max(pair[1] for pair in ranges)
    return (lo, hi) if lo < hi else None


def _unmapped_region_survives(span: dict[str, Any], original: str, text: str, opcodes: list) -> bool:
    """Detect surviving region content that cannot be attributed to unchanged surrounding text."""
    start, end = int(span["start"]), int(span["end"])
    if start < 0:
        return False
    value = original[start:end].strip()
    if not value:
        return False
    position = text.find(value)
    while position >= 0:
        unchanged_elsewhere = False
        for tag, left, right, edited_left, edited_right in opcodes:
            if tag == "equal" and edited_left <= position and position + len(value) <= edited_right:
                source_start = left + position - edited_left
                if source_start + len(value) <= start or source_start >= end:
                    unchanged_elsewhere = True
                    break
        if not unchanged_elsewhere:
            return True
        position = text.find(value, position + 1)
    return False


def _restore_collapsed_spans(row: Any) -> list[dict[str, Any]]:
    """Recover legacy region positions only when rendering reproduces the cached page."""
    collapsed = row["_collapsed_page_text"]
    if "source_text" in collapsed and isinstance(collapsed.get("combined_text"), str):
        text_column = collapsed["text_column"]
        columns = collapsed.get("content_columns")
        columns = tuple(columns) if _is_content_collection(columns) else _CONTENT_COLUMNS
        candidate = dict(row)
        candidate.pop("_collapsed_page_text")
        candidate[text_column] = collapsed["source_text"]
        spans = []
        if _combine_text_with_content(candidate, text_column, columns, spans) == collapsed["combined_text"]:
            return spans
        regions = row.get("ordered_regions")
        if _is_content_collection(regions):
            for column in columns:
                candidate[column] = [
                    dict(region)
                    for region in regions
                    if isinstance(region, dict) and region.get("label_name") == column
                ]
            spans = []
            if _combine_text_with_content(candidate, text_column, columns, spans) == collapsed["combined_text"]:
                return spans
    raise ValueError(
        "Cannot reconcile edited legacy collapsed text without recoverable original region metadata. "
        "Use the original extraction results and collapse them again."
    )


def _reconcile_assembled_text(
    row: Any, text: str, content_columns: Sequence[str], content_spans: list[dict[str, Any]] | None
) -> str:
    """Apply live region changes at tracked positions while retaining page-text edits."""
    collapsed = row["_collapsed_page_text"]
    previous = collapsed.get("content_spans")
    if not _is_content_collection(previous):
        previous = _restore_collapsed_spans(row)
    original = collapsed["combined_text"]
    opcodes = _page_edit_opcodes(original, text)
    patches = []
    retained = []
    consumed = set()
    for old in previous:
        span = dict(old)
        column, field = span["column"], span["field"]
        items = row.get(column)
        box = tuple(_normalize_bbox(span.get("bbox")) or ())
        matches = (
            [
                (index, item)
                for index, item in enumerate(items)
                if isinstance(item, dict) and tuple(_normalize_bbox(item.get("bbox_xyxy_norm")) or ()) == box
            ]
            if _is_content_collection(items)
            else []
        )
        occurrence = int(span["occurrence"])
        match = matches[occurrence] if occurrence < len(matches) else None
        if match is not None:
            consumed.add((column, match[0], field))
        mapped = _map_content_span(span, opcodes)
        if mapped is None:
            if _unmapped_region_survives(span, original, text, opcodes):
                raise ValueError(
                    "Cannot reliably locate structured content after a cross-region page-text edit. "
                    "Edit the original extraction fields and collapse them again."
                )
            span.update(start=-1, end=-1)
            if match is not None and column in content_columns:
                retained.append(span)
            continue
        start, end = mapped
        value = match[1].get(field) if match is not None and column in content_columns else ""
        value = value.strip() if isinstance(value, str) else ""
        if not value:
            if text[end:].startswith("\n\n"):
                end += 2
            elif text[end:].startswith("\n"):
                end += 1
            elif text[:start].endswith("\n\n"):
                start -= 2
            elif text[:start].endswith("\n"):
                start -= 1
            patches.append((start, end, ""))
            continue
        if value != span["source_value"]:
            patches.append((start, end, value))
            span["source_value"] = value
        span.update(start=start, end=end)
        retained.append(span)

    merged = []
    for start, end, value in sorted(patches):
        if merged and start <= merged[-1][1] and not value and not merged[-1][2]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]), "")
        else:
            merged.append((start, end, value))
    if merged and not merged[-1][2] and merged[-1][1] == len(text):
        start, end, value = merged[-1]
        if text[:start].endswith("\n\n"):
            start -= 2
        elif text[:start].endswith("\n"):
            start -= 1
        merged[-1] = (start, end, value)
    parts = []
    cursor = 0
    for start, end, value in merged:
        parts.extend((text[cursor:start], value))
        cursor = end
    parts.append(text[cursor:])
    rendered = "".join(parts)

    def shifted(position: int) -> int:
        return position + sum(len(value) - (end - start) for start, end, value in merged if end <= position)

    for span in retained:
        if span["start"] >= 0:
            span.update(start=shifted(span["start"]), end=shifted(span["end"]))
    for column in content_columns:
        items = row.get(column)
        if not _is_content_collection(items):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            for field in ("text", "caption"):
                value = item.get(field)
                if (column, index, field) in consumed or not isinstance(value, str) or not value.strip():
                    continue
                value = value.strip()
                start = len(rendered) + (2 if rendered else 0)
                rendered += ("\n\n" if rendered else "") + value
                box = _normalize_bbox(item.get("bbox_xyxy_norm")) or []
                occurrence = sum(
                    (_normalize_bbox(other.get("bbox_xyxy_norm")) or []) == box
                    for other in items[:index]
                    if isinstance(other, dict)
                )
                retained.append(
                    {
                        "column": column,
                        "bbox": box,
                        "occurrence": occurrence,
                        "field": field,
                        "source_value": value,
                        "start": start,
                        "end": start + len(value),
                    }
                )
    if content_spans is not None:
        content_spans.extend(retained)
    return rendered


def _normalize_bbox(value: Any) -> Any:
    """Return bbox coordinates in a stable non-tensor container."""
    if hasattr(value, "to_numpy") and callable(value.to_numpy):
        value = value.to_numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _deep_copy_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Copy row dicts without sharing nested mutable values across exploded rows."""
    import copy

    out: Dict[str, Any] = {}
    for key, value in row_dict.items():
        if isinstance(value, np.ndarray) and _is_content_collection(value):
            out[key] = copy.deepcopy(value.tolist())
        elif isinstance(value, (dict, list)):
            out[key] = copy.deepcopy(value)
        else:
            out[key] = value
    return out


def _normalize_bbox_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Force every batch to expose bbox values as ordinary Python objects."""
    values = frame.get("_bbox_xyxy_norm", pd.Series([None] * len(frame.index), index=frame.index))
    frame["_bbox_xyxy_norm"] = pd.Series(
        [_normalize_bbox(value) for value in values],
        index=frame.index,
        dtype=object,
    )
    return frame


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

    if not isinstance(batch_df, pd.DataFrame):
        return batch_df
    if batch_df.empty:
        return _normalize_bbox_column(batch_df)

    any_images = text_mod in IMAGE_MODALITIES or struct_mod in IMAGE_MODALITIES

    if not any(column in batch_df.columns for column in content_columns):
        batch_df = batch_df.copy()
        if text_mod in IMAGE_MODALITIES and "page_image" in batch_df.columns:
            batch_df["_image_b64"] = batch_df["page_image"].apply(
                lambda page_image: inline_image_b64(page_image) if isinstance(page_image, dict) else None
            )
        if "page_image" in batch_df.columns:
            batch_df["_stored_image_uri"] = batch_df["page_image"].apply(
                lambda page_image: page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
            )
        batch_df["_embed_modality"] = text_mod
        return _normalize_bbox_column(batch_df)

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
                page_image_b64 = inline_image_b64(page_image)

        page_text = row_dict.get(text_column)
        if isinstance(page_text, str) and page_text.strip():
            page_row = _deep_copy_row(row_dict)
            page_row["_embed_modality"] = text_mod
            page_row["_content_type"] = "text"
            if text_mod in IMAGE_MODALITIES:
                page_row["_image_b64"] = page_image_b64
            page_row["_stored_image_uri"] = page_stored_uri
            page_row["_bbox_xyxy_norm"] = None
            new_rows.append(page_row)
            exploded_any = True

        for column in content_columns:
            content_list = row_dict.get(column)
            if not _is_content_collection(content_list):
                continue
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                item_b64 = inline_image_b64(item) if struct_mod in IMAGE_MODALITIES else None
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
                        if item_b64:
                            content_row["_image_b64"] = item_b64
                        elif page_image_b64:
                            bbox = _normalize_bbox(item.get("bbox_xyxy_norm"))
                            if bbox is not None and len(bbox) == 4:
                                cropped_b64, _ = _crop_b64_image_by_norm_bbox(page_image_b64, bbox_xyxy_norm=bbox)
                                content_row["_image_b64"] = cropped_b64
                            else:
                                content_row["_image_b64"] = page_image_b64
                        else:
                            content_row["_image_b64"] = None
                    content_row["_stored_image_uri"] = item.get("stored_image_uri") or page_stored_uri
                    content_row["_bbox_xyxy_norm"] = _normalize_bbox(item.get("bbox_xyxy_norm"))
                    new_rows.append(content_row)
                    exploded_any = True

        if not exploded_any:
            preserved = _deep_copy_row(row_dict)
            preserved["_embed_modality"] = text_mod
            preserved["_content_type"] = "text"
            if text_mod in IMAGE_MODALITIES:
                preserved["_image_b64"] = page_image_b64
            preserved["_stored_image_uri"] = page_stored_uri
            preserved["_bbox_xyxy_norm"] = None
            new_rows.append(preserved)

    return _normalize_bbox_column(pd.DataFrame(new_rows).reset_index(drop=True))


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

    has_page_image = "page_image" in batch_df.columns
    new_rows: List[Dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        row_dict = _deep_copy_row(row.to_dict())
        source_text, assembled_text_edited = _page_text_state(row_dict, text_column)
        spans = []
        row_dict[text_column] = _combine_text_with_content(row_dict, text_column, content_columns, spans)
        row_dict["_collapsed_page_text"] = {
            "text_column": text_column,
            "source_text": source_text if isinstance(source_text, str) else None,
            "combined_text": row_dict[text_column],
            "content_columns": list(content_columns),
            "assembled_text_edited": assembled_text_edited,
            "content_spans": spans,
        }
        page_image = row_dict.get("page_image") if has_page_image else None
        if modality in IMAGE_MODALITIES:
            row_dict["_image_b64"] = inline_image_b64(page_image) if isinstance(page_image, dict) else None
        if has_page_image:
            row_dict["_stored_image_uri"] = page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
        row_dict["_embed_modality"] = modality
        new_rows.append(row_dict)

    return _normalize_bbox_column(pd.DataFrame(new_rows).reset_index(drop=True))
