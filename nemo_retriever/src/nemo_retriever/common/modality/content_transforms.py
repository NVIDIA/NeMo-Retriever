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


def _match_ordered_content_items(row: Any, regions: list, content_columns: Sequence[str]) -> dict:
    """Resolve ordered regions to live items without moving surviving shared-box content."""
    spans, positions, occurrences = [], {}, {}
    for position, region in enumerate(regions):
        if not isinstance(region, dict) or region.get("label_name") not in content_columns:
            continue
        column = region["label_name"]
        box = tuple(_normalize_bbox(region.get("bbox_xyxy_norm")) or ())
        key = (column, box)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        positions[(*key, occurrence)] = position
        for field in ("text", "caption"):
            value = region.get(field)
            if field == "caption" and not isinstance(value, str):
                continue
            spans.append(
                {
                    "column": column,
                    "bbox": box,
                    "occurrence": occurrence,
                    "field": field,
                    "source_value": value.strip() if isinstance(value, str) else "",
                }
            )
    matches = _match_edited_content_items(row, spans, content_columns)
    return {positions[key]: match for key, match in matches.items()}


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
    elif ordered and isinstance(row.get("ordered_page_text_source"), str):
        source_text = row["ordered_page_text_source"]
        ordered = (live_text if isinstance(live_text, str) else "") == source_text

    ordered_matches = _match_ordered_content_items(row, regions, content_columns) if ordered else {}
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
        for position, region in enumerate(regions):
            if not isinstance(region, dict):
                continue
            label = region.get("label_name")
            if label in ("text", "title", "header_footer"):
                text = region.get("text")
                if isinstance(text, str) and text.strip():
                    append_piece(selected, text.strip() if paragraphs else text)
            elif label in content_columns:
                match = ordered_matches.get(position)
                if match is not None:
                    index, item, _ = match
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


def _match_edited_content_items(row: Any, spans: list, content_columns: Sequence[str]) -> dict:
    """Match live items to their previous occurrences when shared-box lists change size."""
    originals: dict = {}
    for span in spans:
        if span["column"] not in content_columns:
            continue
        key = (span["column"], tuple(_normalize_bbox(span.get("bbox")) or ()))
        originals.setdefault(key, {}).setdefault(int(span["occurrence"]), {})[span["field"]] = span["source_value"]
    live: dict = {}
    for column in {key[0] for key in originals}:
        items = row.get(column)
        if _is_content_collection(items):
            for index, item in enumerate(items):
                if isinstance(item, dict):
                    key = (column, tuple(_normalize_bbox(item.get("bbox_xyxy_norm")) or ()))
                    matches = live.setdefault(key, [])
                    matches.append((index, item, len(matches)))
    resolved = {}
    for key, occurrences in originals.items():
        matches = live.get(key, [])
        if len(occurrences) == len(matches) == 1:
            resolved[(*key, next(iter(occurrences)))] = matches[0]
            continue
        pending = dict(occurrences)
        available = list(matches)
        for occurrence, fields in occurrences.items():
            candidates = [
                match
                for match in available
                if all(
                    isinstance(match[1].get(field), str) and match[1][field].strip() == value
                    for field, value in fields.items()
                )
            ]
            if len(candidates) != 1:
                continue
            candidate = candidates[0]
            competing = sum(
                all(
                    isinstance(candidate[1].get(field), str) and candidate[1][field].strip() == value
                    for field, value in other.items()
                )
                for other in pending.values()
            )
            if competing == 1:
                resolved[(*key, occurrence)] = candidate
                available = [match for match in available if match[0] != candidate[0]]
                pending.pop(occurrence)
        equivalent_insertions = (
            len(available) > len(pending)
            and len(
                {
                    tuple(
                        item.get(field, "").strip() if isinstance(item.get(field), str) else ""
                        for field in ("text", "caption")
                    )
                    for _, item, _ in available
                }
            )
            == 1
        )
        if pending and available and len(pending) != len(available) and not equivalent_insertions:
            raise ValueError(
                "Cannot identify edited structured items sharing a bounding box after insertion or removal. "
                "Edit the original extraction fields and collapse them again."
            )
        for occurrence, match in zip(pending, available):
            resolved[(*key, occurrence)] = match
    return resolved


def _reconcile_tracked_span(
    old: dict[str, Any], match: Any, original: str, text: str, opcodes: list
) -> tuple[dict[str, Any] | None, tuple | None, bool]:
    """Resolve one saved field to an updated span, a text patch, and its consumption state."""
    span = dict(old)
    if match is not None:
        span["occurrence"] = match[2]
    value = match[1].get(span["field"]) if match is not None else ""
    value = value.strip() if isinstance(value, str) else ""
    mapped = _map_content_span(span, opcodes)
    if mapped is None:
        if _unmapped_region_survives(span, original, text, opcodes):
            raise ValueError(
                "Cannot reliably locate structured content after a cross-region page-text edit. "
                "Edit the original extraction fields and collapse them again."
            )
        span.update(start=-1, end=-1)
        unchanged = match is not None and value == span["source_value"]
        return (span if unchanged else None), None, unchanged
    start, end = mapped
    if not value:
        if text[end:].startswith("\n\n"):
            end += 2
        elif text[end:].startswith("\n"):
            end += 1
        elif text[:start].endswith("\n\n"):
            start -= 2
        elif text[:start].endswith("\n"):
            start -= 1
        return None, (start, end, ""), match is not None
    patch = (start, end, value) if value != span["source_value"] else None
    span.update(start=start, end=end, source_value=value)
    return span, patch, match is not None


def _collect_content_edits(row: Any, text: str, previous: list, content_columns: Sequence[str]) -> tuple:
    """Match live fields and collect independent edits to the assembled page."""
    original = row["_collapsed_page_text"]["combined_text"]
    live_matches = _match_edited_content_items(row, previous, content_columns)
    opcodes = _page_edit_opcodes(original, text)
    patches, retained, consumed = [], [], set()
    for old in previous:
        column, field = old["column"], old["field"]
        box = tuple(_normalize_bbox(old.get("bbox")) or ())
        match = live_matches.get((column, box, int(old["occurrence"])))
        span, patch, used = _reconcile_tracked_span(old, match, original, text, opcodes)
        if used:
            consumed.add((column, match[0], field))
        if span is not None:
            retained.append(span)
        if patch is not None:
            patches.append(patch)
    return patches, retained, consumed


def _merge_content_patches(text: str, patches: list) -> list:
    """Merge adjacent removals and remove the separator before a deleted trailing field."""
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
    return merged


def _apply_content_patches(text: str, patches: list, retained: list) -> str:
    """Apply non-overlapping field edits and shift saved positions to the resulting page."""
    merged = _merge_content_patches(text, patches)
    parts, cursor = [], 0
    for start, end, value in merged:
        parts.extend((text[cursor:start], value))
        cursor = end
    parts.append(text[cursor:])
    for span in retained:
        if span["start"] >= 0:
            for boundary in ("start", "end"):
                position = span[boundary]
                span[boundary] += sum(len(value) - (end - start) for start, end, value in merged if end <= position)
    return "".join(parts)


def _insert_content_field(rendered: str, value: str, span: dict[str, Any], retained: list) -> str:
    """Insert a field beside its retained sibling, or append it when no anchor remains."""
    siblings = [
        other
        for other in retained
        if other["column"] == span["column"]
        and (_normalize_bbox(other.get("bbox")) or []) == span["bbox"]
        and other["occurrence"] == span["occurrence"]
        and other["start"] >= 0
    ]
    if siblings:
        before = span["field"] == "text"
        position = min(other["start"] for other in siblings) if before else max(other["end"] for other in siblings)
        insertion = value + "\n\n" if before else "\n\n" + value
        start = position if before else position + 2
        rendered = rendered[:position] + insertion + rendered[position:]
        for other in retained:
            if other["start"] >= position:
                other["start"] += len(insertion)
                other["end"] += len(insertion)
            elif other["end"] > position:
                other["end"] += len(insertion)
    else:
        start = len(rendered) + (2 if rendered else 0)
        rendered += ("\n\n" if rendered else "") + value
    span.update(start=start, end=start + len(value))
    retained.append(span)
    return rendered


def _append_untracked_fields(
    row: Any, rendered: str, content_columns: Sequence[str], retained: list, consumed: set
) -> str:
    """Render new or corrected live fields that were not retained at a saved position."""
    for column in content_columns:
        items = row.get(column)
        if not _is_content_collection(items):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            box = _normalize_bbox(item.get("bbox_xyxy_norm")) or []
            occurrence = sum(
                (_normalize_bbox(other.get("bbox_xyxy_norm")) or []) == box
                for other in items[:index]
                if isinstance(other, dict)
            )
            for field in ("text", "caption"):
                value = item.get(field)
                if (column, index, field) in consumed or not isinstance(value, str) or not value.strip():
                    continue
                span = {
                    "column": column,
                    "bbox": box,
                    "occurrence": occurrence,
                    "field": field,
                    "source_value": value.strip(),
                }
                rendered = _insert_content_field(rendered, value.strip(), span, retained)
    return rendered


def _reconcile_assembled_text(
    row: Any, text: str, content_columns: Sequence[str], content_spans: list[dict[str, Any]] | None
) -> str:
    """Reconcile live page fields, render their edits, and retain updated field positions."""
    previous = row["_collapsed_page_text"].get("content_spans")
    if not _is_content_collection(previous):
        previous = _restore_collapsed_spans(row)
    patches, retained, consumed = _collect_content_edits(row, text, previous, content_columns)
    rendered = _apply_content_patches(text, patches, retained)
    rendered = _append_untracked_fields(row, rendered, content_columns, retained, consumed)
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
