# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Store extracted images to disk or cloud storage via fsspec."""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from typing import Any, Dict, Optional, Sequence

import pandas as pd
from PIL import Image
from upath import UPath

logger = logging.getLogger(__name__)

_DIRECT_IMAGE_FORMATS = {"png", "jpeg"}
_FORMAT_ALIASES = {"jpg": "jpeg"}

# Known limitation: _safe_stem derives the output subdirectory from the
# filename alone (e.g. "report.pdf" → "report/").  Two source files with
# the same basename but different parent directories will write to the same
# subdirectory and may overwrite each other.  This matches the legacy
# nv-ingest store behaviour.  A future PR should incorporate a short hash
# of the full source path to eliminate collisions.


def _safe_stem(name: str) -> str:
    """Derive a filesystem-safe stem from a source path."""
    s = str(name or "").strip() or "document"
    s = os.path.splitext(os.path.basename(s))[0] or "document"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:160] if len(s) > 160 else s


def _normalize_image_format(image_format: str) -> str:
    fmt = str(image_format or "png").strip().lower()
    fmt = _FORMAT_ALIASES.get(fmt, fmt)
    if fmt not in _DIRECT_IMAGE_FORMATS:
        raise ValueError(f"Unsupported image_format: {image_format!r}. Supported formats: png, jpeg")
    return fmt


def _decode_image_bytes(image_b64: str) -> bytes | None:
    try:
        return base64.b64decode(image_b64)
    except Exception as exc:
        logger.warning("Failed to decode image payload: %s", exc)
        return None


def _write_bytes(dest: UPath, raw: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        f.write(raw)


def _normalized_encoding(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    enc = value.strip().lower()
    enc = _FORMAT_ALIASES.get(enc, enc)
    if enc in _DIRECT_IMAGE_FORMATS:
        return enc
    return None


def _sniff_image_encoding(raw: bytes) -> str | None:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    return None


def _resolve_direct_write_encoding(
    payload: dict[str, Any] | None,
    raw: bytes,
    fallback: str,
) -> str:
    sniffed = _sniff_image_encoding(raw)
    if sniffed:
        return sniffed

    declared = _normalized_encoding(payload.get("encoding")) if isinstance(payload, dict) else None
    if declared:
        return declared

    return fallback


def _crop_and_write(
    dest: UPath,
    page_image: Image.Image,
    bbox_xyxy_norm: Sequence[float],
    image_format: str = "png",
) -> bool:
    """Crop a region from an already-decoded page image and write to *dest*.

    Returns ``True`` on success, ``False`` on skip/failure.
    """
    try:
        w, h = page_image.size
        if w <= 1 or h <= 1:
            return False

        x1n, y1n, x2n, y2n = (float(v) for v in bbox_xyxy_norm)
        x1 = int(min(max(x1n * w, 0), w))
        y1 = int(min(max(y1n * h, 0), h))
        x2 = int(min(max(x2n * w, 0), w))
        y2 = int(min(max(y2n * h, 0), h))
        if x2 <= x1 or y2 <= y1:
            return False

        crop = page_image.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format=image_format.upper())

        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            f.write(buf.getvalue())
        return True
    except Exception as exc:
        logger.warning("Failed to crop and write %s: %s", dest, exc)
        return False


def _decode_page_image(page_image_b64: str) -> Image.Image | None:
    """Decode a base64-encoded page image into a PIL Image (once per row)."""
    try:
        raw = base64.b64decode(page_image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to decode page image: %s", exc)
        return None


def _build_uri_info(
    dest: UPath,
    storage_root: UPath,
    public_base_url: Optional[str],
    uri_key: str = "stored_image_uri",
) -> Dict[str, Optional[str]]:
    """Build a dict with a stored URI and optionally a public URL."""
    relative_key = dest.relative_to(storage_root).as_posix()
    url_key = uri_key.replace("_uri", "_url")
    info: Dict[str, Optional[str]] = {uri_key: dest.as_uri()}
    if public_base_url:
        info[url_key] = f"{public_base_url.rstrip('/')}/{relative_key}"
    return info


def load_image_b64_from_uri(uri: str) -> Optional[str]:
    """Read an image from a stored URI and return its base64 encoding.

    Accepts any fsspec-compatible URI (``file://``, ``s3://``, etc.).
    Returns ``None`` on failure so callers can fall back gracefully.
    """
    try:
        raw = UPath(uri).read_bytes()
        return base64.b64encode(raw).decode("ascii")
    except Exception as exc:
        logger.warning("Failed to load image from %s: %s", uri, exc)
        return None


def store_extracted_images(
    df: pd.DataFrame,
    *,
    storage_uri: str = "stored_images",
    storage_options: dict[str, Any] | None = None,
    public_base_url: str | None = None,
    store_page_images: bool = True,
    store_tables: bool = True,
    store_charts: bool = True,
    store_infographics: bool = True,
    store_images: bool = True,
    store_text: bool = False,
    image_format: str = "png",
    strip_base64: bool = True,
) -> pd.DataFrame:
    """Pipeline task: store extracted images to disk or cloud storage.

    For each row in the DataFrame:

    * Writes the full page image (from ``page_image["image_b64"]``) when
      *store_page_images* is ``True``.
    * Crops and writes sub-page images for tables / charts / infographics
      using ``bbox_xyxy_norm`` from the page image.
    * Writes natural sub-page images from the ``images`` column.
    * Updates the DataFrame in-place with ``stored_image_uri`` (and an
      optional ``stored_image_url`` when *public_base_url* is set).

    Parameters
    ----------
    df : pd.DataFrame
        Primitives DataFrame produced by the extraction pipeline.
    storage_uri : str
        Base URI for storage.  Local path (``"./output"``) or
        fsspec-compatible URI (``"s3://bucket/prefix"``).
    storage_options : dict | None
        Extra options forwarded to fsspec / UPath (auth keys, endpoint, etc.).
    public_base_url : str | None
        When set, each stored item also receives a ``stored_image_url``
        built as ``{public_base_url}/{relative_key}``.
    store_page_images : bool
        Save full page images.
    store_tables : bool
        Save table crops.
    store_charts : bool
        Save chart crops.
    store_infographics : bool
        Save infographic crops.
    store_images : bool
        Save natural sub-page images from the ``images`` column.
    store_text : bool
        Save page text to ``.txt`` files.  Also writes text from structured
        content items (tables, charts, infographics) when present.
        Disabled by default.
    image_format : str
        Output image format for generated crops (default ``"png"``).
        Direct-write payloads preserve their source encoding and file extension.
    strip_base64 : bool
        When ``True`` (the default), clear ``image_b64`` after successful writes
        to reduce memory pressure.  The embed stage loads images from the stored
        URIs when base64 is absent.  Set to ``False`` only if downstream code
        requires inline base64 for a reason other than embedding.

    Returns
    -------
    pd.DataFrame
        The (mutated) input DataFrame with storage URIs added.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    logger.info("Storing extracted images to %s", storage_uri)
    storage_root = UPath(storage_uri, **(storage_options or {})).resolve()
    ext = _normalize_image_format(image_format)
    if strip_base64:
        logger.debug("strip_base64=True: image payloads will be cleared after writing.")

    col_flags: dict[str, str] = {}
    if store_tables:
        col_flags["table"] = "table"
    if store_charts:
        col_flags["chart"] = "chart"
    if store_infographics:
        col_flags["infographic"] = "infographic"

    for idx, row in df.iterrows():
        try:
            source_path = row.get("path") or ""
            stem = _safe_stem(source_path)
            page_num = row.get("page_number", 1)

            page_image = row.get("page_image")
            page_image_b64: str | None = None
            if isinstance(page_image, dict):
                page_image_b64 = page_image.get("image_b64")

            # Decode the page image lazily and reuse for all crops in this row.
            page_pil: Image.Image | None = None
            page_pil_decode_attempted = False

            def _get_page_pil() -> Image.Image | None:
                nonlocal page_pil, page_pil_decode_attempted
                if page_pil_decode_attempted:
                    return page_pil
                page_pil_decode_attempted = True
                if isinstance(page_image_b64, str) and page_image_b64:
                    page_pil = _decode_page_image(page_image_b64)
                return page_pil

            # -- Full page image --
            if store_page_images and isinstance(page_image_b64, str) and page_image_b64:
                raw = _decode_image_bytes(page_image_b64)
                if raw is not None:
                    direct_ext = _resolve_direct_write_encoding(page_image, raw, ext)
                    dest = storage_root / stem / f"page_{page_num}.{direct_ext}"
                    _write_bytes(dest, raw)
                    uri_info = _build_uri_info(dest, storage_root, public_base_url)
                    page_image.update(uri_info)
                    page_image["encoding"] = direct_ext
                    if strip_base64:
                        page_image["image_b64"] = None
                    df.at[idx, "page_image"] = page_image

            # -- Structured content (tables / charts / infographics) --
            for col_name, type_label in col_flags.items():
                content_list = row.get(col_name)
                if not isinstance(content_list, list):
                    continue
                for item_idx, item in enumerate(content_list):
                    if not isinstance(item, dict):
                        continue
                    item_b64 = item.get("image_b64")
                    if isinstance(item_b64, str) and item_b64:
                        raw = _decode_image_bytes(item_b64)
                        if raw is not None:
                            direct_ext = _resolve_direct_write_encoding(item, raw, ext)
                            dest = storage_root / stem / f"page_{page_num}_{type_label}_{item_idx}.{direct_ext}"
                            _write_bytes(dest, raw)
                            item.update(_build_uri_info(dest, storage_root, public_base_url))
                            item["encoding"] = direct_ext
                            if strip_base64:
                                item["image_b64"] = None
                    else:
                        page_pil = _get_page_pil()
                    if page_pil is not None and not (isinstance(item_b64, str) and item_b64):
                        bbox = item.get("bbox_xyxy_norm")
                        if bbox and len(bbox) == 4:
                            dest = storage_root / stem / f"page_{page_num}_{type_label}_{item_idx}.{ext}"
                            if _crop_and_write(dest, page_pil, bbox, image_format=ext):
                                item.update(_build_uri_info(dest, storage_root, public_base_url))
                                item["encoding"] = ext
                df.at[idx, col_name] = content_list

            # -- Natural sub-page images --
            if store_images:
                images_list = row.get("images")
                if isinstance(images_list, list):
                    for img_idx, img_item in enumerate(images_list):
                        if not isinstance(img_item, dict):
                            continue
                        img_b64 = img_item.get("image_b64")
                        if isinstance(img_b64, str) and img_b64:
                            raw = _decode_image_bytes(img_b64)
                            if raw is not None:
                                direct_ext = _resolve_direct_write_encoding(img_item, raw, ext)
                                dest = storage_root / stem / f"page_{page_num}_image_{img_idx}.{direct_ext}"
                                _write_bytes(dest, raw)
                                img_item.update(_build_uri_info(dest, storage_root, public_base_url))
                                img_item["encoding"] = direct_ext
                                if strip_base64:
                                    img_item["image_b64"] = None
                        else:
                            page_pil = _get_page_pil()
                        if page_pil is not None and not (isinstance(img_b64, str) and img_b64):
                            bbox = img_item.get("bbox_xyxy_norm")
                            if bbox and len(bbox) == 4:
                                dest = storage_root / stem / f"page_{page_num}_image_{img_idx}.{ext}"
                                if _crop_and_write(dest, page_pil, bbox, image_format=ext):
                                    img_item.update(_build_uri_info(dest, storage_root, public_base_url))
                                    img_item["encoding"] = ext
                    df.at[idx, "images"] = images_list

            # -- Page text --
            if store_text:
                page_text = row.get("text")
                if isinstance(page_text, str) and page_text.strip():
                    text_dest = storage_root / stem / f"page_{page_num}.txt"
                    _write_bytes(text_dest, page_text.encode("utf-8"))
                    uri_info = _build_uri_info(text_dest, storage_root, public_base_url, uri_key="stored_text_uri")
                    df.at[idx, "stored_text_uri"] = uri_info["stored_text_uri"]

                # Structured content text (tables, charts, infographics)
                for col_name in ("table", "chart", "infographic"):
                    content_list = row.get(col_name)
                    if not isinstance(content_list, list):
                        continue
                    for item_idx, item in enumerate(content_list):
                        if not isinstance(item, dict):
                            continue
                        item_text = item.get("text")
                        if isinstance(item_text, str) and item_text.strip():
                            text_dest = storage_root / stem / f"page_{page_num}_{col_name}_{item_idx}.txt"
                            _write_bytes(text_dest, item_text.encode("utf-8"))
                            item.update(
                                _build_uri_info(text_dest, storage_root, public_base_url, uri_key="stored_text_uri")
                            )
                    df.at[idx, col_name] = content_list

        except Exception as exc:
            logger.exception("Failed to store images for row %s: %s", idx, exc)

    return df
