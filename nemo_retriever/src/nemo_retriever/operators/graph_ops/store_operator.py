# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator for persisting post-embedding row images to storage."""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import fsspec
import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator

logger = logging.getLogger(__name__)

_FORMAT_ALIASES = {"jpg": "jpeg"}
_SUPPORTED_FORMATS = {"png", "jpeg"}


def _normalize_image_format(image_format: str) -> str:
    fmt = str(image_format or "png").strip().lower()
    fmt = _FORMAT_ALIASES.get(fmt, fmt)
    if fmt not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image_format: {image_format!r}. Supported formats: png, jpeg")
    return fmt


def _sniff_image_format(raw: bytes) -> str | None:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    return None


def _decode_image_b64(value: Any) -> bytes | None:
    if not isinstance(value, str) or not value.strip():
        return None

    payload = value.strip()
    if payload.lower().startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        return base64.b64decode(payload)
    except binascii.Error as exc:
        logger.warning("Skipping store row with invalid _image_b64 payload: %s", exc)
        return None


def _build_object_key(*, raw: bytes, extension: str) -> str:
    image_hash = hashlib.sha1(raw).hexdigest()
    return f"{image_hash}.{extension}"


def _join_storage_uri(storage_uri: str, object_key: str) -> str:
    return f"{str(storage_uri).rstrip('/')}/{object_key.lstrip('/')}"


def _stored_uri(dest_uri: str) -> str:
    parsed = urlparse(dest_uri)
    if parsed.scheme:
        return dest_uri
    return Path(dest_uri).resolve().as_uri()


def _row_image_b64_with_source(row: pd.Series) -> tuple[Any, bool]:
    value = row.get("_image_b64")
    if isinstance(value, str) and value.strip():
        return value, False

    page_image = row.get("page_image")
    if isinstance(page_image, dict):
        return page_image.get("image_b64"), True

    return None, False


def _row_image_represents_page(row: pd.Series, *, from_page_image: bool) -> bool:
    if from_page_image:
        return True

    content_type = row.get("_content_type")
    if not isinstance(content_type, str) or not content_type.strip():
        return True
    return content_type == "text"


def _store_row_images(
    df: pd.DataFrame,
    *,
    storage_uri: str,
    storage_options: dict[str, Any] | None = None,
    image_format: str = "png",
    strip_base64: bool = True,
) -> pd.DataFrame:
    """Return a copy of *df* with ``_stored_image_uri`` set for stored rows."""
    if df.empty or ("_image_b64" not in df.columns and "page_image" not in df.columns):
        return df

    out = df.copy()
    fallback_format = _normalize_image_format(image_format)
    fsspec_options = dict(storage_options or {})

    for idx, row in out.iterrows():
        image_b64, from_page_image = _row_image_b64_with_source(row)
        raw = _decode_image_b64(image_b64)
        if raw is None:
            continue

        extension = _sniff_image_format(raw) or fallback_format
        object_key = _build_object_key(raw=raw, extension=extension)
        dest_uri = _join_storage_uri(storage_uri, object_key)

        try:
            with fsspec.open(dest_uri, mode="wb", **fsspec_options) as f:
                f.write(raw)
        except Exception as exc:
            raise RuntimeError(f"Failed to store image for row {idx!r} to {dest_uri!r}: {exc}") from exc

        stored_uri = _stored_uri(dest_uri)
        out.at[idx, "_stored_image_uri"] = stored_uri

        if strip_base64:
            if "_image_b64" in out.columns:
                out.at[idx, "_image_b64"] = None
            page_image = row.get("page_image")
            if isinstance(page_image, dict):
                updated_page_image = dict(page_image)
                updated_page_image["image_b64"] = None
                if _row_image_represents_page(row, from_page_image=from_page_image):
                    updated_page_image["stored_image_uri"] = stored_uri
                out.at[idx, "page_image"] = updated_page_image

    return out


class StoreOperator(AbstractOperator, CPUOperator):
    """Persist row-level image payloads to local or object storage.

    The operator consumes ``_image_b64`` produced by content transforms and
    writes ``_stored_image_uri`` for downstream vector DB upload. By default it
    clears inline base64 after successful writes to avoid carrying page-sized
    payloads into VDB upload.
    """

    def __init__(self, *, params: Any = None) -> None:
        super().__init__()
        self._params = params

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame):
            return data

        if hasattr(self._params, "model_dump"):
            store_kwargs = self._params.model_dump(mode="python")
        elif isinstance(self._params, dict):
            store_kwargs = self._params
        else:
            store_kwargs = {}
        return _store_row_images(
            data,
            storage_uri=store_kwargs.get("storage_uri", "stored_images"),
            storage_options=store_kwargs.get("storage_options") or {},
            image_format=store_kwargs.get("image_format", "png"),
            strip_base64=bool(store_kwargs.get("strip_base64", True)),
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
