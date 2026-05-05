# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image payload helpers used by VL reranking and content transforms."""

from __future__ import annotations

import base64
import logging

from upath import UPath

logger = logging.getLogger(__name__)


def inline_image_b64(container: dict) -> str | None:
    """Return inline base64 image data without reloading stored URIs."""
    value = container.get("image_b64")
    return value if isinstance(value, str) and value.strip() else None


def load_image_b64_from_uri(uri: str) -> str | None:
    """Read an image URI and return its base64 payload."""
    try:
        raw = UPath(uri).read_bytes()
        return base64.b64encode(raw).decode("ascii")
    except Exception as exc:
        logger.warning("Failed to load image from %s: %s", uri, exc)
        return None


def render_page_image_b64(pdf_path: str, page_number: int, *, dpi: int = 300) -> str | None:
    """Render a 1-indexed PDF page to the extraction pipeline's base64 image format."""
    try:
        import pypdfium2 as pdfium

        from nemo_retriever.pdf.extract import _render_page_to_base64

        doc = pdfium.PdfDocument(pdf_path)
        try:
            page_idx = int(page_number) - 1
            if page_idx < 0:
                return None
            page = doc[page_idx]
            render_info = _render_page_to_base64(page, dpi=dpi)
            return render_info.get("image_b64")
        finally:
            doc.close()
    except Exception as exc:
        logger.warning("Failed to render page %s of %s: %s", page_number, pdf_path, exc)
        return None
