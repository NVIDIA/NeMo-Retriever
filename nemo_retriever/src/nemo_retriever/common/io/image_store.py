# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image payload helpers used by VL reranking and content transforms."""

from __future__ import annotations

import base64
import ipaddress
import logging
import mimetypes
import os
from urllib.parse import urlparse

from upath import UPath

logger = logging.getLogger(__name__)

DEFAULT_MAX_IMAGE_BYTES = 50 * 1024 * 1024

_BLOCKED_IMAGE_HOSTS = frozenset(
    {
        "169.254.169.254",
        "169.254.170.2",
        "metadata.google.internal",
        "metadata.azure.com",
    }
)
_IMAGE_EXTENSION_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
}
_ALLOWED_IMAGE_EXTENSIONS = frozenset(_IMAGE_EXTENSION_MIME)


# Known limitation: _safe_stem derives the output subdirectory from the
# filename alone (e.g. "report.pdf" → "report/").  Two source files with
# the same basename but different parent directories will write to the same
# subdirectory and may overwrite each other.  This matches the legacy
# `nemo_retriever.api` store behaviour.  A future PR should incorporate a short hash
# of the full source path to eliminate collisions.
def inline_image_b64(container: dict) -> str | None:
    """Return inline base64 image data without reloading stored URIs."""
    value = container.get("image_b64")
    return value if isinstance(value, str) and value.strip() else None


def image_mime_type_from_uri(uri: str) -> str:
    """Infer an image MIME type from a URI, falling back to PNG."""
    ext = os.path.splitext(urlparse(uri).path)[1].lower()
    return _IMAGE_EXTENSION_MIME.get(ext) or mimetypes.guess_type(uri)[0] or "image/png"


def validate_image_uri(uri: str) -> bool:
    """Return whether an image URI is safe to load."""
    parsed = urlparse(uri)
    host = parsed.hostname or ""
    if parsed.scheme in {"http", "https"}:
        if host in _BLOCKED_IMAGE_HOSTS:
            logger.warning("Blocked request to known metadata endpoint: %s", host)
            return False
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                logger.warning("Blocked request to private/loopback address: %s", host)
                return False
        except ValueError:
            pass
        return True

    if parsed.scheme not in {"", "file"}:
        return True

    ext = os.path.splitext(os.path.realpath(parsed.path))[1].lower()
    if ext not in _ALLOWED_IMAGE_EXTENSIONS:
        logger.warning("Rejected non-image local path (extension %r): %s", ext, uri)
        return False
    return True


def load_image_b64_from_uri(
    uri: str,
    *,
    max_bytes: int | None = None,
    validate: bool = False,
) -> str | None:
    """Read an image URI and return its base64 payload.

    ``validate`` enables image-extension and private-network checks. ``max_bytes``
    bounds the read for callers that accept untrusted or remote image URIs.
    """
    try:
        if validate and not validate_image_uri(uri):
            return None

        path = UPath(uri)
        if max_bytes is None:
            raw = path.read_bytes()
        else:
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            if size is not None and size > max_bytes:
                logger.warning("Skipping oversized image (%d bytes): %s", size, uri)
                return None
            with path.open("rb") as image_file:
                raw = image_file.read(max_bytes + 1)
            if len(raw) > max_bytes:
                logger.warning("Skipping oversized image (>%d bytes): %s", max_bytes, uri)
                return None

        return base64.b64encode(raw).decode("ascii")
    except Exception as exc:
        logger.warning("Failed to load image from %s: %s", uri, exc, exc_info=True)
        return None


def render_page_image_b64(pdf_path: str, page_number: int, *, dpi: int = 300) -> str | None:
    """Render a 1-indexed PDF page to the extraction pipeline's base64 image format."""
    try:
        import pypdfium2 as pdfium

        from nemo_retriever.operators.extract.pdf.extract import _render_page_to_base64

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
        logger.warning("Failed to render page %s of %s: %s", page_number, pdf_path, exc, exc_info=True)
        return None
