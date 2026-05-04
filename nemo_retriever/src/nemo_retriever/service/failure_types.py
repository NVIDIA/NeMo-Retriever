# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured failure-type classification for service-mode ingestion.

Pipeline failures used to surface as raw exception strings on
``WorkerResult.error_message``.  That string is fine for humans, but
clients can't reliably make decisions from it (retry?  bail?  switch
NIM?).  This module defines a small enum of failure categories plus a
``categorize_exception`` helper that maps real exception classes/messages
to those categories.

The categorization is deliberately conservative: when in doubt we return
``UNKNOWN`` so callers don't make wrong assumptions.
"""

from __future__ import annotations

import enum
import re


class FailureType(str, enum.Enum):
    """Categorisation of the cause of a per-page processing failure."""

    PDF_PARSE = "pdf_parse"
    """Failed to load/parse the input file (corrupt PDF, password-protected, etc.)."""

    NIM_TIMEOUT = "nim_timeout"
    """Network call to a NIM endpoint timed out."""

    NIM_5XX = "nim_5xx"
    """NIM endpoint returned a 5xx response."""

    NIM_4XX = "nim_4xx"
    """NIM endpoint returned a 4xx response (auth / bad request)."""

    OOM = "oom"
    """Out-of-memory or RLIMIT_AS violation (CPU or GPU)."""

    INTERNAL = "internal"
    """Bug in the pipeline code (assertion, type error, etc.)."""

    CANCELLED = "cancelled"
    """Page was cancelled before reaching a worker."""

    UNKNOWN = "unknown"
    """Doesn't match any of the heuristics above."""


class EventCategory(str, enum.Enum):
    """Superset of :class:`FailureType` used by the ``event_log`` table.

    Covers every NIM endpoint type, query/rerank operations, file-format
    errors, dedup, spool, auth, and validation — everything the
    provenance UI needs to filter and group on.
    """

    # -- Pipeline / file-level -----------------------------------------
    PDF_PARSE = "pdf_parse"
    HTML = "html"
    DEDUP = "dedup"

    # -- NIM transport --------------------------------------------------
    NIM_UNREACHABLE = "nim_unreachable"
    NIM_TIMEOUT = "nim_timeout"
    NIM_5XX = "nim_5xx"
    NIM_4XX = "nim_4xx"

    # -- NIM endpoint types ---------------------------------------------
    PAGE_ELEMENTS = "page_elements"
    OCR = "ocr"
    TABLE_STRUCTURE = "table_structure"
    GRAPHIC_ELEMENTS = "graphic_elements"
    EMBED = "embed"
    RERANK = "rerank"

    # -- Vector store ---------------------------------------------------
    LANCEDB = "lancedb"

    # -- Infrastructure -------------------------------------------------
    OOM = "oom"
    SPOOL = "spool"
    DISPATCH = "dispatch"

    # -- Lifecycle ------------------------------------------------------
    CANCELLED = "cancelled"
    INTERNAL = "internal"
    AUTH = "auth"
    VALIDATION = "validation"
    UNKNOWN = "unknown"

    @classmethod
    def from_failure_type(cls, ft: FailureType) -> "EventCategory":
        """Map a legacy :class:`FailureType` to the corresponding category."""
        _MAP = {
            FailureType.PDF_PARSE: cls.PDF_PARSE,
            FailureType.NIM_TIMEOUT: cls.NIM_TIMEOUT,
            FailureType.NIM_5XX: cls.NIM_5XX,
            FailureType.NIM_4XX: cls.NIM_4XX,
            FailureType.OOM: cls.OOM,
            FailureType.INTERNAL: cls.INTERNAL,
            FailureType.CANCELLED: cls.CANCELLED,
            FailureType.UNKNOWN: cls.UNKNOWN,
        }
        return _MAP.get(ft, cls.UNKNOWN)


_HTTP_STATUS_RE = re.compile(r"\b([45]\d{2})\b")


def categorize_exception(exc: BaseException) -> FailureType:
    """Best-effort mapping of an exception to a :class:`FailureType`.

    The function inspects ``type(exc).__name__`` first, then falls back to
    string-matching on the message — that way we still classify cleanly
    even when the underlying library wraps everything in a generic
    ``RuntimeError``.
    """
    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    # OOM
    if name in {"memoryerror"} or "out of memory" in msg or "cuda out of memory" in msg:
        return FailureType.OOM

    # PDF / pypdfium2 parse errors
    if "pdfium" in msg or "pdf parse" in msg or "could not open pdf" in msg:
        return FailureType.PDF_PARSE
    if name in {"pdfsyntaxerror", "pdfreaderror"}:
        return FailureType.PDF_PARSE

    # NIM HTTP-status hints
    if "timeout" in name or "timeout" in msg or "timed out" in msg or "deadline exceeded" in msg:
        return FailureType.NIM_TIMEOUT

    status_match = _HTTP_STATUS_RE.search(msg)
    if status_match:
        code = int(status_match.group(1))
        if 500 <= code < 600:
            return FailureType.NIM_5XX
        if 400 <= code < 500:
            return FailureType.NIM_4XX

    if "connection" in msg and ("refused" in msg or "reset" in msg):
        return FailureType.NIM_5XX

    # Code bugs
    if name in {"assertionerror", "typeerror", "attributeerror", "keyerror", "valueerror"}:
        return FailureType.INTERNAL

    return FailureType.UNKNOWN
