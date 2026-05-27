# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Ephemeral per-document result rows on worker pods (split topology).

Worker → gateway completion callbacks intentionally omit ``result_data``
to keep POST bodies small. Rows are held here until the gateway (or a
client polling through the gateway) fetches them via
``GET /v1/internal/document-result/{document_id}``.
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_store: dict[str, list[dict[str, Any]]] = {}


def store_result_data(document_id: str, result_data: list[dict[str, Any]] | None) -> None:
    """Retain *result_data* for a completed document on this worker pod."""
    if not document_id or not result_data:
        return
    with _lock:
        _store[document_id] = result_data


def consume_result_data(document_id: str) -> list[dict[str, Any]] | None:
    """Return stored rows for *document_id* and remove them from memory."""
    with _lock:
        return _store.pop(document_id, None)


def clear_for_tests() -> None:
    """Test helper — drop all cached rows."""
    with _lock:
        _store.clear()
