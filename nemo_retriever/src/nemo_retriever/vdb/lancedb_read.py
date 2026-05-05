# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only LanceDB helpers (table scans, counts) kept under ``vdb``."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Tuple


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def iter_lancedb_rows(uri: str, table_name: str) -> Iterable[Tuple[Any, Dict[str, Any], Dict[str, Any]]]:
    """Yield ``(page_key, meta, row_dict)`` tuples from a LanceDB table."""
    import lancedb  # type: ignore

    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    df = table.to_pandas()[["source_id", "page_number", "metadata"]]

    for row in df.itertuples(index=False):
        source_id = str(getattr(row, "source_id", "") or "")
        page_number = _safe_int(getattr(row, "page_number", -1), default=-1)
        raw_metadata = getattr(row, "metadata", None)
        meta: dict = {}
        if isinstance(raw_metadata, str) and raw_metadata.strip():
            try:
                parsed = json.loads(raw_metadata)
                if isinstance(parsed, dict):
                    meta = parsed
            except Exception:
                pass
        yield (source_id, page_number), meta, {}


def lancedb_row_count(uri: str, table_name: str) -> int:
    """Return ``table.count_rows()`` or 0 on failure."""
    import lancedb  # type: ignore

    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    return int(table.count_rows())


def estimate_processed_pages(uri: str, table_name: str) -> Optional[int]:
    """Estimate pages processed by counting unique (source_id, page_number) pairs.

    Falls back to table row count if page-level fields are unavailable.
    """
    try:
        import lancedb  # type: ignore

        db = lancedb.connect(uri)
        table = db.open_table(table_name)
    except Exception:
        return None

    try:
        df = table.to_pandas()[["source_id", "page_number"]]
        return int(df.dropna(subset=["source_id", "page_number"]).drop_duplicates().shape[0])
    except Exception:
        try:
            return int(table.count_rows())
        except Exception:
            return None
