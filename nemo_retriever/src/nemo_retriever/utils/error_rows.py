# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for detecting error payloads in dataframe rows."""

from __future__ import annotations

import json
from typing import Any


def _is_populated_error_field(key: str, value: Any) -> bool:
    if value is None:
        return False
    if key == "failed" and isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return bool(value)


def has_error(value: Any) -> bool:
    """Recursively detect whether a value contains error-like payloads."""
    if value is None:
        return False
    if isinstance(value, dict):
        for key in ("error", "errors", "exception", "traceback", "failed"):
            if key in value and _is_populated_error_field(key, value.get(key)):
                return True
        return any(has_error(item) for item in value.values())
    if isinstance(value, list):
        return any(has_error(item) for item in value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            try:
                return has_error(json.loads(stripped))
            except Exception:
                pass
        lowered = stripped.lower()
        return any(token in lowered for token in ("error", "exception", "traceback", "failed"))
    return False


def extract_error_rows(batch: Any) -> Any:
    """Return only rows that contain error-like payloads in known columns."""
    if batch is None:
        return batch
    columns = getattr(batch, "columns", None)
    if columns is None:
        return batch

    error_candidate_columns = (
        "error",
        "errors",
        "exception",
        "traceback",
        "metadata",
        "source",
        "embedding",
    )
    cols = [column for column in error_candidate_columns if column in columns]
    if not cols:
        return batch.iloc[0:0]

    mask = batch[cols[0]].apply(has_error).astype(bool)
    for column in cols[1:]:
        mask = mask | batch[column].apply(has_error).astype(bool)
    return batch[mask]
