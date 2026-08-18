# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared detection of structured row-level stage errors."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

ERROR_FIELD_KEYS = ("error", "errors", "exception", "traceback", "failed")


def is_populated_error_field(key: str, value: Any) -> bool:
    """Return whether a conventional stage-error field contains a failure."""
    if value is None:
        return False
    if key == "failed" and isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return bool(value)


def iter_stage_errors_from_value(value: Any, *, path: str = "") -> Iterator[dict[str, Any]]:
    """Yield populated error fields recursively with their value paths."""
    if isinstance(value, dict):
        for key in ERROR_FIELD_KEYS:
            if key in value and is_populated_error_field(key, value.get(key)):
                yield {
                    "path": f"{path}.{key}" if path else key,
                    "error": value.get(key),
                }
        for key, child in value.items():
            if key in ERROR_FIELD_KEYS and is_populated_error_field(key, child):
                continue
            child_path = f"{path}.{key}" if path else str(key)
            yield from iter_stage_errors_from_value(child, path=child_path)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            yield from iter_stage_errors_from_value(child, path=child_path)
