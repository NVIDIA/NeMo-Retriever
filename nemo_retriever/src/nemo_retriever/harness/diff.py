# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemo_retriever.harness.contracts import EXIT_INVALID, FailurePayload, HarnessRunError


def _artifact_file(path_or_dir: Path, name: str) -> Path:
    path = path_or_dir.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if path.is_dir():
        path = path / name
    if not path.exists():
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="invalid_benchmark",
                retryable=False,
                message=f"Artifact file not found: {path}",
            ),
        )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="invalid_benchmark",
                retryable=False,
                message=f"Invalid JSON in {path}: {exc}",
            ),
        ) from exc
    if not isinstance(data, dict):
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="invalid_benchmark",
                retryable=False,
                message=f"Expected JSON object in {path}",
            ),
        )
    return data


def _read_summary(path_or_dir: Path) -> tuple[Path, dict[str, Any]]:
    summary_path = _artifact_file(path_or_dir, "summary_metrics.json")
    return summary_path, _read_json(summary_path)


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def diff_artifact_dirs(left: Path, right: Path) -> dict[str, Any]:
    left_path, left_summary = _read_summary(left)
    right_path, right_summary = _read_summary(right)
    keys = sorted(set(left_summary) | set(right_summary))
    deltas: dict[str, dict[str, Any]] = {}
    for key in keys:
        left_value = left_summary.get(key)
        right_value = right_summary.get(key)
        left_number = _numeric(left_value)
        right_number = _numeric(right_value)
        payload: dict[str, Any] = {
            "left": left_value,
            "right": right_value,
            "changed": left_value != right_value,
        }
        if left_number is not None and right_number is not None:
            delta = right_number - left_number
            payload["delta"] = delta
            payload["percent_delta"] = (delta / left_number * 100.0) if left_number else None
        deltas[key] = payload
    return {
        "left": str(left_path),
        "right": str(right_path),
        "summary_metrics": deltas,
    }
