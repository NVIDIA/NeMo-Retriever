# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemo_retriever.harness.contracts import EXIT_INVALID, FailurePayload, HarnessRunError
from nemo_retriever.harness.json_io import read_json_object


def _read_summary(path_or_dir: Path) -> tuple[Path, dict[str, Any]]:
    try:
        path = path_or_dir.expanduser().resolve()
        if path.is_dir():
            results_path = path / "results.json"
            legacy_summary_path = path / "summary_metrics.json"
        elif path.name == "results.json":
            results_path = path
            legacy_summary_path = path.parent / "summary_metrics.json"
        elif path.name == "summary_metrics.json":
            results_path = path.parent / "results.json"
            legacy_summary_path = path
        else:
            raise ValueError(f"Expected a run directory, results.json, or summary_metrics.json: {path}")

        if results_path.exists():
            payload = read_json_object(results_path)
            summary = payload.get("summary_metrics")
            if not isinstance(summary, dict):
                raise ValueError(f"'summary_metrics' must be an object in {results_path}")
            return results_path, summary
        if legacy_summary_path.exists():
            return legacy_summary_path, read_json_object(legacy_summary_path)
        raise FileNotFoundError(f"No results.json or summary_metrics.json found under {path}")
    except (FileNotFoundError, ValueError) as exc:
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="invalid_benchmark",
                retryable=False,
                message=str(exc),
            ),
        ) from exc


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
