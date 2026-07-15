# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_retriever.harness.json_io import read_json_object

BASELINE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HarnessBaseline:
    name: str
    dataset: str
    metrics: dict[str, int | float]
    environment: dict[str, Any] = field(default_factory=dict)
    comparability: str = "comparable"
    notes: str | None = None
    source: dict[str, Any] = field(default_factory=dict)


def _required_text(item: dict[str, Any], key: str, *, index: int, path: Path) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Baseline at index {index} must define non-empty {key!r} in {path}")
    return value.strip()


def load_baselines(path: Path) -> list[HarnessBaseline]:
    resolved_path = Path(path).expanduser().resolve()
    payload = read_json_object(resolved_path)
    if payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported baseline schema_version in {resolved_path}; expected {BASELINE_SCHEMA_VERSION}")

    raw_baselines = payload.get("baselines")
    if not isinstance(raw_baselines, list) or not raw_baselines:
        raise ValueError(f"'baselines' must be a non-empty list in {resolved_path}")

    baselines: list[HarnessBaseline] = []
    for index, item in enumerate(raw_baselines):
        if not isinstance(item, dict):
            raise ValueError(f"Baseline at index {index} must be an object in {resolved_path}")
        metrics = item.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"Baseline at index {index} must define non-empty 'metrics' in {resolved_path}")
        invalid_metrics = [
            key
            for key, value in metrics.items()
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float))
        ]
        if invalid_metrics:
            raise ValueError(
                f"Baseline at index {index} has non-numeric metric values for {invalid_metrics!r} in {resolved_path}"
            )

        environment = item.get("environment", {})
        source = item.get("source", {})
        if not isinstance(environment, dict):
            raise ValueError(f"Baseline at index {index} 'environment' must be an object in {resolved_path}")
        if not isinstance(source, dict):
            raise ValueError(f"Baseline at index {index} 'source' must be an object in {resolved_path}")

        comparability = item.get("comparability", "comparable")
        if not isinstance(comparability, str) or not comparability.strip():
            raise ValueError(f"Baseline at index {index} 'comparability' must be non-empty text in {resolved_path}")
        notes = item.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ValueError(f"Baseline at index {index} 'notes' must be text in {resolved_path}")

        baselines.append(
            HarnessBaseline(
                name=_required_text(item, "name", index=index, path=resolved_path),
                dataset=_required_text(item, "dataset", index=index, path=resolved_path),
                metrics=dict(metrics),
                environment=dict(environment),
                comparability=comparability.strip(),
                notes=notes.strip() if notes else None,
                source=dict(source),
            )
        )
    return baselines
