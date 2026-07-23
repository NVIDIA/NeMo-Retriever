# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_retriever.harness.json_io import read_json_object


@dataclass(frozen=True)
class ReleaseReference:
    release: str
    dataset: str
    environment: dict[str, Any]
    metrics: dict[str, int | float]


def load_release_references(path: Path) -> list[ReleaseReference]:
    payload = read_json_object(path.expanduser().resolve())
    results = payload.get("baselines")
    if not isinstance(results, list) or not results:
        raise ValueError("Release reference must define a non-empty 'baselines' list")

    references: list[ReleaseReference] = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"Release result at index {index} must be an object")
        release = result.get("name")
        dataset = result.get("dataset")
        environment = result.get("environment", {})
        metrics = result.get("metrics")
        if not isinstance(release, str) or not release.strip():
            raise ValueError(f"Release result at index {index} must define non-empty 'name' text")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError(f"Release result at index {index} must define non-empty 'dataset' text")
        if not isinstance(environment, dict):
            raise ValueError(f"Release result at index {index} 'environment' must be an object")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"Release result at index {index} must define non-empty 'metrics'")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in metrics.values()):
            raise ValueError(f"Release result at index {index} metrics must be numeric")
        references.append(
            ReleaseReference(
                release=release.strip(),
                dataset=dataset.strip(),
                environment=dict(environment),
                metrics=dict(metrics),
            )
        )
    return references
