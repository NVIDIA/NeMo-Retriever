# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from nemo_retriever.harness.benchmark_registry import get_dataset
from nemo_retriever.harness.contracts import EXIT_INVALID, FailurePayload, HarnessRunError


@dataclass(frozen=True)
class DatasetPaths:
    path: Path
    query_file: Path | None = None

    def overrides(self) -> tuple[str, ...]:
        values = [f"dataset.path={json.dumps(str(self.path))}"]
        if self.query_file is not None:
            query_file = json.dumps(str(self.query_file))
            values.extend(
                (
                    f"dataset.query_file={query_file}",
                    f"evaluation.dataset_name={query_file}",
                )
            )
        return tuple(values)

    def to_dict(self) -> dict[str, str]:
        payload = {"path": str(self.path)}
        if self.query_file is not None:
            payload["query_file"] = str(self.query_file)
        return payload


def _invalid(message: str) -> HarnessRunError:
    return HarnessRunError(
        EXIT_INVALID,
        FailurePayload(
            failed_phase="resolve",
            failure_reason="invalid_dataset_paths",
            retryable=False,
            message=message,
        ),
    )


def _resolve_local_path(config_path: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"Dataset path field {field!r} must be a non-empty string.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_dataset_paths(path: Path | None) -> dict[str, DatasetPaths]:
    if path is None:
        return {}

    config_path = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise _invalid(f"Could not read dataset paths file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise _invalid(f"Could not parse dataset paths file {config_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise _invalid("Dataset paths file must contain an object at the top level.")
    unknown_top_level = sorted(set(raw) - {"schema_version", "datasets"})
    if unknown_top_level:
        raise _invalid(f"Dataset paths file contains unknown key {unknown_top_level[0]!r}.")
    if raw.get("schema_version") != 1:
        raise _invalid("Dataset paths file 'schema_version' must be 1.")

    datasets = raw.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise _invalid("Dataset paths file 'datasets' must be a non-empty object.")

    resolved: dict[str, DatasetPaths] = {}
    for raw_name, raw_paths in datasets.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise _invalid("Dataset path names must be non-empty strings.")
        try:
            get_dataset(raw_name)
        except KeyError as exc:
            raise _invalid(f"Dataset paths file references unknown dataset {raw_name!r}.") from exc
        if not isinstance(raw_paths, Mapping):
            raise _invalid(f"Dataset paths entry {raw_name!r} must be an object.")
        unknown_fields = sorted(set(raw_paths) - {"path", "query_file"})
        if unknown_fields:
            raise _invalid(f"Dataset paths entry {raw_name!r} contains unknown key {unknown_fields[0]!r}.")
        if "path" not in raw_paths:
            raise _invalid(f"Dataset paths entry {raw_name!r} must set 'path'.")

        query_file = raw_paths.get("query_file")
        resolved[raw_name] = DatasetPaths(
            path=_resolve_local_path(config_path, raw_paths["path"], field=f"{raw_name}.path"),
            query_file=(
                _resolve_local_path(config_path, query_file, field=f"{raw_name}.query_file")
                if query_file is not None
                else None
            ),
        )
    return resolved
