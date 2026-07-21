# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from nemo_retriever.harness.contracts import (
    EXIT_ARTIFACT_WRITE_FAILURE,
    FailurePayload,
    HarnessRunError,
)


def artifact_write_error(exc: BaseException) -> HarnessRunError:
    return HarnessRunError(
        EXIT_ARTIFACT_WRITE_FAILURE,
        FailurePayload(
            failed_phase="write_artifacts",
            failure_reason="artifact_write_failed",
            retryable=False,
            message=f"{type(exc).__name__}: {exc}",
        ),
    )


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        rendered = json.dumps(jsonable(payload), indent=2, sort_keys=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(rendered)
        assert temporary_path is not None
        os.replace(temporary_path, path)
    except Exception as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise artifact_write_error(exc) from exc


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def artifact_file(path_or_dir: Path, name: str) -> Path:
    path = path_or_dir.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if path.is_dir():
        path = path / name
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path}")
    return path
