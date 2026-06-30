# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


_SECRET_KEY_PARTS = {
    "authorization",
    "credential",
    "credentials",
    "passwd",
    "password",
    "secret",
    "token",
}


def _is_secret_key(key: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
    parts = set(normalized.split("_"))
    return "api_key" in normalized or bool(parts & _SECRET_KEY_PARTS)


def redact(value: Any) -> Any:
    """Return JSON-compatible data with credential values removed.

    In addition to nested credential fields, this covers runfile and CLI
    override strings such as ``query.reranker_api_key=...``. Metric names such
    as ``input_tokens`` are intentionally not treated as credentials.
    """
    if isinstance(value, Mapping):
        out: dict[Any, Any] = {}
        for key, nested in value.items():
            if _is_secret_key(key):
                out[key] = "<redacted>" if nested else nested
            else:
                out[key] = redact(nested)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str) and "=" in value:
        key, nested = value.split("=", 1)
        if _is_secret_key(key):
            return f"{key}=<redacted>" if nested else value
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a redacted JSON object in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(redact(jsonable(payload)), indent=2, sort_keys=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
