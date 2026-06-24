# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from nemo_retriever.harness.contracts import EXIT_INVALID, FailurePayload, HarnessRunError


@dataclass(frozen=True)
class RunFileRequest:
    benchmark: str
    name: str | None
    mode: str | None
    output_dir: str | None
    run_id: str | None
    overrides: tuple[str, ...]
    requirements: tuple[str, ...]
    dry_run: bool | None
    source_path: Path
    payload: Mapping[str, Any]


def _invalid(message: str) -> HarnessRunError:
    return HarnessRunError(
        EXIT_INVALID,
        FailurePayload(
            failed_phase="resolve",
            failure_reason="invalid_runfile",
            retryable=False,
            message=message,
        ),
    )


def _load_payload(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _invalid(f"Could not read runfile {path}: {exc}") from exc

    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(raw)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(raw)
        else:
            raise _invalid("Runfile must use .json, .yaml, or .yml.")
    except json.JSONDecodeError as exc:
        raise _invalid(f"Could not parse JSON runfile {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise _invalid(f"Could not parse YAML runfile {path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise _invalid("Runfile must contain an object at the top level.")
    return payload


def _stringify_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _overrides_from_payload(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        return tuple(f"{key}={_stringify_override_value(value)}" for key, value in raw.items())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        overrides: list[str] = []
        for item in raw:
            if not isinstance(item, str) or "=" not in item:
                raise _invalid("Runfile 'set' entries must be KEY=VALUE strings or an object.")
            overrides.append(item)
        return tuple(overrides)
    raise _invalid("Runfile 'set' must be an object or a list of KEY=VALUE strings.")


def _requirements_from_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    raw = payload.get("require", payload.get("requirements"))
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise _invalid("Runfile 'require' must be a list of metric gate strings.")
    requirements: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise _invalid("Runfile 'require' entries must be strings.")
        requirements.append(item)
    return tuple(requirements)


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid(f"Runfile '{key}' must be a string.")
    return value


def load_runfile(path: Path) -> RunFileRequest:
    source_path = path.expanduser().resolve()
    payload = _load_payload(source_path)
    benchmark = payload.get("benchmark", payload.get("base"))
    if not isinstance(benchmark, str) or not benchmark:
        raise _invalid("Runfile must set 'benchmark' to a registered benchmark name.")
    if "benchmark" in payload and "base" in payload and payload["benchmark"] != payload["base"]:
        raise _invalid("Runfile cannot set conflicting 'benchmark' and 'base' values.")

    dry_run = payload.get("dry_run")
    if dry_run is not None and not isinstance(dry_run, bool):
        raise _invalid("Runfile 'dry_run' must be true or false.")

    return RunFileRequest(
        benchmark=benchmark,
        name=_optional_string(payload, "name"),
        mode=_optional_string(payload, "mode"),
        output_dir=_optional_string(payload, "output_dir"),
        run_id=_optional_string(payload, "run_id"),
        overrides=_overrides_from_payload(payload.get("set")),
        requirements=_requirements_from_payload(payload),
        dry_run=dry_run,
        source_path=source_path,
        payload=payload,
    )
