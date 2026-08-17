# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in persistence of outbound remote-inference requests.

The recorder is deliberately transport and model-schema agnostic.  It records
the final JSON value handed to an HTTP client, not application-level inputs,
so artifacts can be replayed against a compatible NIM endpoint.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Literal
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

CaptureFailureMode = Literal["best_effort", "required"]


@dataclass(frozen=True)
class InferenceCaptureConfig:
    """Configuration for recording remote model requests.

    ``storage_uri`` accepts a local directory or an fsspec-compatible URI.
    The default mode never changes inference behavior when the capture sink is
    unavailable; callers that generate replay fixtures can use ``required``.
    """

    storage_uri: str
    failure_mode: CaptureFailureMode = "best_effort"
    operations: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.storage_uri).strip():
            raise ValueError("inference capture storage_uri must not be empty")
        if self.failure_mode not in {"best_effort", "required"}:
            raise ValueError("inference capture failure_mode must be 'best_effort' or 'required'")

    @classmethod
    def from_value(cls, value: "InferenceCaptureConfig | dict[str, Any] | None") -> "InferenceCaptureConfig | None":
        if value is None or isinstance(value, cls):
            return value
        values = dict(value)
        for name in ("operations", "stages"):
            if name in values and values[name] is not None:
                values[name] = tuple(values[name])
        return cls(**values)


_capture_config: contextvars.ContextVar[InferenceCaptureConfig | None] = contextvars.ContextVar(
    "nemo_retriever_inference_capture", default=None
)
_capture_operation: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nemo_retriever_inference_capture_operation", default=None
)
# ThreadPool workers used by NIM clients do not inherit ContextVars.  Keep a
# process fallback for the active operation; context-local values still win.
_active_config: InferenceCaptureConfig | None = None
_active_operation: str | None = None


@contextlib.contextmanager
def activate_inference_capture(
    config: InferenceCaptureConfig | dict[str, Any] | None,
    *,
    operation: str | None = None,
) -> Iterator[None]:
    """Activate capture for synchronous work in the current execution context."""

    global _active_config, _active_operation
    parsed = InferenceCaptureConfig.from_value(config)
    prior_config, prior_operation = _active_config, _active_operation
    _active_config, _active_operation = parsed, operation
    config_token = _capture_config.set(parsed)
    operation_token = _capture_operation.set(operation)
    try:
        yield
    finally:
        _capture_operation.reset(operation_token)
        _capture_config.reset(config_token)
        _active_config, _active_operation = prior_config, prior_operation


def _safe_endpoint(endpoint: str) -> str:
    parts = urlsplit(str(endpoint))
    if not parts.scheme or not parts.netloc:
        return str(endpoint).split("?", 1)[0].split("#", 1)[0]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _environment_capture_config() -> InferenceCaptureConfig | None:
    uri = os.environ.get("NEMO_RETRIEVER_INFERENCE_CAPTURE_URI", "").strip()
    if not uri:
        return None
    return InferenceCaptureConfig(
        storage_uri=uri,
        failure_mode=os.environ.get("NEMO_RETRIEVER_INFERENCE_CAPTURE_FAILURE_MODE", "best_effort"),
    )


def _matches(config: InferenceCaptureConfig, operation: str | None, stage: str) -> bool:
    return (not config.operations or (operation or "") in config.operations) and (
        not config.stages or stage in config.stages
    )


def stage_from_endpoint(endpoint: str, *, default: str = "remote_nim") -> str:
    """Return a stable, schema-independent stage label from an invoke URL."""
    path = urlsplit(str(endpoint)).path.rstrip("/")
    if not path:
        return default
    return path.rsplit("/", 1)[-1].replace("-", "_") or default


def _write_local(directory: Path, manifest: dict[str, Any], body: bytes, suffix: str) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    try:
        manifest_tmp = directory / ".manifest.tmp"
        request_tmp = directory / ".request.tmp"
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        request_tmp.write_bytes(body)
        os.replace(manifest_tmp, directory / "manifest.json")
        os.replace(request_tmp, directory / f"request.{suffix}")
    except Exception:
        # A partially populated capture is never useful as a replay fixture.
        for item in directory.glob("*"):
            item.unlink(missing_ok=True)
        directory.rmdir()
        raise


def _write_capture(
    config: InferenceCaptureConfig, capture_id: str, manifest: dict[str, Any], body: bytes, suffix: str
) -> None:
    uri = str(config.storage_uri)
    if "://" not in uri:
        _write_local(Path(uri).expanduser().resolve() / capture_id, manifest, body, suffix)
        return

    import fsspec  # noqa: PLC0415

    base = uri.rstrip("/") + "/" + capture_id
    with fsspec.open(base + "/manifest.json", "wt") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    with fsspec.open(base + f"/request.{suffix}", "wb") as handle:
        handle.write(body)


def record_binary_request(
    *,
    stage: str,
    endpoint: str,
    payload: bytes,
    protocol: str,
    model: str | None = None,
    attempt: int = 0,
    metadata: dict[str, Any] | None = None,
    operation: str | None = None,
) -> None:
    """Persist one opaque transport payload, for example Triton gRPC tensors."""
    config = _capture_config.get() or _active_config or _environment_capture_config()
    selected_operation = (
        operation
        or _capture_operation.get()
        or _active_operation
        or os.environ.get("NEMO_RETRIEVER_INFERENCE_CAPTURE_OPERATION")
    )
    if config is None or not _matches(config, selected_operation, stage):
        return
    try:
        capture_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
        manifest = {
            "capture_version": 1,
            "capture_id": capture_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": selected_operation,
            "stage": stage,
            "protocol": protocol,
            "endpoint": _safe_endpoint(endpoint),
            "model": model,
            "attempt": int(attempt),
            "content_type": "application/octet-stream",
            "metadata": metadata or {},
        }
        _write_capture(config, capture_id, manifest, payload, "bin")
    except Exception as exc:
        if config.failure_mode == "required":
            raise RuntimeError(f"Failed to persist inference capture for {stage}: {exc}") from exc
        logger.warning("Failed to persist inference capture for %s: %s", stage, exc)


def record_json_request(
    *,
    stage: str,
    endpoint: str,
    payload: Any,
    method: str = "POST",
    model: str | None = None,
    attempt: int = 0,
    operation: str | None = None,
) -> None:
    """Persist one final JSON request when capture is active.

    Credentials are intentionally not accepted by this function.  Endpoint
    query strings are omitted because they can contain credentials.
    """

    config = _capture_config.get() or _active_config or _environment_capture_config()
    selected_operation = (
        operation
        or _capture_operation.get()
        or _active_operation
        or os.environ.get("NEMO_RETRIEVER_INFERENCE_CAPTURE_OPERATION")
    )
    if config is None or not _matches(config, selected_operation, stage):
        return

    try:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        capture_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
        manifest = {
            "capture_version": 1,
            "capture_id": capture_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": selected_operation,
            "stage": stage,
            "protocol": "http",
            "method": method,
            "endpoint": _safe_endpoint(endpoint),
            "model": model,
            "attempt": int(attempt),
            "content_type": "application/json",
        }
        _write_capture(config, capture_id, manifest, body, "json")
    except Exception as exc:
        if config.failure_mode == "required":
            raise RuntimeError(f"Failed to persist inference capture for {stage}: {exc}") from exc
        logger.warning("Failed to persist inference capture for %s: %s", stage, exc)
