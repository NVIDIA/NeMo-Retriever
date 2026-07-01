# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Retained per-document result rows for split-topology workers.

Worker → gateway completion callbacks intentionally omit ``result_data``
to keep POST bodies small. When ``NEMO_RETRIEVER_RESULTS_DIR`` is set, each
write publishes a uniquely named immutable generation beneath that shared
directory. Reads are idempotent: every gateway can read the same generation
until TTL cleanup removes it.

The storage protocol intentionally has no claims or reader leases:

* a published generation is never renamed or replaced;
* reads never mutate storage;
* cleanup unlinks only the exact immutable generation it inspected; and
* I/O or decode failures preserve the generation and remain retryable.

An in-memory TTL-bounded store provides the same read semantics for local and
non-Helm deployments that do not configure shared storage.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_store: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_RESULTS_DIR_ENV = "NEMO_RETRIEVER_RESULTS_DIR"
_RESULTS_TTL_S_ENV = "NEMO_RETRIEVER_RESULTS_TTL_SECONDS"
_DEFAULT_RESULTS_TTL_S = 8 * 3600  # stale-job window plus terminal-job retention
_SWEEP_INTERVAL_S = 60
_STORE_NAMESPACE = ".worker-results-v1"
_last_sweep_dir: Path | None = None
_last_sweep_at = 0.0


class ResultStoreTemporarilyUnavailable(RuntimeError):
    """Raised when retained result data cannot currently be read safely."""


def _results_dir() -> Path | None:
    value = os.environ.get(_RESULTS_DIR_ENV, "").strip()
    return Path(value) if value else None


def is_shared_result_store_configured() -> bool:
    """Return whether this process is configured to use shared result files."""
    return _results_dir() is not None


def _results_root(results_dir: Path) -> Path:
    return results_dir / _STORE_NAMESPACE


def _document_digest(document_id: str) -> str:
    return hashlib.sha256(document_id.encode("utf-8")).hexdigest()


def _document_dir(results_dir: Path, document_id: str) -> Path:
    """Return a traversal-safe directory for *document_id*."""
    return _results_root(results_dir) / _document_digest(document_id)


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _is_document_dir(path: Path) -> bool:
    return _is_hex(path.name, 64)


def _is_generation(path: Path) -> bool:
    return path.name.endswith(".json") and _is_hex(path.name.removesuffix(".json"), 32)


def _is_temporary(path: Path) -> bool:
    name = path.name
    return name.startswith(".") and name.endswith(".tmp") and _is_hex(name[1:].removesuffix(".tmp"), 32)


def validate_result_store() -> None:
    """Fail startup when the configured shared store lacks required operations."""
    results_dir = _results_dir()
    if results_dir is None:
        return

    root = _results_root(results_dir)
    probe_id = uuid.uuid4().hex
    probe_dir = root / f".probe-{probe_id}"
    source = probe_dir / f".{probe_id}.tmp"
    target = probe_dir / f"{probe_id}.json"
    failure: OSError | None = None
    try:
        probe_dir.mkdir(parents=True, exist_ok=False)
        with source.open("x", encoding="utf-8") as stream:
            stream.write("probe")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(source, target)
        if target.read_text(encoding="utf-8") != "probe":
            raise OSError("shared result store probe read returned unexpected data")
        target.unlink()
        probe_dir.rmdir()
    except OSError as exc:
        failure = exc
    finally:
        for path in (source, target):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                failure = failure or exc
        try:
            probe_dir.rmdir()
        except FileNotFoundError:
            pass
        except OSError as exc:
            failure = failure or exc

    if failure is not None:
        raise RuntimeError(
            f"Shared result directory {results_dir} must support directory and file creation, "
            "fsync, same-directory atomic rename, read, and unlink"
        ) from failure


def _results_ttl_s() -> float:
    value = os.environ.get(_RESULTS_TTL_S_ENV, "").strip()
    if not value:
        return _DEFAULT_RESULTS_TTL_S
    try:
        ttl_s = float(value)
    except ValueError:
        ttl_s = 0
    if not math.isfinite(ttl_s) or ttl_s <= 0:
        logger.warning("Ignoring invalid %s=%r; using %s seconds", _RESULTS_TTL_S_ENV, value, _DEFAULT_RESULTS_TTL_S)
        return _DEFAULT_RESULTS_TTL_S
    return ttl_s


def _remove_expired_file(path: Path, *, cutoff: float) -> bool:
    """Remove an expired immutable generation or interrupted temporary file."""
    try:
        if path.stat().st_mtime > cutoff:
            return False
        path.unlink(missing_ok=True)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.debug("Unable to remove expired shared result file %s", path, exc_info=True)
        return False


def _sweep_expired_files(results_dir: Path, *, now: float, ttl_s: float) -> None:
    """Best-effort removal of expired immutable files owned by this store."""
    root = _results_root(results_dir)
    try:
        document_dirs = list(root.iterdir())
    except FileNotFoundError:
        return
    except OSError:
        logger.warning("Unable to scan shared result directory %s", root, exc_info=True)
        return

    cutoff = now - ttl_s
    removed = 0
    for document_dir in document_dirs:
        if not _is_document_dir(document_dir):
            continue
        try:
            paths = list(document_dir.iterdir())
        except FileNotFoundError:
            continue
        except OSError:
            logger.debug("Unable to scan shared result document directory %s", document_dir, exc_info=True)
            continue
        removed += sum(
            _remove_expired_file(path, cutoff=cutoff) for path in paths if _is_generation(path) or _is_temporary(path)
        )
        try:
            if document_dir.stat().st_mtime <= cutoff:
                document_dir.rmdir()
        except OSError:
            pass
    if removed:
        logger.info("Removed %d expired shared result file(s) from %s", removed, root)


def _maybe_sweep_expired_files(results_dir: Path) -> None:
    """Sweep at most once per interval in this process; other pods may also sweep."""
    global _last_sweep_at, _last_sweep_dir

    monotonic_now = time.monotonic()
    with _lock:
        if _last_sweep_dir == results_dir and monotonic_now - _last_sweep_at < _SWEEP_INTERVAL_S:
            return
        _last_sweep_dir = results_dir
        _last_sweep_at = monotonic_now
    _sweep_expired_files(results_dir, now=time.time(), ttl_s=_results_ttl_s())


def _store_on_filesystem(results_dir: Path, document_id: str, result_data: list[dict[str, Any]]) -> None:
    _maybe_sweep_expired_files(results_dir)
    document_dir = _document_dir(results_dir, document_id)
    document_dir.mkdir(parents=True, exist_ok=True)
    generation_id = uuid.uuid4().hex
    temporary = document_dir / f".{generation_id}.tmp"
    target = document_dir / f"{generation_id}.json"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(result_data, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logger.warning("Unable to remove interrupted shared result write %s", temporary, exc_info=True)


def _generation_candidates(document_dir: Path, document_id: str) -> list[Path]:
    try:
        paths = list(document_dir.iterdir())
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise ResultStoreTemporarilyUnavailable(f"Unable to scan shared results for {document_id!r}") from exc

    candidates: list[tuple[int, str, Path]] = []
    for path in paths:
        if not _is_generation(path):
            continue
        try:
            candidates.append((path.stat().st_mtime_ns, path.name, path))
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ResultStoreTemporarilyUnavailable(f"Unable to inspect shared result for {document_id!r}") from exc
    candidates.sort(reverse=True)
    return [path for _, _, path in candidates]


def _get_from_filesystem(results_dir: Path, document_id: str) -> list[dict[str, Any]] | None:
    _maybe_sweep_expired_files(results_dir)
    candidates = _generation_candidates(_document_dir(results_dir, document_id), document_id)
    for candidate in candidates:
        try:
            with candidate.open(encoding="utf-8") as stream:
                rows = json.load(stream)
        except FileNotFoundError:
            continue
        except (ValueError, OSError) as exc:
            logger.warning("Unable to read shared result payload for %r", document_id, exc_info=True)
            raise ResultStoreTemporarilyUnavailable(f"Unable to read shared result for {document_id!r}") from exc
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            logger.warning("Shared result payload for %r is not a JSON list of objects", document_id)
            raise ResultStoreTemporarilyUnavailable(f"Invalid shared result payload for {document_id!r}")
        return rows
    return None


def _sweep_memory_locked(*, now: float) -> None:
    cutoff = now - _results_ttl_s()
    for document_id, (stored_at, _) in list(_store.items()):
        if stored_at <= cutoff:
            _store.pop(document_id, None)


def store_result_data(document_id: str, result_data: list[dict[str, Any]] | None) -> None:
    """Retain *result_data* for a completed document until TTL cleanup."""
    if not document_id or not result_data:
        return
    if results_dir := _results_dir():
        _store_on_filesystem(results_dir, document_id, result_data)
        return
    with _lock:
        now = time.monotonic()
        _sweep_memory_locked(now=now)
        _store[document_id] = (now, copy.deepcopy(result_data))


def get_result_data(document_id: str) -> list[dict[str, Any]] | None:
    """Return retained rows for *document_id* without consuming them."""
    if results_dir := _results_dir():
        return _get_from_filesystem(results_dir, document_id)
    with _lock:
        _sweep_memory_locked(now=time.monotonic())
        entry = _store.get(document_id)
        return copy.deepcopy(entry[1]) if entry is not None else None


def clear_for_tests() -> None:
    """Test helper — drop all cached rows and sweep timestamps."""
    global _last_sweep_at, _last_sweep_dir
    with _lock:
        _store.clear()
        _last_sweep_dir = None
        _last_sweep_at = 0.0
