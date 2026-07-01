# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Per-document result rows for split-topology workers.

Worker → gateway completion callbacks intentionally omit ``result_data``
to keep POST bodies small. When ``NEMO_RETRIEVER_RESULTS_DIR`` is set, rows
are written atomically to that shared directory so any gateway or worker pod
can consume them by document ID. The in-memory store remains as a fallback
for local and non-Helm deployments that do not configure shared storage.
"""

from __future__ import annotations

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
_store: dict[str, list[dict[str, Any]]] = {}
_RESULTS_DIR_ENV = "NEMO_RETRIEVER_RESULTS_DIR"
_RESULTS_TTL_S_ENV = "NEMO_RETRIEVER_RESULTS_TTL_SECONDS"
_DEFAULT_RESULTS_TTL_S = 8 * 3600  # stale-job window plus terminal-job retention
_SWEEP_INTERVAL_S = 60
_CLAIM_LEASE_S = 60
_last_sweep_dir: Path | None = None
_last_sweep_at = 0.0


class ResultStoreTemporarilyUnavailable(RuntimeError):
    """Raised when shared result data may be retryable after an I/O failure."""


def _results_dir() -> Path | None:
    value = os.environ.get(_RESULTS_DIR_ENV, "").strip()
    return Path(value) if value else None


def _result_path(results_dir: Path, document_id: str) -> Path:
    """Return a traversal-safe, deterministic path for *document_id*."""
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    return results_dir / f"{digest}.json"


def validate_result_store() -> None:
    """Fail startup when the configured shared store lacks required operations."""
    results_dir = _results_dir()
    if results_dir is None:
        return

    probe_id = uuid.uuid4().hex
    source = results_dir / f".result-store-probe-{probe_id}.tmp"
    target = results_dir / f".result-store-probe-{probe_id}.target"
    linked = results_dir / f".result-store-probe-{probe_id}.link"
    failure: OSError | None = None
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        with source.open("x", encoding="utf-8") as stream:
            stream.write("probe")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(source, target)
        os.link(target, linked)
        linked.unlink()
        target.unlink()
    except OSError as exc:
        failure = exc
    finally:
        for path in (source, target, linked):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                failure = failure or exc

    if failure is not None:
        raise RuntimeError(
            f"Shared result directory {results_dir} must support file creation, fsync, "
            "atomic rename, hard links, and unlink"
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


def _is_result_store_file(path: Path) -> bool:
    """Return whether *path* is a result or an intermediate result file."""
    name = path.name
    if name.endswith(".json"):
        digest = name.removesuffix(".json")
        return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
    if not name.startswith(".") or not name.endswith((".tmp", ".claim", ".cleanup")):
        return False
    parts = name[1:].split(".")
    return (
        len(parts) == 4
        and len(parts[0]) == 64
        and all(character in "0123456789abcdef" for character in parts[0])
        and parts[1] == "json"
        and len(parts[2]) == 32
        and all(character in "0123456789abcdef" for character in parts[2])
    )


def _remove_expired_result(path: Path, *, cutoff: float) -> bool:
    """Atomically claim and remove an expired result without deleting a replacement."""
    try:
        if path.stat().st_mtime > cutoff:
            return False
        if not path.name.endswith(".json"):
            path.unlink(missing_ok=True)
            return True

        claimed = path.with_name(f".{path.name}.{uuid.uuid4().hex}.cleanup")
        # Restoring a concurrently replaced result requires an atomic,
        # no-overwrite hard link. Startup validates this operation, while this
        # runtime guard keeps expiry best-effort if mount behavior changes.
        os.link(path, claimed)
        claimed.unlink(missing_ok=True)

        os.replace(path, claimed)
        if claimed.stat().st_mtime <= cutoff:
            claimed.unlink(missing_ok=True)
            return True

        # A writer replaced the stale file between stat() and os.replace().
        # Restore the fresh inode only if no newer result now owns the path.
        try:
            os.link(claimed, path)
        except FileExistsError:
            pass
        claimed.unlink(missing_ok=True)
    except FileNotFoundError:
        pass  # Another pod consumed, replaced, or swept it first.
    except OSError:
        logger.debug("Unable to remove expired shared result file %s", path, exc_info=True)
    return False


def _sweep_expired_files(results_dir: Path, *, now: float, ttl_s: float) -> None:
    """Best-effort removal of expired files owned by this result store."""
    try:
        paths = list(results_dir.iterdir())
    except FileNotFoundError:
        return
    except OSError:
        logger.warning("Unable to scan shared result directory %s", results_dir, exc_info=True)
        return

    cutoff = now - ttl_s
    removed = sum(_remove_expired_result(path, cutoff=cutoff) for path in paths if _is_result_store_file(path))
    if removed:
        logger.info("Removed %d expired shared result file(s) from %s", removed, results_dir)


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
    results_dir.mkdir(parents=True, exist_ok=True)
    _maybe_sweep_expired_files(results_dir)
    target = _result_path(results_dir, document_id)
    temporary = results_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(result_data, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_claim(claimed: Path, document_id: str) -> None:
    """Best-effort removal that must not turn a successful read into an error."""
    try:
        claimed.unlink(missing_ok=True)
    except OSError:
        logger.warning("Unable to remove shared result claim for %r", document_id, exc_info=True)


def _restore_claim(claimed: Path, target: Path, document_id: str) -> bool:
    """Restore a failed claim for retry without replacing an existing result."""
    try:
        os.link(claimed, target)
    except FileExistsError:
        # A replacement already owns the canonical path, so the older claim
        # must not overwrite it.
        _remove_claim(claimed, document_id)
        return True
    except FileNotFoundError:
        try:
            target.stat()
        except FileNotFoundError:
            return False
        except OSError:
            logger.warning("Unable to inspect shared result while restoring %r", document_id, exc_info=True)
            return False
        return True
    except OSError:
        # Preserve the claim when restoration fails. A later request can
        # recover it after its active-consumer lease expires.
        logger.warning("Unable to restore shared result claim for %r", document_id, exc_info=True)
        return False

    _remove_claim(claimed, document_id)
    return True


def _recover_abandoned_claim(results_dir: Path, target: Path, claimed: Path, document_id: str) -> bool:
    """Atomically acquire an abandoned claim after its consumer lease expires."""
    cutoff = time.time() - _CLAIM_LEASE_S
    try:
        candidates = list(results_dir.glob(f".{target.name}.*.claim"))
    except OSError as exc:
        raise ResultStoreTemporarilyUnavailable(f"Unable to scan shared result claims for {document_id!r}") from exc

    for candidate in candidates:
        try:
            if candidate.stat().st_mtime > cutoff:
                continue
            os.replace(candidate, claimed)
            os.utime(claimed, None)
            logger.info("Recovered abandoned shared result claim for %r", document_id)
            return True
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ResultStoreTemporarilyUnavailable(
                f"Unable to recover shared result claim for {document_id!r}"
            ) from exc
    return False


def _consume_from_filesystem(results_dir: Path, document_id: str) -> list[dict[str, Any]] | None:
    _maybe_sweep_expired_files(results_dir)
    target = _result_path(results_dir, document_id)
    claimed = results_dir / f".{target.name}.{uuid.uuid4().hex}.claim"
    try:
        os.replace(target, claimed)
        # A rename preserves the result's original mtime. Touch the claim so
        # other consumers can distinguish an active lease from an abandoned
        # claim left behind by an interrupted or failed reader.
        os.utime(claimed, None)
    except FileNotFoundError:
        if not _recover_abandoned_claim(results_dir, target, claimed, document_id):
            return None
    except OSError as exc:
        logger.warning("Unable to claim shared result for %r", document_id, exc_info=True)
        # Network filesystems can report an error after applying a rename, so
        # restore the known claim path if the canonical path disappeared.
        _restore_claim(claimed, target, document_id)
        raise ResultStoreTemporarilyUnavailable(f"Unable to claim shared result for {document_id!r}") from exc

    discard_claim = True
    try:
        with claimed.open(encoding="utf-8") as stream:
            rows = json.load(stream)
        if not isinstance(rows, list):
            raise ValueError(f"Shared result payload for {document_id!r} is not a JSON list")
        return rows
    except FileNotFoundError:
        logger.debug("Shared result claim disappeared while consuming %r", document_id)
        return None
    except ValueError:
        logger.warning("Unable to decode shared result payload for %r", document_id, exc_info=True)
        return None
    except OSError as exc:
        discard_claim = False
        logger.warning("I/O error while reading shared result for %r", document_id, exc_info=True)
        _restore_claim(claimed, target, document_id)
        raise ResultStoreTemporarilyUnavailable(f"Unable to read shared result for {document_id!r}") from exc
    finally:
        if discard_claim:
            _remove_claim(claimed, document_id)


def store_result_data(document_id: str, result_data: list[dict[str, Any]] | None) -> None:
    """Retain *result_data* for a completed document."""
    if not document_id or not result_data:
        return
    if results_dir := _results_dir():
        _store_on_filesystem(results_dir, document_id, result_data)
        return
    with _lock:
        _store[document_id] = result_data


def consume_result_data(document_id: str) -> list[dict[str, Any]] | None:
    """Return stored rows for *document_id* and remove them from the store."""
    if results_dir := _results_dir():
        return _consume_from_filesystem(results_dir, document_id)
    with _lock:
        return _store.pop(document_id, None)


def clear_for_tests() -> None:
    """Test helper — drop all cached rows."""
    global _last_sweep_at, _last_sweep_dir
    with _lock:
        _store.clear()
        _last_sweep_dir = None
        _last_sweep_at = 0.0
