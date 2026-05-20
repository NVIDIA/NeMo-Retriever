# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory sidecar-metadata store for ``vdb_upload`` requests.

The worker pod has no view into the calling client's filesystem, so a
caller that wants to attach sidecar metadata (the
``meta_dataframe``/``meta_source_field``/``meta_fields`` triple from
``VdbUploadParams``) cannot ship a local path or in-memory DataFrame
directly. The :class:`SidecarStore` lives in the service process; clients
:func:`POST /v1/ingest/sidecar` their dataframe (csv / parquet / json)
and receive an opaque ``sidecar_id`` they can reference in subsequent
ingest requests.

Trust boundary highlights:

* Sidecars are scoped per service instance and per-auth-bearer (when
  auth is enabled) — there is no cross-tenant visibility.
* Each upload has a TTL (default 1 hour) after which the bytes are
  purged. Workers read-and-consume by default; the same sidecar can
  be reused if the upload was created with ``consume_on_read=False``.
* Maximum payload size is bounded by ``ResourceLimitsConfig.max_upload_bytes``.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SidecarEntry:
    """One uploaded sidecar payload.

    The ``content_type`` is the MIME type as reported by the uploader
    (or inferred from the filename extension when absent); the worker
    uses it to pick the right pandas reader.
    """

    sidecar_id: str
    filename: str
    content_type: str
    payload: bytes
    created_at: float
    expires_at: float
    owner_token: Optional[str] = None
    consume_on_read: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


class SidecarStore:
    """Thread-safe in-memory keyed-by-id store with TTL eviction.

    The store is intentionally simple — a dict guarded by a lock. The
    expected working set is small (one entry per active ingest batch
    that needs sidecar metadata) and lifetimes are short (default
    one hour). For larger deployments a Redis-backed implementation
    can plug in via the same interface.
    """

    def __init__(self, *, default_ttl_s: float = 3600.0, max_entries: int = 1024) -> None:
        if default_ttl_s <= 0:
            raise ValueError("default_ttl_s must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._entries: dict[str, SidecarEntry] = {}
        self._lock = threading.Lock()
        self._default_ttl_s = default_ttl_s
        self._max_entries = max_entries

    # ── public API ─────────────────────────────────────────────────

    def put(
        self,
        *,
        filename: str,
        content_type: str,
        payload: bytes,
        owner_token: Optional[str] = None,
        ttl_s: Optional[float] = None,
        consume_on_read: bool = True,
    ) -> SidecarEntry:
        """Store ``payload`` and return the new :class:`SidecarEntry`.

        ``sidecar_id`` is a URL-safe 128-bit token (32 hex chars). The
        chance of collision is negligible for any realistic workload.
        """
        sidecar_id = secrets.token_urlsafe(16)
        now = time.time()
        ttl = float(ttl_s) if ttl_s is not None else self._default_ttl_s
        entry = SidecarEntry(
            sidecar_id=sidecar_id,
            filename=filename,
            content_type=content_type,
            payload=payload,
            created_at=now,
            expires_at=now + ttl,
            owner_token=owner_token,
            consume_on_read=consume_on_read,
        )
        with self._lock:
            self._evict_expired_locked(now)
            if len(self._entries) >= self._max_entries:
                # The cap is high (1024) so this is a last-resort guard
                # against runaway leaks rather than a typical case.
                raise RuntimeError(
                    f"SidecarStore is full ({self._max_entries} entries). "
                    "Existing sidecars must expire or be consumed before new uploads succeed."
                )
            self._entries[sidecar_id] = entry
        logger.info(
            "SidecarStore: stored sidecar_id=%s filename=%s bytes=%d ttl=%.0fs",
            sidecar_id,
            filename,
            len(payload),
            ttl,
        )
        return entry

    def get(self, sidecar_id: str, *, owner_token: Optional[str] = None) -> Optional[SidecarEntry]:
        """Look up a sidecar by id. Returns ``None`` when missing or expired.

        When ``owner_token`` is set, mismatched tokens get ``None`` even
        if the entry exists — a non-owner cannot probe for existence.
        """
        now = time.time()
        with self._lock:
            entry = self._entries.get(sidecar_id)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(sidecar_id, None)
                logger.debug("SidecarStore: sidecar_id=%s expired on read", sidecar_id)
                return None
            if entry.owner_token is not None and owner_token != entry.owner_token:
                logger.warning(
                    "SidecarStore: sidecar_id=%s owner mismatch (expected=%r got=%r)",
                    sidecar_id,
                    entry.owner_token,
                    owner_token,
                )
                return None
            return entry

    def consume(self, sidecar_id: str, *, owner_token: Optional[str] = None) -> Optional[SidecarEntry]:
        """:func:`get` + remove (when ``consume_on_read``).

        The default upload policy is single-use so the worker can
        release the bytes promptly. Lookup and removal run atomically
        under one lock acquisition so concurrent callers cannot both
        receive the same single-use entry.
        """
        now = time.time()
        with self._lock:
            entry = self._entries.get(sidecar_id)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(sidecar_id, None)
                logger.debug("SidecarStore: sidecar_id=%s expired on consume", sidecar_id)
                return None
            if entry.owner_token is not None and owner_token != entry.owner_token:
                logger.warning(
                    "SidecarStore: sidecar_id=%s owner mismatch (expected=%r got=%r)",
                    sidecar_id,
                    entry.owner_token,
                    owner_token,
                )
                return None
            if entry.consume_on_read:
                self._entries.pop(sidecar_id, None)
                logger.debug("SidecarStore: sidecar_id=%s consumed and removed", sidecar_id)
        return entry

    def delete(self, sidecar_id: str) -> bool:
        with self._lock:
            return self._entries.pop(sidecar_id, None) is not None

    def stats(self) -> dict[str, int | float]:
        now = time.time()
        with self._lock:
            self._evict_expired_locked(now)
            total_bytes = sum(len(e.payload) for e in self._entries.values())
            return {
                "entries": len(self._entries),
                "total_bytes": total_bytes,
                "max_entries": self._max_entries,
                "default_ttl_s": self._default_ttl_s,
            }

    # ── internal helpers ───────────────────────────────────────────

    def _evict_expired_locked(self, now: float) -> None:
        """Drop any entry whose TTL has elapsed.

        Caller must hold ``self._lock``.
        """
        stale = [sid for sid, entry in self._entries.items() if entry.expires_at <= now]
        for sid in stale:
            self._entries.pop(sid, None)
        if stale:
            logger.debug("SidecarStore: evicted %d expired sidecar(s)", len(stale))


# ── module-level singleton, mirroring the pattern used elsewhere ────

_instance: SidecarStore | None = None


def init_sidecar_store(*, default_ttl_s: float = 3600.0, max_entries: int = 1024) -> SidecarStore:
    global _instance
    _instance = SidecarStore(default_ttl_s=default_ttl_s, max_entries=max_entries)
    logger.info("SidecarStore initialised (ttl=%.0fs max_entries=%d)", default_ttl_s, max_entries)
    return _instance


def get_sidecar_store() -> SidecarStore | None:
    return _instance


def shutdown_sidecar_store() -> None:
    global _instance
    if _instance is not None:
        stats = _instance.stats()
        logger.info("SidecarStore shut down (entries=%d bytes=%d)", stats["entries"], stats["total_bytes"])
        _instance = None
