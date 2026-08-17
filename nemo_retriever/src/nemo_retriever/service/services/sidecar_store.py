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

import hmac
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_REDIS_PREFIX = "nrl:sidecar:"


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


class RedisSidecarStore:
    """Redis-backed sidecars shared by every split worker replica."""

    _CONSUME_SCRIPT = """
local data = redis.call("HGETALL", KEYS[1])
if #data == 0 then return nil end
local owner = ""
local consume = "0"
for index = 1, #data, 2 do
  if data[index] == "owner_token" then owner = data[index + 1] end
  if data[index] == "consume_on_read" then consume = data[index + 1] end
end
if owner ~= ARGV[1] then return nil end
if consume == "1" then redis.call("DEL", KEYS[1]) end
return data
"""
    _DELETE_SCRIPT = """
local owner = redis.call("HGET", KEYS[1], "owner_token")
if not owner or owner ~= ARGV[1] then return 0 end
return redis.call("DEL", KEYS[1])
"""

    def __init__(
        self, url: str, *, default_ttl_s: float = 3600.0, max_payload_bytes: int = 33_554_432
    ) -> None:
        from redis import Redis

        self._client = Redis.from_url(url, decode_responses=False)
        self._default_ttl_s = default_ttl_s
        self._max_payload_bytes = max_payload_bytes


    @staticmethod
    def _key(sidecar_id: str) -> str:
        return f"{_REDIS_PREFIX}{sidecar_id}"


    @staticmethod
    def _entry_from_data(sidecar_id: str, data: dict[bytes, bytes]) -> SidecarEntry:
        def text(key: str) -> str:
            return data[key.encode()].decode()

        owner_token = text("owner_token") or None
        return SidecarEntry(
            sidecar_id=sidecar_id,
            filename=text("filename"),
            content_type=text("content_type"),
            payload=data[b"payload"],
            created_at=float(text("created_at")),
            expires_at=float(text("expires_at")),
            owner_token=owner_token,
            consume_on_read=text("consume_on_read") == "1",
        )

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
        if len(payload) > self._max_payload_bytes:
            raise ValueError(f"Sidecar payload exceeds Redis limit of {self._max_payload_bytes:,} bytes.")
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
        key = self._key(sidecar_id)
        pipe = self._client.pipeline()
        pipe.hset(
            key,
            mapping={
                "filename": filename,
                "content_type": content_type,
                "payload": payload,
                "created_at": repr(now),
                "expires_at": repr(entry.expires_at),
                "owner_token": owner_token or "",
                "consume_on_read": "1" if consume_on_read else "0",
            },
        )
        pipe.pexpire(key, max(1, int(ttl * 1000)))
        pipe.execute()
        return entry

    def get(self, sidecar_id: str, *, owner_token: Optional[str] = None) -> Optional[SidecarEntry]:
        data = self._client.hgetall(self._key(sidecar_id))
        if not data:
            return None
        stored_owner = data.get(b"owner_token", b"").decode()
        if not hmac.compare_digest(stored_owner, owner_token or ""):
            return None
        return self._entry_from_data(sidecar_id, data)

    def consume(self, sidecar_id: str, *, owner_token: Optional[str] = None) -> Optional[SidecarEntry]:
        raw_data = self._client.eval(self._CONSUME_SCRIPT, 1, self._key(sidecar_id), owner_token or "")
        if not raw_data:
            return None
        data = dict(zip(raw_data[::2], raw_data[1::2]))
        return self._entry_from_data(sidecar_id, data)

    def delete(self, sidecar_id: str, *, owner_token: Optional[str] = None) -> bool:
        return bool(self._client.eval(self._DELETE_SCRIPT, 1, self._key(sidecar_id), owner_token or ""))

    def stats(self) -> dict[str, int | float]:
        return {"entries": 0, "total_bytes": 0, "max_entries": 0, "default_ttl_s": self._default_ttl_s}


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

    def delete(self, sidecar_id: str, *, owner_token: Optional[str] = None) -> bool:
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

_instance: SidecarStore | RedisSidecarStore | None = None


def init_sidecar_store(*, default_ttl_s: float = 3600.0, max_entries: int = 1024, redis_url: str | None = None, max_payload_bytes: int = 33_554_432) -> SidecarStore | RedisSidecarStore:
    global _instance
    _instance = (
        RedisSidecarStore(redis_url, default_ttl_s=default_ttl_s, max_payload_bytes=max_payload_bytes)
        if redis_url
        else SidecarStore(default_ttl_s=default_ttl_s, max_entries=max_entries)
    )
    logger.info("SidecarStore initialised (ttl=%.0fs max_entries=%d)", default_ttl_s, max_entries)
    return _instance


def get_sidecar_store() -> SidecarStore | RedisSidecarStore | None:
    return _instance


def shutdown_sidecar_store() -> None:
    global _instance
    if _instance is not None:
        stats = _instance.stats()
        logger.info("SidecarStore shut down (entries=%d bytes=%d)", stats["entries"], stats["total_bytes"])
        _instance = None
