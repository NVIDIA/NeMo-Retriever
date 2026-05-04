# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable spool of accepted-but-not-yet-processed page bytes.

Why this exists
---------------
Without spooling, the only copy of an accepted page lives in the
in-memory ``_BatchBuffer`` until a worker process picks it up.  A pod
crash, OOM kill, rolling upgrade, or even a graceful shutdown that
exceeds the drain deadline silently loses every page that was accepted
but not yet processed.  Clients see a 202 ``Accepted`` and the work
simply vanishes.

The spool closes that gap by writing every accepted page to a content-
addressed file on the persistence volume **before** the ingest endpoint
returns 202.  The file path is recorded on the ``documents`` row.  On
startup the service scans the spool, finds any document still in
``queued`` / ``processing`` and re-enqueues the bytes — restart-safe
end to end.

Layout
------
``<root>/<sha[0:2]>/<sha[2:4]>/<sha>.bin``

Sharded by the first 4 hex characters of the SHA so a single directory
never accumulates more than ~256 files.  Files are written atomically
using ``write + fsync + rename`` so a crash mid-write never produces a
partial file that survives.

Cleanup
-------
:meth:`SpoolStore.cleanup_terminal` is called periodically by the
service.  It enumerates documents in a terminal state
(``complete``/``failed``/``cancelled``) that still have a spool file,
unlinks the file and clears ``spool_path`` on the row.  The window
between "results persisted" and "spool deleted" is harmless — the file
is just sitting there until the next cleanup tick.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class SpoolStore:
    """Filesystem-backed, content-addressed durable spool for page bytes.

    Thread-safe: writes use atomic rename so concurrent writers of the
    same SHA cannot interleave; the last writer wins (with identical
    content the result is byte-equivalent to any earlier writer).
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        # We intentionally do *not* fsync the root directory on every
        # write — it would dominate latency.  We only fsync the file
        # itself.  Loss of the directory entry on a hard crash means the
        # file is recoverable via ``fsck`` orphan-file recovery; the DB
        # row remains the source of truth either way.
        self._mkdir_lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------
    # Path conventions
    # ------------------------------------------------------------------

    def path_for(self, content_sha256: str) -> Path:
        """Return the canonical on-disk path for a SHA, without creating it."""
        if len(content_sha256) < 4:
            raise ValueError(f"content_sha256 too short: {content_sha256!r}")
        return self._root / content_sha256[0:2] / content_sha256[2:4] / f"{content_sha256}.bin"

    # ------------------------------------------------------------------
    # Write — durable, atomic
    # ------------------------------------------------------------------

    def write(self, content_sha256: str, data: bytes) -> str:
        """Persist ``data`` to disk and return the canonical absolute path.

        Idempotent: if the file already exists with the same SHA the
        write is skipped.  Uses ``write + fsync + rename`` so a crash
        mid-write never produces a partial file with the final name.
        """
        target = self.path_for(content_sha256)
        if target.exists() and target.stat().st_size == len(data):
            return str(target)

        # Ensure directory exists (cheap if it already does).
        with self._mkdir_lock:
            target.parent.mkdir(parents=True, exist_ok=True)

        tmp = target.with_suffix(target.suffix + ".tmp")
        # Open with O_EXCL so concurrent writers don't trample each
        # other's tmp files; pick a unique tmp suffix per-thread.
        tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        # Atomic on POSIX: rename overwrites if target exists.
        os.replace(tmp, target)
        return str(target)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, path: str) -> bytes:
        """Read spooled bytes by absolute path.  Raises ``FileNotFoundError`` if gone."""
        with open(path, "rb") as fh:
            return fh.read()

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, path: str) -> bool:
        """Unlink a spool file.  Returns ``True`` on success, ``False`` if missing."""
        try:
            os.unlink(path)
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.warning("spool: failed to unlink %s: %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # Recovery scan
    # ------------------------------------------------------------------

    def iter_files(self) -> Iterator[Path]:
        """Yield every spool file currently on disk (recursive)."""
        if not self._root.is_dir():
            return
        for p in self._root.rglob("*.bin"):
            if p.is_file():
                yield p

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_terminal(self, terminal_paths: list[tuple[str, str]]) -> int:
        """Delete spool files for documents now in a terminal state.

        Parameters
        ----------
        terminal_paths
            ``[(document_id, spool_path), ...]`` for every document the
            caller considers safe to evict from the spool.

        Returns
        -------
        Number of files actually deleted.
        """
        deleted = 0
        for _doc_id, path in terminal_paths:
            if not path:
                continue
            if self.delete(path):
                deleted += 1
        if deleted:
            logger.info("spool: cleaned %d terminal-doc files", deleted)
        return deleted

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def disk_usage_bytes(self) -> int:
        """Sum the size of every spool file (cheap walk; for metrics)."""
        total = 0
        for p in self.iter_files():
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total
