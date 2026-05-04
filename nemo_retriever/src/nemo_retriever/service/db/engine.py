# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe SQLite connection pool using ``threading.local``.

Multi-process writer safety
---------------------------
Service mode runs N worker *processes*, each opening its own connection.
WAL mode allows concurrent readers and a single writer at a time.  When
multiple workers call ``commit()`` simultaneously, they contend for the
write lock.  We defend against ``OperationalError: database is locked``
at two levels:

1. **Connection-level busy timeout** — ``sqlite3.connect(timeout=60)``
   plus ``PRAGMA busy_timeout=60000`` tells SQLite's built-in busy
   handler to spin/sleep up to 60 s before returning ``SQLITE_BUSY``.

2. **Application-level retry** — :func:`execute_with_retry` wraps
   any callable that touches the DB with exponential-backoff retries
   so that transient lock contention is retried transparently.
"""

from __future__ import annotations

import logging
import random
import sqlite3
import threading
import time
from pathlib import Path
from typing import TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_BUSY_TIMEOUT_MS = 60_000
_CONNECT_TIMEOUT_S = _BUSY_TIMEOUT_MS / 1000

_RETRY_MAX_ATTEMPTS = 8
_RETRY_BASE_DELAY = 0.1
_RETRY_MAX_DELAY = 10.0


def execute_with_retry(fn, *args, **kwargs) -> _T:  # type: ignore[type-var]
    """Call *fn* with retries on ``sqlite3.OperationalError``.

    Uses exponential backoff with jitter.  This catches the cases where
    even the 60 s busy_timeout is exhausted (e.g. a long-running
    checkpoint) or when the error surfaces from Python's connection
    management rather than from SQLite's busy handler.
    """
    last_exc: Exception | None = None
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            last_exc = exc
            delay = min(_RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY)
            delay *= 0.5 + random.random()  # jitter
            logger.warning(
                "SQLite locked (attempt %d/%d), retrying in %.2fs: %s",
                attempt + 1,
                _RETRY_MAX_ATTEMPTS,
                delay,
                exc,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    filename         TEXT NOT NULL,
    content_sha256   TEXT NOT NULL,
    total_pages      INTEGER NOT NULL,
    pages_submitted  INTEGER NOT NULL DEFAULT 0,
    pages_completed  INTEGER NOT NULL DEFAULT 0,
    processing_status TEXT NOT NULL DEFAULT 'queued',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    job_id          TEXT REFERENCES jobs(id),
    filename        TEXT NOT NULL,
    content_type    TEXT NOT NULL DEFAULT 'application/octet-stream',
    content_sha256  TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    page_number     INTEGER,
    total_pages     INTEGER,
    pages_received  INTEGER NOT NULL DEFAULT 0,
    processing_status TEXT NOT NULL DEFAULT 'queued',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    -- Path to the on-disk spooled bytes for this document.  Set at ingest
    -- time (after the multipart upload is fsync'd) so that a pod restart
    -- between accept and processing can re-enqueue the work without data
    -- loss.  Cleared by the spool cleaner once the document reaches a
    -- terminal state and the file has been unlinked.
    spool_path      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(content_sha256);
CREATE INDEX IF NOT EXISTS idx_documents_job ON documents(job_id);

CREATE TABLE IF NOT EXISTS page_results (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    page_number INTEGER NOT NULL,
    content_json TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_page_results_doc ON page_results(document_id);

CREATE TABLE IF NOT EXISTS processing_metrics (
    id                   TEXT PRIMARY KEY,
    document_id          TEXT NOT NULL REFERENCES documents(id),
    model_name           TEXT NOT NULL,
    invocation_count     INTEGER NOT NULL DEFAULT 0,
    pages_processed      INTEGER NOT NULL DEFAULT 0,
    detections_count     INTEGER NOT NULL DEFAULT 0,
    counts_by_label_json TEXT NOT NULL DEFAULT '{}',
    duration_ms          REAL NOT NULL DEFAULT 0.0,
    created_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_doc ON processing_metrics(document_id);

CREATE TABLE IF NOT EXISTS page_processing_log (
    id                     TEXT PRIMARY KEY,
    document_id            TEXT NOT NULL REFERENCES documents(id),
    job_id                 TEXT REFERENCES jobs(id),
    source_file            TEXT NOT NULL,
    page_number            INTEGER NOT NULL,
    status                 TEXT NOT NULL DEFAULT 'complete',
    error_message          TEXT,
    failure_type           TEXT,
    detection_count        INTEGER NOT NULL DEFAULT 0,
    processing_duration_ms REAL NOT NULL DEFAULT 0.0,
    started_at             TEXT NOT NULL,
    completed_at           TEXT NOT NULL,
    created_at             TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ppl_source ON page_processing_log(source_file);
CREATE INDEX IF NOT EXISTS idx_ppl_job ON page_processing_log(job_id);

CREATE TABLE IF NOT EXISTS event_log (
    id           TEXT PRIMARY KEY,
    timestamp    TEXT NOT NULL,
    job_id       TEXT REFERENCES jobs(id),
    document_id  TEXT REFERENCES documents(id),
    source_file  TEXT NOT NULL DEFAULT '',
    page_number  INTEGER,
    category     TEXT NOT NULL,
    severity     TEXT NOT NULL DEFAULT 'error',
    outcome      TEXT NOT NULL DEFAULT 'failed',
    stage        TEXT NOT NULL DEFAULT '',
    summary      TEXT NOT NULL DEFAULT '',
    detail       TEXT NOT NULL DEFAULT '',
    stack_trace  TEXT NOT NULL DEFAULT '',
    endpoint     TEXT NOT NULL DEFAULT '',
    request_id   TEXT NOT NULL DEFAULT '',
    extra_json   TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_log_job ON event_log(job_id);
CREATE INDEX IF NOT EXISTS idx_event_log_doc ON event_log(document_id);
CREATE INDEX IF NOT EXISTS idx_event_log_cat ON event_log(category);
CREATE INDEX IF NOT EXISTS idx_event_log_ts  ON event_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_event_log_req ON event_log(request_id);
"""


def _safe_add_column(conn, table: str, column: str, decl: str) -> None:
    """``ALTER TABLE ... ADD COLUMN`` if the column doesn't already exist.

    SQLite has no ``ADD COLUMN IF NOT EXISTS``, so we introspect first.
    Used to add columns to existing databases without dropping data.
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    existing = {row["name"] for row in cur.fetchall()}
    if column in existing:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    conn.commit()


class DatabaseEngine:
    """Manages thread-local SQLite connections to a single database file."""

    def __init__(self, db_path: str) -> None:
        self._db_path = str(Path(db_path).resolve())
        self._local = threading.local()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path,
            timeout=_CONNECT_TIMEOUT_S,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @property
    def connection(self) -> sqlite3.Connection:
        """Return (or create) the thread-local connection."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def initialize(self) -> None:
        """Create tables and indexes if they do not exist.

        Also runs additive schema migrations for columns added in newer
        service versions so an upgrade does not require dropping the DB.
        """
        conn = self.connection
        conn.executescript(_DDL)
        conn.commit()

        # Additive migrations (safe to run on every startup; no-op when
        # the column already exists).
        _safe_add_column(conn, "page_processing_log", "failure_type", "TEXT")
        _safe_add_column(conn, "documents", "spool_path", "TEXT")

        logger.info("SQLite database initialized at %s", self._db_path)

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
