# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-writer SQLite state for the gateway job tracker and work broker."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1


class DurableStateStore:
    """Synchronous durable state used behind the gateway's existing locks."""

    def __init__(self, spool_directory: Path) -> None:
        self.path = spool_directory / "gateway-state.sqlite3"
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS work_records (
                work_id TEXT PRIMARY KEY, pool TEXT NOT NULL,
                fifo_sequence INTEGER NOT NULL UNIQUE, data TEXT NOT NULL
            );
            """
        )
        row = connection.execute("SELECT version FROM schema_info").fetchone()
        if row is None:
            connection.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
        elif row[0] != SCHEMA_VERSION:
            connection.close()
            raise RuntimeError(f"Unsupported gateway state schema {row[0]}; expected {SCHEMA_VERSION}")
        connection.commit()
        self._connection = connection

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Gateway durable state is not open")
        return self._connection

    def close(self) -> None:
        with self._lock:
            if self._connection is None:
                return
            self._connection.commit()
            self._connection.execute("PRAGMA wal_checkpoint(FULL)")
            self._connection.close()
            self._connection = None

    def persist_tracker_records(
        self,
        *,
        jobs: Iterable[Any] = (),
        documents: Iterable[Any] = (),
        delete_document_ids: Iterable[str] = (),
    ) -> None:
        """Upsert only records touched by one tracker transition."""
        with self._lock, self.connection:
            self.connection.executemany(
                "INSERT INTO jobs(job_id, data) VALUES (?, ?) " "ON CONFLICT(job_id) DO UPDATE SET data=excluded.data",
                ((job.job_id, job.model_dump_json()) for job in jobs),
            )
            self.connection.executemany(
                "INSERT INTO documents(document_id, job_id, data) VALUES (?, ?, ?) "
                "ON CONFLICT(document_id) DO UPDATE SET "
                "job_id=excluded.job_id, data=excluded.data",
                ((doc.id, doc.job_id, doc.model_dump_json()) for doc in documents),
            )
            self.connection.executemany(
                "DELETE FROM documents WHERE document_id = ?",
                ((document_id,) for document_id in delete_document_ids),
            )

    def delete_job(self, job_id: str) -> None:
        with self._lock, self.connection:
            self.connection.execute("DELETE FROM documents WHERE job_id = ?", (job_id,))
            self.connection.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

    def load_tracker(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self._lock:
            jobs = [json.loads(row[0]) for row in self.connection.execute("SELECT data FROM jobs ORDER BY rowid")]
            documents = [
                json.loads(row[0]) for row in self.connection.execute("SELECT data FROM documents ORDER BY rowid")
            ]
        return jobs, documents

    def save_work(self, work_id: str, pool: str, fifo_sequence: int, data: dict[str, Any]) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO work_records(work_id, pool, fifo_sequence, data) VALUES (?, ?, ?, ?)",
                (work_id, pool, fifo_sequence, json.dumps(data, separators=(",", ":"), sort_keys=True)),
            )

    def delete_work(self, work_id: str) -> None:
        with self._lock, self.connection:
            self.connection.execute("DELETE FROM work_records WHERE work_id = ?", (work_id,))

    def load_work(self) -> list[tuple[int, dict[str, Any]]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT fifo_sequence, data FROM work_records ORDER BY fifo_sequence"
            ).fetchall()
        return [(int(sequence), json.loads(data)) for sequence, data in rows]
