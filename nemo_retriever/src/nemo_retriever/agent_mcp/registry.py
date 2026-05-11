# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath

from nemo_retriever.agent_mcp.models import (
    AgentMcpError,
    AgentMcpErrorCode,
    CollectionRecord,
    CollectionStatus,
    IngestJobRecord,
    JobStatus,
    utc_now,
)


class CollectionRegistry:
    def __init__(self, db_path: str | Path, *, data_root: str | Path) -> None:
        self.db_path = Path(db_path)
        self.data_root = Path(data_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _validate_collection_name(self, name: str) -> None:
        posix_path = PurePosixPath(name)
        windows_path = PureWindowsPath(name)
        unsafe_parts = {".", ".."}
        if (
            not name
            or name in unsafe_parts
            or any(part in unsafe_parts for part in posix_path.parts)
            or any(part in unsafe_parts for part in windows_path.parts)
            or posix_path.parts != (name,)
            or windows_path.parts != (name,)
            or windows_path.drive
        ):
            raise AgentMcpError(
                AgentMcpErrorCode.PATH_OUTSIDE_ALLOWED_ROOT,
                f"Collection name '{name}' is not a safe path component.",
                details={"collection": name, "data_root": str(self.data_root)},
            )

    def _collection_paths(self, name: str, temporary: bool) -> tuple[Path, Path, Path]:
        self._validate_collection_name(name)
        namespace = "tmp" if temporary else "collections"
        base = self.data_root / namespace / name
        return base, base / "lancedb", base / "artifacts"

    def create_collection(
        self,
        name: str = "default",
        temporary: bool = False,
        hybrid: bool = False,
        metadata: dict | None = None,
    ) -> CollectionRecord:
        base, lancedb, artifacts = self._collection_paths(name, temporary)
        lancedb.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)

        record = CollectionRecord(
            name=name,
            root_path=str(base),
            temporary=temporary,
            vdb_uri=str(lancedb),
            artifact_root=str(artifacts),
            hybrid=hybrid,
            metadata=dict(metadata or {}),
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO collections (name, payload, updated_at) VALUES (?, ?, ?)",
                    (record.name, record.model_dump_json(), record.updated_at.isoformat()),
                )
        except sqlite3.IntegrityError as exc:
            raise AgentMcpError(
                AgentMcpErrorCode.COLLECTION_ALREADY_EXISTS,
                f"Collection '{name}' already exists.",
                details={"collection": name},
            ) from exc
        return record

    def get_collection(self, name: str = "default") -> CollectionRecord:
        self._validate_collection_name(name)
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM collections WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise AgentMcpError(
                AgentMcpErrorCode.COLLECTION_NOT_FOUND,
                f"Collection '{name}' was not found.",
                details={"collection": name},
            )
        return CollectionRecord.model_validate_json(row["payload"])

    def get_or_create_collection(self, name: str = "default") -> CollectionRecord:
        try:
            return self.get_collection(name)
        except AgentMcpError as exc:
            if exc.code is not AgentMcpErrorCode.COLLECTION_NOT_FOUND:
                raise
        return self.create_collection(name)

    def save_collection(self, record: CollectionRecord) -> CollectionRecord:
        self._validate_collection_name(record.name)
        updated = record.model_copy(update={"updated_at": utc_now()})
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE collections SET payload = ?, updated_at = ? WHERE name = ?",
                (updated.model_dump_json(), updated.updated_at.isoformat(), updated.name),
            )
        if cursor.rowcount == 0:
            raise AgentMcpError(
                AgentMcpErrorCode.COLLECTION_NOT_FOUND,
                f"Collection '{record.name}' was not found.",
                details={"collection": record.name},
            )
        return updated

    def mark_collection_queryable(self, name: str, row_count: int | None = None) -> CollectionRecord:
        record = self.get_collection(name)
        metadata = dict(record.metadata)
        if row_count is not None:
            metadata["row_count"] = row_count
        updated = record.model_copy(
            update={
                "queryable": True,
                "status": CollectionStatus.QUERYABLE,
                "metadata": metadata,
            }
        )
        return self.save_collection(updated)

    def list_collections(self) -> list[CollectionRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM collections ORDER BY name").fetchall()
        return [CollectionRecord.model_validate_json(row["payload"]) for row in rows]

    def delete_collection(self, name: str) -> CollectionRecord:
        record = self.get_collection(name)
        with self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE collection = ?", (name,))
            conn.execute("DELETE FROM collections WHERE name = ?", (name,))
        return record

    def create_job(self, collection: str, source_count: int = 0) -> IngestJobRecord:
        self.get_collection(collection)
        job = IngestJobRecord(job_id=str(uuid.uuid4()), collection=collection, source_count=source_count)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, collection, payload, updated_at) VALUES (?, ?, ?, ?)",
                (job.job_id, job.collection, job.model_dump_json(), job.updated_at.isoformat()),
            )
        return job

    def get_job(self, job_id: str) -> IngestJobRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise AgentMcpError(
                AgentMcpErrorCode.INGEST_JOB_NOT_FOUND,
                f"Ingest job '{job_id}' was not found.",
                details={"job_id": job_id},
            )
        return IngestJobRecord.model_validate_json(row["payload"])

    def update_job(self, job_id: str, **updates: object) -> IngestJobRecord:
        job = self.get_job(job_id)
        payload = job.model_dump()
        payload.update(updates)
        payload["updated_at"] = utc_now()
        updated = IngestJobRecord.model_validate(payload)
        if "collection" in updates:
            self.get_collection(updated.collection)
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET collection = ?, payload = ?, updated_at = ? WHERE job_id = ?",
                (updated.collection, updated.model_dump_json(), updated.updated_at.isoformat(), updated.job_id),
            )
        return updated

    def list_jobs(self, collection: str) -> list[IngestJobRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM jobs WHERE collection = ? ORDER BY updated_at DESC",
                (collection,),
            ).fetchall()
        return [IngestJobRecord.model_validate_json(row["payload"]) for row in rows]
