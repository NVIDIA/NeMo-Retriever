# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone VectorDB microservice backed by LanceDB.

Provides three endpoints:

- ``POST /internal/vectordb/write`` -- append embedding rows from ingest workers
- ``POST /v1/query``               -- embed query text and search the index
- ``GET  /v1/health``              -- liveness probe

Run with a remote NIM embed endpoint::

    python -m nemo_retriever.service.vectordb_app \\
        --lancedb-uri /data/vectordb \\
        --embed-endpoint http://nemo-retriever-nim-embed-0...:8000/v1/embeddings \\
        --port 7671

Run with in-pod Hugging Face query embedding (requires ``[local]`` extras + GPU)::

    python -m nemo_retriever.service.vectordb_app \\
        --lancedb-uri /data/vectordb \\
        --local-embed \\
        --local-embed-backend hf \\
        --embed-model nvidia/llama-nemotron-embed-vl-1b-v2 \\
        --port 7671
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import threading
import hmac
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Union

import lancedb
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel

from nemo_retriever.common.schemas.collections import (
    CollectionDeleteResult,
    CollectionCreateRequest,
    CollectionInfo,
    CollectionPage,
    CollectionUpdateRequest,
    DocumentId,
    DocumentDeleteResult,
    DocumentInfo,
    DocumentPage,
)

from nemo_retriever.query.evidence import build_evidence_result
from nemo_retriever.service.query_schema import (
    EvidenceQueryResponse,
    EvidenceResult,
    QueryRequest,
    QueryResponse,
    QueryResult,
)

# /v1/query is dense vector search only; report that honestly in coverage.
_QUERY_STRATEGIES = ["dense"]

logger = logging.getLogger(__name__)

# ── Request / response models ────────────────────────────────────────


class WriteRequest(BaseModel):
    rows: list[dict[str, Any]]
    scope: str | None = None
    collection_name: str | None = None
    document_id: str | None = None
    job_id: str | None = None
    filename: str | None = None
    content_sha256: str | None = None
    operation: str = "append"
    artifact_prefix: str | None = None


class WriteResponse(BaseModel):
    written: int
    total_rows: int


_COLLECTIONS_TABLE = "_nrl_collections"
_DOCUMENTS_TABLE = "_nrl_documents"
_CATALOG_SCHEMA_VERSION = 2
_CATALOG_SCAN_LIMIT = 100_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _physical_table(scope: str, collection_name: str) -> str:
    digest = hashlib.sha256(f"{scope}\0{collection_name}".encode()).hexdigest()
    return f"nrl_{digest[:40]}"


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _cursor(resource: str, scope: str, collection: str | None, last: list[str]) -> str:
    payload = {
        "v": 1,
        "resource": resource,
        "scope": scope,
        "collection": collection,
        "last": last,
    }
    return (
        base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )


def _decode_cursor(
    token: str | None,
    *,
    resource: str,
    scope: str,
    collection: str | None,
) -> list[str] | None:
    if not token:
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode())
    except Exception as exc:
        raise HTTPException(422, "Invalid continuation token") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("v") != 1
        or payload.get("resource") != resource
        or payload.get("scope") != scope
        or payload.get("collection") != collection
        or not isinstance(payload.get("last"), list)
    ):
        raise HTTPException(422, "Continuation token does not match this resource context")
    return [str(value) for value in payload["last"]]


# ── Embedding helpers ────────────────────────────────────────────────


def _tensor_to_embedding_rows(tensor: Any) -> list[list[float]]:
    """Convert a local embedder tensor output to JSON-serializable vectors."""
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "tolist"):
        rows = tensor.tolist()
        if rows and isinstance(rows[0], (int, float)):
            return [rows]
        return rows
    return list(tensor)


def _embed_queries_remote(
    texts: list[str],
    *,
    embed_model: str,
    embed_endpoint: str,
    embed_api_key: str,
    embed_model_provider_prefix: str | None = None,
) -> list[list[float]]:
    from nemo_retriever.models.nim.util import infer_microservice

    return infer_microservice(
        texts,
        model_name=embed_model,
        model_provider_prefix=embed_model_provider_prefix,
        embedding_endpoint=embed_endpoint,
        nvidia_api_key=embed_api_key or None,
        input_type="query",
        grpc=False,
    )


# ── VectorDB state ───────────────────────────────────────────────────


class VectorDBState:
    """Thread-safe wrapper around a LanceDB connection."""

    def __init__(
        self,
        lancedb_uri: str,
        table_name: str,
        embed_endpoint: str,
        embed_model: str,
        embed_api_key: str,
        *,
        embed_model_provider_prefix: str | None = None,
        local_embed: bool = False,
        local_embed_backend: str = "hf",
        hf_cache_dir: str | None = None,
        device: str | None = None,
        gpu_memory_utilization: float = 0.45,
        collection_artifact_root: str | None = None,
        artifact_storage_options_file: str | None = None,
        expiration_cleanup_enabled: bool = True,
    ) -> None:
        self.lancedb_uri = lancedb_uri
        self.table_name = table_name
        self.embed_endpoint = embed_endpoint
        self.embed_model = embed_model
        self.embed_model_provider_prefix = embed_model_provider_prefix
        self.embed_api_key = embed_api_key
        self.local_embed = local_embed
        self.local_embed_backend = local_embed_backend
        self.hf_cache_dir = hf_cache_dir
        self.device = device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.collection_artifact_root = (collection_artifact_root or "").rstrip("/") or None
        self.artifact_storage_options = self._load_storage_options(artifact_storage_options_file)
        self.expiration_cleanup_enabled = expiration_cleanup_enabled
        self.reconciliation_successes = 0
        self.reconciliation_failures = 0
        self._write_lock = threading.Lock()
        self._embed_lock = threading.Lock()
        self._local_embedder: Any | None = None
        self._db = lancedb.connect(uri=lancedb_uri)
        self._collection_tables: dict[tuple[str, str], str] = {}
        self._opened_tables: dict[str, Any] = {}
        self._table_exists = False
        try:
            self._db.open_table(table_name)
            self._table_exists = True
            logger.info("Opened existing LanceDB table '%s' at %s", table_name, lancedb_uri)
        except Exception:
            logger.info("LanceDB table '%s' does not exist yet at %s", table_name, lancedb_uri)
        self._ensure_catalogs()

    @staticmethod
    def _load_storage_options(path: str | None) -> dict[str, Any]:
        if not path:
            return {}
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Unable to load artifact storage-options secret file: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("artifact storage-options secret file must contain a JSON object")
        return payload

    def _ensure_catalogs(self) -> None:
        import pyarrow as pa

        collection_schema = pa.schema(
            [
                pa.field("scope", pa.string()),
                pa.field("name", pa.string()),
                pa.field("physical_table", pa.string()),
                pa.field("status", pa.string()),
                pa.field("description", pa.string()),
                pa.field("metadata_json", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("expires_at", pa.string()),
                pa.field("deletion_phase", pa.string()),
                pa.field("retry_count", pa.int64()),
                pa.field("next_retry_at", pa.string()),
                pa.field("last_error", pa.string()),
                pa.field("delete_started_at", pa.string()),
            ]
        )
        document_schema = pa.schema(
            [
                pa.field("scope", pa.string()),
                pa.field("collection_name", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("job_id", pa.string()),
                pa.field("filename", pa.string()),
                pa.field("content_sha256", pa.string()),
                pa.field("document_version", pa.string()),
                pa.field("status", pa.string()),
                pa.field("chunk_count", pa.int64()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("error", pa.string()),
                pa.field("current_document_version", pa.string()),
                pa.field("pending_document_version", pa.string()),
                pa.field("current_artifact_prefix", pa.string()),
                pa.field("pending_artifact_prefixes_json", pa.string()),
                pa.field("recovery_state", pa.string()),
            ]
        )
        for name, schema in (
            (_COLLECTIONS_TABLE, collection_schema),
            (_DOCUMENTS_TABLE, document_schema),
        ):
            try:
                table = self._db.open_table(name)
            except Exception:
                table = self._db.create_table(name, schema=schema, mode="create")
            existing = {field.name: field for field in table.schema}
            for field in schema:
                if field.name in existing and existing[field.name].type != field.type:
                    raise RuntimeError(
                        f"Incompatible {name} catalog column {field.name!r}: "
                        f"expected {field.type}, found {existing[field.name].type}"
                    )
            missing = [field for field in schema if field.name not in existing]
            if missing:
                table.add_columns(missing)
                logger.info("Migrated %s catalog with %d additive column(s)", name, len(missing))
            index_columns = (
                ("scope", "name", "status", "expires_at")
                if name == _COLLECTIONS_TABLE
                else ("scope", "collection_name", "document_id", "status")
            )
            for column in index_columns:
                table.create_scalar_index(column, replace=True)

    def _rows(
        self,
        table_name: str,
        where: str | None = None,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = self._db.open_table(table_name).search()
        if where:
            query = query.where(where)
        if columns:
            query = query.select(columns)
        return query.limit(_CATALOG_SCAN_LIMIT).to_list()

    def _has_table(self, table_name: str) -> bool:
        return table_name in self._db.list_tables().tables

    def _open_table(self, table_name: str) -> Any:
        table = self._opened_tables.get(table_name)
        if table is None:
            table = self._db.open_table(table_name)
            self._opened_tables[table_name] = table
        return table

    def _collection_row(self, scope: str, name: str, *, active: bool = False) -> dict[str, Any] | None:
        where = f"scope = {_quoted(scope)} AND name = {_quoted(name)}"
        rows = self._rows(_COLLECTIONS_TABLE, where)
        row = rows[0] if rows else None
        if row and active and row["status"] != "active":
            raise HTTPException(409, f"Collection {name!r} is {row['status']}")
        if row and active and row.get("expires_at"):
            expires = datetime.fromisoformat(str(row["expires_at"]))
            if expires <= datetime.now(timezone.utc):
                raise HTTPException(409, f"Collection {name!r} is expired")
        return row

    @staticmethod
    def _collection_info(row: dict[str, Any]) -> CollectionInfo:
        return CollectionInfo(
            name=row["name"],
            scope=row["scope"],
            status=row["status"],
            description=row.get("description") or None,
            metadata=json.loads(row.get("metadata_json") or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row.get("expires_at") or None,
        )

    @staticmethod
    def _document_info(row: dict[str, Any]) -> DocumentInfo:
        return DocumentInfo(**{k: row.get(k) for k in DocumentInfo.model_fields})

    def create_collection(self, scope: str, req: CollectionCreateRequest) -> CollectionInfo:
        with self._write_lock:
            if self._collection_row(scope, req.name):
                raise HTTPException(409, f"Collection {req.name!r} already exists")
            now = _now()
            row = {
                "scope": scope,
                "name": req.name,
                "physical_table": _physical_table(scope, req.name),
                "status": "active",
                "description": req.description or "",
                "metadata_json": json.dumps(req.metadata, sort_keys=True),
                "created_at": now,
                "updated_at": now,
                "expires_at": req.expires_at or "",
                "deletion_phase": "",
                "retry_count": 0,
                "next_retry_at": "",
                "last_error": "",
                "delete_started_at": "",
            }
            self._db.open_table(_COLLECTIONS_TABLE).add([row])
            self._collection_tables[(scope, req.name)] = row["physical_table"]
            return self._collection_info(row)

    def get_collection(self, scope: str, name: str) -> CollectionInfo:
        row = self._collection_row(scope, name)
        if not row:
            raise HTTPException(404, "Collection not found")
        return self._collection_info(row)

    def list_collections(self, scope: str, limit: int, token: str | None) -> CollectionPage:
        rows = sorted(
            self._rows(_COLLECTIONS_TABLE, f"scope = {_quoted(scope)}"),
            key=lambda r: r["name"],
        )
        last = _decode_cursor(token, resource="collections", scope=scope, collection=None)
        if last:
            rows = [row for row in rows if row["name"] > last[0]]
        page = rows[:limit]
        next_token = _cursor("collections", scope, None, [page[-1]["name"]]) if len(rows) > limit and page else None
        return CollectionPage(items=[self._collection_info(r) for r in page], next_token=next_token)

    def update_collection(self, scope: str, name: str, req: CollectionUpdateRequest) -> CollectionInfo:
        with self._write_lock:
            old = self._collection_row(scope, name, active=True)
            if not old:
                raise HTTPException(404, "Collection not found")
            update = req.model_dump(exclude_unset=True)
            old["description"] = update.get("description", old["description"]) or ""
            if "metadata" in update:
                old["metadata_json"] = json.dumps(update["metadata"] or {}, sort_keys=True)
            old["expires_at"] = update.get("expires_at", old["expires_at"]) or ""
            old["updated_at"] = _now()
            (
                self._db.open_table(_COLLECTIONS_TABLE)
                .merge_insert(["scope", "name"])
                .when_matched_update_all()
                .execute([old])
            )
            return self._collection_info(old)

    def _persist_collection_row(self, row: dict[str, Any]) -> None:
        (
            self._db.open_table(_COLLECTIONS_TABLE)
            .merge_insert(["scope", "name"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([row])
        )

    def _collection_artifact_prefix(self, scope: str, name: str) -> str | None:
        if not self.collection_artifact_root:
            return None
        from urllib.parse import quote

        return "/".join((self.collection_artifact_root, quote(scope, safe=""), quote(name, safe="")))

    def _delete_owned_artifacts(self, prefix: str | None) -> None:
        if not prefix or not self.collection_artifact_root:
            return
        controlled = self.collection_artifact_root.rstrip("/")
        if prefix != controlled and not prefix.startswith(controlled + "/"):
            logger.warning("Refused cleanup for artifact prefix outside the configured root")
            return
        from fsspec.core import url_to_fs

        fs, path = url_to_fs(prefix, **self.artifact_storage_options)
        if fs.exists(path):
            fs.rm(path, recursive=True)

    @staticmethod
    def _retry_at(retry_count: int) -> str:
        delay = min(3600, 2 ** min(max(retry_count, 1), 12))
        return (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()

    def _schedule_collection_retry(self, row: dict[str, Any], phase: str, exc: Exception) -> None:
        retries = int(row.get("retry_count") or 0) + 1
        row.update(
            {
                "status": "deleting",
                "deletion_phase": phase,
                "retry_count": retries,
                "next_retry_at": self._retry_at(retries),
                "last_error": str(exc)[:2000],
                "updated_at": _now(),
            }
        )
        self._persist_collection_row(row)

    def _cleanup_collection_locked(self, row: dict[str, Any]) -> bool:
        phase = str(row.get("deletion_phase") or "drop_table")
        try:
            if phase == "drop_table":
                self._db.drop_table(row["physical_table"], ignore_missing=True)
                self._opened_tables.pop(row["physical_table"], None)
                row["deletion_phase"] = phase = "delete_artifacts"
                row["updated_at"] = _now()
                self._persist_collection_row(row)
            if phase == "delete_artifacts":
                self._delete_owned_artifacts(self._collection_artifact_prefix(row["scope"], row["name"]))
                row["deletion_phase"] = phase = "delete_catalog"
                row["updated_at"] = _now()
                self._persist_collection_row(row)
            if phase == "delete_catalog":
                self._db.open_table(_DOCUMENTS_TABLE).delete(
                    f"scope = {_quoted(row['scope'])} AND collection_name = {_quoted(row['name'])}"
                )
                self._db.open_table(_COLLECTIONS_TABLE).delete(
                    f"scope = {_quoted(row['scope'])} AND name = {_quoted(row['name'])}"
                )
                self._collection_tables.pop((row["scope"], row["name"]), None)
            return True
        except Exception as exc:
            logger.exception("Collection cleanup paused at phase %s", phase)
            self._schedule_collection_retry(row, phase, exc)
            return False

    def delete_collection(self, scope: str, name: str, if_exists: bool) -> CollectionDeleteResult:
        with self._write_lock:
            row = self._collection_row(scope, name)
            if not row:
                if if_exists:
                    return CollectionDeleteResult(
                        name=name,
                        scope=scope,
                        existed=False,
                        deleted=False,
                        status="deleted",
                        cleanup_pending=False,
                    )
                raise HTTPException(404, "Collection not found")
            if row["status"] != "deleting":
                now = _now()
                row.update(
                    {
                        "status": "deleting",
                        "deletion_phase": "drop_table",
                        "retry_count": 0,
                        "next_retry_at": "",
                        "last_error": "",
                        "delete_started_at": now,
                        "updated_at": now,
                    }
                )
                self._persist_collection_row(row)
            deleted = self._cleanup_collection_locked(row)
            return CollectionDeleteResult(
                name=name,
                scope=scope,
                existed=True,
                deleted=deleted,
                status="deleted" if deleted else "deleting",
                cleanup_pending=not deleted,
            )

    def _resolved_table(self, scope: str, name: str) -> str:
        row = self._collection_row(scope, name, active=True)
        if not row:
            raise HTTPException(404, "Collection not found")
        table = row["physical_table"]
        self._collection_tables[(scope, name)] = table
        return table

    @property
    def embed_mode(self) -> str:
        if self.embed_endpoint:
            return "remote"
        if self.local_embed:
            return "local"
        return "none"

    @property
    def table_exists(self) -> bool:
        return self._table_exists

    def write_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        scope: str | None = None,
        collection_name: str | None = None,
        document_id: str | None = None,
        job_id: str | None = None,
        filename: str | None = None,
        content_sha256: str | None = None,
        operation: str = "append",
        artifact_prefix: str | None = None,
    ) -> int:
        """Append rows to the LanceDB table (creates table on first write)."""
        if not rows:
            return 0

        from nemo_retriever.common.vdb.lancedb_schema import (
            create_or_append_lancedb_table,
            infer_vector_dim,
            lancedb_schema,
        )

        table_name = self.table_name
        collection_mode = collection_name is not None
        if collection_mode:
            table_name = self._resolved_table(scope or "default", collection_name)

        with self._write_lock:
            existing: list[dict[str, Any]] = []
            version = str(rows[0].get("document_version") or content_sha256 or "1")
            if collection_mode and document_id:
                existing = self._rows(
                    _DOCUMENTS_TABLE,
                    f"scope = {_quoted(scope or 'default')} AND collection_name = {_quoted(collection_name)} "
                    f"AND document_id = {_quoted(document_id)}",
                )
                if operation == "replace" and existing:
                    marker = dict(existing[0])
                    marker.update(
                        {
                            "status": "replacing",
                            "pending_document_version": version,
                            "pending_artifact_prefixes_json": json.dumps(
                                {
                                    "old": (
                                        [marker.get("current_artifact_prefix")]
                                        if marker.get("current_artifact_prefix")
                                        else []
                                    ),
                                    "new": artifact_prefix or "",
                                }
                            ),
                            "recovery_state": "replacing",
                            "updated_at": _now(),
                            "error": "",
                        }
                    )
                    (
                        self._db.open_table(_DOCUMENTS_TABLE)
                        .merge_insert(["scope", "collection_name", "document_id"])
                        .when_matched_update_all()
                        .execute([marker])
                    )
            table_exists = self._table_exists if not collection_mode else self._has_table(table_name)
            if not table_exists:
                dim = infer_vector_dim(rows)
                if dim == 0:
                    logger.warning("Cannot infer vector dimension from rows; skipping write")
                    return 0
                schema = lancedb_schema(vector_dim=dim, collection_managed=collection_mode)
                table = create_or_append_lancedb_table(
                    self._db,
                    table_name,
                    rows,
                    schema,
                    overwrite=True,
                )
                self._opened_tables[table_name] = table
                if not collection_mode:
                    self._table_exists = True
                logger.info(
                    "Created LanceDB table '%s' with %d rows (dim=%d)",
                    table_name,
                    len(rows),
                    dim,
                )
            else:
                table = self._open_table(table_name)
                if operation == "replace":
                    if not document_id:
                        raise HTTPException(422, "replace requires document_id")
                    predicate = f"document_id = {_quoted(document_id)}"
                    (
                        table.merge_insert("chunk_id")
                        .when_matched_update_all()
                        .when_not_matched_insert_all()
                        .when_not_matched_by_source_delete(predicate)
                        .execute(rows)
                    )
                else:
                    table.add(rows)
                logger.info(
                    "Wrote %d rows to table '%s' operation=%s",
                    len(rows),
                    table_name,
                    operation,
                )

            if collection_mode and document_id:
                now = _now()
                created_at = existing[0]["created_at"] if existing else now
                doc = {
                    "scope": scope or "default",
                    "collection_name": collection_name,
                    "document_id": document_id,
                    "job_id": job_id or "",
                    "filename": filename or "",
                    "content_sha256": content_sha256 or "",
                    "document_version": version,
                    "status": "completed",
                    "chunk_count": len(rows),
                    "created_at": created_at,
                    "updated_at": now,
                    "error": "",
                    "current_document_version": version,
                    "pending_document_version": "",
                    "current_artifact_prefix": artifact_prefix or "",
                    "pending_artifact_prefixes_json": (
                        existing[0].get("pending_artifact_prefixes_json") or "{}"
                        if operation == "replace" and existing
                        else "{}"
                    ),
                    "recovery_state": ("cleanup" if operation == "replace" and existing else ""),
                }
                self._persist_document_row(doc)
                if doc["recovery_state"] == "cleanup":
                    self._reconcile_document_row_locked(doc, table_name)

        return len(rows)

    def total_rows(self, *, scope: str | None = None, collection_name: str | None = None) -> int:
        table_name = self.table_name
        exists = self._table_exists
        if collection_name:
            table_name = self._resolved_table(scope or "default", collection_name)
            exists = self._has_table(table_name)
        if not exists:
            return 0
        try:
            table = self._open_table(table_name)
            return table.count_rows()
        except Exception:
            return 0

    def search(
        self,
        vectors: list[list[float]],
        top_k: int,
        *,
        scope: str | None = None,
        collection_name: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Search the LanceDB table with precomputed query vectors."""
        table_name = self.table_name
        exists = self._table_exists
        if collection_name:
            table_name = self._resolved_table(scope or "default", collection_name)
            exists = self._has_table(table_name)
        if not exists:
            return [[] for _ in vectors]

        from nemo_retriever.common.vdb.records import normalize_retrieval_results

        table = self._open_table(table_name)
        raw_results = []
        for vector in vectors:
            results = table.search(vector).limit(top_k).to_list()
            raw_results.append(results)

        return normalize_retrieval_results(raw_results)

    def list_documents(self, scope: str, collection: str, limit: int, token: str | None) -> DocumentPage:
        self._resolved_table(scope, collection)
        where = f"scope = {_quoted(scope)} AND collection_name = {_quoted(collection)}"
        rows = sorted(
            self._rows(_DOCUMENTS_TABLE, where),
            key=lambda r: (r["created_at"], r["document_id"]),
        )
        last = _decode_cursor(token, resource="documents", scope=scope, collection=collection)
        if last:
            if len(last) != 2:
                raise HTTPException(422, "Invalid document continuation token")
            rows = [row for row in rows if (row["created_at"], row["document_id"]) > (last[0], last[1])]
        page = rows[:limit]
        return DocumentPage(
            items=[self._document_info(r) for r in page],
            next_token=(
                _cursor(
                    "documents",
                    scope,
                    collection,
                    [page[-1]["created_at"], page[-1]["document_id"]],
                )
                if len(rows) > limit and page
                else None
            ),
        )

    def get_document(self, scope: str, collection: str, document_id: str) -> DocumentInfo:
        self._resolved_table(scope, collection)
        where = (
            f"scope = {_quoted(scope)} AND collection_name = {_quoted(collection)} "
            f"AND document_id = {_quoted(document_id)}"
        )
        rows = self._rows(_DOCUMENTS_TABLE, where)
        if not rows:
            raise HTTPException(404, "Document not found")
        return self._document_info(rows[0])

    def _persist_document_row(self, row: dict[str, Any]) -> None:
        (
            self._db.open_table(_DOCUMENTS_TABLE)
            .merge_insert(["scope", "collection_name", "document_id"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([row])
        )

    @staticmethod
    def _artifact_recovery_payload(row: dict[str, Any]) -> dict[str, Any]:
        try:
            payload = json.loads(row.get("pending_artifact_prefixes_json") or "{}")
        except (TypeError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _reconcile_document_row_locked(self, row: dict[str, Any], table_name: str) -> bool:
        state = str(row.get("recovery_state") or "")
        try:
            if state == "replacing":
                pending = str(row.get("pending_document_version") or "")
                versions: set[str] = set()
                if self._has_table(table_name):
                    chunks = self._rows(
                        table_name,
                        f"document_id = {_quoted(row['document_id'])}",
                        ["document_version"],
                    )
                    versions = {str(chunk.get("document_version") or "") for chunk in chunks}
                artifacts = self._artifact_recovery_payload(row)
                if pending and pending in versions:
                    row.update(
                        {
                            "document_version": pending,
                            "current_document_version": pending,
                            "content_sha256": pending,
                            "current_artifact_prefix": str(artifacts.get("new") or ""),
                            "pending_document_version": "",
                            "status": "completed",
                            "recovery_state": "cleanup",
                            "updated_at": _now(),
                            "error": "",
                        }
                    )
                    self._persist_document_row(row)
                    state = "cleanup"
                else:
                    self._delete_owned_artifacts(str(artifacts.get("new") or "") or None)
                    row.update(
                        {
                            "pending_document_version": "",
                            "pending_artifact_prefixes_json": "{}",
                            "status": "completed",
                            "recovery_state": "",
                            "updated_at": _now(),
                            "error": "",
                        }
                    )
                    self._persist_document_row(row)
                    return True
            if state == "cleanup":
                artifacts = self._artifact_recovery_payload(row)
                for prefix in artifacts.get("old", []):
                    if prefix and prefix != row.get("current_artifact_prefix"):
                        self._delete_owned_artifacts(str(prefix))
                row.update(
                    {
                        "pending_artifact_prefixes_json": "{}",
                        "recovery_state": "",
                        "status": "completed",
                        "updated_at": _now(),
                        "error": "",
                    }
                )
                self._persist_document_row(row)
                return True
            if state == "deleting_chunks":
                if self._has_table(table_name):
                    self._open_table(table_name).delete(f"document_id = {_quoted(row['document_id'])}")
                row.update({"recovery_state": "deleting_artifacts", "updated_at": _now()})
                self._persist_document_row(row)
                state = "deleting_artifacts"
            if state == "deleting_artifacts":
                self._delete_owned_artifacts(row.get("current_artifact_prefix") or None)
                artifacts = self._artifact_recovery_payload(row)
                self._delete_owned_artifacts(str(artifacts.get("new") or "") or None)
                for prefix in artifacts.get("old", []):
                    self._delete_owned_artifacts(str(prefix) or None)
                self._db.open_table(_DOCUMENTS_TABLE).delete(
                    f"scope = {_quoted(row['scope'])} AND collection_name = {_quoted(row['collection_name'])} "
                    f"AND document_id = {_quoted(row['document_id'])}"
                )
                return True
            return state == ""
        except Exception as exc:
            row.update({"error": str(exc)[:2000], "updated_at": _now()})
            self._persist_document_row(row)
            logger.exception("Document reconciliation paused in state %s", state)
            return False

    def delete_document(self, scope: str, collection: str, document_id: str, if_exists: bool) -> DocumentDeleteResult:
        table_name = self._resolved_table(scope, collection)
        try:
            self.get_document(scope, collection, document_id)
        except HTTPException:
            if if_exists:
                return DocumentDeleteResult(
                    document_id=document_id,
                    collection_name=collection,
                    scope=scope,
                    existed=False,
                    deleted=False,
                    status="deleted",
                    cleanup_pending=False,
                )
            raise
        with self._write_lock:
            rows = self._rows(
                _DOCUMENTS_TABLE,
                f"scope = {_quoted(scope)} AND collection_name = {_quoted(collection)} "
                f"AND document_id = {_quoted(document_id)}",
            )
            row = rows[0]
            if row.get("recovery_state") not in (
                "deleting_chunks",
                "deleting_artifacts",
            ):
                row.update(
                    {
                        "status": "deleting",
                        "recovery_state": "deleting_chunks",
                        "updated_at": _now(),
                        "error": "",
                    }
                )
                self._persist_document_row(row)
            deleted = self._reconcile_document_row_locked(row, table_name)
        return DocumentDeleteResult(
            document_id=document_id,
            collection_name=collection,
            scope=scope,
            existed=True,
            deleted=deleted,
            status="deleted" if deleted else "deleting",
            cleanup_pending=not deleted,
        )

    def reconcile(self) -> dict[str, int]:
        """Resume recoverable lifecycle work and expire due collections."""
        successes = 0
        failures = 0
        now = datetime.now(timezone.utc)
        with self._write_lock:
            documents = self._rows(_DOCUMENTS_TABLE)
            for row in documents:
                if not row.get("recovery_state"):
                    continue
                collection = self._collection_row(row["scope"], row["collection_name"])
                if not collection:
                    continue
                if self._reconcile_document_row_locked(row, collection["physical_table"]):
                    successes += 1
                else:
                    failures += 1

            collections = self._rows(_COLLECTIONS_TABLE)
            for row in collections:
                if (
                    self.expiration_cleanup_enabled
                    and row.get("status") == "active"
                    and row.get("expires_at")
                    and datetime.fromisoformat(str(row["expires_at"])) <= now
                ):
                    started = _now()
                    row.update(
                        {
                            "status": "deleting",
                            "deletion_phase": "drop_table",
                            "retry_count": 0,
                            "next_retry_at": "",
                            "last_error": "",
                            "delete_started_at": started,
                            "updated_at": started,
                        }
                    )
                    self._persist_collection_row(row)
                if row.get("status") != "deleting":
                    continue
                retry_at = str(row.get("next_retry_at") or "")
                if retry_at and datetime.fromisoformat(retry_at) > now:
                    continue
                if self._cleanup_collection_locked(row):
                    successes += 1
                else:
                    failures += 1
        self.reconciliation_successes += successes
        self.reconciliation_failures += failures
        return {"successes": successes, "failures": failures}

    def operational_health(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        collections = self._rows(_COLLECTIONS_TABLE, columns=["status", "expires_at", "delete_started_at"])
        documents = self._rows(_DOCUMENTS_TABLE, columns=["recovery_state", "updated_at"])
        active = sum(row.get("status") == "active" for row in collections)
        deleting = sum(row.get("status") == "deleting" for row in collections)
        expired = sum(
            bool(row.get("expires_at")) and datetime.fromisoformat(str(row["expires_at"])) <= now for row in collections
        )
        pending_times: list[datetime] = []
        for row in collections:
            if row.get("status") == "deleting" and row.get("delete_started_at"):
                pending_times.append(datetime.fromisoformat(str(row["delete_started_at"])))
        for row in documents:
            if row.get("recovery_state") and row.get("updated_at"):
                pending_times.append(datetime.fromisoformat(str(row["updated_at"])))
        oldest_age = max(((now - started).total_seconds() for started in pending_times), default=0.0)
        return {
            "catalog": {"healthy": True, "schema_version": _CATALOG_SCHEMA_VERSION},
            "collections": {"active": active, "deleting": deleting, "expired": expired},
            "cleanup": {
                "pending": len(pending_times),
                "oldest_age_seconds": round(oldest_age, 3),
            },
            "reconciliation": {
                "successes": self.reconciliation_successes,
                "failures": self.reconciliation_failures,
            },
            "open_table_cache_count": len(self._opened_tables),
        }

    def _get_local_embedder(self) -> Any:
        if self._local_embedder is None:
            from nemo_retriever.models import create_local_embedder

            self._local_embedder = create_local_embedder(
                self.embed_model,
                backend=self.local_embed_backend,
                device=self.device,
                hf_cache_dir=self.hf_cache_dir,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
            logger.info(
                "Loaded local query embedder model=%s backend=%s",
                self.embed_model,
                self.local_embed_backend,
            )
        return self._local_embedder

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed query texts via remote NIM or in-pod Hugging Face."""
        if self.embed_endpoint:
            return _embed_queries_remote(
                texts,
                embed_model=self.embed_model,
                embed_model_provider_prefix=self.embed_model_provider_prefix,
                embed_endpoint=self.embed_endpoint,
                embed_api_key=self.embed_api_key,
            )
        if self.local_embed:
            with self._embed_lock:
                embedder = self._get_local_embedder()
                tensor = embedder.embed_queries(texts)
            return _tensor_to_embedding_rows(tensor)
        raise RuntimeError("No embedding backend configured (remote endpoint or --local-embed).")


# ── FastAPI app ──────────────────────────────────────────────────────

_state: VectorDBState | None = None
_query_semaphore: asyncio.Semaphore | None = None

MAX_CONCURRENT_QUERIES = 4


def create_vectordb_app(
    lancedb_uri: str = "/data/vectordb",
    table_name: str = "nemo_retriever",
    embed_endpoint: str = "",
    embed_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2",
    embed_model_provider_prefix: str | None = None,
    embed_api_key: str = "",
    *,
    local_embed: bool = False,
    local_embed_backend: str = "hf",
    hf_cache_dir: str | None = None,
    device: str | None = None,
    gpu_memory_utilization: float = 0.45,
    internal_api_token: str | None = None,
    collection_artifact_root: str | None = None,
    artifact_storage_options_file: str | None = None,
    reconciliation_interval_seconds: int = 60,
    expiration_cleanup_enabled: bool = True,
) -> FastAPI:
    """Build the VectorDB FastAPI application."""

    if reconciliation_interval_seconds < 0:
        raise ValueError("reconciliation_interval_seconds must be non-negative")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        global _state, _query_semaphore
        _state = VectorDBState(
            lancedb_uri=lancedb_uri,
            table_name=table_name,
            embed_endpoint=embed_endpoint,
            embed_model=embed_model,
            embed_model_provider_prefix=embed_model_provider_prefix,
            embed_api_key=embed_api_key,
            local_embed=local_embed,
            local_embed_backend=local_embed_backend,
            hf_cache_dir=hf_cache_dir,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            collection_artifact_root=collection_artifact_root,
            artifact_storage_options_file=artifact_storage_options_file,
            expiration_cleanup_enabled=expiration_cleanup_enabled,
        )
        _query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
        logger.info(
            "VectorDB service started: uri=%s table=%s embed_mode=%s max_concurrent_queries=%d",
            lancedb_uri,
            table_name,
            _state.embed_mode,
            MAX_CONCURRENT_QUERIES,
        )
        if _state.embed_mode == "none":
            logger.error(
                "VectorDB started without an embedding backend; /v1/query will "
                "return HTTP 501 until --embed-endpoint or --local-embed is "
                "configured."
            )

        async def reconciliation_loop() -> None:
            while True:
                try:
                    await asyncio.to_thread(_state.reconcile)
                except Exception:
                    logger.exception("VectorDB reconciliation iteration failed")
                    if _state is not None:
                        _state.reconciliation_failures += 1
                await asyncio.sleep(reconciliation_interval_seconds)

        reconciliation_task = (
            asyncio.create_task(reconciliation_loop()) if reconciliation_interval_seconds > 0 else None
        )
        try:
            yield
        finally:
            if reconciliation_task is not None:
                reconciliation_task.cancel()
                try:
                    await reconciliation_task
                except asyncio.CancelledError:
                    pass
            _state = None
            _query_semaphore = None
            logger.info("VectorDB service stopped")

    app = FastAPI(
        title="NeMo Retriever VectorDB",
        description="LanceDB-backed vector storage and retrieval",
        version="1.0.0",
        lifespan=lifespan,
    )

    required_internal_token = (internal_api_token or "").strip()

    @app.middleware("http")
    async def require_internal_credential(request: Request, call_next):
        if request.url.path == "/v1/health" or not required_internal_token:
            return await call_next(request)
        supplied = request.headers.get("X-NRL-Internal-Token", "")
        if not supplied or not hmac.compare_digest(supplied, required_internal_token):
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid internal credential."},
            )
        return await call_next(request)

    @app.get("/v1/health", tags=["system"])
    async def health() -> dict[str, Any]:
        rows = _state.total_rows() if _state else 0
        health_payload = {
            "status": "ok",
            "total_rows": rows,
            "table_exists": _state.table_exists if _state else False,
            "embed_mode": _state.embed_mode if _state else "none",
        }
        if _state:
            health_payload.update(_state.operational_health())
        return health_payload

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        from prometheus_client import CollectorRegistry, Gauge, generate_latest
        from fastapi.responses import Response

        registry = CollectorRegistry()
        health = (
            _state.operational_health()
            if _state
            else {
                "collections": {"active": 0, "deleting": 0, "expired": 0},
                "cleanup": {"pending": 0, "oldest_age_seconds": 0},
                "reconciliation": {"successes": 0, "failures": 0},
                "open_table_cache_count": 0,
            }
        )
        collection_gauge = Gauge(
            "nrl_vectordb_collections",
            "Collection count by lifecycle status",
            ["status"],
            registry=registry,
        )
        for status, value in health["collections"].items():
            collection_gauge.labels(status=status).set(value)
        Gauge(
            "nrl_vectordb_cleanup_pending",
            "Pending lifecycle cleanup",
            registry=registry,
        ).set(health["cleanup"]["pending"])
        Gauge(
            "nrl_vectordb_cleanup_oldest_age_seconds",
            "Oldest pending cleanup age",
            registry=registry,
        ).set(health["cleanup"]["oldest_age_seconds"])
        Gauge(
            "nrl_vectordb_reconciliation_successes_total",
            "Successful reconciliations",
            registry=registry,
        ).set(health["reconciliation"]["successes"])
        Gauge(
            "nrl_vectordb_reconciliation_failures_total",
            "Failed reconciliations",
            registry=registry,
        ).set(health["reconciliation"]["failures"])
        Gauge(
            "nrl_vectordb_open_table_cache",
            "Open collection-table cache size",
            registry=registry,
        ).set(health["open_table_cache_count"])
        return Response(generate_latest(registry), media_type="text/plain; version=0.0.4")

    @app.post("/internal/vectordb/write", response_model=WriteResponse, tags=["internal"])
    async def write(req: WriteRequest) -> WriteResponse:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        written = await asyncio.to_thread(
            _state.write_rows,
            req.rows,
            scope=req.scope,
            collection_name=req.collection_name,
            document_id=req.document_id,
            job_id=req.job_id,
            filename=req.filename,
            content_sha256=req.content_sha256,
            operation=req.operation,
            artifact_prefix=req.artifact_prefix,
        )
        return WriteResponse(
            written=written,
            total_rows=_state.total_rows(scope=req.scope, collection_name=req.collection_name),
        )

    def _scope(value: str | None) -> str:
        return (value or "default").strip() or "default"

    @app.post(
        "/v1/collections",
        response_model=CollectionInfo,
        status_code=201,
        tags=["collections"],
    )
    async def create_collection(req: CollectionCreateRequest, x_nrl_scope: str | None = Header(None)) -> CollectionInfo:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return await asyncio.to_thread(_state.create_collection, _scope(x_nrl_scope), req)

    @app.get("/v1/collections", response_model=CollectionPage, tags=["collections"])
    async def list_collections(
        limit: int = Query(100, ge=1, le=200),
        continuation_token: str | None = None,
        x_nrl_scope: str | None = Header(None),
    ) -> CollectionPage:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return await asyncio.to_thread(_state.list_collections, _scope(x_nrl_scope), limit, continuation_token)

    @app.get("/v1/collections/{name}", response_model=CollectionInfo, tags=["collections"])
    async def get_collection(name: str, x_nrl_scope: str | None = Header(None)) -> CollectionInfo:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return await asyncio.to_thread(_state.get_collection, _scope(x_nrl_scope), name)

    @app.patch("/v1/collections/{name}", response_model=CollectionInfo, tags=["collections"])
    async def update_collection(
        name: str,
        req: CollectionUpdateRequest,
        x_nrl_scope: str | None = Header(None),
    ) -> CollectionInfo:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return await asyncio.to_thread(_state.update_collection, _scope(x_nrl_scope), name, req)

    @app.delete(
        "/v1/collections/{name}",
        response_model=CollectionDeleteResult,
        tags=["collections"],
    )
    async def delete_collection(
        response: Response,
        name: str,
        if_exists: bool = False,
        x_nrl_scope: str | None = Header(None),
    ) -> CollectionDeleteResult:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        result = await asyncio.to_thread(_state.delete_collection, _scope(x_nrl_scope), name, if_exists)
        response.status_code = 202 if result.cleanup_pending else 200
        return result

    @app.get(
        "/v1/collections/{name}/documents",
        response_model=DocumentPage,
        tags=["collections"],
    )
    async def list_documents(
        name: str,
        limit: int = Query(100, ge=1, le=200),
        continuation_token: str | None = None,
        x_nrl_scope: str | None = Header(None),
    ) -> DocumentPage:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return await asyncio.to_thread(_state.list_documents, _scope(x_nrl_scope), name, limit, continuation_token)

    @app.get(
        "/v1/collections/{name}/documents/{document_id}",
        response_model=DocumentInfo,
        tags=["collections"],
    )
    async def get_document(name: str, document_id: DocumentId, x_nrl_scope: str | None = Header(None)) -> DocumentInfo:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return await asyncio.to_thread(_state.get_document, _scope(x_nrl_scope), name, document_id)

    @app.delete(
        "/v1/collections/{name}/documents/{document_id}",
        response_model=DocumentDeleteResult,
        tags=["collections"],
    )
    async def delete_document(
        response: Response,
        name: str,
        document_id: DocumentId,
        if_exists: bool = False,
        x_nrl_scope: str | None = Header(None),
    ) -> DocumentDeleteResult:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        result = await asyncio.to_thread(
            _state.delete_document,
            _scope(x_nrl_scope),
            name,
            document_id,
            if_exists,
        )
        response.status_code = 202 if result.cleanup_pending else 200
        return result

    @app.post(
        "/v1/query",
        response_model=Union[QueryResponse, EvidenceQueryResponse],
        tags=["query"],
    )
    async def query(
        req: QueryRequest,
        x_nrl_scope: str | None = Header(None),
    ) -> QueryResponse | EvidenceQueryResponse:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")

        if _state.embed_mode == "none":
            raise HTTPException(
                501,
                "No embedding backend configured. Set --embed-endpoint for a remote "
                "NIM or --local-embed for in-pod Hugging Face query embedding.",
            )

        collection_exists = req.collection_name is not None
        if not collection_exists and not _state.table_exists:
            raise HTTPException(
                status_code=422,
                detail="No data has been ingested yet. Ingest documents first, then query.",
            )

        queries = req.query if isinstance(req.query, list) else [req.query]
        if not queries:
            if req.format == "evidence":
                return EvidenceQueryResponse(results=[])
            return QueryResponse(results=[])

        async with _query_semaphore:
            vectors = await asyncio.to_thread(_state.embed_queries, queries)
            hits_per_query = await asyncio.to_thread(
                _state.search,
                vectors,
                req.top_k,
                scope=_scope(x_nrl_scope),
                collection_name=req.collection_name,
            )

        if req.format == "evidence":
            return EvidenceQueryResponse(
                results=[EvidenceResult(**build_evidence_result(hits, _QUERY_STRATEGIES)) for hits in hits_per_query]
            )

        return QueryResponse(results=[QueryResult(hits=hits) for hits in hits_per_query])

    return app


# ── CLI entry point ──────────────────────────────────────────────────


def main() -> None:
    internal_token = os.environ.get("NRL_INTERNAL_VDB_TOKEN", "")
    if not internal_token and (token_file := os.environ.get("NRL_INTERNAL_VDB_TOKEN_FILE")):
        internal_token = Path(token_file).read_text(encoding="utf-8").strip()
    parser = argparse.ArgumentParser(description="NeMo Retriever VectorDB service")
    parser.add_argument("--lancedb-uri", default="/data/vectordb", help="LanceDB directory")
    parser.add_argument("--table-name", default="nemo_retriever", help="LanceDB table name")
    parser.add_argument("--embed-endpoint", default="", help="Remote NIM/OpenAI-compatible embed URL")
    parser.add_argument("--embed-model", default="nvidia/llama-nemotron-embed-vl-1b-v2")
    parser.add_argument(
        "--embed-model-provider-prefix",
        default="",
        help="Optional LiteLLM provider prefix",
    )
    parser.add_argument("--embed-api-key", default="")
    parser.add_argument(
        "--internal-api-token",
        default=internal_token,
        help="Dedicated internal credential (prefer NRL_INTERNAL_VDB_TOKEN from a Secret).",
    )
    parser.add_argument(
        "--collection-artifact-root",
        default=os.environ.get("NRL_COLLECTION_ARTIFACT_ROOT", ""),
    )
    parser.add_argument(
        "--artifact-storage-options-file",
        default=os.environ.get("NRL_ARTIFACT_STORAGE_OPTIONS_FILE", ""),
    )
    parser.add_argument(
        "--reconciliation-interval-seconds",
        type=int,
        default=int(os.environ.get("NRL_RECONCILIATION_INTERVAL_SECONDS", "60")),
        help="Lifecycle reconciliation interval; zero disables the local loop.",
    )
    parser.add_argument(
        "--disable-expiration-cleanup",
        action="store_true",
        help="Disable automatic expiration cleanup (enabled by default).",
    )
    parser.add_argument(
        "--local-embed",
        action="store_true",
        help="Load Hugging Face embedder in-pod for /v1/query (requires [local] extras + GPU).",
    )
    parser.add_argument(
        "--local-embed-backend",
        default="hf",
        choices=("hf", "vllm"),
        help="Backend for --local-embed (default: hf).",
    )
    parser.add_argument("--hf-cache-dir", default="", help="Hugging Face model cache directory")
    parser.add_argument("--device", default="", help="Torch device for --local-embed (e.g. cuda:0)")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.45,
        help="vLLM GPU memory fraction when --local-embed-backend=vllm.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7671)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    if args.embed_endpoint and args.local_embed:
        parser.error("Use either --embed-endpoint or --local-embed, not both.")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app = create_vectordb_app(
        lancedb_uri=args.lancedb_uri,
        table_name=args.table_name,
        embed_endpoint=args.embed_endpoint,
        embed_model=args.embed_model,
        embed_model_provider_prefix=args.embed_model_provider_prefix or None,
        embed_api_key=args.embed_api_key,
        local_embed=args.local_embed,
        local_embed_backend=args.local_embed_backend,
        hf_cache_dir=args.hf_cache_dir or None,
        device=args.device or None,
        gpu_memory_utilization=args.gpu_memory_utilization,
        internal_api_token=args.internal_api_token or None,
        collection_artifact_root=args.collection_artifact_root or None,
        artifact_storage_options_file=args.artifact_storage_options_file or None,
        reconciliation_interval_seconds=args.reconciliation_interval_seconds,
        expiration_cleanup_enabled=not args.disable_expiration_cleanup,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
