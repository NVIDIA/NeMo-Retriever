# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qdrant storage for the service collection API.

Collection and document records live in the vector-less ``_nrl_catalog`` collection. Chunks live in shared
``_nrl_chunks_*`` collections, one per embedding model and vector width, filtered by the ``scope`` and
``collection_name`` tenant fields. Qdrant has no multi-point transactions, so writes follow the LanceDB
store's recovery states and ``reconcile_collections`` finishes or rolls back interrupted work.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from qdrant_client import models

from nemo_retriever.common.schemas.collections import (
    CollectionCreateRequest,
    CollectionInfo,
    CollectionPage,
    CollectionUpdateRequest,
    DocumentPage,
    IngestOperation,
)
from nemo_retriever.common.vdb.adt_vdb import (
    CollectionWriteContext,
    CollectionWriteResult,
    VDBInvalidRequest,
    VDBResourceConflict,
    VDBResourceNotFound,
)
from nemo_retriever.common.vdb.lancedb_collections import (
    LanceDBCollectionStore,
    _collection_rows,
    _decode_cursor,
    _encode_cursor,
    _is_uncommitted_initial_append,
    _normalize_collection_results,
    _now,
    _public_collection_hit,
)
from nemo_retriever.common.vdb.lancedb_schema import infer_vector_dim
from nemo_retriever.common.vdb.qdrant import (
    DENSE_VECTOR_NAME,
    _metadata_key,
    point_batches,
    query_batch,
    scroll_points,
)
from nemo_retriever.common.vdb.records import RetrievalContractError

logger = logging.getLogger(__name__)

CATALOG_COLLECTION = "_nrl_catalog"
_CHUNKS_PREFIX = "_nrl_chunks_"
_CATALOG_SCHEMA_VERSION = 2
_SCHEMA_VERSION_KEY = _metadata_key("catalog_schema_version")
_EMBEDDING_MODEL_KEY = _metadata_key("embedding_model_name")
_POINT_NAMESPACE = uuid.UUID("0f3c5a4e-6d1b-4c55-9a51-3b6c0f9e8d21")
_CATALOG_INDEXES = ("kind", "scope", "name", "collection_name", "document_id", "status", "recovery_state")


def _point_id(*parts: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, "\0".join(parts)))


def _collection_point(scope: str, name: str) -> str:
    return _point_id("collection", scope, name)


def _document_point(scope: str, collection_name: str, document_id: str) -> str:
    return _point_id("document", scope, collection_name, document_id)


def _chunks_collection(model: str, vector_dim: int) -> str:
    return _CHUNKS_PREFIX + hashlib.sha256(f"{model}\0{vector_dim}".encode()).hexdigest()[:16]


def _generation(version: Any, content_sha256: Any) -> str:
    """Key one written version of a document. A replace may keep the version and change the content."""
    return f"{version or ''}\0{content_sha256 or ''}"


def _match(**fields: Any) -> list[Any]:
    return [models.FieldCondition(key=key, match=models.MatchValue(value=value)) for key, value in fields.items()]


def _where(**fields: Any) -> Any:
    return models.Filter(must=_match(**fields))


class QdrantCollectionStore:
    """Collection API for the Qdrant backend."""

    def __init__(self, backend: Any, *, expiration_cleanup_enabled: bool = True) -> None:
        self._backend = backend
        self._client = backend.client
        self.expiration_cleanup_enabled = expiration_cleanup_enabled
        self.reconciliation_successes = 0
        self.reconciliation_failures = 0
        self._write_lock = threading.Lock()
        self._collection_write_lock = threading.Lock()
        self._table_user_condition = threading.Condition(self._write_lock)
        self._active_table_users: dict[str, int] = {}
        # Queries between their visibility snapshot and their search, and writers waiting to change visibility.
        self._active_queries: Counter[str] = Counter()
        self._pending_publishes: Counter[str] = Counter()
        self._ensure_catalog()

    def _ensure_catalog(self) -> None:
        if self._client.collection_exists(CATALOG_COLLECTION):
            config = self._client.get_collection(CATALOG_COLLECTION).config
            version = (config.metadata or {}).get(_SCHEMA_VERSION_KEY)
            if config.params.vectors or config.params.sparse_vectors or version != _CATALOG_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Incompatible Qdrant collection catalog {CATALOG_COLLECTION!r}: expected a vector-less "
                    f"collection with {_SCHEMA_VERSION_KEY}={_CATALOG_SCHEMA_VERSION}, found schema version "
                    f"{version!r}."
                )
            return
        try:
            self._client.create_collection(
                CATALOG_COLLECTION,
                vectors_config={},
                metadata={_SCHEMA_VERSION_KEY: _CATALOG_SCHEMA_VERSION},
            )
        except Exception:
            # Another process may have created the catalog concurrently.
            if not self._client.collection_exists(CATALOG_COLLECTION):
                raise
            self._ensure_catalog()
            return
        for field in _CATALOG_INDEXES:
            self._client.create_payload_index(CATALOG_COLLECTION, field, models.PayloadSchemaType.KEYWORD)

    def _scroll(self, collection: str, scroll_filter: Any, with_payload: Any = True) -> list[dict[str, Any]]:
        points = scroll_points(self._client, collection, scroll_filter=scroll_filter, with_payload=with_payload)
        return [dict(point.payload or {}) for point in points]

    def _rows(self, kind: str, scroll_filter: Any = None, with_payload: Any = True) -> list[dict[str, Any]]:
        conditions = _match(kind=kind)
        if scroll_filter is not None:
            conditions.append(scroll_filter)
        return self._scroll(CATALOG_COLLECTION, models.Filter(must=conditions), with_payload)

    def _catalog_row(self, point_id: str) -> dict[str, Any] | None:
        points = self._client.retrieve(CATALOG_COLLECTION, [point_id])
        return dict(points[0].payload or {}) if points else None

    def _upsert_catalog(self, point_id: str, row: dict[str, Any]) -> None:
        self._client.upsert(CATALOG_COLLECTION, [models.PointStruct(id=point_id, vector={}, payload=row)], wait=True)

    def _delete_catalog(self, point_id: str) -> None:
        self._client.delete(CATALOG_COLLECTION, points_selector=models.PointIdsList(points=[point_id]), wait=True)

    # Storage-independent steps are shared with the LanceDB store.
    _acquire_table_user_locked = LanceDBCollectionStore._acquire_table_user_locked
    _release_table_user = LanceDBCollectionStore._release_table_user
    _wait_for_table_users_locked = LanceDBCollectionStore._wait_for_table_users_locked
    _document_info = staticmethod(LanceDBCollectionStore._document_info)
    _refresh_collection_activity_row = staticmethod(LanceDBCollectionStore._refresh_collection_activity_row)
    _refresh_collection_activity_locked = LanceDBCollectionStore._refresh_collection_activity_locked
    _retry_at = staticmethod(LanceDBCollectionStore._retry_at)
    _schedule_collection_retry = LanceDBCollectionStore._schedule_collection_retry
    get_collection = LanceDBCollectionStore.get_collection
    delete_collection = LanceDBCollectionStore.delete_collection
    write_collection = LanceDBCollectionStore.write_collection
    empty_health = staticmethod(LanceDBCollectionStore.empty_health)
    get_document = LanceDBCollectionStore.get_document
    delete_document = LanceDBCollectionStore.delete_document

    def _begin_query_locked(self, key: str) -> None:
        while self._pending_publishes[key]:
            self._table_user_condition.wait()
        self._active_queries[key] += 1

    def _end_query_locked(self, key: str) -> None:
        self._active_queries[key] -= 1
        self._table_user_condition.notify_all()

    def _publish_locked(self, key: str, row: dict[str, Any]) -> None:
        """Persist a document row that changes chunk visibility once no query is mid-search."""
        self._pending_publishes[key] += 1
        try:
            while self._active_queries[key]:
                self._table_user_condition.wait()
            self._persist_document_row(row)
        finally:
            self._pending_publishes[key] -= 1
            self._table_user_condition.notify_all()

    @staticmethod
    def _lease_key(scope: str, name: str) -> str:
        return f"{scope}\0{name}"

    def _collection_row(self, scope: str, name: str, *, active: bool = False) -> dict[str, Any] | None:
        row = self._catalog_row(_collection_point(scope, name))
        if row and active and row["status"] != "active":
            raise VDBInvalidRequest(f"Collection {name!r} is {row['status']}")
        if row and active and row.get("expires_at"):
            if datetime.fromisoformat(str(row["expires_at"])) <= datetime.now(timezone.utc):
                raise VDBInvalidRequest(f"Collection {name!r} is expired")
        return row

    @staticmethod
    def _collection_info(row: dict[str, Any]) -> CollectionInfo:
        return CollectionInfo(
            name=row["name"],
            scope=row["scope"],
            status=row["status"],
            description=row.get("description") or None,
            metadata=row.get("metadata") or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row.get("expires_at") or None,
        )

    def _persist_collection_row(self, row: dict[str, Any]) -> None:
        self._upsert_catalog(_collection_point(row["scope"], row["name"]), row)

    def create_collection(self, scope: str, request: CollectionCreateRequest) -> CollectionInfo:
        with self._write_lock:
            if self._collection_row(scope, request.name):
                raise VDBResourceConflict(f"Collection {request.name!r} already exists")
            now = _now()
            row = {
                "kind": "collection",
                "scope": scope,
                "name": request.name,
                "status": "active",
                "description": request.description or "",
                "metadata": request.metadata,
                "created_at": now,
                "updated_at": now,
                "expires_at": request.expires_at or "",
                "chunks_collection": "",
                "embedding_model": "",
                "vector_dim": 0,
                "deletion_phase": "",
                "retry_count": 0,
                "next_retry_at": "",
                "last_error": "",
                "delete_started_at": "",
            }
            self._persist_collection_row(row)
            return self._collection_info(row)

    def list_collections(self, scope: str, limit: int, continuation_token: str | None) -> CollectionPage:
        rows = sorted(self._rows("collection", _where(scope=scope)), key=lambda row: row["name"])
        last = _decode_cursor(continuation_token, resource="collections", scope=scope, collection=None)
        if last is not None:
            if len(last) != 1:
                raise VDBInvalidRequest("Invalid collection continuation token")
            rows = [row for row in rows if row["name"] > last[0]]
        page = rows[:limit]
        next_token = (
            _encode_cursor("collections", scope, None, [page[-1]["name"]]) if len(rows) > limit and page else None
        )
        return CollectionPage(items=[self._collection_info(row) for row in page], next_token=next_token)

    def update_collection(self, scope: str, name: str, request: CollectionUpdateRequest) -> CollectionInfo:
        with self._write_lock:
            row = self._collection_row(scope, name, active=True)
            if not row:
                raise VDBResourceNotFound("Collection not found")
            update = request.model_dump(exclude_unset=True)
            row["description"] = update.get("description", row["description"]) or ""
            if "metadata" in update:
                row["metadata"] = update["metadata"] or {}
            now = _now()
            if "expires_at" in update:
                row["expires_at"] = update["expires_at"] or ""
                row["updated_at"] = now
            else:
                self._refresh_collection_activity_row(row, activity_at=now)
            self._persist_collection_row(row)
            return self._collection_info(row)

    def _mark_collection_deleting_locked(self, row: dict[str, Any]) -> None:
        now = _now()
        row.update(
            {
                "status": "deleting",
                "deletion_phase": "delete_chunks",
                "retry_count": 0,
                "next_retry_at": "",
                "last_error": "",
                "delete_started_at": now,
                "updated_at": now,
            }
        )
        self._persist_collection_row(row)

    def _delete_chunks(self, row: dict[str, Any], **fields: Any) -> None:
        chunks = row.get("chunks_collection")
        if chunks and self._client.collection_exists(chunks):
            selector = models.FilterSelector(filter=_where(scope=row["scope"], collection_name=row["name"], **fields))
            self._client.delete(chunks, points_selector=selector, wait=True)

    @staticmethod
    def _stale_chunks(row: dict[str, Any]) -> Any:
        """Match a document's chunks outside its committed generation or chunk count."""
        return models.Filter(
            must=_match(scope=row["scope"], collection_name=row["collection_name"], document_id=row["document_id"]),
            should=[
                models.Filter(
                    must_not=_match(generation=_generation(row["current_document_version"], row["content_sha256"]))
                ),
                models.FieldCondition(key="chunk_index", range=models.Range(gte=int(row["chunk_count"]))),
            ],
        )

    def _delete_stale_chunks(self, chunks: str, row: dict[str, Any]) -> None:
        selector = models.FilterSelector(filter=self._stale_chunks(row))
        self._client.delete(chunks, points_selector=selector, wait=True)

    def _cleanup_collection_locked(self, row: dict[str, Any]) -> bool:
        phase = str(row.get("deletion_phase") or "delete_chunks")
        try:
            if phase == "delete_chunks":
                self._wait_for_table_users_locked(self._lease_key(row["scope"], row["name"]))
                self._delete_chunks(row)
                row["deletion_phase"] = phase = "delete_catalog"
                row["updated_at"] = _now()
                self._persist_collection_row(row)
            if phase == "delete_catalog":
                self._client.delete(
                    CATALOG_COLLECTION,
                    points_selector=models.FilterSelector(
                        filter=_where(kind="document", scope=row["scope"], collection_name=row["name"])
                    ),
                    wait=True,
                )
                self._delete_catalog(_collection_point(row["scope"], row["name"]))
            return True
        except Exception as exc:
            logger.exception("Collection cleanup paused at phase %s", phase)
            self._schedule_collection_retry(row, phase, exc)
            return False

    def _resolved_table(self, scope: str, name: str) -> dict[str, Any]:
        """Return the active collection's catalog row (the LanceDB store returns a table name)."""
        row = self._collection_row(scope, name, active=True)
        if not row:
            raise VDBResourceNotFound("Collection not found")
        return row

    def _validate_embedding_model(self, row: dict[str, Any]) -> None:
        configured = self._backend.embedding_model_name
        if not configured or not row.get("chunks_collection"):
            return
        stored = str(row.get("embedding_model") or "").strip()
        if not stored:
            raise VDBInvalidRequest(
                f"Existing Qdrant collection {row['name']!r} does not record its embedding model, "
                "so query compatibility cannot be verified. Rebuild and re-ingest the collection with "
                "the configured embedding model."
            )

        from nemo_retriever.models import resolve_embed_model

        stored_model = resolve_embed_model(stored)
        expected_model = resolve_embed_model(configured)
        if stored_model != expected_model:
            raise VDBInvalidRequest(
                f"Existing Qdrant collection {row['name']!r} uses embedding model {stored_model!r}, "
                f"but the VectorDB service is configured for {expected_model!r}. Use the index model or "
                "rebuild and re-ingest the collection with the configured model."
            )

    def _ensure_chunks_collection(self, name: str, vector_dim: int, model: str) -> None:
        if not self._client.collection_exists(name):
            try:
                self._client.create_collection(
                    name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
                    },
                    metadata={_EMBEDDING_MODEL_KEY: model, _SCHEMA_VERSION_KEY: _CATALOG_SCHEMA_VERSION},
                )
            except Exception:
                if not self._client.collection_exists(name):
                    raise
            else:
                tenant = models.KeywordIndexParams(type="keyword", is_tenant=True)
                self._client.create_payload_index(name, "scope", tenant)
                self._client.create_payload_index(name, "collection_name", tenant)
                for field in ("document_id", "generation"):
                    self._client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
                self._client.create_payload_index(name, "chunk_index", models.PayloadSchemaType.INTEGER)
        vectors = self._client.get_collection(name).config.params.vectors
        dense = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
        if dense is None:
            raise VDBInvalidRequest(f"Qdrant collection {name!r} has no {DENSE_VECTOR_NAME!r} vector; recreate it")
        size = dense.size
        if size != vector_dim:
            raise VDBInvalidRequest(f"Collection vectors have dimension {vector_dim}, expected {size}")

    def _persist_document_row(self, row: dict[str, Any]) -> None:
        self._upsert_catalog(_document_point(row["scope"], row["collection_name"], row["document_id"]), row)

    def _document_rows(self, scope: str, collection_name: str, document_id: str) -> list[dict[str, Any]]:
        row = self._catalog_row(_document_point(scope, collection_name, document_id))
        return [row] if row else []

    def _write_collection_serialized(self, records: list, *, context: CollectionWriteContext) -> CollectionWriteResult:
        rows = _collection_rows(records, context=context)
        completed_row: dict[str, Any] | None = None
        lease = self._lease_key(context.scope, context.collection_name)
        with self._write_lock:
            collection = self._resolved_table(context.scope, context.collection_name)
            if records and not rows:
                raise VDBInvalidRequest("Collection records produced no writable vector rows")
            existing = self._document_rows(context.scope, context.collection_name, context.document_id)
            if context.operation is IngestOperation.REPLACE:
                if not existing:
                    raise VDBResourceNotFound("Document not found")
            elif existing:
                document = existing[0]
                known_versions = {
                    str(document.get(field) or "")
                    for field in ("document_version", "current_document_version", "pending_document_version")
                    if document.get(field)
                }
                if document.get("recovery_state") not in {"", "appending"} or known_versions != {
                    context.document_version
                }:
                    raise VDBResourceConflict("append cannot change an existing document; use replace")
                stored_hash = str(document.get("content_sha256") or "")
                if stored_hash and stored_hash != context.content_sha256:
                    raise VDBResourceConflict("append content does not match the existing document; use replace")

            if rows:
                vector_dim = infer_vector_dim(rows)
                if vector_dim == 0:
                    raise VDBInvalidRequest("Cannot infer vector dimension from collection records")
                if collection.get("chunks_collection"):
                    self._validate_embedding_model(collection)
                    if int(collection.get("vector_dim") or 0) != vector_dim:
                        raise VDBInvalidRequest(
                            f"Collection vectors have dimension {vector_dim}, "
                            f"expected {collection.get('vector_dim')}"
                        )
                else:
                    model = self._backend.embedding_model_name or ""
                    chunks = _chunks_collection(model, vector_dim)
                    self._ensure_chunks_collection(chunks, vector_dim, model)
                    collection.update({"chunks_collection": chunks, "embedding_model": model, "vector_dim": vector_dim})
                    self._persist_collection_row(collection)

                now = _now()
                completed_row = {
                    "kind": "document",
                    "scope": context.scope,
                    "collection_name": context.collection_name,
                    "document_id": context.document_id,
                    "job_id": context.job_id or "",
                    "filename": context.filename,
                    "content_sha256": context.content_sha256,
                    "document_version": context.document_version,
                    "status": "completed",
                    "chunk_count": len(rows),
                    "created_at": existing[0]["created_at"] if existing else now,
                    "updated_at": now,
                    "error": "",
                    "current_document_version": context.document_version,
                    "pending_document_version": "",
                    "pending_content_sha256": "",
                    "pending_chunk_count": 0,
                    "recovery_state": "refreshing_collection_activity",
                }
                if existing:
                    marker = dict(existing[0])
                else:
                    marker = {
                        **completed_row,
                        "status": "appending",
                        "document_version": "",
                        "current_document_version": "",
                        "chunk_count": 0,
                    }
                if context.operation is IngestOperation.APPEND:
                    marker.update({"job_id": context.job_id or "", "recovery_state": "appending"})
                else:
                    marker.update({"status": "replacing", "recovery_state": "replacing"})
                marker.update(
                    {
                        "pending_document_version": context.document_version,
                        "pending_content_sha256": context.content_sha256,
                        "pending_chunk_count": len(rows),
                        "updated_at": now,
                        "error": "",
                    }
                )
                self._publish_locked(lease, marker)
            chunks_collection = collection.get("chunks_collection") or ""
            self._acquire_table_user_locked(lease)

        try:
            if rows:
                generation = _generation(context.document_version, context.content_sha256)
                points = [
                    models.PointStruct(
                        id=_point_id(context.scope, context.collection_name, row["chunk_id"], context.content_sha256),
                        vector={DENSE_VECTOR_NAME: list(map(float, row["vector"]))},
                        payload={
                            **{key: value for key, value in row.items() if key != "vector"},
                            "scope": context.scope,
                            "collection_name": context.collection_name,
                            "chunk_index": index,
                            "generation": generation,
                        },
                    )
                    for index, row in enumerate(rows)
                ]
                for batch in point_batches(points, 256):
                    self._client.upload_points(chunks_collection, batch, batch_size=len(batch), wait=True)
                if context.operation is IngestOperation.REPLACE:
                    # Queries only see the committed generation, so commit before pruning.
                    with self._write_lock:
                        self._publish_locked(lease, {**completed_row, "recovery_state": "pruning_chunks"})
                    self._delete_stale_chunks(chunks_collection, completed_row)
                logger.info(
                    "Wrote %d chunks to Qdrant collection %r operation=%s",
                    len(rows),
                    context.collection_name,
                    context.operation,
                )

            total_rows = 0
            if chunks_collection:
                total_rows = self._client.count(
                    chunks_collection,
                    count_filter=_where(scope=context.scope, collection_name=context.collection_name),
                    exact=True,
                ).count
            with self._write_lock:
                if completed_row is not None:
                    self._persist_document_row(completed_row)
                    self._refresh_collection_activity_locked(
                        context.scope, context.collection_name, activity_at=completed_row["updated_at"]
                    )
                    completed_row["recovery_state"] = ""
                    self._persist_document_row(completed_row)
            return CollectionWriteResult(written=len(rows), total_rows=total_rows)
        finally:
            self._release_table_user(lease)

    def retrieve_collection(
        self,
        vectors: list,
        *,
        scope: str,
        collection_name: str,
        query_texts: list[str],
        top_k: int,
        **kwargs: Any,
    ) -> tuple[list[list[dict[str, Any]]], list[str]]:
        """Run scoped dense retrieval. ``distance`` is the cosine distance ``1 - similarity``."""
        if len(query_texts) != len(vectors):
            raise RetrievalContractError("query_texts must contain one entry per query vector")
        lease = self._lease_key(scope, collection_name)
        with self._write_lock:
            collection = self._resolved_table(scope, collection_name)
            chunks_collection = collection.get("chunks_collection") or ""
            if not chunks_collection:
                return ([[] for _ in vectors], ["dense"])
            self._validate_embedding_model(collection)
            self._begin_query_locked(lease)
            try:
                in_flux = self._rows(
                    "document",
                    models.Filter(
                        must=_match(scope=scope, collection_name=collection_name),
                        must_not=_match(recovery_state=""),
                    ),
                )
            except BaseException:
                self._end_query_locked(lease)
                raise
            self._acquire_table_user_locked(lease)
        try:
            # Documents with unfinished writes show only their committed generation.
            hidden: list[Any] = []
            uncommitted = [row["document_id"] for row in in_flux if not row.get("current_document_version")]
            if uncommitted:
                hidden.append(models.FieldCondition(key="document_id", match=models.MatchAny(any=uncommitted)))
            hidden.extend(
                self._stale_chunks(row)
                for row in in_flux
                if row.get("current_document_version") and row.get("recovery_state") != "deleting_chunks"
            )
            query_filter, params, _ = self._backend._search_options({**kwargs, "top_k": top_k})
            visible = models.Filter(
                must=[*_match(scope=scope, collection_name=collection_name), *([query_filter] if query_filter else [])],
                must_not=hidden or None,
            )
            requests = [
                models.QueryRequest(
                    query=list(map(float, vector)),
                    using=DENSE_VECTOR_NAME,
                    filter=visible,
                    params=params,
                    limit=int(top_k),
                    with_payload=True,
                )
                for vector in vectors
            ]
            responses = query_batch(self._client, chunks_collection, requests)
            raw_results = [
                [{**(hit.payload or {}), "_distance": max(0.0, 1.0 - hit.score)} for hit in response.points]
                for response in responses
            ]
            normalized = _normalize_collection_results(raw_results, expected_queries=len(vectors))
            return [[_public_collection_hit(hit) for hit in hits] for hits in normalized], ["dense"]
        finally:
            self._release_table_user(lease)
            with self._write_lock:
                self._end_query_locked(lease)

    def list_documents(
        self, scope: str, collection_name: str, limit: int, continuation_token: str | None
    ) -> DocumentPage:
        self._resolved_table(scope, collection_name)
        rows = self._rows("document", _where(scope=scope, collection_name=collection_name))
        rows = [row for row in rows if not _is_uncommitted_initial_append(row)]
        rows.sort(key=lambda row: (row["created_at"], row["document_id"]))
        last = _decode_cursor(continuation_token, resource="documents", scope=scope, collection=collection_name)
        if last is not None:
            if len(last) != 2:
                raise VDBInvalidRequest("Invalid document continuation token")
            rows = [row for row in rows if (row["created_at"], row["document_id"]) > (last[0], last[1])]
        page = rows[:limit]
        return DocumentPage(
            items=[self._document_info(row) for row in page],
            next_token=(
                _encode_cursor("documents", scope, collection_name, [page[-1]["created_at"], page[-1]["document_id"]])
                if len(rows) > limit and page
                else None
            ),
        )

    def _reconcile_document_row_locked(self, row: dict[str, Any], collection: dict[str, Any]) -> bool:
        activity_refresh = "refreshing_collection_activity"
        state = str(row.get("recovery_state") or "")
        scope = str(row["scope"])
        collection_name = str(row["collection_name"])
        document_id = str(row["document_id"])
        chunks_collection = collection.get("chunks_collection") or ""
        try:
            if state in {"appending", "replacing"}:
                pending = str(row.get("pending_document_version") or "")
                pending_generation = _generation(pending, row.get("pending_content_sha256"))
                expected = int(row.get("pending_chunk_count") or 0)
                pending_chunks: list[dict[str, Any]] = []
                if pending and chunks_collection and self._client.collection_exists(chunks_collection):
                    pending_chunks = self._scroll(
                        chunks_collection,
                        _where(
                            scope=scope,
                            collection_name=collection_name,
                            document_id=document_id,
                            generation=pending_generation,
                        ),
                        with_payload=["filename", "chunk_index"],
                    )
                    pending_chunks = [
                        chunk for chunk in pending_chunks if int(chunk.get("chunk_index") or 0) < max(expected, 1)
                    ]
                if pending_chunks and len(pending_chunks) >= expected:
                    row.update(
                        {
                            "document_version": pending,
                            "current_document_version": pending,
                            "content_sha256": row.get("pending_content_sha256") or row.get("content_sha256") or "",
                            "filename": pending_chunks[0].get("filename") or row.get("filename") or "",
                            "chunk_count": len(pending_chunks),
                            "pending_document_version": "",
                            "pending_content_sha256": "",
                            "pending_chunk_count": 0,
                            "status": "completed",
                            "recovery_state": "pruning_chunks" if state == "replacing" else activity_refresh,
                            "updated_at": _now(),
                            "error": "",
                        }
                    )
                    self._persist_document_row(row)
                    state = str(row["recovery_state"])
                else:
                    # Roll back to the committed generation.
                    if not row.get("current_document_version"):
                        self._delete_chunks(collection, document_id=document_id)
                        self._delete_catalog(_document_point(scope, collection_name, document_id))
                        return True
                    if chunks_collection and self._client.collection_exists(chunks_collection):
                        self._delete_stale_chunks(chunks_collection, row)
                    row.update(
                        {
                            "pending_document_version": "",
                            "pending_content_sha256": "",
                            "pending_chunk_count": 0,
                            "status": "completed",
                            "recovery_state": "",
                            "updated_at": _now(),
                            "error": "",
                        }
                    )
                    self._persist_document_row(row)
                    return True
            if state == "pruning_chunks":
                if chunks_collection and self._client.collection_exists(chunks_collection):
                    self._delete_stale_chunks(chunks_collection, row)
                row.update({"recovery_state": activity_refresh, "error": ""})
                self._persist_document_row(row)
                state = activity_refresh
            if state == activity_refresh:
                self._refresh_collection_activity_locked(scope, collection_name, activity_at=row["updated_at"])
                row.update({"recovery_state": "", "error": ""})
                self._persist_document_row(row)
                return True
            if state == "deleting_chunks":
                self._wait_for_table_users_locked(self._lease_key(scope, collection_name))
                self._delete_chunks(collection, document_id=document_id)
                self._delete_catalog(_document_point(scope, collection_name, document_id))
                return True
            return state == ""
        except Exception as exc:
            row["error"] = str(exc)[:2000]
            if state == activity_refresh:
                row["recovery_state"] = activity_refresh
            else:
                row["updated_at"] = _now()
            self._persist_document_row(row)
            logger.exception("Document reconciliation paused in state %s", state)
            return False

    def reconcile_collections(self) -> dict[str, int]:
        successes = 0
        failures = 0
        now = datetime.now(timezone.utc)

        recovering = models.Filter(must_not=_match(recovery_state=""))
        for candidate in self._rows("document", recovering):
            with self._write_lock:
                rows = self._document_rows(candidate["scope"], candidate["collection_name"], candidate["document_id"])
                if not rows or not rows[0].get("recovery_state"):
                    continue
                collection = self._collection_row(rows[0]["scope"], rows[0]["collection_name"])
                if not collection:
                    continue
                self._wait_for_table_users_locked(self._lease_key(collection["scope"], collection["name"]))
                rows = self._document_rows(candidate["scope"], candidate["collection_name"], candidate["document_id"])
                if not rows or not rows[0].get("recovery_state"):
                    continue
                if self._reconcile_document_row_locked(rows[0], collection):
                    successes += 1
                else:
                    failures += 1

        def due(row: dict[str, Any]) -> bool:
            if row.get("status") == "deleting":
                retry_at = str(row.get("next_retry_at") or "")
                return not retry_at or datetime.fromisoformat(retry_at) <= now
            expires_at = str(row.get("expires_at") or "")
            return (
                self.expiration_cleanup_enabled
                and row.get("status") == "active"
                and bool(expires_at)
                and datetime.fromisoformat(expires_at) <= now
            )

        for candidate in [row for row in self._rows("collection") if due(row)]:
            with self._write_lock:
                row = self._collection_row(candidate["scope"], candidate["name"])
                if not row or not due(row):
                    continue
                if row.get("status") == "active":
                    self._mark_collection_deleting_locked(row)
                if self._cleanup_collection_locked(row):
                    successes += 1
                else:
                    failures += 1

        with self._write_lock:
            self.reconciliation_successes += successes
            self.reconciliation_failures += failures
        return {"successes": successes, "failures": failures}

    def health(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        collections = self._rows("collection", with_payload=["status", "expires_at", "delete_started_at"])
        documents = self._rows(
            "document", models.Filter(must_not=_match(recovery_state="")), ["recovery_state", "updated_at"]
        )
        expired = sum(
            bool(row.get("expires_at")) and datetime.fromisoformat(str(row["expires_at"])) <= now for row in collections
        )
        pending_times = [
            datetime.fromisoformat(str(row["delete_started_at"]))
            for row in collections
            if row.get("status") == "deleting" and row.get("delete_started_at")
        ]
        pending_times += [
            datetime.fromisoformat(str(row["updated_at"]))
            for row in documents
            if row.get("recovery_state") and row.get("updated_at")
        ]
        oldest_age = max(((now - started).total_seconds() for started in pending_times), default=0.0)
        return {
            "catalog": {"healthy": True, "initialized": True, "schema_version": _CATALOG_SCHEMA_VERSION},
            "collections": {
                "active": sum(row.get("status") == "active" for row in collections),
                "deleting": sum(row.get("status") == "deleting" for row in collections),
                "expired": expired,
            },
            "cleanup": {"pending": len(pending_times), "oldest_age_seconds": round(oldest_age, 3)},
            "reconciliation": {
                "successes": self.reconciliation_successes,
                "failures": self.reconciliation_failures,
            },
            "open_table_cache_count": 0,
        }
