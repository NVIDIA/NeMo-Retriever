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
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Union

import lancedb
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from nemo_retriever.common.schemas.collections import (
    CollectionCreateRequest,
    CollectionInfo,
    CollectionPage,
    CollectionUpdateRequest,
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


class WriteResponse(BaseModel):
    written: int
    total_rows: int


_COLLECTIONS_TABLE = "_nrl_collections"
_DOCUMENTS_TABLE = "_nrl_documents"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _physical_table(scope: str, collection_name: str) -> str:
    digest = hashlib.sha256(f"{scope}\0{collection_name}".encode()).hexdigest()
    return f"nrl_{digest[:40]}"


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _token(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode()).decode().rstrip("=")


def _offset(token: str | None) -> int:
    if not token:
        return 0
    try:
        return int(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode())
    except Exception as exc:
        raise HTTPException(422, "Invalid continuation token") from exc


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
            ]
        )
        for name, schema in ((_COLLECTIONS_TABLE, collection_schema), (_DOCUMENTS_TABLE, document_schema)):
            try:
                self._db.open_table(name)
            except Exception:
                self._db.create_table(name, schema=schema, mode="create")

    def _rows(self, table_name: str, where: str | None = None) -> list[dict[str, Any]]:
        query = self._db.open_table(table_name).search()
        if where:
            query = query.where(where)
        return query.to_list()

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
        return row

    @staticmethod
    def _collection_info(row: dict[str, Any]) -> CollectionInfo:
        return CollectionInfo(
            name=row["name"], scope=row["scope"], status=row["status"],
            description=row.get("description") or None,
            metadata=json.loads(row.get("metadata_json") or "{}"),
            created_at=row["created_at"], updated_at=row["updated_at"],
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
                "scope": scope, "name": req.name, "physical_table": _physical_table(scope, req.name),
                "status": "active", "description": req.description or "",
                "metadata_json": json.dumps(req.metadata, sort_keys=True), "created_at": now,
                "updated_at": now, "expires_at": req.expires_at or "",
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
        rows = sorted(self._rows(_COLLECTIONS_TABLE, f"scope = {_quoted(scope)}"), key=lambda r: r["name"])
        start = _offset(token)
        page = rows[start : start + limit]
        next_token = _token(start + limit) if start + limit < len(rows) else None
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

    def delete_collection(self, scope: str, name: str, if_exists: bool) -> CollectionInfo | None:
        with self._write_lock:
            row = self._collection_row(scope, name)
            if not row:
                if if_exists:
                    return None
                raise HTTPException(404, "Collection not found")
            row["status"] = "deleting"
            row["updated_at"] = _now()
            (
                self._db.open_table(_COLLECTIONS_TABLE)
                .merge_insert(["scope", "name"])
                .when_matched_update_all()
                .execute([row])
            )
            try:
                self._db.drop_table(row["physical_table"], ignore_missing=True)
                self._db.open_table(_DOCUMENTS_TABLE).delete(
                    f"scope = {_quoted(scope)} AND collection_name = {_quoted(name)}"
                )
                self._db.open_table(_COLLECTIONS_TABLE).delete(
                    f"scope = {_quoted(scope)} AND name = {_quoted(name)}"
                )
                self._collection_tables.pop((scope, name), None)
                self._opened_tables.pop(row["physical_table"], None)
            except Exception:
                logger.exception("Collection deletion is incomplete and may be retried")
                raise
            return self._collection_info(row)

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
                    (table.merge_insert("chunk_id").when_matched_update_all().when_not_matched_insert_all()
                     .when_not_matched_by_source_delete(predicate).execute(rows))
                else:
                    table.add(rows)
                logger.info("Wrote %d rows to table '%s' operation=%s", len(rows), table_name, operation)

            if collection_mode and document_id:
                now = _now()
                existing = self._rows(
                    _DOCUMENTS_TABLE,
                    f"scope = {_quoted(scope or 'default')} AND collection_name = {_quoted(collection_name)} "
                    f"AND document_id = {_quoted(document_id)}",
                )
                created_at = existing[0]["created_at"] if existing else now
                version = str(rows[0].get("document_version") or content_sha256 or "1")
                doc = {
                    "scope": scope or "default", "collection_name": collection_name,
                    "document_id": document_id, "job_id": job_id or "", "filename": filename or "",
                    "content_sha256": content_sha256 or "", "document_version": version,
                    "status": "completed", "chunk_count": len(rows), "created_at": created_at,
                    "updated_at": now, "error": "",
                }
                (self._db.open_table(_DOCUMENTS_TABLE).merge_insert(["scope", "collection_name", "document_id"])
                 .when_matched_update_all().when_not_matched_insert_all().execute([doc]))

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
        self, vectors: list[list[float]], top_k: int, *, scope: str | None = None,
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
        rows = sorted(self._rows(_DOCUMENTS_TABLE, where), key=lambda r: (r["created_at"], r["document_id"]))
        start = _offset(token)
        page = rows[start : start + limit]
        return DocumentPage(
            items=[self._document_info(r) for r in page],
            next_token=_token(start + limit) if start + limit < len(rows) else None,
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

    def delete_document(self, scope: str, collection: str, document_id: str, if_exists: bool) -> DocumentDeleteResult:
        table_name = self._resolved_table(scope, collection)
        try:
            self.get_document(scope, collection, document_id)
        except HTTPException:
            if if_exists:
                return DocumentDeleteResult(document_id=document_id, collection_name=collection, deleted=False)
            raise
        with self._write_lock:
            if self._has_table(table_name):
                self._open_table(table_name).delete(f"document_id = {_quoted(document_id)}")
            self._db.open_table(_DOCUMENTS_TABLE).delete(
                f"scope = {_quoted(scope)} AND collection_name = {_quoted(collection)} "
                f"AND document_id = {_quoted(document_id)}"
            )
        return DocumentDeleteResult(document_id=document_id, collection_name=collection, deleted=True)

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
) -> FastAPI:
    """Build the VectorDB FastAPI application."""

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
        yield
        _state = None
        _query_semaphore = None
        logger.info("VectorDB service stopped")

    app = FastAPI(
        title="NeMo Retriever VectorDB",
        description="LanceDB-backed vector storage and retrieval",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/v1/health", tags=["system"])
    async def health() -> dict[str, Any]:
        rows = _state.total_rows() if _state else 0
        return {
            "status": "ok",
            "table": table_name,
            "total_rows": rows,
            "table_exists": _state.table_exists if _state else False,
            "embed_mode": _state.embed_mode if _state else "none",
        }

    @app.post("/internal/vectordb/write", response_model=WriteResponse, tags=["internal"])
    async def write(req: WriteRequest) -> WriteResponse:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        written = await asyncio.to_thread(
            _state.write_rows, req.rows, scope=req.scope, collection_name=req.collection_name,
            document_id=req.document_id, job_id=req.job_id, filename=req.filename,
            content_sha256=req.content_sha256, operation=req.operation,
        )
        return WriteResponse(
            written=written,
            total_rows=_state.total_rows(scope=req.scope, collection_name=req.collection_name),
        )

    def _scope(value: str | None) -> str:
        return (value or "default").strip() or "default"

    @app.post("/v1/collections", response_model=CollectionInfo, status_code=201, tags=["collections"])
    async def create_collection(req: CollectionCreateRequest, x_nrl_scope: str | None = Header(None)) -> CollectionInfo:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return await asyncio.to_thread(_state.create_collection, _scope(x_nrl_scope), req)

    @app.get("/v1/collections", response_model=CollectionPage, tags=["collections"])
    async def list_collections(
        limit: int = Query(100, ge=1, le=200), continuation_token: str | None = None,
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
        name: str, req: CollectionUpdateRequest, x_nrl_scope: str | None = Header(None),
    ) -> CollectionInfo:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return await asyncio.to_thread(_state.update_collection, _scope(x_nrl_scope), name, req)

    @app.delete("/v1/collections/{name}", response_model=CollectionInfo | None, tags=["collections"])
    async def delete_collection(
        name: str, if_exists: bool = False, x_nrl_scope: str | None = Header(None),
    ) -> CollectionInfo | None:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return await asyncio.to_thread(_state.delete_collection, _scope(x_nrl_scope), name, if_exists)

    @app.get("/v1/collections/{name}/documents", response_model=DocumentPage, tags=["collections"])
    async def list_documents(
        name: str, limit: int = Query(100, ge=1, le=200), continuation_token: str | None = None,
        x_nrl_scope: str | None = Header(None),
    ) -> DocumentPage:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return await asyncio.to_thread(_state.list_documents, _scope(x_nrl_scope), name, limit, continuation_token)

    @app.get("/v1/collections/{name}/documents/{document_id}", response_model=DocumentInfo, tags=["collections"])
    async def get_document(name: str, document_id: str, x_nrl_scope: str | None = Header(None)) -> DocumentInfo:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return await asyncio.to_thread(_state.get_document, _scope(x_nrl_scope), name, document_id)

    @app.delete(
        "/v1/collections/{name}/documents/{document_id}", response_model=DocumentDeleteResult,
        tags=["collections"],
    )
    async def delete_document(
        name: str, document_id: str, if_exists: bool = False, x_nrl_scope: str | None = Header(None),
    ) -> DocumentDeleteResult:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        return await asyncio.to_thread(
            _state.delete_document, _scope(x_nrl_scope), name, document_id, if_exists,
        )

    @app.post("/v1/query", response_model=Union[QueryResponse, EvidenceQueryResponse], tags=["query"])
    async def query(
        req: QueryRequest, x_nrl_scope: str | None = Header(None),
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
                _state.search, vectors, req.top_k, scope=_scope(x_nrl_scope),
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
    parser = argparse.ArgumentParser(description="NeMo Retriever VectorDB service")
    parser.add_argument("--lancedb-uri", default="/data/vectordb", help="LanceDB directory")
    parser.add_argument("--table-name", default="nemo_retriever", help="LanceDB table name")
    parser.add_argument("--embed-endpoint", default="", help="Remote NIM/OpenAI-compatible embed URL")
    parser.add_argument("--embed-model", default="nvidia/llama-nemotron-embed-vl-1b-v2")
    parser.add_argument("--embed-model-provider-prefix", default="", help="Optional LiteLLM provider prefix")
    parser.add_argument("--embed-api-key", default="")
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
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
