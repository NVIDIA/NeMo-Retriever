# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone VectorDB microservice backed by LanceDB.

Provides three endpoints:

- ``POST /internal/vectordb/write`` -- append embedding rows from ingest workers
- ``POST /v1/query``               -- embed query text and search the index
- ``GET  /v1/health``              -- liveness probe

Run standalone::

    python -m nemo_retriever.service.vectordb_app \\
        --lancedb-uri /data/vectordb \\
        --embed-endpoint http://nemo-retriever-nim-embed-0...:8000/v1/embeddings \\
        --port 7671
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import lancedb
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Request / response models ────────────────────────────────────────


class WriteRequest(BaseModel):
    rows: list[dict[str, Any]]


class WriteResponse(BaseModel):
    written: int
    total_rows: int


class QueryRequest(BaseModel):
    query: str | list[str]
    top_k: int = Field(default=10, ge=1, le=1000)


class QueryResponse(BaseModel):
    results: list[dict[str, Any]]


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
    ) -> None:
        self.lancedb_uri = lancedb_uri
        self.table_name = table_name
        self.embed_endpoint = embed_endpoint
        self.embed_model = embed_model
        self.embed_api_key = embed_api_key
        self._write_lock = threading.Lock()
        self._db = lancedb.connect(uri=lancedb_uri)
        self._table_exists = False
        try:
            self._db.open_table(table_name)
            self._table_exists = True
            logger.info("Opened existing LanceDB table '%s' at %s", table_name, lancedb_uri)
        except Exception:
            logger.info("LanceDB table '%s' does not exist yet at %s", table_name, lancedb_uri)

    @property
    def table_exists(self) -> bool:
        return self._table_exists

    def write_rows(self, rows: list[dict[str, Any]]) -> int:
        """Append rows to the LanceDB table (creates table on first write)."""
        if not rows:
            return 0

        from nemo_retriever.vdb.lancedb_schema import (
            create_or_append_lancedb_table,
            infer_vector_dim,
            lancedb_schema,
        )

        with self._write_lock:
            if not self._table_exists:
                dim = infer_vector_dim(rows)
                if dim == 0:
                    logger.warning("Cannot infer vector dimension from rows; skipping write")
                    return 0
                schema = lancedb_schema(vector_dim=dim)
                create_or_append_lancedb_table(
                    self._db,
                    self.table_name,
                    rows,
                    schema,
                    overwrite=True,
                )
                self._table_exists = True
                logger.info(
                    "Created LanceDB table '%s' with %d rows (dim=%d)",
                    self.table_name,
                    len(rows),
                    dim,
                )
            else:
                table = self._db.open_table(self.table_name)
                table.add(rows)
                logger.info("Appended %d rows to table '%s'", len(rows), self.table_name)

        return len(rows)

    def total_rows(self) -> int:
        if not self._table_exists:
            return 0
        try:
            table = self._db.open_table(self.table_name)
            return table.count_rows()
        except Exception:
            return 0

    def search(self, vectors: list[list[float]], top_k: int) -> list[list[dict[str, Any]]]:
        """Search the LanceDB table with precomputed query vectors."""
        if not self._table_exists:
            return [[] for _ in vectors]

        from nemo_retriever.vdb.records import normalize_retrieval_results

        table = self._db.open_table(self.table_name)
        raw_results = []
        for vector in vectors:
            results = table.search(vector).limit(top_k).to_list()
            raw_results.append(results)

        return normalize_retrieval_results(raw_results)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed query texts using the configured NIM endpoint."""
        from nemo_retriever.api.util.nim import infer_microservice

        embeddings = infer_microservice(
            texts,
            model_name=self.embed_model,
            embedding_endpoint=self.embed_endpoint,
            nvidia_api_key=self.embed_api_key or None,
            input_type="query",
            grpc=False,
        )
        return embeddings


# ── FastAPI app ──────────────────────────────────────────────────────

_state: VectorDBState | None = None
_query_semaphore: asyncio.Semaphore | None = None

MAX_CONCURRENT_QUERIES = 4


def create_vectordb_app(
    lancedb_uri: str = "/data/vectordb",
    table_name: str = "nemo_retriever",
    embed_endpoint: str = "",
    embed_model: str = "nvidia/llama-nemotron-embed-1b-v2",
    embed_api_key: str = "",
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
            embed_api_key=embed_api_key,
        )
        _query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
        logger.info(
            "VectorDB service started: uri=%s table=%s embed=%s max_concurrent_queries=%d",
            lancedb_uri,
            table_name,
            embed_endpoint or "(none)",
            MAX_CONCURRENT_QUERIES,
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
        }

    @app.post("/internal/vectordb/write", response_model=WriteResponse, tags=["internal"])
    async def write(req: WriteRequest) -> WriteResponse:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        written = await asyncio.to_thread(_state.write_rows, req.rows)
        return WriteResponse(written=written, total_rows=_state.total_rows())

    @app.post("/v1/query", response_model=QueryResponse, tags=["query"])
    async def query(req: QueryRequest) -> QueryResponse:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")

        if not _state.embed_endpoint:
            raise HTTPException(
                501,
                "No embedding endpoint configured. Set embed_invoke_url in the vectordb config.",
            )

        if not _state.table_exists:
            raise HTTPException(
                404,
                "No data has been ingested yet. Ingest documents first, then query.",
            )

        queries = req.query if isinstance(req.query, list) else [req.query]
        if not queries:
            return QueryResponse(results=[])

        async with _query_semaphore:
            vectors = await asyncio.to_thread(_state.embed_queries, queries)
            hits_per_query = await asyncio.to_thread(_state.search, vectors, req.top_k)

        results = []
        for hits in hits_per_query:
            results.append({"hits": hits})

        return QueryResponse(results=results)

    return app


# ── CLI entry point ──────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="NeMo Retriever VectorDB service")
    parser.add_argument("--lancedb-uri", default="/data/vectordb", help="LanceDB directory")
    parser.add_argument("--table-name", default="nemo_retriever", help="LanceDB table name")
    parser.add_argument("--embed-endpoint", default="", help="NIM embed endpoint URL")
    parser.add_argument("--embed-model", default="nvidia/llama-nemotron-embed-1b-v2")
    parser.add_argument("--embed-api-key", default="")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7671)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    app = create_vectordb_app(
        lancedb_uri=args.lancedb_uri,
        table_name=args.table_name,
        embed_endpoint=args.embed_endpoint,
        embed_model=args.embed_model,
        embed_api_key=args.embed_api_key,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
