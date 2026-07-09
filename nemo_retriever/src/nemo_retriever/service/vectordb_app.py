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
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Union

import lancedb
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nemo_retriever.common.vdb.lancedb_capabilities import LanceRetrievalMode, inspect_lancedb_table_object
from nemo_retriever.query.evidence import build_evidence_result
from nemo_retriever.service.config import ServiceVectorRetrievalMode
from nemo_retriever.service.query_schema import (
    EvidenceQueryResponse,
    EvidenceResult,
    QueryRequest,
    QueryResponse,
    QueryResult,
)

logger = logging.getLogger(__name__)

# ── Request / response models ────────────────────────────────────────


class WriteRequest(BaseModel):
    rows: list[dict[str, Any]]


class WriteResponse(BaseModel):
    written: int
    total_rows: int


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


def _strategies_for_retrieval_mode(mode: LanceRetrievalMode | str) -> list[str]:
    if mode == "hybrid":
        return ["hybrid"]
    return ["dense"]


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
        retrieval_mode: ServiceVectorRetrievalMode = "dense",
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
        self.retrieval_mode = retrieval_mode
        self._write_lock = threading.Lock()
        self._embed_lock = threading.Lock()
        self._local_embedder: Any | None = None
        self._db = lancedb.connect(uri=lancedb_uri)
        self._table_exists = False
        try:
            self._db.open_table(table_name)
            self._table_exists = True
            logger.info("Opened existing LanceDB table '%s' at %s", table_name, lancedb_uri)
            if self.retrieval_mode == "hybrid":
                with self._write_lock:
                    self._ensure_hybrid_indexes()
        except Exception:
            logger.info("LanceDB table '%s' does not exist yet at %s", table_name, lancedb_uri)

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

    def _table_capabilities(self):
        if not self._table_exists:
            return None
        table = self._db.open_table(self.table_name)
        return inspect_lancedb_table_object(table)

    def resolve_effective_retrieval_mode(self, caps=None) -> LanceRetrievalMode:
        """Resolve the configured retrieval mode against table capabilities.

        Callers that already hold a capabilities object (e.g. ``search``) may
        pass it in to avoid re-opening the LanceDB table.
        """
        if not self._table_exists:
            return "dense"

        if caps is None:
            caps = self._table_capabilities()
        if caps is None:
            raise ValueError(
                f"Unable to inspect LanceDB table {self.table_name!r} at {self.lancedb_uri!r}: "
                "capabilities could not be determined."
            )

        if self.retrieval_mode == "auto":
            mode: LanceRetrievalMode = caps.retrieval_mode
        else:
            mode = self.retrieval_mode

        if mode == "unknown":
            raise ValueError(
                f"LanceDB table {self.table_name!r} at {self.lancedb_uri!r} is not queryable: "
                "no vector column or FTS index was detected."
            )
        if mode == "dense" and not caps.has_vector:
            raise ValueError(
                f"LanceDB table {self.table_name!r} at {self.lancedb_uri!r} cannot run dense retrieval: "
                "no vector column was detected."
            )
        if mode == "hybrid" and (not caps.has_vector or not caps.has_fts):
            raise ValueError(
                f"LanceDB table {self.table_name!r} at {self.lancedb_uri!r} cannot run hybrid retrieval: "
                "both a vector column and FTS index are required."
            )
        if mode == "sparse":
            raise ValueError("Sparse retrieval is not supported by the VectorDB service. Use dense, hybrid, or auto.")
        return mode

    def _ensure_hybrid_indexes(self) -> None:
        """Build FTS indexes when hybrid retrieval is configured or auto-selected.

        Must be called while holding ``_write_lock``. This does not gate on
        ``self._table_exists`` so it can run against a freshly created table
        before that flag is published to concurrent readers.
        """
        if self.retrieval_mode == "dense":
            return

        try:
            table = self._db.open_table(self.table_name)
        except Exception:
            return
        caps = inspect_lancedb_table_object(table)
        if not caps.has_vector or caps.has_fts:
            return

        from nemo_retriever.common.vdb.lancedb import LanceDB

        vdb = LanceDB(uri=self.lancedb_uri, table_name=self.table_name, overwrite=False, hybrid=True)
        vdb.write_to_index(records=None, table=table, hybrid=True)
        logger.info(
            "Built FTS index on LanceDB table '%s' at %s for hybrid retrieval",
            self.table_name,
            self.lancedb_uri,
        )

    def write_rows(self, rows: list[dict[str, Any]]) -> int:
        """Append rows to the LanceDB table (creates table on first write)."""
        if not rows:
            return 0

        from nemo_retriever.common.vdb.lancedb_schema import (
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
                # Build the FTS index before publishing `_table_exists` so a
                # concurrent hybrid query never observes a table that is missing
                # its FTS index (which would otherwise fail with a 422).
                self._ensure_hybrid_indexes()
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
                self._ensure_hybrid_indexes()
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

    def search(
        self,
        vectors: list[list[float]],
        query_texts: list[str],
        top_k: int,
    ) -> tuple[list[list[dict[str, Any]]], list[str]]:
        """Search the LanceDB table with precomputed query vectors."""
        if not self._table_exists:
            return [[] for _ in vectors], _strategies_for_retrieval_mode("dense")

        caps = self._table_capabilities()
        mode = self.resolve_effective_retrieval_mode(caps)
        strategies = _strategies_for_retrieval_mode(mode)

        from nemo_retriever.common.vdb.lancedb import LanceDB
        from nemo_retriever.common.vdb.records import normalize_retrieval_results

        hybrid = mode == "hybrid"
        vdb = LanceDB(uri=self.lancedb_uri, table_name=self.table_name, overwrite=False, hybrid=hybrid)
        retrieval_kwargs: dict[str, Any] = {"top_k": top_k, "hybrid": hybrid}
        if hybrid:
            retrieval_kwargs["query_texts"] = query_texts

        if caps is not None and caps.vector_column and caps.vector_column != "vector":
            retrieval_kwargs["vector_column_name"] = caps.vector_column

        raw_results = vdb.retrieval(vectors, **retrieval_kwargs)
        return normalize_retrieval_results(raw_results), strategies

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
    retrieval_mode: ServiceVectorRetrievalMode = "dense",
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
            retrieval_mode=retrieval_mode,
        )
        _query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
        logger.info(
            "VectorDB service started: uri=%s table=%s embed_mode=%s retrieval_mode=%s max_concurrent_queries=%d",
            lancedb_uri,
            table_name,
            _state.embed_mode,
            retrieval_mode,
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
        effective_mode: str | None = None
        if _state is not None and _state.table_exists:
            try:
                effective_mode = _state.resolve_effective_retrieval_mode()
            except ValueError:
                effective_mode = "unknown"
        return {
            "status": "ok",
            "table": table_name,
            "total_rows": rows,
            "table_exists": _state.table_exists if _state else False,
            "embed_mode": _state.embed_mode if _state else "none",
            "retrieval_mode": _state.retrieval_mode if _state else retrieval_mode,
            "effective_retrieval_mode": effective_mode,
        }

    @app.post("/internal/vectordb/write", response_model=WriteResponse, tags=["internal"])
    async def write(req: WriteRequest) -> WriteResponse:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")
        written = await asyncio.to_thread(_state.write_rows, req.rows)
        return WriteResponse(written=written, total_rows=_state.total_rows())

    @app.post("/v1/query", response_model=Union[QueryResponse, EvidenceQueryResponse], tags=["query"])
    async def query(req: QueryRequest) -> QueryResponse | EvidenceQueryResponse:
        if _state is None:
            raise HTTPException(503, "VectorDB not initialised")

        if _state.embed_mode == "none":
            raise HTTPException(
                501,
                "No embedding backend configured. Set --embed-endpoint for a remote "
                "NIM or --local-embed for in-pod Hugging Face query embedding.",
            )

        if not _state.table_exists:
            raise HTTPException(
                status_code=422,
                detail="No data has been ingested yet. Ingest documents first, then query.",
            )

        queries = req.query if isinstance(req.query, list) else [req.query]
        if not queries:
            if req.format == "evidence":
                return EvidenceQueryResponse(results=[])
            return QueryResponse(results=[])

        try:
            async with _query_semaphore:
                vectors = await asyncio.to_thread(_state.embed_queries, queries)
                hits_per_query, strategies = await asyncio.to_thread(
                    _state.search,
                    vectors,
                    queries,
                    req.top_k,
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if req.format == "evidence":
            return EvidenceQueryResponse(
                results=[EvidenceResult(**build_evidence_result(hits, strategies)) for hits in hits_per_query]
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
    parser.add_argument(
        "--retrieval-mode",
        default="dense",
        choices=("dense", "hybrid", "auto"),
        help="LanceDB retrieval mode for /v1/query (default: dense).",
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
        retrieval_mode=args.retrieval_mode,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
