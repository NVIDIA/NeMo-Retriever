# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POST /v1/query — embed user queries via the NIM and search LanceDB."""

from __future__ import annotations

import ast
import asyncio
import logging
import traceback
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from nemo_retriever.service.config import ServiceConfig, VectorStoreConfig
from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_logger import record_event
from nemo_retriever.service.failure_types import EventCategory
from nemo_retriever.service.models.event_log import EventOutcome, EventSeverity
from nemo_retriever.service.models.requests import QueryRequest
from nemo_retriever.service.models.responses import QueryHit, QueryResponse, QueryResultSet

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

_embed_rr_idx = 0


def _pick_endpoint(csv_url: str) -> str:
    """Round-robin select one URL from a comma-separated NIM endpoint string."""
    global _embed_rr_idx
    urls = [u.strip() for u in csv_url.split(",") if u.strip()]
    if not urls:
        return csv_url
    if len(urls) == 1:
        return urls[0]
    chosen = urls[_embed_rr_idx % len(urls)]
    _embed_rr_idx += 1
    return chosen


_KEEP_KEYS = frozenset(
    {
        "text",
        "metadata",
        "source",
        "page_number",
        "pdf_page",
        "pdf_basename",
        "source_id",
        "path",
        "stored_image_uri",
        "content_type",
        "bbox_xyxy_norm",
    }
)


# ------------------------------------------------------------------
# Embedding helper
# ------------------------------------------------------------------


def _embed_queries(
    query_texts: list[str],
    *,
    endpoint: str,
    model: str,
    api_key: str,
) -> list[list[float]]:
    """Call the embedding NIM endpoint for one or more query strings.

    Returns a list of embedding vectors (one per input text).
    Raises with a descriptive message on any failure so the caller can
    surface it to the client.
    """
    from nv_ingest_api.util.nim import infer_microservice

    try:
        embeddings = infer_microservice(
            query_texts,
            model_name=model,
            embedding_endpoint=endpoint,
            nvidia_api_key=api_key,
            input_type="query",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Embedding NIM request failed (endpoint={endpoint!r}, model={model!r}): " f"{type(exc).__name__}: {exc}"
        ) from exc

    if not embeddings or len(embeddings) == 0:
        raise RuntimeError(
            f"Embedding NIM returned an empty response (endpoint={endpoint!r}, model={model!r}). "
            "Verify the NIM is healthy and the model name is correct."
        )

    if len(embeddings) != len(query_texts):
        raise RuntimeError(
            f"Embedding NIM returned {len(embeddings)} vectors for {len(query_texts)} "
            f"input texts (endpoint={endpoint!r}, model={model!r}). "
            "The model may have dropped some inputs."
        )

    out: list[list[float]] = []
    for vector in embeddings:
        if isinstance(vector, np.ndarray):
            out.append(vector.astype("float32").tolist())
        else:
            out.append(list(vector))
    return out


# ------------------------------------------------------------------
# LanceDB search helper
# ------------------------------------------------------------------


def _search_lancedb(
    *,
    lancedb_uri: str,
    lancedb_table: str,
    query_vectors: list[list[float]],
    query_texts: list[str],
    top_k: int,
    vector_column_name: str,
    nprobes: int,
    refine_factor: int,
    hybrid: bool,
) -> list[list[dict[str, Any]]]:
    """Run a vector (or hybrid) search against a LanceDB table.

    Returns a list of hit-lists, one per input query.  Raises with a
    descriptive message on any failure.
    """
    import lancedb as lancedb_lib  # type: ignore

    try:
        db = lancedb_lib.connect(lancedb_uri)
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to LanceDB at {lancedb_uri!r}: " f"{type(exc).__name__}: {exc}") from exc

    try:
        table_names = db.table_names()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to list tables in LanceDB at {lancedb_uri!r}: " f"{type(exc).__name__}: {exc}"
        ) from exc

    if lancedb_table not in table_names:
        available = ", ".join(sorted(table_names)) if table_names else "<none>"
        raise RuntimeError(
            f"Table {lancedb_table!r} does not exist in LanceDB at {lancedb_uri!r}. "
            f"Available tables: {available}. "
            "Has the ingestion pipeline been run with embedding enabled?"
        )

    try:
        table = db.open_table(lancedb_table)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open table {lancedb_table!r} in LanceDB: " f"{type(exc).__name__}: {exc}"
        ) from exc

    table_columns = {f.name for f in table.schema}

    if vector_column_name not in table_columns:
        raise RuntimeError(
            f"Vector column {vector_column_name!r} not found in table {lancedb_table!r}. "
            f"Available columns: {', '.join(sorted(table_columns))}."
        )

    select_cols = [
        col
        for col in (
            "text",
            "metadata",
            "source",
            "page_number",
            "_distance",
            "pdf_page",
            "pdf_basename",
            "source_id",
            "path",
            "stored_image_uri",
            "content_type",
            "bbox_xyxy_norm",
        )
        if col in table_columns or col == "_distance"
    ]

    effective_nprobes = nprobes
    if effective_nprobes <= 0:
        try:
            for idx in table.list_indices():
                num_parts = getattr(idx, "num_partitions", None)
                if num_parts and int(num_parts) > 0:
                    effective_nprobes = int(num_parts)
                    break
        except Exception:
            pass
        if effective_nprobes <= 0:
            effective_nprobes = 16

    results: list[list[dict[str, Any]]] = []
    for i, vector in enumerate(query_vectors):
        try:
            q = np.asarray(vector, dtype="float32")

            if hybrid:
                from lancedb.rerankers import RRFReranker  # type: ignore

                hits = (
                    table.search(query_type="hybrid")
                    .vector(q)
                    .text(query_texts[i])
                    .nprobes(effective_nprobes)
                    .refine_factor(int(refine_factor))
                    .limit(int(top_k))
                    .rerank(RRFReranker())
                    .to_list()
                )
            else:
                hits = (
                    table.search(q, vector_column_name=vector_column_name)
                    .nprobes(effective_nprobes)
                    .refine_factor(int(refine_factor))
                    .select(select_cols)
                    .limit(int(top_k))
                    .to_list()
                )

            results.append([{k: v for k, v in h.items() if k in _KEEP_KEYS or k == "_distance"} for h in hits])
        except Exception as exc:
            raise RuntimeError(
                f"Vector search failed on table {lancedb_table!r} for query #{i + 1}: " f"{type(exc).__name__}: {exc}"
            ) from exc

    return results


# ------------------------------------------------------------------
# Hit conversion helper
# ------------------------------------------------------------------


def _hit_to_query_hit(hit: dict[str, Any], *, distance: float) -> QueryHit:
    """Convert a raw LanceDB hit dict into a typed :class:`QueryHit`."""
    metadata_raw = hit.get("metadata", "")
    if isinstance(metadata_raw, str):
        try:
            metadata_dict = ast.literal_eval(metadata_raw) if metadata_raw else {}
        except Exception:
            metadata_dict = {"_raw": metadata_raw} if metadata_raw else {}
    elif isinstance(metadata_raw, dict):
        metadata_dict = metadata_raw
    else:
        metadata_dict = {}

    return QueryHit(
        text=str(hit.get("text", "")),
        source=str(hit.get("source", "")),
        page_number=int(hit.get("page_number", 0)),
        pdf_page=str(hit.get("pdf_page", "")),
        pdf_basename=str(hit.get("pdf_basename", "")),
        source_id=str(hit.get("source_id", "")),
        path=str(hit.get("path", "")),
        stored_image_uri=str(hit.get("stored_image_uri", "")),
        content_type=str(hit.get("content_type", "")),
        bbox_xyxy_norm=str(hit.get("bbox_xyxy_norm", "")),
        score=float(distance),
        metadata=metadata_dict,
    )


# ------------------------------------------------------------------
# Route
# ------------------------------------------------------------------


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Embed text queries and search the vector store",
    responses={
        400: {
            "description": "Invalid request (empty query, bad parameters).",
        },
        503: {
            "description": "Embedding NIM is unreachable or LanceDB is unavailable.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Embedding NIM request failed: ConnectionError: ...",
                    }
                }
            },
        },
    },
)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    config: ServiceConfig = request.app.state.config
    vs: VectorStoreConfig = config.vector_store

    # --- Validate embed endpoint ---
    embed_raw = (config.nim_endpoints.embed_invoke_url or "").strip()
    embed_endpoint = _pick_endpoint(embed_raw) if embed_raw else ""
    if not embed_endpoint:
        logger.error("Query rejected: nim_endpoints.embed_invoke_url is not configured")
        raise HTTPException(
            status_code=503,
            detail=(
                "The embedding NIM endpoint is not configured on this server. "
                "Set nim_endpoints.embed_invoke_url in retriever-service.yaml "
                "or deploy with nims.embed.enabled=true."
            ),
        )

    # --- Normalize inputs ---
    if isinstance(body.query, str):
        query_texts = [body.query.strip()]
    else:
        query_texts = [q.strip() for q in body.query]

    if not query_texts or all(not q for q in query_texts):
        raise HTTPException(status_code=400, detail="At least one non-empty query string is required.")

    empty_indices = [i for i, q in enumerate(query_texts) if not q]
    if empty_indices:
        raise HTTPException(
            status_code=400,
            detail=f"Query strings at indices {empty_indices} are empty or whitespace-only.",
        )

    lancedb_uri = body.lancedb_uri or vs.lancedb_uri
    lancedb_table = body.lancedb_table or vs.lancedb_table
    top_k = body.top_k or vs.top_k
    embedding_model = vs.embedding_model
    api_key = (config.nim_endpoints.api_key or "").strip()

    logger.info(
        "Query request: %d queries, top_k=%d, table=%s, uri=%s, hybrid=%s",
        len(query_texts),
        top_k,
        lancedb_table,
        lancedb_uri,
        body.hybrid,
    )

    repo: Repository = request.app.state.repository
    req_id = getattr(request.state, "request_id", "")

    # --- Step 1: embed all queries via the NIM ---
    try:
        query_vectors = await asyncio.to_thread(
            _embed_queries,
            query_texts,
            endpoint=embed_endpoint,
            model=embedding_model,
            api_key=api_key,
        )
    except RuntimeError as exc:
        logger.error("Embedding failed: %s", exc)
        record_event(
            repo,
            category=EventCategory.EMBED.value,
            severity=EventSeverity.ERROR,
            outcome=EventOutcome.FAILED,
            summary=f"Embedding NIM failure: {exc}",
            detail=str(exc),
            stack_trace=traceback.format_exc(),
            stage="embed",
            endpoint="/v1/query",
            request_id=req_id,
        )
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Unexpected error during embedding: %s\n%s", exc, tb)
        record_event(
            repo,
            category=EventCategory.EMBED.value,
            severity=EventSeverity.ERROR,
            outcome=EventOutcome.FAILED,
            summary=f"Unexpected embedding error: {type(exc).__name__}: {exc}",
            detail=str(exc),
            stack_trace=tb,
            stage="embed",
            endpoint="/v1/query",
            request_id=req_id,
        )
        raise HTTPException(
            status_code=503,
            detail=(f"Unexpected error while embedding the queries: " f"{type(exc).__name__}: {exc}"),
        )

    if not query_vectors or any(not v for v in query_vectors):
        logger.error("Embedding NIM returned one or more zero-length vectors")
        raise HTTPException(
            status_code=503,
            detail="Embedding NIM returned one or more zero-length vectors. The model may not be loaded.",
        )

    logger.debug("Queries embedded: count=%d, dim=%d", len(query_vectors), len(query_vectors[0]))

    # --- Step 2: search LanceDB ---
    try:
        all_hits = await asyncio.to_thread(
            _search_lancedb,
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
            query_vectors=query_vectors,
            query_texts=query_texts,
            top_k=top_k,
            vector_column_name=vs.vector_column_name,
            nprobes=vs.nprobes,
            refine_factor=vs.refine_factor,
            hybrid=body.hybrid,
        )
    except RuntimeError as exc:
        logger.error("LanceDB search failed: %s", exc)
        record_event(
            repo,
            category=EventCategory.LANCEDB.value,
            severity=EventSeverity.ERROR,
            outcome=EventOutcome.FAILED,
            summary=f"LanceDB search failure: {exc}",
            detail=str(exc),
            stack_trace=traceback.format_exc(),
            stage="lancedb_search",
            endpoint="/v1/query",
            request_id=req_id,
        )
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Unexpected error during LanceDB search: %s\n%s", exc, tb)
        record_event(
            repo,
            category=EventCategory.LANCEDB.value,
            severity=EventSeverity.ERROR,
            outcome=EventOutcome.FAILED,
            summary=f"Unexpected LanceDB error: {type(exc).__name__}: {exc}",
            detail=str(exc),
            stack_trace=tb,
            stage="lancedb_search",
            endpoint="/v1/query",
            request_id=req_id,
        )
        raise HTTPException(
            status_code=503,
            detail=(f"Unexpected error while searching the vector store: " f"{type(exc).__name__}: {exc}"),
        )

    # --- Step 3: build per-query result sets ---
    result_sets: list[QueryResultSet] = []
    for i, (query_text, hits) in enumerate(zip(query_texts, all_hits)):
        typed_hits = [_hit_to_query_hit(h, distance=h.get("_distance", 0.0)) for h in hits]
        result_sets.append(
            QueryResultSet(
                query=query_text,
                hits=typed_hits,
                total_hits=len(typed_hits),
            )
        )

    total_hits = sum(rs.total_hits for rs in result_sets)
    logger.info(
        "Query complete: %d queries, %d total hits",
        len(query_texts),
        total_hits,
    )

    return QueryResponse(
        results=result_sets,
        total_queries=len(query_texts),
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
        embedding_model=embedding_model,
    )
