# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service boundary for in-process agentic retrieval over the VectorDB query API."""

from __future__ import annotations

from typing import Any

from nemo_retriever.query.options import (
    QueryAgenticOptions,
    QueryEmbedOptions,
    QueryRequest as WorkflowQueryRequest,
    QueryRetrievalOptions,
    QueryStorageOptions,
)
from nemo_retriever.query.workflow import agentic_query_documents
from nemo_retriever.service.config import AgenticConfig
from nemo_retriever.service.query_schema import QueryResponse, QueryResult


def agentic_ranked_to_hits(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map agentic document ranks onto the ``/v1/query`` hits envelope.

    Agentic retrieval selects documents (``doc_id``), not chunks. Each hit places
    ``doc_id`` in ``source``, records ``result_source`` / ``rank`` under
    ``metadata``, and leaves chunk-level fields (``text``, ``page_number``,
    scores, …) unset.

    Rows without a non-empty ``doc_id`` are a contract violation: the agentic
    workflow already skips blank ids on retrieve hops, and a hit with
    ``source=null`` is useless to clients. Raise rather than emit a null source.
    """
    hits: list[dict[str, Any]] = []
    for item in ranked:
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError(
                "agentic ranked result is missing a non-empty doc_id "
                f"(rank={item.get('rank')!r}, result_source={item.get('result_source')!r})"
            )
        hits.append(
            {
                "text": None,
                "metadata": {
                    "result_source": item.get("result_source"),
                    "rank": item.get("rank"),
                },
                "source": doc_id,
                "source_id": None,
                "path": None,
                "page_number": None,
                "pdf_basename": None,
                "pdf_page": None,
            }
        )
    return hits


def build_agentic_query_request(
    *,
    query: str,
    top_k: int,
    config: AgenticConfig,
    lancedb_uri: str,
    table_name: str,
    embed_endpoint: str,
    embed_model: str,
    embed_model_provider_prefix: str | None,
    embed_api_key: str,
) -> WorkflowQueryRequest:
    """Map server-owned service settings onto the shared agentic query request."""
    return WorkflowQueryRequest(
        query=query,
        retrieval=QueryRetrievalOptions(top_k=top_k),
        embed=QueryEmbedOptions(
            embed_invoke_url=embed_endpoint or None,
            embed_model_name=embed_model or None,
            embed_model_provider_prefix=embed_model_provider_prefix,
            embed_api_key=embed_api_key or None,
        ),
        storage=QueryStorageOptions(
            lancedb_uri=lancedb_uri,
            table_name=table_name,
        ),
        agentic=QueryAgenticOptions(
            enabled=True,
            llm_model=config.llm_model,
            invoke_url=config.invoke_url,
            api_key=config.api_key,
            reasoning_effort=config.reasoning_effort,
            backend_top_k=config.backend_top_k,
            react_max_steps=config.react_max_steps,
            text_truncation=config.text_truncation,
            temperature=config.temperature,
        ),
    )


def run_agentic_query(
    *,
    query: str,
    top_k: int,
    config: AgenticConfig,
    lancedb_uri: str,
    table_name: str,
    embed_endpoint: str,
    embed_model: str,
    embed_model_provider_prefix: str | None,
    embed_api_key: str,
) -> QueryResponse:
    """Execute one agentic retrieval query and return ``QueryResponse`` hits."""
    query_request = build_agentic_query_request(
        query=query,
        top_k=top_k,
        config=config,
        lancedb_uri=lancedb_uri,
        table_name=table_name,
        embed_endpoint=embed_endpoint,
        embed_model=embed_model,
        embed_model_provider_prefix=embed_model_provider_prefix,
        embed_api_key=embed_api_key,
    )
    ranked = agentic_query_documents(query_request)
    return QueryResponse(results=[QueryResult(hits=agentic_ranked_to_hits(ranked))])
