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


#: Annotations the agentic workflow layers on top of the classic hit fields.
_AGENTIC_ANNOTATION_FIELDS = frozenset({"doc_id", "rank", "result_source"})

#: Classic fields reported as null when no retrieve hop captured the selected
#: document, so the envelope keeps a stable key set either way.
_UNRESOLVED_HIT_FIELDS = ("text", "metadata", "source_id", "path", "page_number", "pdf_basename", "pdf_page")


def agentic_ranked_to_hits(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map agentic ranked hits onto the ``/v1/query`` hits envelope.

    Agentic retrieval selects documents (``doc_id``), not chunks, but each hit
    carries the classic chunk-level fields (``text``, ``metadata``, ``source``,
    ``page_number``, scores, …) rehydrated from the retrieve hop that returned the
    document, plus ``doc_id`` / ``rank`` / ``result_source``. When no hop captured
    the document, those fields are null and ``source`` falls back to ``doc_id``.

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
        hit = {key: value for key, value in item.items() if key not in _AGENTIC_ANNOTATION_FIELDS}
        if not hit:
            hit = {field: None for field in _UNRESOLVED_HIT_FIELDS}
        if not str(hit.get("source") or "").strip():
            hit["source"] = doc_id
        hit.update(
            {
                "doc_id": doc_id,
                "rank": item.get("rank"),
                "result_source": item.get("result_source"),
            }
        )
        hits.append(hit)
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
    return QueryResponse(
        results=[QueryResult(hits=agentic_ranked_to_hits(ranked))],
        query_mode="agentic",
    )
