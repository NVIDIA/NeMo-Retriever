# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service boundary for the existing in-process agentic retrieval workflow."""

from __future__ import annotations

from nemo_retriever.query.options import (
    QueryAgenticOptions,
    QueryEmbedOptions,
    QueryRequest,
    QueryRetrievalOptions,
    QueryStorageOptions,
)
from nemo_retriever.query.workflow import agentic_query_documents
from nemo_retriever.service.config import AgenticConfig
from nemo_retriever.service.query_schema import AgenticQueryRequest, AgenticQueryResponse


def build_agentic_query_request(
    request: AgenticQueryRequest,
    *,
    config: AgenticConfig,
    lancedb_uri: str,
    table_name: str,
    embed_endpoint: str,
    embed_model: str,
    embed_model_provider_prefix: str | None,
    embed_api_key: str,
) -> QueryRequest:
    """Map server-owned service settings onto the shared agentic query request."""
    return QueryRequest(
        query=request.query,
        retrieval=QueryRetrievalOptions(top_k=request.top_k),
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
    request: AgenticQueryRequest,
    *,
    config: AgenticConfig,
    lancedb_uri: str,
    table_name: str,
    embed_endpoint: str,
    embed_model: str,
    embed_model_provider_prefix: str | None,
    embed_api_key: str,
) -> AgenticQueryResponse:
    """Execute one agentic retrieval query without changing agent internals."""
    query_request = build_agentic_query_request(
        request,
        config=config,
        lancedb_uri=lancedb_uri,
        table_name=table_name,
        embed_endpoint=embed_endpoint,
        embed_model=embed_model,
        embed_model_provider_prefix=embed_model_provider_prefix,
        embed_api_key=embed_api_key,
    )
    return AgenticQueryResponse(results=agentic_query_documents(query_request))
