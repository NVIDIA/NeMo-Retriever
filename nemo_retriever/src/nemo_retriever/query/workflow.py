# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_retriever.common.params import build_embed_option_kwargs
from nemo_retriever.query.options import QueryRequest, QueryRerankOptions
from nemo_retriever.graph.retriever import Retriever
from nemo_retriever.common.remote_auth import resolve_remote_api_key
from nemo_retriever.common.vdb.records import RetrievalHit

_LOCAL_VL_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"


def _build_rerank_kwargs(options: QueryRerankOptions) -> dict[str, str]:
    """Build kwargs for the rerank stage using the existing root query behavior."""
    reranker_url = (options.reranker_invoke_url or "").strip()
    if reranker_url:
        rerank_kwargs: dict[str, str] = {"rerank_invoke_url": reranker_url}
        if options.reranker_model_name:
            rerank_kwargs["model_name"] = options.reranker_model_name
        api_key = resolve_remote_api_key(options.reranker_api_key)
        if api_key is not None:
            rerank_kwargs["api_key"] = api_key
        return rerank_kwargs

    local: dict[str, str] = {"model_name": options.reranker_model_name or _LOCAL_VL_RERANK_MODEL}
    if options.reranker_backend:
        local["local_reranker_backend"] = options.reranker_backend
    return local


def _build_retriever_kwargs(request: QueryRequest) -> dict[str, Any]:
    embed_kwargs = build_embed_option_kwargs(request.embed.embed_invoke_url, request.embed.embed_model_name)
    vdb_kwargs: dict[str, Any] = {
        "uri": request.storage.lancedb_uri,
        "table_name": request.storage.table_name,
    }
    # Only inject hybrid when opted in, so the vector-only path stays byte-for-byte legacy.
    if request.retrieval.hybrid:
        vdb_kwargs["hybrid"] = True
    retriever_kwargs: dict[str, Any] = {
        "top_k": request.retrieval.top_k,
        "vdb_kwargs": vdb_kwargs,
    }
    if embed_kwargs:
        retriever_kwargs["embed_kwargs"] = embed_kwargs
    if request.rerank.enabled:
        rerank_kwargs = _build_rerank_kwargs(request.rerank)
        retriever_kwargs["rerank"] = True
        if rerank_kwargs:
            retriever_kwargs["rerank_kwargs"] = rerank_kwargs
    return retriever_kwargs


def query_documents(request: QueryRequest) -> list[RetrievalHit]:
    """Run the SDK query path used by the root CLI."""
    retriever = Retriever(**_build_retriever_kwargs(request))
    return retriever.query(
        request.query,
        candidate_k=request.retrieval.candidate_k,
        page_dedup=request.retrieval.page_dedup,
        content_types=request.retrieval.content_types,
    )
