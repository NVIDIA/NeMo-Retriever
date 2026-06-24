# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ResolvedQueryPlan:
    """Resolved Retriever query configuration reusable across many queries."""

    top_k: int
    candidate_k: int | None
    page_dedup: bool
    content_types: str | None
    lancedb_uri: str
    table_name: str
    embed_kwargs: dict[str, Any]
    hybrid: bool
    rerank: bool
    rerank_kwargs: dict[str, Any]

    def retriever_kwargs(self) -> dict[str, Any]:
        vdb_kwargs: dict[str, Any] = {
            "uri": self.lancedb_uri,
            "table_name": self.table_name,
        }
        if self.hybrid:
            vdb_kwargs["hybrid"] = True

        kwargs: dict[str, Any] = {
            "top_k": self.top_k,
            "vdb_kwargs": vdb_kwargs,
        }
        if self.embed_kwargs:
            kwargs["embed_kwargs"] = dict(self.embed_kwargs)
        if self.rerank:
            kwargs["rerank"] = True
            if self.rerank_kwargs:
                kwargs["rerank_kwargs"] = dict(self.rerank_kwargs)
        return kwargs

    def create_retriever(self) -> Retriever:
        return Retriever(**self.retriever_kwargs())

    def query_kwargs(self) -> dict[str, Any]:
        return {
            "candidate_k": self.candidate_k,
            "page_dedup": self.page_dedup,
            "content_types": self.content_types,
        }


def resolve_query_plan(request: QueryRequest) -> ResolvedQueryPlan:
    """Resolve root query options once so callers can reuse a Retriever."""
    embed_kwargs = build_embed_option_kwargs(request.embed.embed_invoke_url, request.embed.embed_model_name)
    rerank_kwargs = _build_rerank_kwargs(request.rerank) if request.rerank.enabled else {}
    content_types = request.retrieval.content_types
    if content_types is not None and not isinstance(content_types, str):
        content_types = ",".join(str(value) for value in content_types)
    return ResolvedQueryPlan(
        top_k=int(request.retrieval.top_k),
        candidate_k=request.retrieval.candidate_k,
        page_dedup=bool(request.retrieval.page_dedup),
        content_types=content_types,
        lancedb_uri=str(request.storage.lancedb_uri),
        table_name=str(request.storage.table_name),
        embed_kwargs=embed_kwargs,
        hybrid=bool(request.retrieval.hybrid),
        rerank=bool(request.rerank.enabled),
        rerank_kwargs=rerank_kwargs,
    )


def query_documents(request: QueryRequest) -> list[RetrievalHit]:
    """Run the SDK query path used by the root CLI."""
    retriever = resolve_query_plan(request).create_retriever()
    return retriever.query(
        request.query,
        candidate_k=request.retrieval.candidate_k,
        page_dedup=request.retrieval.page_dedup,
        content_types=request.retrieval.content_types,
    )
