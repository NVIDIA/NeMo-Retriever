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


def agentic_query_documents(request: QueryRequest) -> list[dict[str, Any]]:
    """Run agentic (ReAct) retrieval for a single query and return the agent's
    ranked document IDs.

    Unlike the dense ``query_documents`` path (which returns enriched hits with
    text), the agent operates at the document-ID granularity of the configured
    index, so the result is the ranked ``doc_id`` list the agent selected,
    annotated with the source that produced it (``final_results`` / ``rrf`` /
    ``selection_agent``). The LanceDB ``uri``/``table_name``, embedding config,
    and (when ``--rerank`` is enabled) reranker config are passed straight
    through to the wrapped ``Retriever`` that backs the agent's ``retrieve``
    tool. Reranking therefore applies per agent retrieval hop.
    """
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

    api_key = resolve_remote_api_key()
    cfg_kwargs: dict[str, Any] = {
        "vdb_op": "lancedb",
        "vdb_kwargs": {"uri": request.storage.lancedb_uri, "table_name": request.storage.table_name},
        "top_k": int(request.retrieval.top_k),
        "embedding_endpoint": request.embed.embed_invoke_url,
        "embedding_api_key": api_key or "",
        "llm_model": request.agentic.llm_model,
        "invoke_url": request.agentic.invoke_url,
        "api_key": api_key,
        "reasoning_effort": request.agentic.reasoning_effort,
        "backend_top_k": int(request.agentic.backend_top_k),
        "react_max_steps": int(request.agentic.react_max_steps),
        "text_truncation": int(request.agentic.text_truncation),
        "temperature": float(request.agentic.temperature),
    }
    if request.embed.embed_model_name:
        cfg_kwargs["query_embedder"] = request.embed.embed_model_name
    if request.rerank.enabled:
        # `reranker` doubles as the on/off gate (rerank=bool(cfg.reranker)) and the
        # model name, so fall back to the default model when only --rerank is given.
        cfg_kwargs["reranker"] = request.rerank.reranker_model_name or _LOCAL_VL_RERANK_MODEL
        cfg_kwargs["reranker_endpoint"] = request.rerank.reranker_invoke_url
        cfg_kwargs["reranker_api_key"] = resolve_remote_api_key(request.rerank.reranker_api_key) or ""
        if request.rerank.reranker_backend:
            cfg_kwargs["local_reranker_backend"] = request.rerank.reranker_backend

    result = AgenticRetriever(AgenticRetrievalConfig(**cfg_kwargs)).retrieve(["0"], [str(request.query)])
    if "rank" in result.columns:
        result = result.sort_values("rank")
    ranked: list[dict[str, Any]] = []
    for _, row in result.iterrows():
        ranked.append(
            {
                "rank": int(row.get("rank", len(ranked) + 1)),
                "doc_id": str(row.get("doc_id", "")),
                "result_source": str(row.get("result_source", "")),
            }
        )
        if len(ranked) >= request.retrieval.top_k:
            break
    return ranked
