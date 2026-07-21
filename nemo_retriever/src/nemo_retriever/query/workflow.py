# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_retriever.query.agentic import AgenticRetrievalConfig, AgenticRetriever

from nemo_retriever.common.params import build_embed_option_kwargs
from nemo_retriever.common.remote_auth import resolve_remote_api_key
from nemo_retriever.common.vdb.lancedb_capabilities import LanceRetrievalMode
from nemo_retriever.common.vdb.lancedb_schema import normalize_content_type_values
from nemo_retriever.common.vdb.records import RetrievalHit
from nemo_retriever.graph.retriever import Retriever
from nemo_retriever.models import VL_RERANK_MODEL
from nemo_retriever.query.options import QueryRequest, QueryRerankOptions

_LOCAL_VL_RERANK_MODEL = VL_RERANK_MODEL


@dataclass(frozen=True)
class QueryDocumentsResult:
    hits: list[RetrievalHit]
    strategies: list[str]


def _strategies_for_retrieval_mode(mode: LanceRetrievalMode | None) -> list[str]:
    if mode == "hybrid":
        return ["semantic", "lexical"]
    if mode == "sparse":
        return ["lexical"]
    return ["semantic"]


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
    return resolve_query_plan(request).retriever_kwargs()


@dataclass(frozen=True)
class ResolvedQueryPlan:
    """Resolved Retriever query configuration reusable across many queries."""

    top_k: int
    candidate_k: int | None
    page_dedup: bool
    content_types: str | None
    lancedb_uri: str
    table_name: str
    retrieval_mode: str
    embed_kwargs: dict[str, Any]
    rerank: bool
    rerank_kwargs: dict[str, Any]

    def retriever_kwargs(self) -> dict[str, Any]:
        vdb_kwargs: dict[str, Any] = {
            "uri": self.lancedb_uri,
            "table_name": self.table_name,
        }
        if self.retrieval_mode != "auto":
            vdb_kwargs["retrieval_mode"] = self.retrieval_mode

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
    embed_kwargs = build_embed_option_kwargs(
        request.embed.embed_invoke_url,
        request.embed.embed_model_name,
        embed_model_provider_prefix=request.embed.embed_model_provider_prefix,
    )
    rerank_kwargs = _build_rerank_kwargs(request.rerank) if request.rerank.enabled else {}
    content_type_values = normalize_content_type_values(request.retrieval.content_types)
    content_types = ",".join(content_type_values) if content_type_values is not None else None
    return ResolvedQueryPlan(
        top_k=int(request.retrieval.top_k),
        candidate_k=request.retrieval.candidate_k,
        page_dedup=bool(request.retrieval.page_dedup),
        content_types=content_types,
        lancedb_uri=str(request.storage.lancedb_uri),
        table_name=str(request.storage.table_name),
        retrieval_mode=str(request.retrieval.retrieval_mode),
        embed_kwargs=embed_kwargs,
        rerank=bool(request.rerank.enabled),
        rerank_kwargs=rerank_kwargs,
    )


def query_documents_with_metadata(request: QueryRequest) -> QueryDocumentsResult:
    """Run the SDK query path and return hits plus resolved retrieval strategy metadata."""
    plan = resolve_query_plan(request)
    retriever = plan.create_retriever()
    mode: LanceRetrievalMode | None = None
    resolve_mode = getattr(retriever, "_resolve_lancedb_query_mode", None)
    if callable(resolve_mode):
        lancedb_mode = resolve_mode(None)
        if lancedb_mode is not None:
            mode = lancedb_mode[0]
    hits = retriever.query(
        request.query,
        **plan.query_kwargs(),
    )
    return QueryDocumentsResult(hits=hits, strategies=_strategies_for_retrieval_mode(mode))


def query_documents(request: QueryRequest) -> list[RetrievalHit]:
    """Run the SDK query path used by the root CLI."""
    return query_documents_with_metadata(request).hits


def build_agentic_config(request: QueryRequest, *, top_k: int | None = None) -> "AgenticRetrievalConfig":
    """Build an :class:`AgenticRetrievalConfig` from a :class:`QueryRequest`.

    Shared by the single-query CLI path (:func:`agentic_query_documents`) and the
    batch harness BEIR path so agentic config derivation lives in one place. The
    LanceDB ``uri``/``table_name``, embedding config, and (when ``rerank`` is
    enabled) reranker config are passed straight through to the wrapped
    ``Retriever`` that backs the agent's ``retrieve`` tool. ``top_k`` overrides the
    final document count the agent targets (the harness sets this to the deepest
    BEIR metric ``k`` so recall at the largest cutoff is computable).
    """
    from nemo_retriever.query.agentic import AgenticRetrievalConfig

    api_key = resolve_remote_api_key()
    vdb_kwargs: dict[str, Any] = {"uri": request.storage.lancedb_uri, "table_name": request.storage.table_name}
    if request.retrieval.retrieval_mode != "auto":
        vdb_kwargs["retrieval_mode"] = request.retrieval.retrieval_mode
    content_type_values = normalize_content_type_values(request.retrieval.content_types)
    cfg_kwargs: dict[str, Any] = {
        "vdb_op": "lancedb",
        "vdb_kwargs": vdb_kwargs,
        "top_k": int(top_k if top_k is not None else request.retrieval.top_k),
        "candidate_k": request.retrieval.candidate_k,
        "page_dedup": bool(request.retrieval.page_dedup),
        "content_types": ",".join(content_type_values) if content_type_values is not None else None,
        "embedding_endpoint": request.embed.embed_invoke_url,
        "embedding_api_key": api_key or "",
        "llm_model": request.agentic.llm_model,
        "invoke_url": request.agentic.invoke_url,
        "api_key": api_key,
        "reasoning_effort": request.agentic.reasoning_effort,
        "backend_top_k": int(request.agentic.backend_top_k),
        "react_max_steps": int(request.agentic.react_max_steps),
        "text_truncation": int(request.agentic.text_truncation),
        "num_concurrent": int(request.agentic.num_concurrent),
        "temperature": float(request.agentic.temperature),
        "trace_enabled": bool(request.agentic.trace_enabled),
        "trace_path": request.agentic.trace_path,
    }
    if request.embed.embed_model_name:
        cfg_kwargs["query_embedder"] = request.embed.embed_model_name
    if request.embed.embed_model_provider_prefix:
        cfg_kwargs["query_embedder_provider_prefix"] = request.embed.embed_model_provider_prefix
    if request.rerank.enabled:
        # `reranker` doubles as the on/off gate (rerank=bool(cfg.reranker)) and the
        # model name, so fall back to the default model when only --rerank is given.
        cfg_kwargs["reranker"] = request.rerank.reranker_model_name or _LOCAL_VL_RERANK_MODEL
        cfg_kwargs["reranker_endpoint"] = request.rerank.reranker_invoke_url
        cfg_kwargs["reranker_api_key"] = resolve_remote_api_key(request.rerank.reranker_api_key) or ""
        if request.rerank.reranker_backend:
            cfg_kwargs["local_reranker_backend"] = request.rerank.reranker_backend

    return AgenticRetrievalConfig(**cfg_kwargs)


def build_agentic_retriever(request: QueryRequest) -> "AgenticRetriever":
    """Construct an :class:`AgenticRetriever` from a :class:`QueryRequest`."""
    from nemo_retriever.query.agentic import AgenticRetriever

    return AgenticRetriever(build_agentic_config(request))


def agentic_query_documents(request: QueryRequest) -> list[dict[str, Any]]:
    """Run agentic (ReAct) retrieval for a single query and return ranked hits.

    Each result carries the full ``RetrievalHit`` metadata (``text``, ``source``,
    ``page_number``, ``pdf_page``, ``metadata``, …) — re-hydrated by ``doc_id`` from
    the hits captured at retrieval time — plus the agentic annotations ``doc_id``,
    ``rank``, and ``result_source`` (``final_results`` / ``rrf`` / ``selection_agent``).
    This matches the metadata the dense ``query_documents`` path returns. The LanceDB
    ``uri``/``table_name``, embedding config, and (when ``--rerank`` is enabled)
    reranker config are passed through to the wrapped ``Retriever`` that backs the
    agent's ``retrieve`` tool; reranking applies per agent retrieval hop.
    """
    result = build_agentic_retriever(request).retrieve(["0"], [str(request.query)])
    if "rank" in result.columns:
        result = result.sort_values("rank")
    ranked: list[dict[str, Any]] = []
    for _, row in result.iterrows():
        hit = row.get("hit") if "hit" in result.columns else None
        enriched: dict[str, Any] = dict(hit) if isinstance(hit, dict) else {}
        enriched.update(
            {
                "doc_id": str(row.get("doc_id", "")),
                "rank": int(row.get("rank", len(ranked) + 1)),
                "result_source": str(row.get("result_source", "")),
            }
        )
        ranked.append(enriched)
        if len(ranked) >= request.retrieval.top_k:
            break
    return ranked
