# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Sequence

from nemo_retriever.params import build_embed_option_kwargs
from nemo_retriever.retriever import Retriever
from nemo_retriever.utils.remote_auth import resolve_remote_api_key
from nemo_retriever.vdb.records import RetrievalHit

_LOCAL_VL_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"


def _build_rerank_kwargs(
    reranker_invoke_url: str | None,
    reranker_model_name: str | None = None,
    reranker_backend: str | None = None,
) -> dict[str, str]:
    """Build kwargs for the rerank stage. Mirrors :func:`build_embed_option_kwargs`:
    if ``reranker_invoke_url`` is given the remote NIM path is configured;
    otherwise the local GPU reranker runs with ``reranker_model_name`` (or the
    matching VL default to pair with the local VL embedder).

    ``reranker_backend`` only applies to the local path and selects the local
    inference backend (``"vllm"`` or ``"hf"``); ``None`` defers to the library
    default in ``_default_rerank_actor_kwargs``.
    """
    reranker_url = (reranker_invoke_url or "").strip()
    if reranker_url:
        rerank_kwargs: dict[str, str] = {"rerank_invoke_url": reranker_url}
        if reranker_model_name:
            rerank_kwargs["model_name"] = reranker_model_name
        api_key = resolve_remote_api_key()
        if api_key is not None:
            rerank_kwargs["api_key"] = api_key
        return rerank_kwargs

    # Local GPU reranker - VL by default to pair with the local VL embedder.
    # ``NemotronRerankGPUActor`` loads the model once per actor; the rerank
    # model is ~2 GB and coexists with the vLLM embedder (which respects
    # ``gpu_memory_utilization=0.45``).
    local: dict[str, str] = {"model_name": reranker_model_name or _LOCAL_VL_RERANK_MODEL}
    if reranker_backend:
        local["local_reranker_backend"] = reranker_backend
    return local


def query_documents(
    query: str,
    *,
    top_k: int = 10,
    candidate_k: int | None = None,
    page_dedup: bool = False,
    content_types: str | Sequence[str] | None = None,
    lancedb_uri: str = "lancedb",
    table_name: str = "nemo-retriever",
    embed_invoke_url: str | None = None,
    embed_model_name: str | None = None,
    reranker_invoke_url: str | None = None,
    reranker_model_name: str | None = None,
    reranker_backend: str | None = None,
    rerank: bool = False,
) -> list[RetrievalHit]:
    """Run the minimal SDK query path used by the root CLI.

    Reranking is opt-in: pass ``rerank=True`` (or any of the rerank-related
    args via the CLI, which implicitly set ``rerank=True``) to enable.
    """
    embed_kwargs = build_embed_option_kwargs(embed_invoke_url, embed_model_name)
    retriever_kwargs: dict[str, Any] = {
        "top_k": top_k,
        "vdb_kwargs": {"uri": lancedb_uri, "table_name": table_name},
    }
    if embed_kwargs:
        retriever_kwargs["embed_kwargs"] = embed_kwargs
    if rerank:
        rerank_kwargs = _build_rerank_kwargs(reranker_invoke_url, reranker_model_name, reranker_backend)
        retriever_kwargs["rerank"] = True
        if rerank_kwargs:
            retriever_kwargs["rerank_kwargs"] = rerank_kwargs

    retriever = Retriever(**retriever_kwargs)
    return retriever.query(
        query,
        candidate_k=candidate_k,
        page_dedup=page_dedup,
        content_types=content_types,
    )
