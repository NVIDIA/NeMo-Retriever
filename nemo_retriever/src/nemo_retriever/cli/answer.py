# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level answer-generation command."""

from __future__ import annotations

import json
import os
from typing import cast

import typer

from nemo_retriever.cli.query import options as opts
from nemo_retriever.cli.shared import (
    ROOT_CLI_ERRORS,
    api_key_from_env_option,
    build_retrieval_options,
    quiet_capture,
    resolve_retrieval_mode,
    silence_noisy_libraries,
)
from nemo_retriever.query.options import (
    QueryEmbedOptions,
    QueryRequest,
    QueryRerankOptions,
    QueryRetrievalMode,
    QueryStorageOptions,
)
from nemo_retriever.query.workflow import resolve_query_plan

ANSWER_HELP = (
    "Retrieve context from a local LanceDB index and generate an answer with an LLM. "
    "Add --multimodal to use a Vision-Language Model: visual chunks (image, chart, "
    "infographic, table) that have a stored image URI are loaded and sent inline alongside "
    "their text captions. Outputs JSON with 'answer', 'model', 'latency_s', 'chunk_count'. "
    "Pass --reference to add automatic token-F1 / context-coverage scoring."
)


def _build_request(
    query: str,
    *,
    top_k: int,
    candidate_k: int | None,
    page_dedup: bool,
    content_types: str | None,
    lancedb_uri: str,
    table_name: str,
    embed_invoke_url: str | None,
    embed_model_name: str | None,
    reranker_invoke_url: str | None,
    reranker_api_key: str | None,
    reranker_model_name: str | None,
    reranker_backend: str | None,
    rerank: bool,
    retrieval_mode: str,
) -> QueryRequest:
    return QueryRequest(
        query=query,
        retrieval=build_retrieval_options(
            top_k=top_k,
            candidate_k=candidate_k,
            page_dedup=page_dedup,
            content_types=content_types,
            retrieval_mode=cast(QueryRetrievalMode, retrieval_mode),
        ),
        embed=QueryEmbedOptions(
            embed_invoke_url=embed_invoke_url,
            embed_model_name=embed_model_name,
        ),
        rerank=QueryRerankOptions(
            enabled=rerank,
            reranker_invoke_url=reranker_invoke_url,
            reranker_model_name=reranker_model_name,
            reranker_backend=reranker_backend,
            reranker_api_key=reranker_api_key,
        ),
        storage=QueryStorageOptions(lancedb_uri=lancedb_uri, table_name=table_name),
    )


def answer_command(
    ctx: typer.Context,
    query: opts.QueryArgument,
    top_k: opts.TopKOption = 5,
    candidate_k: opts.CandidateKOption = None,
    page_dedup: opts.PageDedupOption = False,
    content_types: opts.ContentTypesOption = None,
    lancedb_uri: opts.LanceDbUriOption = "lancedb",
    table_name: opts.TableNameOption = "nemo-retriever",
    embed_invoke_url: opts.EmbedInvokeUrlOption = None,
    embed_model_name: opts.EmbedModelNameOption = None,
    reranker_invoke_url: opts.RerankerInvokeUrlOption = None,
    reranker_api_key_env: opts.RerankerApiKeyEnvOption = None,
    reranker_model_name: opts.RerankerModelNameOption = None,
    reranker_backend: opts.RerankerBackendOption = None,
    rerank: opts.RerankOption = False,
    retrieval_mode: opts.RetrievalModeOption = "auto",
    hybrid: opts.HybridOption = False,
    answer_llm_model: opts.AnswerLlmModelOption = None,
    answer_llm_invoke_url: opts.AnswerLlmInvokeUrlOption = None,
    answer_llm_api_key_env: opts.AnswerLlmApiKeyEnvOption = None,
    answer_llm_max_tokens: opts.AnswerLlmMaxTokensOption = 4096,
    answer_llm_temperature: opts.AnswerLlmTemperatureOption = 0.0,
    answer_reasoning: opts.AnswerReasoningOption = None,
    reference: opts.AnswerReferenceOption = None,
    multimodal: opts.AnswerMultimodalOption = False,
) -> None:
    from nemo_retriever.models.llm.clients.litellm import LiteLLMClient
    from nemo_retriever.models.llm.clients.vlm_litellm import LiteVLMClient

    if reranker_invoke_url is None:
        reranker_invoke_url = os.environ.get("RERANKER_INVOKE_URL") or None
    if embed_invoke_url is None:
        embed_invoke_url = os.environ.get("EMBED_INVOKE_URL") or None
    rerank = (
        rerank
        or bool(reranker_invoke_url)
        or bool(reranker_model_name)
        or bool(reranker_backend)
    )
    silence_noisy_libraries()

    try:
        reranker_api_key = (
            api_key_from_env_option(reranker_api_key_env)
            if reranker_invoke_url
            else None
        )
        answer_llm_api_key = api_key_from_env_option(answer_llm_api_key_env)
        effective_retrieval_mode = resolve_retrieval_mode(ctx, retrieval_mode, hybrid)
        request = _build_request(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            page_dedup=page_dedup,
            content_types=content_types,
            lancedb_uri=lancedb_uri,
            table_name=table_name,
            embed_invoke_url=embed_invoke_url,
            embed_model_name=embed_model_name,
            reranker_invoke_url=reranker_invoke_url,
            reranker_api_key=reranker_api_key,
            reranker_model_name=reranker_model_name,
            reranker_backend=reranker_backend,
            rerank=rerank,
            retrieval_mode=effective_retrieval_mode,
        )
        retriever = resolve_query_plan(request).create_retriever()

        llm_kwargs: dict[str, object] = {
            "temperature": answer_llm_temperature,
            "max_tokens": answer_llm_max_tokens,
        }
        if answer_llm_model:
            llm_kwargs["model"] = answer_llm_model
        if answer_llm_invoke_url:
            llm_kwargs["api_base"] = answer_llm_invoke_url
        if answer_llm_api_key:
            llm_kwargs["api_key"] = answer_llm_api_key
        if answer_reasoning is not None:
            llm_kwargs["reasoning_enabled"] = answer_reasoning

        client_cls = LiteVLMClient if multimodal else LiteLLMClient
        llm = client_cls.from_kwargs(**llm_kwargs)  # type: ignore[arg-type]
        with quiet_capture():
            result = retriever.answer(
                query,
                llm=llm,
                top_k=top_k,
                reference=reference,
                reasoning_enabled=answer_reasoning,
                multimodal=multimodal,
            )
    except ROOT_CLI_ERRORS as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        json.dumps(
            result.model_dump(exclude_none=True), indent=2, sort_keys=True, default=str
        )
    )
