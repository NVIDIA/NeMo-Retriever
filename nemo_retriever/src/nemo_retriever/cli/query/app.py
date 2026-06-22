# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os

import click
import typer
from typer.core import TyperGroup

from nemo_retriever.cli.evidence import build_evidence_result
from nemo_retriever.cli.query import options as opts
from nemo_retriever.cli.query_workflow import query_documents as query_local_documents
from nemo_retriever.cli.shared import (
    ROOT_CLI_ERRORS,
    quiet_capture,
    silence_noisy_libraries,
)
from nemo_retriever.common.vdb.records import RetrievalHit
from nemo_retriever.query.options import (
    QueryEmbedOptions,
    QueryRerankOptions,
    QueryRequest,
    QueryRetrievalOptions,
    QueryServiceOptions,
    QueryStorageOptions,
    ServiceQueryRequest,
)
from nemo_retriever.query.service import query_documents as query_service_documents

_DEFAULT_COMMAND = "_local"
_GROUP_OPTIONS = {"--help", "-h", "--install-completion", "--show-completion"}


class DefaultLocalQueryGroup(TyperGroup):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in _GROUP_OPTIONS:
            args = [_DEFAULT_COMMAND, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=DefaultLocalQueryGroup,
    help="Query Retriever indexes. Omitting a mode queries local LanceDB.",
    no_args_is_help=True,
)


def _query_cli_hit(hit: RetrievalHit, max_text_chars: int | None = None) -> dict[str, object]:
    metadata = hit.get("metadata") or {}
    modality = hit.get("content_type") or metadata.get("type") or "text"
    if "_score" in hit and hit["_score"] is not None:
        score: object = hit["_score"]
    elif "_distance" in hit and hit["_distance"] is not None:
        score = hit["_distance"]
    else:
        score = None
    text = hit.get("text", "")
    if max_text_chars is not None and max_text_chars >= 0 and len(text) > max_text_chars:
        text = text[:max_text_chars] + ("…" if max_text_chars > 0 else "")
    return {
        "source": hit.get("source", ""),
        "page_number": hit.get("page_number"),
        "text": text,
        "modality": modality,
        "score": score,
    }


def _api_key_from_env_option(env_key: str | None) -> str | None:
    key = (env_key or "").strip()
    if not key:
        return None
    value = os.environ.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is not set or is empty.")
    return value


def _validate_output_options(output_format: str, max_text_chars: int | None) -> None:
    if output_format not in ("hits", "evidence"):
        typer.echo(f"Error: unknown --format {output_format!r} (use 'hits' or 'evidence').", err=True)
        raise typer.Exit(1)
    if max_text_chars is not None and output_format != "hits":
        typer.echo("Error: --max-text-chars only applies to --format hits.", err=True)
        raise typer.Exit(1)


def _emit_query_output(
    hits: list[RetrievalHit],
    *,
    strategies: list[str],
    output_format: str,
    max_text_chars: int | None,
) -> None:
    if output_format == "evidence":
        result = build_evidence_result(hits, strategies)
        typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(
            json.dumps([_query_cli_hit(hit, max_text_chars) for hit in hits], indent=2, sort_keys=True, default=str)
        )


def _retrieval_options(
    *,
    top_k: int,
    candidate_k: int | None,
    page_dedup: bool,
    content_types: str | None,
    hybrid: bool = False,
) -> QueryRetrievalOptions:
    return QueryRetrievalOptions(
        top_k=top_k,
        candidate_k=candidate_k,
        page_dedup=page_dedup,
        content_types=content_types,
        hybrid=hybrid,
    )


@app.command("_local", hidden=True)
def _local_command(
    query: opts.QueryArgument,
    top_k: opts.TopKOption = 10,
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
    hybrid: opts.HybridOption = False,
    output_format: opts.OutputFormatOption = "hits",
    max_text_chars: opts.MaxTextCharsOption = None,
) -> None:
    _validate_output_options(output_format, max_text_chars)
    if reranker_invoke_url is None:
        reranker_invoke_url = os.environ.get("RERANKER_INVOKE_URL") or None
    if embed_invoke_url is None:
        embed_invoke_url = os.environ.get("EMBED_INVOKE_URL") or None
    rerank = rerank or bool(reranker_invoke_url) or bool(reranker_model_name) or bool(reranker_backend)
    silence_noisy_libraries()

    try:
        reranker_api_key = _api_key_from_env_option(reranker_api_key_env) if reranker_invoke_url else None

        def _request(use_hybrid: bool) -> QueryRequest:
            return QueryRequest(
                query=query,
                retrieval=_retrieval_options(
                    top_k=top_k,
                    candidate_k=candidate_k,
                    page_dedup=page_dedup,
                    content_types=content_types,
                    hybrid=use_hybrid,
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
                storage=QueryStorageOptions(
                    lancedb_uri=lancedb_uri,
                    table_name=table_name,
                ),
            )

        with quiet_capture():
            if hybrid:
                try:
                    hits = query_local_documents(_request(True))
                    strategies = ["semantic", "lexical"]
                except Exception:  # noqa: BLE001 - preserve root query's vector fallback on missing FTS.
                    hits = query_local_documents(_request(False))
                    strategies = ["semantic"]
            else:
                hits = query_local_documents(_request(False))
                strategies = ["semantic"]
    except ROOT_CLI_ERRORS as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    _emit_query_output(hits, strategies=strategies, output_format=output_format, max_text_chars=max_text_chars)


@app.command("service")
def _service_command(
    query: opts.QueryArgument,
    service_url: opts.ServiceUrlOption = "http://localhost:7670",
    service_api_token: opts.ServiceApiTokenOption = None,
    top_k: opts.TopKOption = 10,
    candidate_k: opts.CandidateKOption = None,
    page_dedup: opts.PageDedupOption = False,
    content_types: opts.ContentTypesOption = None,
    output_format: opts.OutputFormatOption = "hits",
    max_text_chars: opts.MaxTextCharsOption = None,
) -> None:
    _validate_output_options(output_format, max_text_chars)
    silence_noisy_libraries()
    try:
        with quiet_capture():
            hits = query_service_documents(
                ServiceQueryRequest(
                    query=query,
                    retrieval=_retrieval_options(
                        top_k=top_k,
                        candidate_k=candidate_k,
                        page_dedup=page_dedup,
                        content_types=content_types,
                    ),
                    service=QueryServiceOptions(
                        service_url=service_url,
                        service_api_token=service_api_token,
                    ),
                )
            )
    except ROOT_CLI_ERRORS as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    _emit_query_output(hits, strategies=["semantic"], output_format=output_format, max_text_chars=max_text_chars)
