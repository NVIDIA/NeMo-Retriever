# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import logging
import os

import typer

from nemo_retriever.query.agentic_options import (
    AGENTIC_OPENAI_COMPATIBLE_TEMPERATURE_MAX,
    agentic_backend_top_k_error,
    agentic_temperature_error,
)
from nemo_retriever.cli.ingest import app as ingest_app
from nemo_retriever.cli.shared import (
    ROOT_CLI_ERRORS as _ROOT_CLI_ERRORS,
    quiet_capture as _quiet_capture,
    silence_noisy_libraries as _silence_noisy_libraries,
)
from nemo_retriever.cli.evidence import build_evidence_result
from nemo_retriever.cli.query_workflow import agentic_query_documents, query_documents
from nemo_retriever.query.options import (
    QueryAgenticOptions,
    QueryEmbedOptions,
    QueryRerankOptions,
    QueryRequest,
    QueryRetrievalOptions,
    QueryStorageOptions,
)
from nemo_retriever.common.vdb.records import RetrievalHit
from nemo_retriever.version import get_version_info

logger = logging.getLogger(__name__)

app = typer.Typer(help="Retriever")

# Service sub-app is always available (lightweight, no GPU deps).
from nemo_retriever.service.cli import app as service_app  # noqa: E402

app.add_typer(service_app, name="service")
app.add_typer(ingest_app, name="ingest")

# All other sub-apps are registered lazily so that missing optional
# dependencies (tritonclient, torch, …) don't prevent the service
# from starting.
_LAZY_SUBAPPS: list[tuple[str, str, str]] = [
    ("audio", "nemo_retriever.cli.audio.cli", "app"),
    ("image", "nemo_retriever.cli.image", "app"),
    ("pdf", "nemo_retriever.cli.pdf.__main__", "app"),
    ("local", "nemo_retriever.cli.local", "app"),
    ("chart", "nemo_retriever.cli.chart.commands", "app"),
    ("compare", "nemo_retriever.cli.compare", "app"),
    ("eval", "nemo_retriever.tools.evaluation.cli", "app"),
    ("benchmark", "nemo_retriever.tools.benchmark", "app"),
    ("harness", "nemo_retriever.harness", "app"),
    ("recall", "nemo_retriever.tools.recall", "app"),
    ("skill-eval", "nemo_retriever.tools.skill_eval", "app"),
    ("txt", "nemo_retriever.cli.txt.__main__", "app"),
    ("html", "nemo_retriever.cli.html.__main__", "app"),
    ("pipeline", "nemo_retriever.cli.pipeline.__main__", "app"),
]

for _name, _module, _attr in _LAZY_SUBAPPS:
    try:
        _mod = importlib.import_module(_module)
        app.add_typer(getattr(_mod, _attr), name=_name)
    except Exception:
        logger.debug("Skipping '%s' sub-command (import failed)", _name)


def _query_cli_hit(hit: RetrievalHit, max_text_chars: int | None = None) -> dict[str, object]:
    metadata = hit.get("metadata") or {}
    modality = hit.get("content_type") or metadata.get("type") or "text"
    # Relevance the engine ranked by: hybrid/rerank score if present, else the
    # vector distance, else null. Hit ORDER is authoritative; score is informational.
    if "_score" in hit and hit["_score"] is not None:
        score: object = hit["_score"]
    elif "_distance" in hit and hit["_distance"] is not None:
        score = hit["_distance"]
    else:
        score = None
    text = hit.get("text", "")
    # Compact output: truncate to max_text_chars (0 = metadata-only). None = full text.
    if max_text_chars is not None and max_text_chars >= 0 and len(text) > max_text_chars:
        text = text[:max_text_chars] + ("…" if max_text_chars > 0 else "")
    return {
        "source": hit.get("source", ""),
        "page_number": hit.get("page_number"),
        "text": text,
        "modality": modality,
        "score": score,
    }


def _version_callback(value: bool) -> None:
    if not value:
        return
    info = get_version_info()
    typer.echo(info["full_version"])
    raise typer.Exit()


def _api_key_from_env_option(env_key: str | None) -> str | None:
    key = (env_key or "").strip()
    if not key:
        return None
    value = os.environ.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is not set or is empty.")
    return value


def main() -> None:
    app()


@app.command("query")
def query_command(
    query: str = typer.Argument(..., help="Query text."),
    top_k: int = typer.Option(10, "--top-k", min=1, help="Final number of hits to return."),
    candidate_k: int | None = typer.Option(
        None,
        "--candidate-k",
        min=1,
        help=(
            "Candidate pool size before page deduplication or content-type filtering; "
            "must be greater than or equal to --top-k."
        ),
    ),
    page_dedup: bool = typer.Option(
        False,
        "--page-dedup/--no-page-dedup",
        help="Collapse hits to unique document pages.",
    ),
    content_types: str | None = typer.Option(
        None,
        "--content-types",
        help="Comma-separated content types to keep, such as text,table; untyped hits are excluded.",
    ),
    lancedb_uri: str = typer.Option(
        "lancedb",
        "--lancedb-uri",
        help="LanceDB database URI to read; match the value used for retriever ingest --lancedb-uri.",
    ),
    table_name: str = typer.Option(
        "nemo-retriever",
        "--table-name",
        help="LanceDB table name to read; match the value used for retriever ingest --table-name.",
    ),
    embed_invoke_url: str | None = typer.Option(None, "--embed-invoke-url", help="Embedding NIM endpoint URL."),
    embed_model_name: str | None = typer.Option(
        None,
        "--embed-model-name",
        help="Optional embedding model name override.",
    ),
    reranker_invoke_url: str | None = typer.Option(None, "--reranker-invoke-url", help="Reranker endpoint URL."),
    reranker_api_key_env: str | None = typer.Option(
        None,
        "--reranker-api-key-env",
        help=(
            "Environment variable containing the bearer token for --reranker-invoke-url. "
            "If omitted, NVIDIA_API_KEY / NGC_API_KEY is used when set."
        ),
    ),
    reranker_model_name: str | None = typer.Option(
        None,
        "--reranker-model-name",
        help="Optional reranker model name override (used by the local GPU reranker).",
    ),
    reranker_backend: str | None = typer.Option(
        None,
        "--reranker-backend",
        help=(
            "Backend for the local GPU reranker when no --reranker-invoke-url is given: "
            "'vllm' (default — high-throughput batch) or 'hf' (HuggingFace, faster cold "
            "start; preferred for ad-hoc / single-query CLI use)."
        ),
    ),
    rerank: bool = typer.Option(
        False,
        "--rerank/--no-rerank",
        help=(
            "Enable reranking after vector retrieval. Default off. Implicitly enabled when "
            "any of --reranker-invoke-url / --reranker-model-name / --reranker-backend is set."
        ),
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid/--no-hybrid",
        help=(
            "Fused vector + full-text (BM25) retrieval; falls back to vector-only if the table "
            "has no FTS index. Opt-in (default off) — preserves the legacy vector-only default."
        ),
    ),
    output_format: str = typer.Option(
        "hits",
        "--format",
        help=(
            "'hits' (default): raw ranked hit list (source/page/text/modality/score) — the legacy "
            "output. 'evidence': answer-ready, fidelity-tagged, cited evidence + coverage (opt-in)."
        ),
    ),
    max_text_chars: int | None = typer.Option(
        None,
        "--max-text-chars",
        help="('hits' format only) Truncate each hit's text to N chars (0 = metadata-only). Default: full text.",
    ),
    agentic: bool = typer.Option(
        False,
        "--agentic",
        help="Run an LLM-driven agentic (ReAct) retrieval loop instead of the default dense pass.",
    ),
    agentic_llm_model: str | None = typer.Option(
        None,
        "--agentic-llm-model",
        help="Chat model the agent drives. Required when --agentic is set.",
    ),
    agentic_invoke_url: str | None = typer.Option(
        None,
        "--agentic-invoke-url",
        help="OpenAI-compatible chat-completions endpoint for the agent LLM (agentic mode).",
    ),
    agentic_reasoning_effort: str | None = typer.Option(
        None,
        "--agentic-reasoning-effort",
        help="reasoning_effort forwarded on agentic LLM calls when set.",
    ),
    agentic_backend_top_k: int = typer.Option(
        20,
        "--agentic-backend-top-k",
        min=1,
        help="Backend retrieve-pool depth per agentic retrieval call.",
    ),
    agentic_react_max_steps: int = typer.Option(
        50,
        "--agentic-react-max-steps",
        min=1,
        help="Maximum ReAct loop iterations for the agentic query.",
    ),
    agentic_text_truncation: int = typer.Option(
        0,
        "--agentic-text-truncation",
        min=0,
        help="Max characters of each candidate shown to the agent; 0 disables truncation.",
    ),
    agentic_temperature: float = typer.Option(
        0.0,
        "--agentic-temperature",
        min=0.0,
        max=AGENTIC_OPENAI_COMPATIBLE_TEMPERATURE_MAX,
        help="Sampling temperature for agentic LLM calls (0.0 = greedy). NVIDIA endpoints allow up to 1.0; other OpenAI-compatible endpoints allow up to 2.0.",
    ),
) -> None:
    if output_format not in ("hits", "evidence"):
        typer.echo(f"Error: unknown --format {output_format!r} (use 'hits' or 'evidence').", err=True)
        raise typer.Exit(1)
    if max_text_chars is not None and output_format != "hits":
        typer.echo("Error: --max-text-chars only applies to --format hits.", err=True)
        raise typer.Exit(1)
    if reranker_invoke_url is None:
        reranker_invoke_url = os.environ.get("RERANKER_INVOKE_URL") or None
    if embed_invoke_url is None:
        embed_invoke_url = os.environ.get("EMBED_INVOKE_URL") or None
    rerank = rerank or bool(reranker_invoke_url) or bool(reranker_model_name) or bool(reranker_backend)
    _silence_noisy_libraries()
    if agentic and not agentic_llm_model:
        typer.echo("Error: --agentic requires --agentic-llm-model.", err=True)
        raise typer.Exit(1)
    if agentic:
        backend_error = agentic_backend_top_k_error(agentic_backend_top_k, target_top_k=top_k)
        if backend_error:
            typer.echo(f"Error: {backend_error}", err=True)
            raise typer.Exit(1)
        temperature_error = agentic_temperature_error(agentic_temperature, invoke_url=agentic_invoke_url)
        if temperature_error:
            typer.echo(f"Error: {temperature_error}", err=True)
            raise typer.Exit(1)
        request = QueryRequest(
            query=query,
            retrieval=QueryRetrievalOptions(
                top_k=top_k,
                candidate_k=candidate_k,
                page_dedup=page_dedup,
                content_types=content_types,
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
            ),
            storage=QueryStorageOptions(
                lancedb_uri=lancedb_uri,
                table_name=table_name,
            ),
            agentic=QueryAgenticOptions(
                enabled=agentic,
                llm_model=agentic_llm_model,
                invoke_url=agentic_invoke_url,
                reasoning_effort=agentic_reasoning_effort,
                backend_top_k=agentic_backend_top_k,
                react_max_steps=agentic_react_max_steps,
                text_truncation=agentic_text_truncation,
                temperature=agentic_temperature,
            ),
        )
        try:
            with _quiet_capture():
                ranked = agentic_query_documents(request)
        except _ROOT_CLI_ERRORS as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        typer.echo(json.dumps(ranked, indent=2, sort_keys=True, default=str))
        return

    try:
        reranker_api_key = _api_key_from_env_option(reranker_api_key_env) if reranker_invoke_url else None

        def _run(use_hybrid: bool) -> list:
            return query_documents(
                QueryRequest(
                    query=query,
                    retrieval=QueryRetrievalOptions(
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
            )

        with _quiet_capture():
            if hybrid:
                try:
                    hits = _run(True)
                    strategies = ["semantic", "lexical"]
                except Exception:  # noqa: BLE001 — e.g. table has no FTS index; degrade to vector-only
                    hits = _run(False)
                    strategies = ["semantic"]
            else:
                hits = _run(False)
                strategies = ["semantic"]
    except _ROOT_CLI_ERRORS as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if output_format == "evidence":
        result = build_evidence_result(hits, strategies)
        typer.echo(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(
            json.dumps([_query_cli_hit(hit, max_text_chars) for hit in hits], indent=2, sort_keys=True, default=str)
        )


@app.callback()
def _callback(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show retriever version metadata and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    _ = version
