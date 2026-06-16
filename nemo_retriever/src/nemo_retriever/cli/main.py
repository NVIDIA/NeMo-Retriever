# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import logging
import os

import typer

from nemo_retriever.cli.ingest import app as ingest_app
from nemo_retriever.cli.shared import (
    ROOT_CLI_ERRORS as _ROOT_CLI_ERRORS,
    quiet_capture as _quiet_capture,
    silence_noisy_libraries as _silence_noisy_libraries,
)
from nemo_retriever.cli.query_workflow import query_documents
from nemo_retriever.query.options import (
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


def _query_cli_hit(hit: RetrievalHit) -> dict[str, object]:
    return {
        "source": hit.get("source", ""),
        "page_number": hit.get("page_number"),
        "text": hit.get("text", ""),
    }


def _version_callback(value: bool) -> None:
    if not value:
        return
    info = get_version_info()
    typer.echo(info["full_version"])
    raise typer.Exit()


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
    reranker_invoke_url: str | None = typer.Option(None, "--reranker-invoke-url", help="Reranker NIM endpoint URL."),
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
) -> None:
    if reranker_invoke_url is None:
        reranker_invoke_url = os.environ.get("RERANKER_INVOKE_URL") or None
    if embed_invoke_url is None:
        embed_invoke_url = os.environ.get("EMBED_INVOKE_URL") or None
    rerank = rerank or bool(reranker_invoke_url) or bool(reranker_model_name) or bool(reranker_backend)
    _silence_noisy_libraries()
    try:
        with _quiet_capture():
            hits = query_documents(
                QueryRequest(
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
                )
            )
    except _ROOT_CLI_ERRORS as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(json.dumps([_query_cli_hit(hit) for hit in hits], indent=2, sort_keys=True, default=str))


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
