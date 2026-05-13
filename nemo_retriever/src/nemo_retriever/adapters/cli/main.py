# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import logging

import typer

from nemo_retriever.adapters.cli.sdk_workflow import IngestRunModeValue, ingest_documents, query_documents
from nemo_retriever.version import get_version_info

logger = logging.getLogger(__name__)

app = typer.Typer(help="Retriever")

# Service sub-app is always available (lightweight, no GPU deps).
from nemo_retriever.service.cli import app as service_app  # noqa: E402

app.add_typer(service_app, name="service")

# All other sub-apps are registered lazily so that missing optional
# dependencies (tritonclient, torch, …) don't prevent the service
# from starting.
_LAZY_SUBAPPS: list[tuple[str, str, str]] = [
    ("audio", "nemo_retriever.audio", "app"),
    ("image", "nemo_retriever.utils.image", "app"),
    ("pdf", "nemo_retriever.pdf", "app"),
    ("local", "nemo_retriever.local", "app"),
    ("chart", "nemo_retriever.chart", "app"),
    ("compare", "nemo_retriever.utils.compare", "app"),
    ("eval", "nemo_retriever.evaluation.cli", "app"),
    ("benchmark", "nemo_retriever.utils.benchmark", "app"),
    ("harness", "nemo_retriever.harness", "app"),
    ("vector-store", "nemo_retriever.vector_store", "app"),
    ("recall", "nemo_retriever.recall", "app"),
    ("txt", "nemo_retriever.txt.__main__", "app"),
    ("html", "nemo_retriever.html.__main__", "app"),
    ("pipeline", "nemo_retriever.pipeline.__main__", "app"),
]

for _name, _module, _attr in _LAZY_SUBAPPS:
    try:
        _mod = importlib.import_module(_module)
        app.add_typer(getattr(_mod, _attr), name=_name)
    except Exception:
        logger.debug("Skipping '%s' sub-command (import failed)", _name)


def _version_callback(value: bool) -> None:
    if not value:
        return
    info = get_version_info()
    typer.echo(info["full_version"])
    raise typer.Exit()


def main() -> None:
    app()


@app.command("ingest")
def ingest_command(
    documents: list[str] = typer.Argument(
        ...,
        help="One or more PDF file paths, directories containing PDFs, or PDF globs to ingest.",
    ),
    lancedb_uri: str = typer.Option("lancedb", "--lancedb-uri", help="LanceDB database URI."),
    table_name: str = typer.Option("nv-ingest", "--table-name", help="LanceDB table name."),
    run_mode: IngestRunModeValue = typer.Option(
        "inprocess",
        "--run-mode",
        help="Execution mode for the SDK ingestor.",
    ),
) -> None:
    try:
        summary = ingest_documents(
            documents,
            run_mode=run_mode,
            lancedb_uri=lancedb_uri,
            table_name=table_name,
        )
    except (FileNotFoundError, IsADirectoryError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        f"Ingested {len(summary['documents'])} document(s) into LanceDB "
        f"{summary['lancedb_uri']}/{summary['table_name']}."
    )


@app.command("query")
def query_command(
    query: str = typer.Argument(..., help="Query text."),
    top_k: int = typer.Option(10, "--top-k", min=1, help="Number of hits to retrieve."),
    lancedb_uri: str = typer.Option("lancedb", "--lancedb-uri", help="LanceDB database URI."),
    table_name: str = typer.Option("nv-ingest", "--table-name", help="LanceDB table name."),
) -> None:
    try:
        hits = query_documents(query, top_k=top_k, lancedb_uri=lancedb_uri, table_name=table_name)
    except (FileNotFoundError, IsADirectoryError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(json.dumps(list(hits), indent=2, sort_keys=True, default=str))


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
