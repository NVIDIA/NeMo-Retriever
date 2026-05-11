# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import typer

from nemo_retriever.adapters.cli.sdk_workflow import IngestRunModeValue, ingest_documents, query_documents
from nemo_retriever.audio import app as audio_app
from nemo_retriever.utils.benchmark import app as benchmark_app
from nemo_retriever.chart import app as chart_app
from nemo_retriever.utils.compare import app as compare_app
from nemo_retriever.evaluation.cli import app as eval_app
from nemo_retriever.harness import app as harness_app
from nemo_retriever.html import __main__ as html_main
from nemo_retriever.utils.image import app as image_app
from nemo_retriever.local import app as local_app
from nemo_retriever.pdf import app as pdf_app
from nemo_retriever.pipeline import __main__ as pipeline_main
from nemo_retriever.recall import app as recall_app
from nemo_retriever.service.cli import app as service_app
from nemo_retriever.txt import __main__ as txt_main
from nemo_retriever.vector_store import app as vector_store_app
from nemo_retriever.version import get_version_info


app = typer.Typer(help="Retriever")
app.add_typer(audio_app, name="audio")
app.add_typer(image_app, name="image")
app.add_typer(pdf_app, name="pdf")
app.add_typer(local_app, name="local")
app.add_typer(chart_app, name="chart")
app.add_typer(compare_app, name="compare")
app.add_typer(eval_app, name="eval")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(harness_app, name="harness")
app.add_typer(vector_store_app, name="vector-store")
app.add_typer(recall_app, name="recall")
app.add_typer(service_app, name="service")
app.add_typer(txt_main.app, name="txt")
app.add_typer(html_main.app, name="html")
app.add_typer(pipeline_main.app, name="pipeline")


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
