# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import logging

from pydantic import ValidationError
import typer

from nemo_retriever.adapters.cli.sdk_workflow import (
    IngestRunModeValue,
    OcrLangValue,
    OcrVersionValue,
    ingest_documents,
    query_documents,
)
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
    ("recall", "nemo_retriever.recall", "app"),
    ("skill-eval", "nemo_retriever.skill_eval", "app"),
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

_ROOT_CLI_ERRORS = (OSError, RuntimeError, ValueError, ValidationError)


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
    overwrite: bool = typer.Option(
        True,
        "--overwrite/--append",
        help=(
            "Overwrite the target LanceDB table by default. Use --append to add rows to an existing "
            "table without duplicate checks; rerunning the same inputs in append mode creates duplicates."
        ),
    ),
    ray_address: str | None = typer.Option(None, "--ray-address", help="Ray address for batch run mode."),
    ray_log_to_driver: bool | None = typer.Option(
        None,
        "--ray-log-to-driver/--no-ray-log-to-driver",
        help="Forward Ray worker logs to the driver in batch run mode.",
    ),
    page_elements_invoke_url: str | None = typer.Option(
        None,
        "--page-elements-invoke-url",
        help="Page-elements NIM endpoint URL.",
    ),
    ocr_invoke_url: str | None = typer.Option(None, "--ocr-invoke-url", help="OCR NIM endpoint URL."),
    ocr_version: OcrVersionValue | None = typer.Option(
        None,
        "--ocr-version",
        help="OCR engine version for extraction.",
    ),
    ocr_lang: OcrLangValue | None = typer.Option(
        None,
        "--ocr-lang",
        help="OCR v2 language selector for local extraction.",
    ),
    graphic_elements_invoke_url: str | None = typer.Option(
        None,
        "--graphic-elements-invoke-url",
        help="Graphic-elements NIM endpoint URL.",
    ),
    table_structure_invoke_url: str | None = typer.Option(
        None,
        "--table-structure-invoke-url",
        help="Table-structure NIM endpoint URL.",
    ),
    embed_invoke_url: str | None = typer.Option(None, "--embed-invoke-url", help="Embedding NIM endpoint URL."),
    embed_model_name: str | None = typer.Option(
        None,
        "--embed-model-name",
        help="Optional embedding model name override.",
    ),
    pdf_extract_workers: int | None = typer.Option(
        None,
        "--pdf-extract-workers",
        min=1,
        help="Maximum Ray tasks for PDF extraction in batch mode.",
    ),
    pdf_extract_batch_size: int | None = typer.Option(
        None,
        "--pdf-extract-batch-size",
        min=1,
        help="PDF extraction batch size per Ray task in batch mode.",
    ),
    pdf_extract_cpus_per_task: float | None = typer.Option(
        None,
        "--pdf-extract-cpus-per-task",
        min=0.0,
        help="CPUs reserved per PDF extraction Ray task in batch mode.",
    ),
    page_elements_workers: int | None = typer.Option(
        None,
        "--page-elements-workers",
        min=1,
        help="Number of Ray actors for page-element detection in batch mode.",
    ),
    page_elements_batch_size: int | None = typer.Option(
        None,
        "--page-elements-batch-size",
        min=1,
        help="Page-element detection batch size per actor in batch mode.",
    ),
    page_elements_cpus_per_actor: float | None = typer.Option(
        None,
        "--page-elements-cpus-per-actor",
        min=0.0,
        help="CPUs reserved per page-element detection actor in batch mode.",
    ),
    ocr_workers: int | None = typer.Option(
        None,
        "--ocr-workers",
        min=1,
        help="Number of Ray actors for OCR inference in batch mode.",
    ),
    ocr_batch_size: int | None = typer.Option(
        None,
        "--ocr-batch-size",
        min=1,
        help="OCR inference batch size per actor in batch mode.",
    ),
    ocr_cpus_per_actor: float | None = typer.Option(
        None,
        "--ocr-cpus-per-actor",
        min=0.0,
        help="CPUs reserved per OCR actor in batch mode.",
    ),
    embed_workers: int | None = typer.Option(
        None,
        "--embed-workers",
        min=1,
        help="Number of Ray actors for embedding in batch mode.",
    ),
    embed_batch_size: int | None = typer.Option(
        None,
        "--embed-batch-size",
        min=1,
        help="Embedding batch size per actor in batch mode.",
    ),
    embed_cpus_per_actor: float | None = typer.Option(
        None,
        "--embed-cpus-per-actor",
        min=0.0,
        help="CPUs reserved per embedding actor in batch mode.",
    ),
) -> None:
    try:
        summary = ingest_documents(
            documents,
            run_mode=run_mode,
            ray_address=ray_address,
            ray_log_to_driver=ray_log_to_driver,
            lancedb_uri=lancedb_uri,
            table_name=table_name,
            overwrite=overwrite,
            page_elements_invoke_url=page_elements_invoke_url,
            ocr_invoke_url=ocr_invoke_url,
            ocr_version=ocr_version,
            ocr_lang=ocr_lang,
            graphic_elements_invoke_url=graphic_elements_invoke_url,
            table_structure_invoke_url=table_structure_invoke_url,
            embed_invoke_url=embed_invoke_url,
            embed_model_name=embed_model_name,
            pdf_extract_workers=pdf_extract_workers,
            pdf_extract_batch_size=pdf_extract_batch_size,
            pdf_extract_cpus_per_task=pdf_extract_cpus_per_task,
            page_elements_workers=page_elements_workers,
            page_elements_batch_size=page_elements_batch_size,
            page_elements_cpus_per_actor=page_elements_cpus_per_actor,
            ocr_workers=ocr_workers,
            ocr_batch_size=ocr_batch_size,
            ocr_cpus_per_actor=ocr_cpus_per_actor,
            embed_workers=embed_workers,
            embed_batch_size=embed_batch_size,
            embed_cpus_per_actor=embed_cpus_per_actor,
        )
    except _ROOT_CLI_ERRORS as exc:
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
    embed_invoke_url: str | None = typer.Option(None, "--embed-invoke-url", help="Embedding NIM endpoint URL."),
    embed_model_name: str | None = typer.Option(
        None,
        "--embed-model-name",
        help="Optional embedding model name override.",
    ),
    reranker_invoke_url: str | None = typer.Option(None, "--reranker-invoke-url", help="Reranker NIM endpoint URL."),
) -> None:
    try:
        hits = query_documents(
            query,
            top_k=top_k,
            lancedb_uri=lancedb_uri,
            table_name=table_name,
            embed_invoke_url=embed_invoke_url,
            embed_model_name=embed_model_name,
            reranker_invoke_url=reranker_invoke_url,
        )
    except _ROOT_CLI_ERRORS as exc:
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
