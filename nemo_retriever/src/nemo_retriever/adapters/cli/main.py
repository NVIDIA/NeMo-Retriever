# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import importlib
import io
import json
import logging
import os
import sys
import tempfile

from pydantic import ValidationError
import typer

from nemo_retriever.adapters.cli.sdk_workflow import (
    IngestInputTypeValue,
    IngestRunModeValue,
    LocalIngestEmbedBackendValue,
    OcrLangValue,
    OcrVersionValue,
    TableOutputFormatValue,
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


def _silence_noisy_libraries() -> None:
    # vLLM/transformers/HuggingFace otherwise emit dozens of INFO-level lines
    # + tqdm progress bars (CUDA kernel compile, weight download, "Loading
    # safetensors checkpoint shards", "Capturing CUDA graphs (PIECEWISE)").
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("vllm").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)


@contextlib.contextmanager
def _quiet_capture():
    """Capture stdout AND stderr at the OS fd level inside the ``with``
    block (so output from C libraries and child processes is captured too,
    not just Python prints). On normal exit the captured buffer is
    discarded. On any exception the buffer is flushed to the real stderr
    before the exception propagates, so an agent or human can debug the
    failure.

    When stdout/stderr aren't real OS-level streams (e.g. under pytest's
    sys-capture, where they're StringIO), skip the fd dance and yield
    plainly."""
    try:
        stdout_fd, stderr_fd = sys.stdout.fileno(), sys.stderr.fileno()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        yield
        return

    saved_stdout = saved_stderr = buf = None
    try:
        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)
        buf = tempfile.TemporaryFile(mode="w+b")
        try:
            try:
                os.dup2(buf.fileno(), stdout_fd)
                os.dup2(buf.fileno(), stderr_fd)
                yield
            finally:
                # Always restore; if a dup2 above failed, dup2-ing saved_* back
                # over the still-original fd is a harmless no-op.
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_stdout, stdout_fd)
                os.dup2(saved_stderr, stderr_fd)
        except BaseException:
            buf.seek(0)
            sys.stderr.buffer.write(buf.read())
            sys.stderr.flush()
            raise
    finally:
        if buf is not None:
            buf.close()
        if saved_stderr is not None:
            os.close(saved_stderr)
        if saved_stdout is not None:
            os.close(saved_stdout)


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
        help="One or more file paths, directories, or globs to ingest.",
    ),
    input_type: IngestInputTypeValue = typer.Option(
        "auto",
        "--input-type",
        help="Input type: auto, pdf, doc, txt, html, image, audio, or video.",
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
    table_output_format: TableOutputFormatValue | None = typer.Option(
        None,
        "--table-output-format",
        help="Table text format. 'markdown' enables local table-structure extraction.",
    ),
    embed_invoke_url: str | None = typer.Option(None, "--embed-invoke-url", help="Embedding NIM endpoint URL."),
    embed_model_name: str | None = typer.Option(
        None,
        "--embed-model-name",
        help="Optional embedding model name override.",
    ),
    local_ingest_embed_backend: LocalIngestEmbedBackendValue | None = typer.Option(
        None,
        "--local-ingest-embed-backend",
        help="Local ingest-time text embedder when --embed-invoke-url is unset.",
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
    page_elements_gpus_per_actor: float | None = typer.Option(
        None,
        "--page-elements-gpus-per-actor",
        min=0.0,
        help="GPUs reserved per local page-element detection actor in batch mode.",
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
    ocr_gpus_per_actor: float | None = typer.Option(
        None,
        "--ocr-gpus-per-actor",
        min=0.0,
        help="GPUs reserved per local OCR actor in batch mode.",
    ),
    table_structure_workers: int | None = typer.Option(
        None,
        "--table-structure-workers",
        min=1,
        help="Number of Ray actors for table-structure extraction in batch mode.",
    ),
    table_structure_batch_size: int | None = typer.Option(
        None,
        "--table-structure-batch-size",
        min=1,
        help="Table-structure extraction batch size per actor in batch mode.",
    ),
    table_structure_cpus_per_actor: float | None = typer.Option(
        None,
        "--table-structure-cpus-per-actor",
        min=0.0,
        help="CPUs reserved per table-structure actor in batch mode.",
    ),
    table_structure_gpus_per_actor: float | None = typer.Option(
        None,
        "--table-structure-gpus-per-actor",
        min=0.0,
        help="GPUs reserved per local table-structure actor in batch mode.",
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
    embed_gpus_per_actor: float | None = typer.Option(
        None,
        "--embed-gpus-per-actor",
        min=0.0,
        help="GPUs reserved per local embedding actor in batch mode.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help=(
            "Suppress verbose progress output (progress bars, HuggingFace "
            "downloads, vLLM init logs). On success, prints only the final "
            "summary line. On error, flushes all captured output to stderr "
            "for debugging."
        ),
    ),
) -> None:
    if quiet:
        _silence_noisy_libraries()
    capture = _quiet_capture() if quiet else contextlib.nullcontext()
    try:
        with capture:
            summary = ingest_documents(
                documents,
                input_type=input_type,
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
                table_output_format=table_output_format,
                embed_invoke_url=embed_invoke_url,
                embed_model_name=embed_model_name,
                local_ingest_embed_backend=local_ingest_embed_backend,
                pdf_extract_workers=pdf_extract_workers,
                pdf_extract_batch_size=pdf_extract_batch_size,
                pdf_extract_cpus_per_task=pdf_extract_cpus_per_task,
                page_elements_workers=page_elements_workers,
                page_elements_batch_size=page_elements_batch_size,
                page_elements_cpus_per_actor=page_elements_cpus_per_actor,
                page_elements_gpus_per_actor=page_elements_gpus_per_actor,
                ocr_workers=ocr_workers,
                ocr_batch_size=ocr_batch_size,
                ocr_cpus_per_actor=ocr_cpus_per_actor,
                ocr_gpus_per_actor=ocr_gpus_per_actor,
                table_structure_workers=table_structure_workers,
                table_structure_batch_size=table_structure_batch_size,
                table_structure_cpus_per_actor=table_structure_cpus_per_actor,
                table_structure_gpus_per_actor=table_structure_gpus_per_actor,
                embed_workers=embed_workers,
                embed_batch_size=embed_batch_size,
                embed_cpus_per_actor=embed_cpus_per_actor,
                embed_gpus_per_actor=embed_gpus_per_actor,
            )
    except _ROOT_CLI_ERRORS as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    # Report input-file count alongside the actual landed-row count from the
    # LanceDB table — they diverge whenever one document explodes into multiple
    # chunks (PDFs → page elements, video → audio_visual segments) or
    # shrinks to zero rows when every NIM call failed. The previous message
    # only reported inputs and hid both cases. ``n_rows`` is None when the
    # table read itself failed (caller can still see file count + URI).
    n_files = len(summary["documents"])
    table_path = f"{summary['lancedb_uri']}/{summary['table_name']}"
    n_rows = summary.get("n_rows")
    if n_rows is None:
        typer.echo(f"Ingested {n_files} file(s) into LanceDB {table_path} (row count unavailable).")
    else:
        typer.echo(f"Ingested {n_files} file(s) → {n_rows} row(s) in LanceDB {table_path}.")


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
                query,
                top_k=top_k,
                lancedb_uri=lancedb_uri,
                table_name=table_name,
                embed_invoke_url=embed_invoke_url,
                embed_model_name=embed_model_name,
                reranker_invoke_url=reranker_invoke_url,
                reranker_model_name=reranker_model_name,
                reranker_backend=reranker_backend,
                rerank=rerank,
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
