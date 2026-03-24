# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Remote service CLI: start the Ray Serve + FastAPI REST API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(
    help="Remote service: Ray Serve + FastAPI REST API (serve) and CLI to submit documents (submit).",
)


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host for the HTTP server."),
    port: int = typer.Option(7670, "--port", help="Port for the HTTP server."),
    ray_address: Optional[str] = typer.Option(None, "--ray-address", help="Ray cluster address (default: local)."),
    lancedb_uri: str = typer.Option("lancedb", "--lancedb-uri", help="LanceDB URI for retrieval."),
    lancedb_table: str = typer.Option("nv-ingest", "--lancedb-table", help="LanceDB table name."),
    embedding_model: str = typer.Option(
        "nvidia/llama-nemotron-embed-1b-v2", "--embedding-model", help="Embedding model name."
    ),
    embedding_endpoint: Optional[str] = typer.Option(
        None, "--embedding-endpoint", help="Remote embedding NIM endpoint URL."
    ),
    embedding_api_key: str = typer.Option("", "--embedding-api-key", help="API key for remote embedding endpoint."),
    top_k: int = typer.Option(10, "--top-k", help="Default number of retrieval results."),
    reranker: bool = typer.Option(False, "--reranker", help="Enable reranking on retrieval results."),
    reranker_endpoint: Optional[str] = typer.Option(
        None, "--reranker-endpoint", help="Remote reranker NIM endpoint URL."
    ),
) -> None:
    """Start the NeMo Retriever remote Ray Serve + FastAPI REST API."""
    try:
        import ray
        from ray import serve
        from ray.serve.config import HTTPOptions
    except ImportError as e:
        typer.echo(f"Ray Serve is required: {e}", err=True)
        raise typer.Exit(1)

    from nemo_retriever.adapters.service.app import RetrieverAPIDeployment

    # Prevent Ray from packaging the working directory into a runtime env zip.
    # The editable path deps (../src, ../api, ../client) in pyproject.toml
    # don't resolve inside Ray's temp staging area. Local workers already
    # have the installed packages available.
    runtime_env = {"working_dir": None}

    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True, runtime_env=runtime_env)
    else:
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    serve.start(http_options=HTTPOptions(host=host, port=port))

    serve.run(
        RetrieverAPIDeployment.bind(
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
            embedding_model=embedding_model,
            embedding_endpoint=embedding_endpoint,
            embedding_api_key=embedding_api_key,
            top_k=top_k,
            reranker=reranker,
            reranker_endpoint=reranker_endpoint,
        ),
        name="retriever_api",
        route_prefix="/",
        blocking=True,
    )


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".html",
    ".htm",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".rst",
    ".xml",
    ".yaml",
    ".yml",
    ".docx",
    ".pptx",
    ".xlsx",
}


def _collect_files(paths: list[Path]) -> list[Path]:
    """Expand directories into individual files, filtering to supported types."""
    collected: list[Path] = []
    for p in paths:
        if p.is_dir():
            collected.extend(
                f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif p.is_file():
            collected.append(p)
    return collected


@app.command("ingest")
def ingest_cmd(
    paths: List[Path] = typer.Argument(
        ...,
        help="Files or directories to ingest. Directories are searched recursively.",
        path_type=Path,
        exists=True,
    ),
    run_mode: str = typer.Option(
        "inprocess",
        "--run-mode",
        "-m",
        help="Ingestor run mode.",
        show_default=True,
    ),
    base_url: str = typer.Option(
        "http://localhost:7670",
        "--base-url",
        "-u",
        help="Base URL of the running service.",
    ),
    timeout: int = typer.Option(
        300,
        "--timeout",
        "-t",
        help="Request timeout in seconds.",
    ),
) -> None:
    """Send documents to the /ingest endpoint of a running NeMo Retriever service."""
    import requests

    files = _collect_files(paths)
    if not files:
        typer.echo("No supported files found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Ingesting {len(files)} file(s) [run_mode={run_mode}]")
    for f in files:
        typer.echo(f"  {f.name}", err=True)

    url = f"{base_url.rstrip('/')}/ingest"
    handles = [("files", (f.name, f.open("rb"))) for f in files]
    try:
        resp = requests.post(
            url,
            files=handles,
            params={"run_mode": run_mode},
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        num_docs = body.get("num_documents", "?")
        pages = body.get("pages", [])
        typer.echo(f"OK — {num_docs} document(s), {len(pages)} page(s) ingested.")
        for pg in pages:
            text_preview = (pg.get("text") or "")[:120]
            if text_preview:
                text_preview = text_preview.replace("\n", " ").strip()
                text_preview = f"  {text_preview}{'...' if len(pg.get('text', '')) > 120 else ''}"
            typer.echo(f"  p{pg['page_number']}: {pg.get('filename', '')}")
            if text_preview:
                typer.echo(text_preview)
    except requests.RequestException as e:
        typer.echo(f"Error: {e}", err=True)
        if hasattr(e, "response") and e.response is not None:
            typer.echo(e.response.text, err=True)
        sys.exit(1)
    finally:
        for _, (_, fh) in handles:
            fh.close()


@app.command("stream-pdf")
def stream_pdf_cmd(
    files: List[Path] = typer.Argument(
        ...,
        help="PDF files to stream (POST /stream-pdf, print NDJSON per page).",
        path_type=Path,
        exists=True,
    ),
    base_url: str = typer.Option(
        "http://localhost:7670",
        "--base-url",
        "-u",
        help="Base URL of the running service.",
    ),
) -> None:
    """Submit PDFs to POST /stream-pdf and print the streaming page-by-page text (NDJSON)."""
    import requests

    url = f"{base_url.rstrip('/')}/stream-pdf"
    for path in files:
        path = path.resolve()
        typer.echo(f"--- {path} ---", err=True)
        try:
            with path.open("rb") as file_handle:
                response = requests.post(
                    url,
                    files={"file": (path.name, file_handle, "application/pdf")},
                    stream=True,
                    timeout=60,
                )
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    page = obj.get("page", "?")
                    text = obj.get("text", "")
                    typer.echo(f"Page {page}:")
                    typer.echo(text if text else "(no text)")
                except json.JSONDecodeError:
                    typer.echo(line)
        except requests.RequestException as e:
            typer.echo(f"Error: {e}", err=True)
            sys.exit(1)
        typer.echo("", err=True)


def main() -> None:
    app()
