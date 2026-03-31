# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ingest a directory of PDFs via the remote REST API, then query the results.

Requires a running NeMo Retriever service (``retriever remote serve``).

Run with::

    python -m nemo_retriever.examples.remote_ingest_and_retrieve ./my_pdfs/

Or with explicit options::

    python -m nemo_retriever.examples.remote_ingest_and_retrieve ./my_pdfs/ \\
        --base-url http://localhost:7670 \\
        --query "What are the key findings?" \\
        --top-k 5 \\
        --stream
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import typer

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing PDF files to ingest.",
        path_type=Path,
    ),
    base_url: str = typer.Option(
        "http://localhost:7670",
        "--base-url",
        help="Base URL of the NeMo Retriever service.",
    ),
    query: str = typer.Option(
        "Summarize the main topics covered in these documents.",
        "--query",
        help="Query to run after ingestion completes.",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        help="Number of retrieval results to return.",
    ),
    stream: bool = typer.Option(
        False,
        "--stream/--no-stream",
        help="Use the streaming ingest endpoint for page-by-page progress.",
    ),
    lancedb_uri: Optional[str] = typer.Option(
        None,
        "--lancedb-uri",
        help="Override the server-side LanceDB URI.",
    ),
    lancedb_table: Optional[str] = typer.Option(
        None,
        "--lancedb-table",
        help="Override the server-side LanceDB table name.",
    ),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise typer.BadParameter(f"Not a directory: {input_dir}")

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise typer.BadParameter(f"No PDF files found in {input_dir}")

    logger.info("Found %d PDF(s) in %s", len(pdf_files), input_dir)

    # ------------------------------------------------------------------
    # 1. Ingest
    # ------------------------------------------------------------------
    from nemo_retriever.client import RemoteIngestor, RemoteRetriever

    ingestor = RemoteIngestor(base_url).files([str(p) for p in pdf_files]).extract().embed().vdb_upload()

    t0 = time.perf_counter()

    if stream:
        logger.info("Starting streaming ingestion ...")
        pages_ok = 0
        pages_err = 0
        for record in ingestor.ingest_stream():
            if record.get("_summary"):
                logger.info(
                    "Stream complete: %d pages processed, %d errors, vdb_rows=%s",
                    record.get("pages_ok", 0),
                    record.get("pages_error", 0),
                    record.get("vdb_rows_written"),
                )
            elif record.get("_error"):
                pages_err += 1
                logger.warning("Page error: %s", record.get("error"))
            else:
                pages_ok += 1
                text_preview = (record.get("text") or "")[:80]
                logger.info(
                    "Page %s of %s: %s…",
                    record.get("page_number", "?"),
                    record.get("source", "?"),
                    text_preview,
                )
        ingest_elapsed = time.perf_counter() - t0
        logger.info(
            "Streaming ingest finished in %.2fs (%d ok, %d errors).",
            ingest_elapsed,
            pages_ok,
            pages_err,
        )
    else:
        logger.info("Submitting async ingest job ...")
        job = ingestor.ingest_async()
        logger.info("Job submitted: %s", job.job_id)
        job.wait(poll_interval=2.0)
        ingest_elapsed = time.perf_counter() - t0

        status = job.status()
        job_status = status.get("status")
        logger.info(
            "Job %s finished with status '%s' in %.2fs.",
            job.job_id,
            job_status,
            ingest_elapsed,
        )

        if job_status == "failed":
            progress = status.get("progress") or {}
            error = progress.get("error") or status.get("error") or "unknown"
            logger.error("Ingestion failed: %s", error)
            raise typer.Exit(code=1)

        if job_status == "completed":
            results = job.results()
            logger.info("Ingestion produced %d result records.", len(results))

    # ------------------------------------------------------------------
    # 2. Retrieve
    # ------------------------------------------------------------------
    logger.info("Running query: %r (top_k=%d)", query, top_k)

    retriever = RemoteRetriever(
        base_url,
        top_k=top_k,
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
    )

    hits = retriever.query(
        query,
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
    )

    print(f"\n{'=' * 72}")
    print(f"Query: {query}")
    print(f"{'=' * 72}")
    for i, hit in enumerate(hits, 1):
        source = hit.get("source") or hit.get("pdf_basename") or hit.get("path") or "unknown"
        page = hit.get("page_number") or hit.get("pdf_page") or "?"
        score = hit.get("_distance") or hit.get("_rerank_score") or ""
        text = (hit.get("text") or "")[:200]
        print(f"\n--- Result {i} (source={source}, page={page}, score={score}) ---")
        print(text)
    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    app()
