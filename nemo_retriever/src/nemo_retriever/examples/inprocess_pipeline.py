# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Graph-based in-process ingestion pipeline using AbstractOperator, Node, Graph,
and InprocessExecutor.

This example mirrors the PDF path of ``inprocess_pipeline.py`` but constructs
the pipeline as an explicit operator graph instead of the fluent ingestor API.
All processing runs in a single process on pandas DataFrames — no Ray.

Run with::

    source /opt/retriever_runtime/bin/activate
    python -m nemo_retriever.examples.graph_inprocess_pipeline <input-dir-or-file>
"""

from __future__ import annotations

import logging
import pandas as pd
import time
from pathlib import Path
from typing import Optional

import typer

from nemo_retriever import create_ingestor
from nemo_retriever.params import EmbedParams, ExtractParams

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        help="File or directory containing PDFs to ingest.",
        path_type=Path,
    ),
    page_elements_invoke_url: Optional[str] = typer.Option(
        None,
        "--page-elements-invoke-url",
        help="Remote endpoint URL for page-elements model inference.",
    ),
    ocr_invoke_url: Optional[str] = typer.Option(
        None,
        "--ocr-invoke-url",
        help="Remote endpoint URL for OCR model inference.",
    ),
    embed_invoke_url: Optional[str] = typer.Option(
        None,
        "--embed-invoke-url",
        help="Remote endpoint URL for embedding model inference.",
    ),
    embed_model_name: str = typer.Option(
        "nvidia/llama-nemotron-embed-1b-v2",
        "--embed-model-name",
        help="Embedding model name.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Bearer token for remote NIM endpoints.",
    ),
    method: str = typer.Option(
        "pdfium",
        "--method",
        help="PDF text extraction method.",
    ),
    dpi: int = typer.Option(
        300,
        "--dpi",
        help="Render DPI for PDF page images.",
    ),
    lancedb_uri: str = typer.Option(
        "lancedb_graph_inprocess",
        "--lancedb-uri",
        help="LanceDB URI/path for this run.",
    ),
    query_csv: Path = typer.Option(
        "./data/bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help="Path to query CSV for recall evaluation.",
    ),
    recall_match_mode: str = typer.Option(
        "pdf_page",
        "--recall-match-mode",
        help="Recall match mode: 'pdf_page' or 'pdf_only'.",
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid/--no-hybrid",
        help="Enable LanceDB hybrid mode (dense + FTS text).",
    ),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    # -- Resolve input files ---------------------------------------------------
    input_path = Path(input_path)
    if input_path.is_file():
        file_patterns = [str(input_path)]
    elif input_path.is_dir():
        import glob as _glob

        file_patterns = _glob.glob(str(input_path / "*.pdf"))
        if not file_patterns:
            raise typer.BadParameter(f"No PDF files found in {input_path}")
    else:
        raise typer.BadParameter(f"Path does not exist: {input_path}")

    from nemo_retriever.utils.remote_auth import resolve_remote_api_key

    remote_api_key = resolve_remote_api_key(api_key)
    extract_params = ExtractParams(
        method=method,
        dpi=dpi,
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_page_as_image=True,
        page_elements_invoke_url=page_elements_invoke_url,
        ocr_invoke_url=ocr_invoke_url,
        api_key=remote_api_key,
    )
    embed_params = EmbedParams(
        model_name=embed_model_name,
        embed_invoke_url=embed_invoke_url,
        api_key=remote_api_key if embed_invoke_url else None,
    )

    logger.info("Starting in-process ingestion of %s ...", input_path)
    t0 = time.perf_counter()
    ingestor = create_ingestor(run_mode="inprocess").files(file_patterns).extract(extract_params).embed(embed_params)
    results = ingestor.ingest()
    result_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    ingestion_time = time.perf_counter() - t0
    row_count = len(result_df)
    logger.info("Ingestion complete: %d rows in %.2f seconds.", row_count, ingestion_time)

    # -- Write to LanceDB ------------------------------------------------------
    from nemo_retriever.vector_store.lancedb_store import handle_lancedb
    from nemo_retriever.vector_store.lancedb_utils import lancedb_schema

    lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
    lancedb_table = "nv-ingest"

    import lancedb as _lancedb_mod
    import pyarrow as pa

    Path(lancedb_uri).mkdir(parents=True, exist_ok=True)
    _db = _lancedb_mod.connect(lancedb_uri)
    try:
        _db.open_table(lancedb_table)
    except Exception:
        schema = lancedb_schema()
        empty = pa.table({f.name: [] for f in schema}, schema=schema)
        _db.create_table(lancedb_table, data=empty, schema=schema, mode="create")

    # Convert DataFrame to list of dicts for handle_lancedb
    local_results = result_df.to_dict("records")

    lancedb_start = time.perf_counter()
    handle_lancedb(local_results, lancedb_uri, lancedb_table, hybrid=hybrid, mode="overwrite")
    lancedb_time = time.perf_counter() - lancedb_start
    logger.info("LanceDB write: %.2f seconds.", lancedb_time)

    # -- Recall evaluation -----------------------------------------------------
    query_csv = Path(query_csv)
    if query_csv.exists():
        from nemo_retriever.model import resolve_embed_model
        from nemo_retriever.recall.core import RecallConfig, retrieve_and_score
        from nemo_retriever.utils.detection_summary import print_run_summary

        _recall_model = resolve_embed_model(str(embed_model_name))
        embed_remote_api_key = remote_api_key if embed_invoke_url else None

        recall_cfg = RecallConfig(
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
            embedding_model=_recall_model,
            embedding_http_endpoint=embed_invoke_url,
            embedding_api_key=embed_remote_api_key or "",
            top_k=10,
            ks=(1, 5, 10),
            hybrid=hybrid,
            match_mode=recall_match_mode,
        )

        recall_start = time.perf_counter()
        _df_query, _gold, _raw_hits, _retrieved_keys, evaluation_metrics = retrieve_and_score(
            query_csv=query_csv,
            cfg=recall_cfg,
        )
        recall_time = time.perf_counter() - recall_start

        total_time = time.perf_counter() - t0

        # Count unique source_ids for page count
        num_pages = result_df["source_id"].nunique() if "source_id" in result_df.columns else row_count

        print_run_summary(
            num_pages,
            input_path,
            hybrid,
            lancedb_uri,
            lancedb_table,
            total_time,
            ingestion_time,
            0.0,  # no ray download time
            lancedb_time,
            recall_time,
            evaluation_metrics,
            evaluation_label="Recall",
            evaluation_count=len(_df_query.index),
        )
    else:
        logger.warning("Query CSV not found at %s; skipping recall evaluation.", query_csv)
        logger.info("Total time: %.2f seconds.", time.perf_counter() - t0)


if __name__ == "__main__":
    app()
