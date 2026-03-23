# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Graph-based batch ingestion pipeline using AbstractOperator, Node, Graph,
and RayDataExecutor.

This example mirrors the PDF path of ``batch_pipeline.py`` but constructs the
pipeline as an explicit operator graph instead of the fluent ingestor API.

Run with::

    source /opt/retriever_runtime/bin/activate
    python -m nemo_retriever.examples.graph_pipeline <input-dir-or-file> \\
        --page-elements-invoke-url <url> \\
        --ocr-invoke-url <url> \\
        --embed-invoke-url <url>
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import typer

from nemo_retriever.utils.executor import RayDataExecutor

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
    ray_address: Optional[str] = typer.Option(
        None,
        "--ray-address",
        help="Ray cluster address. Omit for local Ray.",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help="Default Ray Data batch size for map_batches stages.",
    ),
    lancedb_uri: str = typer.Option(
        "lancedb_graph",
        "--lancedb-uri",
        help="LanceDB URI/path for this run.",
    ),
    query_csv: Path = typer.Option(
        "./data/bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help="Path to query CSV for recall evaluation. Recall is skipped if the file does not exist.",
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

    # -- Operator classes and their kwargs (deferred construction on workers) --
    from nemo_retriever.utils.convert.to_pdf import DocToPdfConversionActor
    from nemo_retriever.pdf.split import PDFSplitActor
    from nemo_retriever.pdf.extract import PDFExtractionActor
    from nemo_retriever.page_elements.page_elements import PageElementDetectionActor
    from nemo_retriever.ocr.ocr import OCRActor
    from nemo_retriever.ingest_modes.batch import ExplodeContentActor, _BatchEmbedActor
    from nemo_retriever.params import EmbedParams

    extract_kwargs: dict[str, Any] = {
        "method": method,
        "dpi": dpi,
        "extract_text": True,
        "extract_tables": True,
        "extract_charts": True,
        "extract_page_as_image": True,
    }

    detect_kwargs: dict[str, Any] = {}
    if page_elements_invoke_url:
        detect_kwargs["invoke_url"] = page_elements_invoke_url
    if remote_api_key:
        detect_kwargs["api_key"] = remote_api_key

    ocr_kwargs: dict[str, Any] = {
        "extract_tables": True,
        "extract_charts": True,
    }
    if ocr_invoke_url:
        ocr_kwargs["ocr_invoke_url"] = ocr_invoke_url
    if remote_api_key:
        ocr_kwargs["api_key"] = remote_api_key

    embed_params = EmbedParams(
        model_name=embed_model_name,
        embed_invoke_url=embed_invoke_url,
        api_key=remote_api_key if embed_invoke_url else None,
    )

    # -- Build the graph using >> chaining -------------------------------------
    # Operators are auto-wrapped in Nodes by the >> operator.
    # The RayDataExecutor uses operator_class + operator_kwargs to
    # reconstruct them on each Ray worker (deferred construction), so heavy
    # GPU models are loaded there instead of being pickled from the driver.
    explode_kwargs: dict[str, Any] = {"modality": "text"}

    graph = (
        DocToPdfConversionActor()
        >> PDFSplitActor()
        >> PDFExtractionActor(**extract_kwargs)
        >> PageElementDetectionActor(**detect_kwargs)
        >> OCRActor(**ocr_kwargs)
        >> ExplodeContentActor(**explode_kwargs)
        >> _BatchEmbedActor(params=embed_params)
    )

    logger.info("Pipeline graph: %s", graph)

    # -- Configure per-node Ray Data overrides ---------------------------------
    # Node names default to the operator class name when auto-wrapped.
    node_overrides: dict[str, dict[str, Any]] = {
        "DocToPdfConversionActor": {"batch_size": 1, "num_cpus": 1},
        "PDFSplitActor": {"batch_size": 1, "num_cpus": 1},
        "PDFExtractionActor": {"batch_size": 4, "num_cpus": 1},
        "PageElementDetectionActor": {"batch_size": 8, "num_gpus": 0.0 if page_elements_invoke_url else 0.5},
        "OCRActor": {"batch_size": 8, "num_gpus": 0.0 if ocr_invoke_url else 0.5},
        "ExplodeContentActor": {"batch_size": 256, "num_cpus": 1, "num_gpus": 0},
        "_BatchEmbedActor": {"batch_size": 256, "num_gpus": 0.0 if embed_invoke_url else 0.5},
    }

    # -- Execute ---------------------------------------------------------------
    executor = RayDataExecutor(
        graph,
        ray_address=ray_address,
        batch_size=batch_size,
        node_overrides=node_overrides,
    )

    logger.info("Starting ingestion of %s ...", input_path)
    t0 = time.perf_counter()

    result_ds = executor.ingest(file_patterns)

    ingestion_time = time.perf_counter() - t0
    ingest_local_results = result_ds.take_all()
    logger.info("Ingestion complete: %d rows in %.2f seconds.", len(ingest_local_results), ingestion_time)

    # -- Download results from Ray ---------------------------------------------
    import ray

    download_start = time.perf_counter()
    # ingest_local_results = result_ds.take_all()
    download_time = time.perf_counter() - download_start
    logger.info("Ray dataset download: %.2f seconds.", download_time)

    # -- Write to LanceDB ------------------------------------------------------
    from nemo_retriever.vector_store.lancedb_store import handle_lancedb
    from nemo_retriever.ingest_modes.lancedb_utils import lancedb_schema

    lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
    lancedb_table = "nv-ingest"

    # Ensure table exists
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

    lancedb_start = time.perf_counter()
    handle_lancedb(ingest_local_results, lancedb_uri, lancedb_table, hybrid=hybrid, mode="overwrite")
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
        evaluation_query_count = len(_df_query.index)

        num_rows = result_ds.groupby("source_id").count().count()
        total_time = time.perf_counter() - t0

        ray.shutdown()

        print_run_summary(
            num_rows,
            input_path,
            hybrid,
            lancedb_uri,
            lancedb_table,
            total_time,
            ingestion_time,
            download_time,
            lancedb_time,
            recall_time,
            evaluation_metrics,
            evaluation_label="Recall",
            evaluation_count=evaluation_query_count,
        )
    else:
        ray.shutdown()
        logger.warning("Query CSV not found at %s; skipping recall evaluation.", query_csv)
        logger.info("Total time: %.2f seconds.", time.perf_counter() - t0)


if __name__ == "__main__":
    app()
