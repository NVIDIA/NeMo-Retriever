# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Graph-based batch ingestion pipeline with CLI parity to ``batch_pipeline.py``.

This example keeps the explicit graph/executor wiring while exposing the same
command-line flags as the higher-level batch example for the PDF/doc path.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, TextIO

import typer

from nemo_retriever import create_ingestor
from nemo_retriever.params import CaptionParams
from nemo_retriever.params import DedupParams
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.params.models import BatchTuningParams
from nemo_retriever.recall.beir import evaluate_lancedb_beir
from nemo_retriever.utils.detection_summary import (
    collect_detection_summary_from_df,
    print_run_summary,
    write_detection_summary,
)
from nemo_retriever.utils.remote_auth import resolve_remote_api_key
from nemo_retriever.vector_store.lancedb_store import handle_lancedb

logger = logging.getLogger(__name__)
app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


def _lancedb():
    import lancedb

    return lancedb


class _TeeStream:
    def __init__(self, primary: TextIO, mirror: TextIO) -> None:
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(getattr(self._primary, "fileno")())

    def writable(self) -> bool:
        return bool(getattr(self._primary, "writable", lambda: True)())

    @property
    def encoding(self) -> str:
        return str(getattr(self._primary, "encoding", "utf-8"))


def _configure_logging(log_file: Optional[Path], *, debug: bool = False) -> tuple[Optional[TextIO], TextIO, TextIO]:
    original_stdout = os.sys.stdout
    original_stderr = os.sys.stderr
    log_level = logging.DEBUG if debug else logging.INFO
    if log_file is None:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,
        )
        return None, original_stdout, original_stderr

    target = Path(log_file).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fh = open(target, "a", encoding="utf-8", buffering=1)
    os.sys.stdout = _TeeStream(os.sys.__stdout__, fh)
    os.sys.stderr = _TeeStream(os.sys.__stderr__, fh)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(os.sys.stdout)],
        force=True,
    )
    logging.getLogger(__name__).info("Writing combined pipeline logs to %s", str(target))
    return fh, original_stdout, original_stderr


def _ensure_lancedb_table(uri: str, table_name: str) -> None:
    from nemo_retriever.vector_store.lancedb_utils import lancedb_schema
    import lancedb
    import pyarrow as pa

    Path(uri).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(uri)
    try:
        db.open_table(table_name)
        return
    except Exception:
        pass

    schema = lancedb_schema()
    empty = pa.table({f.name: [] for f in schema}, schema=schema)
    db.create_table(table_name, data=empty, schema=schema, mode="create")


def _input_file_patterns(input_path: Path, input_type: str) -> list[str]:
    input_path = Path(input_path)
    if input_path.is_file():
        return [str(input_path)]

    if not input_path.is_dir():
        raise typer.BadParameter(f"Path does not exist: {input_path}")

    ext_map = {
        "pdf": ["*.pdf"],
        "doc": ["*.docx", "*.pptx"],
    }
    exts = ext_map.get(input_type)
    if exts is None:
        raise typer.BadParameter(
            f"graph_pipeline currently supports only input_type='pdf' or 'doc'; got {input_type!r}"
        )

    import glob as _glob

    patterns = [str(input_path / ext) for ext in exts]
    file_patterns = [pattern for pattern in patterns if _glob.glob(pattern)]
    if not file_patterns:
        raise typer.BadParameter(f"No files found for input_type={input_type!r} in {input_path}")
    return file_patterns


@app.command()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(
        False, "--debug/--no-debug", help="Enable debug-level logging for this full pipeline run."
    ),
    dpi: int = typer.Option(300, "--dpi", min=72, help="Render DPI for PDF page images (default: 300)."),
    input_path: Path = typer.Argument(
        ...,
        help="File or directory containing PDFs, .txt, .html, or .doc/.pptx files to ingest.",
        path_type=Path,
    ),
    detection_summary_file: Optional[Path] = typer.Option(
        None,
        "--detection-summary-file",
        path_type=Path,
        dir_okay=False,
        help="Write a JSON detection summary for the ingested output rows to this file.",
    ),
    recall_match_mode: str = typer.Option(
        "pdf_page", "--recall-match-mode", help="Recall match mode: 'pdf_page' or 'pdf_only'."
    ),
    evaluation_mode: str = typer.Option(
        "recall", "--evaluation-mode", help="Evaluation mode to run after ingest: 'recall' or 'beir'."
    ),
    no_recall_details: bool = typer.Option(
        False, "--no-recall-details", help="Accepted for CLI parity; detailed recall output is not emitted here."
    ),
    embed_actors: Optional[int] = typer.Option(0, "--embed-actors", help="Number of embedding workers/actors to use."),
    embed_batch_size: Optional[int] = typer.Option(0, "--embed-batch-size", help="Embedding inference batch size."),
    embed_cpus_per_actor: Optional[float] = typer.Option(
        0.0, "--embed-cpus-per-actor", help="CPU resources reserved per embedding actor."
    ),
    embed_gpus_per_actor: Optional[float] = typer.Option(
        0.0, "--embed-gpus-per-actor", max=1.0, help="GPU fraction reserved per embedding actor."
    ),
    embed_granularity: str = typer.Option(
        "element", "--embed-granularity", help="Embedding granularity, such as 'element' or chunk-level output."
    ),
    beir_loader: Optional[str] = typer.Option(
        None, "--beir-loader", help="BEIR dataset loader name to use when --evaluation-mode=beir."
    ),
    beir_dataset_name: Optional[str] = typer.Option(
        None, "--beir-dataset-name", help="BEIR dataset name to evaluate when --evaluation-mode=beir."
    ),
    beir_split: str = typer.Option("test", "--beir-split", help="BEIR dataset split to evaluate."),
    beir_query_language: Optional[str] = typer.Option(
        None, "--beir-query-language", help="Optional BEIR query language filter."
    ),
    beir_doc_id_field: str = typer.Option(
        "pdf_basename", "--beir-doc-id-field", help="Document id field used when scoring BEIR retrieval results."
    ),
    beir_k: list[int] = typer.Option(
        [], "--beir-k", help="Top-k values to score for BEIR evaluation. Repeat the flag to provide multiple values."
    ),
    graphic_elements_invoke_url: Optional[str] = typer.Option(
        None, "--graphic-elements-invoke-url", help="Remote endpoint URL for graphic-elements model inference."
    ),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Bearer token for remote NIM endpoints."),
    embed_invoke_url: Optional[str] = typer.Option(
        None, "--embed-invoke-url", help="Remote endpoint URL for embedding model inference."
    ),
    embed_model_name: str = typer.Option(
        "nvidia/llama-nemotron-embed-1b-v2", "--embed-model-name", help="Embedding model name."
    ),
    embed_modality: str = typer.Option(
        "text", "--embed-modality", help="Embedding modality to generate, for example 'text'."
    ),
    hybrid: bool = typer.Option(False, "--hybrid/--no-hybrid", help="Enable LanceDB hybrid mode (dense + FTS text)."),
    input_type: str = typer.Option(
        "pdf", "--input-type", help="Input type to ingest from a directory. Supported here: 'pdf' or 'doc'."
    ),
    lancedb_uri: str = typer.Option(LANCEDB_URI, "--lancedb-uri", help="LanceDB URI/path for this run."),
    method: str = typer.Option("pdfium", "--method", help="PDF text extraction method."),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", path_type=Path, dir_okay=False, help="Optional file to mirror stdout, stderr, and logs."
    ),
    nemotron_parse_actors: Optional[int] = typer.Option(
        0, "--nemotron-parse-actors", help="Number of Nemotron Parse workers/actors to use."
    ),
    nemotron_parse_gpus_per_actor: Optional[float] = typer.Option(
        0.0,
        "--nemotron-parse-gpus-per-actor",
        min=0.0,
        max=1.0,
        help="GPU fraction reserved per Nemotron Parse actor.",
    ),
    nemotron_parse_batch_size: Optional[int] = typer.Option(
        0, "--nemotron-parse-batch-size", help="Nemotron Parse inference batch size."
    ),
    ocr_actors: Optional[int] = typer.Option(0, "--ocr-actors", help="Number of OCR workers/actors to use."),
    ocr_batch_size: Optional[int] = typer.Option(0, "--ocr-batch-size", help="OCR inference batch size."),
    ocr_cpus_per_actor: Optional[float] = typer.Option(
        0.0, "--ocr-cpus-per-actor", help="CPU resources reserved per OCR actor."
    ),
    ocr_gpus_per_actor: Optional[float] = typer.Option(
        0.0, "--ocr-gpus-per-actor", min=0.0, max=1.0, help="GPU fraction reserved per OCR actor."
    ),
    ocr_invoke_url: Optional[str] = typer.Option(
        None, "--ocr-invoke-url", help="Remote endpoint URL for OCR model inference."
    ),
    page_elements_actors: Optional[int] = typer.Option(
        0, "--page-elements-actors", help="Number of page-elements workers/actors to use."
    ),
    page_elements_batch_size: Optional[int] = typer.Option(
        0, "--page-elements-batch-size", help="Page-elements inference batch size."
    ),
    page_elements_cpus_per_actor: Optional[float] = typer.Option(
        0.0, "--page-elements-cpus-per-actor", help="CPU resources reserved per page-elements actor."
    ),
    page_elements_gpus_per_actor: Optional[float] = typer.Option(
        0.0,
        "--page-elements-gpus-per-actor",
        min=0.0,
        max=1.0,
        help="GPU fraction reserved per page-elements actor.",
    ),
    page_elements_invoke_url: Optional[str] = typer.Option(
        None, "--page-elements-invoke-url", help="Remote endpoint URL for page-elements model inference."
    ),
    pdf_extract_batch_size: Optional[int] = typer.Option(
        0, "--pdf-extract-batch-size", help="Batch size for PDF extraction tasks."
    ),
    pdf_extract_cpus_per_task: Optional[float] = typer.Option(
        0.0, "--pdf-extract-cpus-per-task", help="CPU resources reserved per PDF extraction task."
    ),
    pdf_extract_tasks: Optional[int] = typer.Option(
        0, "--pdf-extract-tasks", help="Number of parallel PDF extraction tasks to use."
    ),
    pdf_split_batch_size: int = typer.Option(
        1, "--pdf-split-batch-size", min=1, help="Batch size for PDF page splitting."
    ),
    query_csv: Path = typer.Option(
        "./data/bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help="Path to query CSV for recall evaluation.",
    ),
    ray_address: Optional[str] = typer.Option(
        None, "--ray-address", help="Ray cluster address. Leave unset to start or connect using defaults."
    ),
    ray_log_to_driver: bool = typer.Option(
        True, "--ray-log-to-driver/--no-ray-log-to-driver", help="Forward Ray worker logs to the driver process."
    ),
    runtime_metrics_dir: Optional[Path] = typer.Option(
        None,
        "--runtime-metrics-dir",
        path_type=Path,
        file_okay=False,
        dir_okay=True,
        help="Accepted for CLI parity; runtime metrics are not emitted by this graph example.",
    ),
    runtime_metrics_prefix: Optional[str] = typer.Option(
        None, "--runtime-metrics-prefix", help="Accepted for CLI parity with the batch example."
    ),
    reranker: Optional[bool] = typer.Option(
        False, "--reranker/--no-reranker", help="Enable reranking during evaluation."
    ),
    reranker_model_name: str = typer.Option(
        "nvidia/llama-nemotron-rerank-1b-v2", "--reranker-model-name", help="Reranker model name."
    ),
    structured_elements_modality: Optional[str] = typer.Option(
        None, "--structured-elements-modality", help="Embedding modality label for structured/table-like elements."
    ),
    text_elements_modality: Optional[str] = typer.Option(
        None, "--text-elements-modality", help="Embedding modality label for text elements."
    ),
    use_graphic_elements: bool = typer.Option(
        False, "--use-graphic-elements", help="Enable graphic-elements detection during extraction."
    ),
    use_table_structure: bool = typer.Option(
        False, "--use-table-structure", help="Enable table-structure detection during extraction."
    ),
    table_output_format: Optional[str] = typer.Option(
        None, "--table-output-format", help="Requested output format for extracted tables."
    ),
    table_structure_invoke_url: Optional[str] = typer.Option(
        None, "--table-structure-invoke-url", help="Remote endpoint URL for table-structure model inference."
    ),
    extract_text: bool = typer.Option(
        True, "--extract-text/--no-extract-text", help="Enable text extraction from documents."
    ),
    extract_tables: bool = typer.Option(
        True, "--extract-tables/--no-extract-tables", help="Enable table extraction from documents."
    ),
    extract_charts: bool = typer.Option(
        True, "--extract-charts/--no-extract-charts", help="Enable chart extraction from documents."
    ),
    extract_infographics: bool = typer.Option(
        False, "--extract-infographics/--no-extract-infographics", help="Enable infographic extraction from documents."
    ),
    extract_page_as_image: bool = typer.Option(
        True,
        "--extract-page-as-image/--no-extract-page-as-image",
        help="Include rendered page images in extraction output.",
    ),
    caption: bool = typer.Option(False, "--caption/--no-caption", help="Generate captions for extracted images."),
    caption_invoke_url: Optional[str] = typer.Option(
        None, "--caption-invoke-url", help="Remote endpoint URL for caption model inference."
    ),
    caption_model_name: str = typer.Option(
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "--caption-model-name", help="Caption model name."
    ),
    caption_device: Optional[str] = typer.Option(
        None, "--caption-device", help="Device to use for local caption inference, such as 'cuda' or 'cpu'."
    ),
    caption_context_text_max_chars: int = typer.Option(
        0,
        "--caption-context-text-max-chars",
        help="Maximum amount of surrounding text context to pass into captioning.",
    ),
    caption_gpu_memory_utilization: float = typer.Option(
        0.5,
        "--caption-gpu-memory-utilization",
        help="Fraction of GPU memory vLLM may use for the caption model (0.0–1.0).",
    ),
    dedup: Optional[bool] = typer.Option(
        None,
        "--dedup/--no-dedup",
        help="Remove duplicate/overlapping images before captioning. "
        "Defaults to on when captioning is enabled, off otherwise.",
    ),
    dedup_iou_threshold: float = typer.Option(
        0.45,
        "--dedup-iou-threshold",
        help="IoU threshold for bbox-based image dedup (0.0–1.0).",
    ),
    text_chunk: bool = typer.Option(
        False,
        "--text-chunk",
        help=(
            "Re-chunk extracted page text by token count before embedding. "
            "Uses --text-chunk-max-tokens and --text-chunk-overlap-tokens (defaults: 1024, 150)."
        ),
    ),
    text_chunk_max_tokens: Optional[int] = typer.Option(
        None, "--text-chunk-max-tokens", help="Maximum tokens per text chunk."
    ),
    text_chunk_overlap_tokens: Optional[int] = typer.Option(
        None, "--text-chunk-overlap-tokens", help="Token overlap between adjacent text chunks."
    ),
) -> None:
    _ = (ctx, no_recall_details)
    log_handle, original_stdout, original_stderr = _configure_logging(log_file, debug=bool(debug))
    try:
        if recall_match_mode not in {"pdf_page", "pdf_only"}:
            raise ValueError(f"Unsupported --recall-match-mode: {recall_match_mode}")
        if evaluation_mode not in {"recall", "beir"}:
            raise ValueError(f"Unsupported --evaluation-mode: {evaluation_mode}")

        if runtime_metrics_dir is not None or runtime_metrics_prefix is not None:
            logger.warning("runtime metrics flags are accepted for CLI parity but are not emitted by graph_pipeline.")

        os.environ["RAY_LOG_TO_DRIVER"] = "1" if ray_log_to_driver else "0"
        lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
        _ensure_lancedb_table(lancedb_uri, LANCEDB_TABLE)

        remote_api_key = resolve_remote_api_key(api_key)
        extract_remote_api_key = (
            remote_api_key
            if any((page_elements_invoke_url, ocr_invoke_url, graphic_elements_invoke_url, table_structure_invoke_url))
            else None
        )
        embed_remote_api_key = remote_api_key if embed_invoke_url else None

        if (
            any(
                (
                    page_elements_invoke_url,
                    ocr_invoke_url,
                    graphic_elements_invoke_url,
                    table_structure_invoke_url,
                    embed_invoke_url,
                )
            )
            and remote_api_key is None
        ):
            logger.warning("Remote endpoint URL(s) were configured without an API key.")

        if page_elements_invoke_url and float(page_elements_gpus_per_actor or 0.0) != 0.0:
            logger.warning("Forcing page-elements GPUs to 0.0 because --page-elements-invoke-url is set.")
            page_elements_gpus_per_actor = 0.0
        if ocr_invoke_url and float(ocr_gpus_per_actor or 0.0) != 0.0:
            logger.warning("Forcing OCR GPUs to 0.0 because --ocr-invoke-url is set.")
            ocr_gpus_per_actor = 0.0
        if embed_invoke_url and float(embed_gpus_per_actor or 0.0) != 0.0:
            logger.warning("Forcing embed GPUs to 0.0 because --embed-invoke-url is set.")
            embed_gpus_per_actor = 0.0

        file_patterns = _input_file_patterns(Path(input_path), input_type)
        extract_batch_tuning_kwargs = {
            "pdf_split_batch_size": pdf_split_batch_size,
            "pdf_extract_batch_size": pdf_extract_batch_size or None,
            "pdf_extract_workers": pdf_extract_tasks or None,
            "pdf_extract_num_cpus": pdf_extract_cpus_per_task or None,
            "page_elements_batch_size": page_elements_batch_size or None,
            "page_elements_workers": page_elements_actors or None,
            "page_elements_cpus_per_actor": page_elements_cpus_per_actor or None,
            "gpu_page_elements": (
                0.0
                if page_elements_invoke_url
                else (
                    page_elements_gpus_per_actor
                    if page_elements_gpus_per_actor and page_elements_gpus_per_actor > 0
                    else None
                )
            ),
            "ocr_inference_batch_size": ocr_batch_size or None,
            "ocr_workers": ocr_actors or None,
            "ocr_cpus_per_actor": ocr_cpus_per_actor or None,
            "gpu_ocr": (
                0.0
                if ocr_invoke_url
                else (ocr_gpus_per_actor if ocr_gpus_per_actor and ocr_gpus_per_actor > 0 else None)
            ),
            "nemotron_parse_batch_size": nemotron_parse_batch_size or None,
            "nemotron_parse_workers": nemotron_parse_actors or None,
            "gpu_nemotron_parse": (
                nemotron_parse_gpus_per_actor
                if nemotron_parse_gpus_per_actor and nemotron_parse_gpus_per_actor > 0
                else None
            ),
        }
        extract_batch_tuning = BatchTuningParams(
            **{key: value for key, value in extract_batch_tuning_kwargs.items() if value is not None}
        )
        extract_params_kwargs = {
            "method": method,
            "dpi": int(dpi),
            "extract_text": extract_text,
            "extract_tables": extract_tables,
            "extract_charts": extract_charts,
            "extract_infographics": extract_infographics,
            "extract_page_as_image": extract_page_as_image,
            "api_key": extract_remote_api_key,
            "page_elements_invoke_url": page_elements_invoke_url,
            "ocr_invoke_url": ocr_invoke_url,
            "graphic_elements_invoke_url": graphic_elements_invoke_url,
            "table_structure_invoke_url": table_structure_invoke_url,
            "use_graphic_elements": use_graphic_elements,
            "use_table_structure": use_table_structure,
            "table_output_format": table_output_format,
            "inference_batch_size": page_elements_batch_size or None,
            "batch_tuning": extract_batch_tuning,
        }
        extract_params = ExtractParams(
            **{key: value for key, value in extract_params_kwargs.items() if value is not None}
        )
        embed_batch_tuning_kwargs = {
            "embed_batch_size": embed_batch_size or None,
            "embed_workers": embed_actors or None,
            "embed_cpus_per_actor": embed_cpus_per_actor or None,
            "gpu_embed": (
                0.0
                if embed_invoke_url
                else (embed_gpus_per_actor if embed_gpus_per_actor and embed_gpus_per_actor > 0 else None)
            ),
        }
        embed_batch_tuning = BatchTuningParams(
            **{key: value for key, value in embed_batch_tuning_kwargs.items() if value is not None}
        )
        embed_params_kwargs = {
            "model_name": str(embed_model_name),
            "embed_invoke_url": embed_invoke_url,
            "api_key": embed_remote_api_key,
            "embed_modality": embed_modality,
            "text_elements_modality": text_elements_modality,
            "structured_elements_modality": structured_elements_modality,
            "embed_granularity": embed_granularity,
            "batch_tuning": embed_batch_tuning,
            "inference_batch_size": embed_batch_size or None,
        }
        embed_params = EmbedParams(**{key: value for key, value in embed_params_kwargs.items() if value is not None})
        text_chunk_params = TextChunkParams(
            max_tokens=text_chunk_max_tokens or 1024,
            overlap_tokens=text_chunk_overlap_tokens if text_chunk_overlap_tokens is not None else 150,
        )
        ingestor = create_ingestor(
            run_mode="batch",
            ray_address=ray_address,
            ray_log_to_driver=ray_log_to_driver,
            debug=bool(debug),
        )

        if input_type == "txt":
            ingestor = ingestor.files(file_patterns).extract_txt(text_chunk_params)
        elif input_type == "html":
            ingestor = ingestor.files(file_patterns).extract_html(text_chunk_params)
        elif input_type == "image":
            ingestor = ingestor.files(file_patterns).extract_image_files(extract_params)
        elif input_type == "doc":
            ingestor = ingestor.files(file_patterns).extract(extract_params)
        else:
            ingestor = ingestor.files(file_patterns).extract(extract_params)

        enable_text_chunk = text_chunk or text_chunk_max_tokens is not None or text_chunk_overlap_tokens is not None
        if enable_text_chunk:
            ingestor = ingestor.split(text_chunk_params)

        enable_caption = caption or caption_invoke_url is not None
        enable_dedup = dedup if dedup is not None else enable_caption
        if enable_dedup:
            ingestor = ingestor.dedup(
                DedupParams(
                    iou_threshold=dedup_iou_threshold,
                )
            )

        if enable_caption:
            ingestor = ingestor.caption(
                CaptionParams(
                    endpoint_url=caption_invoke_url,
                    model_name=caption_model_name,
                    device=caption_device,
                    context_text_max_chars=caption_context_text_max_chars,
                    gpu_memory_utilization=caption_gpu_memory_utilization,
                )
            )
        ingestor = ingestor.embed(embed_params)

        logger.info("Starting ingestion of %s ...", input_path)
        ingest_start = time.perf_counter()
        ingest_result = ingestor.ingest()
        dataset_holder = ingest_result if hasattr(ingest_result, "get_dataset") else ingestor
        result_ds = dataset_holder.get_dataset()
        if result_ds is None:
            raise RuntimeError("Batch ingestor did not produce a dataset.")
        result_ds = result_ds.materialize()
        ingestion_only_total_time = time.perf_counter() - ingest_start

        import ray

        ray_dataset_download_start = time.perf_counter()
        ingest_local_results = result_ds.take_all()
        ray_dataset_download_time = time.perf_counter() - ray_dataset_download_start

        error_rows = ingestor.get_error_rows(result_ds).materialize()
        error_count = int(error_rows.count())
        if error_count > 0:
            error_file = Path("ingest_errors.json").resolve()
            error_rows_to_write = error_rows.take(min(5, error_count))
            with error_file.open("w", encoding="utf-8") as fh:
                json.dump(error_rows_to_write, fh, indent=2, default=str)
                fh.write("\n")
            logger.error(
                "Detected %d error row(s) in graph ingest results. Wrote first %d row(s) to %s.",
                error_count,
                len(error_rows_to_write),
                str(error_file),
            )
            ray.shutdown()
            raise typer.Exit(code=1)

        if detection_summary_file is not None:
            import pandas as pd

            write_detection_summary(
                Path(detection_summary_file), collect_detection_summary_from_df(pd.DataFrame(ingest_local_results))
            )

        lancedb_write_start = time.perf_counter()
        handle_lancedb(ingest_local_results, lancedb_uri, LANCEDB_TABLE, hybrid=hybrid, mode="overwrite")
        lancedb_write_time = time.perf_counter() - lancedb_write_start

        from nemo_retriever.model import resolve_embed_model
        from nemo_retriever.recall.beir import BeirConfig
        from nemo_retriever.recall.core import RecallConfig, retrieve_and_score

        _recall_model = resolve_embed_model(str(embed_model_name))
        evaluation_label = "Recall"
        evaluation_total_time = 0.0
        evaluation_metrics: dict[str, float] = {}
        evaluation_query_count: Optional[int] = None
        query_csv_path = Path(query_csv)

        if evaluation_mode == "beir":
            if not beir_loader:
                raise ValueError("--beir-loader is required when --evaluation-mode=beir")
            if not beir_dataset_name:
                raise ValueError("--beir-dataset-name is required when --evaluation-mode=beir")
        else:
            if not query_csv_path.exists():
                logger.warning("Query CSV not found at %s; skipping recall evaluation.", query_csv_path)
                ray.shutdown()
                return

        db = _lancedb().connect(lancedb_uri)
        table = db.open_table(LANCEDB_TABLE)
        if int(table.count_rows()) == 0:
            logger.warning("LanceDB table is empty; skipping %s evaluation.", evaluation_mode)
            ray.shutdown()
            return

        if evaluation_mode == "beir":
            cfg = BeirConfig(
                lancedb_uri=str(lancedb_uri),
                lancedb_table=str(LANCEDB_TABLE),
                embedding_model=_recall_model,
                loader=str(beir_loader),
                dataset_name=str(beir_dataset_name),
                split=str(beir_split),
                query_language=beir_query_language,
                doc_id_field=str(beir_doc_id_field),
                ks=tuple(beir_k) if beir_k else (1, 3, 5, 10),
                embedding_http_endpoint=embed_invoke_url,
                embedding_api_key=embed_remote_api_key or "",
                hybrid=hybrid,
                reranker=bool(reranker),
                reranker_model_name=str(reranker_model_name),
            )
            evaluation_start = time.perf_counter()
            beir_dataset, _raw_hits, _run, evaluation_metrics = evaluate_lancedb_beir(cfg)
            evaluation_total_time = time.perf_counter() - evaluation_start
            evaluation_label = "BEIR"
            evaluation_query_count = len(beir_dataset.query_ids)
        else:
            cfg = RecallConfig(
                lancedb_uri=str(lancedb_uri),
                lancedb_table=str(LANCEDB_TABLE),
                embedding_model=_recall_model,
                embedding_http_endpoint=embed_invoke_url,
                embedding_api_key=embed_remote_api_key or "",
                top_k=10,
                ks=(1, 5, 10),
                hybrid=hybrid,
                match_mode=recall_match_mode,
                reranker=reranker_model_name if reranker else None,
            )
            evaluation_start = time.perf_counter()
            _df_query, _gold, _raw_hits, _retrieved_keys, evaluation_metrics = retrieve_and_score(
                query_csv=query_csv_path, cfg=cfg
            )
            evaluation_total_time = time.perf_counter() - evaluation_start
            evaluation_query_count = len(_df_query.index)

        total_time = time.perf_counter() - ingest_start
        num_rows = result_ds.groupby("source_id").count().count()
        ray.shutdown()

        print_run_summary(
            num_rows,
            Path(input_path),
            hybrid,
            lancedb_uri,
            LANCEDB_TABLE,
            total_time,
            ingestion_only_total_time,
            ray_dataset_download_time,
            lancedb_write_time,
            evaluation_total_time,
            evaluation_metrics,
            evaluation_label=evaluation_label,
            evaluation_count=evaluation_query_count,
        )
    finally:
        os.sys.stdout = original_stdout
        os.sys.stderr = original_stderr
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    app()
