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
from functools import partial
from pathlib import Path
from typing import Any, Optional, TextIO

import typer

from nemo_retriever.graph import UDFOperator
from nemo_retriever.graph.executor import RayDataExecutor
from nemo_retriever.utils.detection_summary import (
    collect_detection_summary_from_df,
    print_run_summary,
    write_detection_summary,
)
from nemo_retriever.utils.remote_auth import resolve_remote_api_key

logger = logging.getLogger(__name__)
app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


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
    from nemo_retriever.ingest_modes.lancedb_utils import lancedb_schema
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


def _override(batch_size: Optional[int] = None, num_cpus: Optional[float] = None, num_gpus: Optional[float] = None,
              concurrency: Optional[int] = None, target_num_rows_per_block: Optional[int] = None,
              **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if batch_size is not None and batch_size > 0:
        out["batch_size"] = batch_size
    if num_cpus is not None and num_cpus > 0:
        out["num_cpus"] = num_cpus
    if num_gpus is not None and num_gpus >= 0:
        out["num_gpus"] = num_gpus
    if concurrency is not None and concurrency > 0:
        out["concurrency"] = concurrency
    if target_num_rows_per_block is not None and target_num_rows_per_block > 0:
        out["target_num_rows_per_block"] = target_num_rows_per_block
    out.update(extra)
    return out


@app.command()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Enable debug-level logging for this full pipeline run."),
    dpi: int = typer.Option(300, "--dpi", min=72, help="Render DPI for PDF page images (default: 300)."),
    input_path: Path = typer.Argument(
        ...,
        help="File or directory containing PDFs, .txt, .html, or .doc/.pptx files to ingest.",
        path_type=Path,
    ),
    detection_summary_file: Optional[Path] = typer.Option(None, "--detection-summary-file", path_type=Path, dir_okay=False),
    recall_match_mode: str = typer.Option("pdf_page", "--recall-match-mode"),
    evaluation_mode: str = typer.Option("recall", "--evaluation-mode"),
    no_recall_details: bool = typer.Option(False, "--no-recall-details"),
    embed_actors: Optional[int] = typer.Option(0, "--embed-actors"),
    embed_batch_size: Optional[int] = typer.Option(0, "--embed-batch-size"),
    embed_cpus_per_actor: Optional[float] = typer.Option(0.0, "--embed-cpus-per-actor"),
    embed_gpus_per_actor: Optional[float] = typer.Option(0.0, "--embed-gpus-per-actor", max=1.0),
    embed_granularity: str = typer.Option("element", "--embed-granularity"),
    beir_loader: Optional[str] = typer.Option(None, "--beir-loader"),
    beir_dataset_name: Optional[str] = typer.Option(None, "--beir-dataset-name"),
    beir_split: str = typer.Option("test", "--beir-split"),
    beir_query_language: Optional[str] = typer.Option(None, "--beir-query-language"),
    beir_doc_id_field: str = typer.Option("pdf_basename", "--beir-doc-id-field"),
    beir_k: list[int] = typer.Option([], "--beir-k"),
    graphic_elements_invoke_url: Optional[str] = typer.Option(None, "--graphic-elements-invoke-url"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    embed_invoke_url: Optional[str] = typer.Option(None, "--embed-invoke-url"),
    embed_model_name: str = typer.Option("nvidia/llama-nemotron-embed-1b-v2", "--embed-model-name"),
    embed_modality: str = typer.Option("text", "--embed-modality"),
    hybrid: bool = typer.Option(False, "--hybrid/--no-hybrid"),
    input_type: str = typer.Option("pdf", "--input-type"),
    lancedb_uri: str = typer.Option(LANCEDB_URI, "--lancedb-uri"),
    method: str = typer.Option("pdfium", "--method"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", path_type=Path, dir_okay=False),
    nemotron_parse_actors: Optional[int] = typer.Option(0, "--nemotron-parse-actors"),
    nemotron_parse_gpus_per_actor: Optional[float] = typer.Option(0.0, "--nemotron-parse-gpus-per-actor", min=0.0, max=1.0),
    nemotron_parse_batch_size: Optional[int] = typer.Option(0, "--nemotron-parse-batch-size"),
    ocr_actors: Optional[int] = typer.Option(0, "--ocr-actors"),
    ocr_batch_size: Optional[int] = typer.Option(0, "--ocr-batch-size"),
    ocr_cpus_per_actor: Optional[float] = typer.Option(0.0, "--ocr-cpus-per-actor"),
    ocr_gpus_per_actor: Optional[float] = typer.Option(0.0, "--ocr-gpus-per-actor", min=0.0, max=1.0),
    ocr_invoke_url: Optional[str] = typer.Option(None, "--ocr-invoke-url"),
    page_elements_actors: Optional[int] = typer.Option(0, "--page-elements-actors"),
    page_elements_batch_size: Optional[int] = typer.Option(0, "--page-elements-batch-size"),
    page_elements_cpus_per_actor: Optional[float] = typer.Option(0.0, "--page-elements-cpus-per-actor"),
    page_elements_gpus_per_actor: Optional[float] = typer.Option(0.0, "--page-elements-gpus-per-actor", min=0.0, max=1.0),
    page_elements_invoke_url: Optional[str] = typer.Option(None, "--page-elements-invoke-url"),
    pdf_extract_batch_size: Optional[int] = typer.Option(0, "--pdf-extract-batch-size"),
    pdf_extract_cpus_per_task: Optional[float] = typer.Option(0.0, "--pdf-extract-cpus-per-task"),
    pdf_extract_tasks: Optional[int] = typer.Option(0, "--pdf-extract-tasks"),
    pdf_split_batch_size: int = typer.Option(1, "--pdf-split-batch-size", min=1),
    query_csv: Path = typer.Option("./data/bo767_query_gt.csv", "--query-csv", path_type=Path),
    ray_address: Optional[str] = typer.Option(None, "--ray-address"),
    ray_log_to_driver: bool = typer.Option(True, "--ray-log-to-driver/--no-ray-log-to-driver"),
    runtime_metrics_dir: Optional[Path] = typer.Option(None, "--runtime-metrics-dir", path_type=Path, file_okay=False, dir_okay=True),
    runtime_metrics_prefix: Optional[str] = typer.Option(None, "--runtime-metrics-prefix"),
    reranker: Optional[bool] = typer.Option(False, "--reranker/--no-reranker"),
    reranker_model_name: str = typer.Option("nvidia/llama-nemotron-rerank-1b-v2", "--reranker-model-name"),
    structured_elements_modality: Optional[str] = typer.Option(None, "--structured-elements-modality"),
    text_elements_modality: Optional[str] = typer.Option(None, "--text-elements-modality"),
    use_graphic_elements: bool = typer.Option(False, "--use-graphic-elements"),
    use_table_structure: bool = typer.Option(False, "--use-table-structure"),
    table_output_format: Optional[str] = typer.Option(None, "--table-output-format"),
    table_structure_invoke_url: Optional[str] = typer.Option(None, "--table-structure-invoke-url"),
    extract_text: bool = typer.Option(True, "--extract-text/--no-extract-text"),
    extract_tables: bool = typer.Option(True, "--extract-tables/--no-extract-tables"),
    extract_charts: bool = typer.Option(True, "--extract-charts/--no-extract-charts"),
    extract_infographics: bool = typer.Option(False, "--extract-infographics/--no-extract-infographics"),
    extract_page_as_image: bool = typer.Option(True, "--extract-page-as-image/--no-extract-page-as-image"),
    caption: bool = typer.Option(False, "--caption/--no-caption"),
    caption_invoke_url: Optional[str] = typer.Option(None, "--caption-invoke-url"),
    caption_model_name: str = typer.Option("nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "--caption-model-name"),
    caption_device: Optional[str] = typer.Option(None, "--caption-device"),
    caption_context_text_max_chars: int = typer.Option(0, "--caption-context-text-max-chars"),
    caption_gpu_memory_utilization: float = typer.Option(0.5, "--caption-gpu-memory-utilization"),
    text_chunk: bool = typer.Option(False, "--text-chunk"),
    text_chunk_max_tokens: Optional[int] = typer.Option(None, "--text-chunk-max-tokens"),
    text_chunk_overlap_tokens: Optional[int] = typer.Option(None, "--text-chunk-overlap-tokens"),
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
            remote_api_key if any((page_elements_invoke_url, ocr_invoke_url, graphic_elements_invoke_url, table_structure_invoke_url)) else None
        )
        embed_remote_api_key = remote_api_key if embed_invoke_url else None

        if any((page_elements_invoke_url, ocr_invoke_url, graphic_elements_invoke_url, table_structure_invoke_url, embed_invoke_url)) and remote_api_key is None:
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

        from nemo_retriever.audio.asr_actor import ASRActor  # noqa: F401
        from nemo_retriever.caption.caption import CaptionActor
        from nemo_retriever.chart.chart_detection import GraphicElementsActor
        from nemo_retriever.graph import Graph
        from nemo_retriever.infographic.infographic_detection import InfographicDetectionActor  # noqa: F401
        from nemo_retriever.ingest_modes.batch import BatchIngestor, _BatchEmbedActor
        from nemo_retriever.ingest_modes.inprocess import (
            _CONTENT_COLUMNS,
            collapse_content_to_page_rows,
            explode_content_to_rows,
        )
        from nemo_retriever.ocr.ocr import NemotronParseActor, OCRActor
        from nemo_retriever.page_elements.page_elements import PageElementDetectionActor
        from nemo_retriever.params import CaptionParams, EmbedParams, TextChunkParams
        from nemo_retriever.pdf.extract import PDFExtractionActor
        from nemo_retriever.pdf.split import PDFSplitActor
        from nemo_retriever.table.table_detection import TableStructureActor
        from nemo_retriever.txt.ray_data import TextChunkActor
        from nemo_retriever.utils.convert.to_pdf import DocToPdfConversionActor

        graph = Graph()
        if input_type == "doc":
            graph = graph >> DocToPdfConversionActor()
        graph = graph >> PDFSplitActor()

        parse_mode = method == "nemotron_parse" or (
            int(nemotron_parse_actors or 0) > 0
            and float(nemotron_parse_gpus_per_actor or 0.0) > 0.0
            and int(nemotron_parse_batch_size or 0) > 0
        )

        extract_kwargs: dict[str, Any] = {
            "method": method,
            "dpi": int(dpi),
            "extract_text": extract_text,
            "extract_tables": extract_tables,
            "extract_charts": extract_charts,
            "extract_infographics": extract_infographics,
            "extract_page_as_image": extract_page_as_image,
            "api_key": extract_remote_api_key,
        }

        detect_kwargs: dict[str, Any] = {}
        if page_elements_invoke_url:
            detect_kwargs["page_elements_invoke_url"] = page_elements_invoke_url
        if extract_remote_api_key:
            detect_kwargs["api_key"] = extract_remote_api_key
        if page_elements_batch_size:
            detect_kwargs["inference_batch_size"] = int(page_elements_batch_size)

        ocr_kwargs: dict[str, Any] = {}
        if method in ("pdfium_hybrid", "ocr") and extract_text:
            ocr_kwargs["extract_text"] = True
        if extract_tables and not use_table_structure:
            ocr_kwargs["extract_tables"] = True
        if extract_charts and not use_graphic_elements:
            ocr_kwargs["extract_charts"] = True
        if extract_infographics:
            ocr_kwargs["extract_infographics"] = True
        ocr_kwargs["use_graphic_elements"] = use_graphic_elements
        if ocr_invoke_url:
            ocr_kwargs["ocr_invoke_url"] = ocr_invoke_url
        if extract_remote_api_key:
            ocr_kwargs["api_key"] = extract_remote_api_key
        if ocr_batch_size:
            ocr_kwargs["inference_batch_size"] = int(ocr_batch_size)
        needs_ocr = any(
            bool(ocr_kwargs.get(key))
            for key in ("extract_text", "extract_tables", "extract_charts", "extract_infographics")
        )

        table_kwargs: dict[str, Any] = {}
        if table_structure_invoke_url:
            table_kwargs["table_structure_invoke_url"] = table_structure_invoke_url
        if ocr_invoke_url:
            table_kwargs["ocr_invoke_url"] = ocr_invoke_url
        if extract_remote_api_key:
            table_kwargs["api_key"] = extract_remote_api_key
        if table_output_format:
            table_kwargs["table_output_format"] = table_output_format

        graphic_kwargs: dict[str, Any] = {}
        if graphic_elements_invoke_url:
            graphic_kwargs["graphic_elements_invoke_url"] = graphic_elements_invoke_url
        if ocr_invoke_url:
            graphic_kwargs["ocr_invoke_url"] = ocr_invoke_url
        if extract_remote_api_key:
            graphic_kwargs["api_key"] = extract_remote_api_key

        if parse_mode:
            parse_kwargs: dict[str, Any] = {
                "extract_text": extract_text,
                "extract_tables": extract_tables,
                "extract_charts": extract_charts,
                "extract_infographics": extract_infographics,
            }
            if remote_api_key:
                parse_kwargs["api_key"] = remote_api_key
            graph = graph >> NemotronParseActor(**parse_kwargs)
        else:
            graph = graph >> PDFExtractionActor(**extract_kwargs) >> PageElementDetectionActor(**detect_kwargs)
            if use_table_structure and extract_tables:
                graph = graph >> TableStructureActor(**table_kwargs)
            if use_graphic_elements and extract_charts:
                graph = graph >> GraphicElementsActor(**graphic_kwargs)
            if needs_ocr:
                graph = graph >> OCRActor(**ocr_kwargs)

        enable_caption = caption or caption_invoke_url is not None
        if enable_caption:
            graph = graph >> CaptionActor(
                CaptionParams(
                    endpoint_url=caption_invoke_url,
                    model_name=caption_model_name,
                    device=caption_device,
                    context_text_max_chars=caption_context_text_max_chars,
                    gpu_memory_utilization=caption_gpu_memory_utilization,
                )
            )

        content_columns = (_CONTENT_COLUMNS + ("images",)) if enable_caption else _CONTENT_COLUMNS
        if embed_granularity == "page":
            graph = graph >> UDFOperator(
                partial(
                    collapse_content_to_page_rows,
                    modality=embed_modality,
                    content_columns=content_columns,
                ),
                name="CollapseContentToPageRows",
            )
        else:
            graph = graph >> UDFOperator(
                partial(
                    explode_content_to_rows,
                    modality=embed_modality,
                    text_elements_modality=text_elements_modality or embed_modality,
                    structured_elements_modality=structured_elements_modality or embed_modality,
                    content_columns=content_columns,
                ),
                name="ExplodeContentToRows",
            )

        if text_chunk or text_chunk_max_tokens is not None or text_chunk_overlap_tokens is not None:
            graph = graph >> TextChunkActor(
                TextChunkParams(
                    max_tokens=text_chunk_max_tokens or 1024,
                    overlap_tokens=text_chunk_overlap_tokens if text_chunk_overlap_tokens is not None else 150,
                )
            )

        graph = graph >> _BatchEmbedActor(
            params=EmbedParams(
                model_name=str(embed_model_name),
                embed_invoke_url=embed_invoke_url,
                api_key=embed_remote_api_key,
                embed_modality=embed_modality,
                text_elements_modality=text_elements_modality,
                structured_elements_modality=structured_elements_modality,
                embed_granularity=embed_granularity,
            )
        )

        logger.info("Pipeline graph: %s", graph)

        node_overrides: dict[str, dict[str, Any]] = {
            "DocToPdfConversionActor": _override(batch_size=1, num_cpus=1),
            "PDFSplitActor": _override(batch_size=pdf_split_batch_size, num_cpus=1),
            "PDFExtractionActor": _override(
                batch_size=pdf_extract_batch_size or 4,
                num_cpus=pdf_extract_cpus_per_task or 1,
                concurrency=pdf_extract_tasks or None,
            ),
            "PageElementDetectionActor": _override(
                batch_size=page_elements_batch_size or 8,
                target_num_rows_per_block=page_elements_batch_size or 8,
                num_cpus=page_elements_cpus_per_actor or 1,
                num_gpus=0.0 if page_elements_invoke_url else (page_elements_gpus_per_actor if page_elements_gpus_per_actor and page_elements_gpus_per_actor > 0 else 0.5),
                concurrency=page_elements_actors or None,
            ),
            "TableStructureActor": _override(
                batch_size=ocr_batch_size or 8,
                num_cpus=ocr_cpus_per_actor or 1,
                num_gpus=0.0 if table_structure_invoke_url else (ocr_gpus_per_actor if ocr_gpus_per_actor and ocr_gpus_per_actor > 0 else 0.5),
            ),
            "GraphicElementsActor": _override(
                batch_size=ocr_batch_size or 8,
                num_cpus=ocr_cpus_per_actor or 1,
                num_gpus=0.0 if graphic_elements_invoke_url else (ocr_gpus_per_actor if ocr_gpus_per_actor and ocr_gpus_per_actor > 0 else 0.5),
            ),
            "OCRActor": _override(
                batch_size=ocr_batch_size or 8,
                num_cpus=ocr_cpus_per_actor or 1,
                num_gpus=0.0 if ocr_invoke_url else (ocr_gpus_per_actor if ocr_gpus_per_actor and ocr_gpus_per_actor > 0 else 0.5),
                concurrency=ocr_actors or None,
            ),
            "NemotronParseActor": _override(
                batch_size=nemotron_parse_batch_size or page_elements_batch_size or 8,
                target_num_rows_per_block=nemotron_parse_batch_size or page_elements_batch_size or 8,
                num_cpus=1,
                num_gpus=nemotron_parse_gpus_per_actor if nemotron_parse_gpus_per_actor and nemotron_parse_gpus_per_actor > 0 else 0.5,
                concurrency=nemotron_parse_actors or None,
            ),
            "CaptionActor": _override(
                batch_size=8,
                num_cpus=1,
                num_gpus=0.0 if caption_invoke_url else 0.5,
            ),
            "ExplodeContentToRows": _override(batch_size=256, num_cpus=1, num_gpus=0.0),
            "CollapseContentToPageRows": _override(batch_size=256, num_cpus=1, num_gpus=0.0),
            "TextChunkActor": _override(batch_size=256, num_cpus=1, num_gpus=0.0),
            "_BatchEmbedActor": _override(
                batch_size=embed_batch_size or 256,
                target_num_rows_per_block=embed_batch_size or 256,
                num_cpus=embed_cpus_per_actor or 1,
                num_gpus=0.0 if embed_invoke_url else (embed_gpus_per_actor if embed_gpus_per_actor and embed_gpus_per_actor > 0 else 0.5),
                concurrency=embed_actors or None,
            ),
        }

        executor = RayDataExecutor(graph, ray_address=ray_address, batch_size=1, node_overrides=node_overrides)

        logger.info("Starting ingestion of %s ...", input_path)
        ingest_start = time.perf_counter()
        result_ds = executor.ingest(file_patterns)
        ingestion_only_total_time = time.perf_counter() - ingest_start

        import ray

        ray_dataset_download_start = time.perf_counter()
        ingest_local_results = result_ds.take_all()
        ray_dataset_download_time = time.perf_counter() - ray_dataset_download_start

        error_rows = result_ds.map_batches(BatchIngestor.extract_error_rows, batch_format="pandas").materialize()
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

            write_detection_summary(Path(detection_summary_file), collect_detection_summary_from_df(pd.DataFrame(ingest_local_results)))

        from nemo_retriever.vector_store.lancedb_store import handle_lancedb

        lancedb_write_start = time.perf_counter()
        handle_lancedb(ingest_local_results, lancedb_uri, LANCEDB_TABLE, hybrid=hybrid, mode="overwrite")
        lancedb_write_time = time.perf_counter() - lancedb_write_start

        db = __import__("lancedb").connect(lancedb_uri)
        table = db.open_table(LANCEDB_TABLE)
        if int(table.count_rows()) == 0:
            logger.warning("LanceDB table is empty; skipping %s evaluation.", evaluation_mode)
            ray.shutdown()
            return

        from nemo_retriever.model import resolve_embed_model
        from nemo_retriever.recall.beir import BeirConfig, evaluate_lancedb_beir
        from nemo_retriever.recall.core import RecallConfig, retrieve_and_score

        _recall_model = resolve_embed_model(str(embed_model_name))
        evaluation_label = "Recall"
        evaluation_total_time = 0.0
        evaluation_metrics: dict[str, float] = {}
        evaluation_query_count: Optional[int] = None

        if evaluation_mode == "beir":
            if not beir_loader:
                raise ValueError("--beir-loader is required when --evaluation-mode=beir")
            if not beir_dataset_name:
                raise ValueError("--beir-dataset-name is required when --evaluation-mode=beir")

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
            query_csv = Path(query_csv)
            if not query_csv.exists():
                logger.warning("Query CSV not found at %s; skipping recall evaluation.", query_csv)
                ray.shutdown()
                return

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
            _df_query, _gold, _raw_hits, _retrieved_keys, evaluation_metrics = retrieve_and_score(query_csv=query_csv, cfg=cfg)
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
