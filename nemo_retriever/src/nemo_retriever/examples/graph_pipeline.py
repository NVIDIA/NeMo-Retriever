# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-based ingestion pipeline using the operator graph API directly.

This script uses :class:`~nemo_retriever.graph_ingestor.GraphIngestor` which
builds an operator :class:`~nemo_retriever.graph.Graph` via
:func:`~nemo_retriever.graph.ingestor_runtime.build_graph` (Ray Data)
or :func:`~nemo_retriever.graph.ingestor_runtime.build_inprocess_graph`
(single-process pandas) and then calls the appropriate executor.

Run with::

    source /opt/retriever_runtime/bin/activate
    python -m nemo_retriever.examples.graph_pipeline <input-dir-or-file> [OPTIONS]

Examples::

    # Batch mode (Ray) with PDF extraction + embedding
    python -m nemo_retriever.examples.graph_pipeline /data/pdfs \\
        --run-mode batch \\
        --embed-invoke-url http://localhost:8000/v1

    # In-process mode (no Ray) for quick local testing
    python -m nemo_retriever.examples.graph_pipeline /data/pdfs \\
        --run-mode inprocess \\
        --ocr-invoke-url http://localhost:9000/v1

    # Save extraction Parquet for full-page markdown (e.g. page index + export)
    python -m nemo_retriever.examples.graph_pipeline /data/pdfs \\
        --lancedb-uri lancedb \\
        --save-intermediate /path/to/extracted_parquet_dir
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, TextIO

import typer

from nemo_retriever.audio import asr_params_from_env
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.params import AudioChunkParams
from nemo_retriever.params import CaptionParams
from nemo_retriever.params import DedupParams
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import StoreParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.model import VL_EMBED_MODEL, VL_RERANK_MODEL
from nemo_retriever.params.models import BatchTuningParams
from nemo_retriever.utils.input_files import resolve_input_patterns
from nemo_retriever.utils.remote_auth import resolve_remote_api_key
from nemo_retriever.vector_store.lancedb_store import handle_lancedb

logger = logging.getLogger(__name__)
app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


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
    logger.info("Writing combined pipeline logs to %s", str(target))
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


def _write_runtime_summary(
    runtime_metrics_dir: Optional[Path],
    runtime_metrics_prefix: Optional[str],
    payload: dict[str, object],
) -> None:
    if runtime_metrics_dir is None and not runtime_metrics_prefix:
        return

    target_dir = Path(runtime_metrics_dir or Path.cwd()).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    prefix = (runtime_metrics_prefix or "run").strip() or "run"
    target = target_dir / f"{prefix}.runtime.summary.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _count_input_units(result_df) -> int:
    if "source_id" in result_df.columns:
        return int(result_df["source_id"].nunique())
    if "source_path" in result_df.columns:
        return int(result_df["source_path"].nunique())
    return int(len(result_df.index))


def _resolve_file_patterns(input_path: Path, input_type: str) -> list[str]:
    import glob as _glob

    input_path = Path(input_path)
    if input_path.is_file():
        return [str(input_path)]
    if not input_path.is_dir():
        raise typer.BadParameter(f"Path does not exist: {input_path}")

    if input_type not in {"pdf", "doc", "txt", "html", "image", "audio"}:
        raise typer.BadParameter(f"Unsupported --input-type: {input_type!r}")

    patterns = resolve_input_patterns(input_path, input_type)
    matched = [p for p in patterns if _glob.glob(p, recursive=True)]
    if not matched:
        raise typer.BadParameter(f"No files found for input_type={input_type!r} in {input_path}")
    return matched


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@app.command()
def main(
    ctx: typer.Context,
    input_path: Path = typer.Argument(
        ...,
        help="File or directory of documents to ingest.",
        path_type=Path,
    ),
    run_mode: str = typer.Option(
        "batch",
        "--run-mode",
        help="Execution mode: 'batch' (Ray Data) or 'inprocess' (pandas, no Ray).",
    ),
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Enable debug-level logging."),
    dpi: int = typer.Option(300, "--dpi", min=72, help="Render DPI for PDF page images."),
    input_type: str = typer.Option(
        "pdf", "--input-type", help="Input type: 'pdf', 'doc', 'txt', 'html', 'image', or 'audio'."
    ),
    method: str = typer.Option("pdfium", "--method", help="PDF text extraction method."),
    extract_text: bool = typer.Option(True, "--extract-text/--no-extract-text"),
    extract_tables: bool = typer.Option(True, "--extract-tables/--no-extract-tables"),
    extract_charts: bool = typer.Option(True, "--extract-charts/--no-extract-charts"),
    extract_infographics: bool = typer.Option(False, "--extract-infographics/--no-extract-infographics"),
    extract_page_as_image: bool = typer.Option(True, "--extract-page-as-image/--no-extract-page-as-image"),
    use_graphic_elements: bool = typer.Option(False, "--use-graphic-elements"),
    use_table_structure: bool = typer.Option(False, "--use-table-structure"),
    table_output_format: Optional[str] = typer.Option(None, "--table-output-format"),
    # Remote endpoints
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Bearer token for remote NIM endpoints."),
    page_elements_invoke_url: Optional[str] = typer.Option(None, "--page-elements-invoke-url"),
    ocr_invoke_url: Optional[str] = typer.Option(None, "--ocr-invoke-url"),
    graphic_elements_invoke_url: Optional[str] = typer.Option(None, "--graphic-elements-invoke-url"),
    table_structure_invoke_url: Optional[str] = typer.Option(None, "--table-structure-invoke-url"),
    embed_invoke_url: Optional[str] = typer.Option(None, "--embed-invoke-url"),
    # Embedding
    embed_model_name: str = typer.Option(VL_EMBED_MODEL, "--embed-model-name"),
    embed_modality: str = typer.Option("text", "--embed-modality"),
    embed_granularity: str = typer.Option("element", "--embed-granularity"),
    text_elements_modality: Optional[str] = typer.Option(None, "--text-elements-modality"),
    structured_elements_modality: Optional[str] = typer.Option(None, "--structured-elements-modality"),
    # Dedup / caption
    dedup: Optional[bool] = typer.Option(None, "--dedup/--no-dedup"),
    dedup_iou_threshold: float = typer.Option(0.45, "--dedup-iou-threshold"),
    caption: bool = typer.Option(False, "--caption/--no-caption"),
    caption_invoke_url: Optional[str] = typer.Option(None, "--caption-invoke-url"),
    caption_model_name: str = typer.Option("nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "--caption-model-name"),
    caption_device: Optional[str] = typer.Option(None, "--caption-device"),
    caption_context_text_max_chars: int = typer.Option(0, "--caption-context-text-max-chars"),
    caption_gpu_memory_utilization: float = typer.Option(0.5, "--caption-gpu-memory-utilization"),
    caption_gpus_per_actor: Optional[float] = typer.Option(None, "--caption-gpus-per-actor", max=1.0),
    caption_temperature: float = typer.Option(1.0, "--caption-temperature"),
    caption_top_p: Optional[float] = typer.Option(None, "--caption-top-p"),
    caption_max_tokens: int = typer.Option(1024, "--caption-max-tokens"),
    # Text chunking
    store_images_uri: Optional[str] = typer.Option(
        None, "--store-images-uri", help="Store extracted images to this URI."
    ),
    store_text: bool = typer.Option(False, "--store-text/--no-store-text", help="Also store extracted text."),
    strip_base64: bool = typer.Option(True, "--strip-base64/--no-strip-base64", help="Strip base64 after storing."),
    text_chunk: bool = typer.Option(False, "--text-chunk"),
    text_chunk_max_tokens: Optional[int] = typer.Option(None, "--text-chunk-max-tokens"),
    text_chunk_overlap_tokens: Optional[int] = typer.Option(None, "--text-chunk-overlap-tokens"),
    # Ray / batch tuning
    # NOTE: *_gpus_per_actor defaults are None (not 0.0) so we can distinguish
    # "not set → use heuristic" from "explicitly 0 → no GPU".  Other tuning
    # defaults use 0/0.0 because those values are never valid explicit choices.
    ray_address: Optional[str] = typer.Option(None, "--ray-address"),
    ray_log_to_driver: bool = typer.Option(True, "--ray-log-to-driver/--no-ray-log-to-driver"),
    ocr_actors: Optional[int] = typer.Option(0, "--ocr-actors"),
    ocr_batch_size: Optional[int] = typer.Option(0, "--ocr-batch-size"),
    ocr_cpus_per_actor: Optional[float] = typer.Option(0.0, "--ocr-cpus-per-actor"),
    ocr_gpus_per_actor: Optional[float] = typer.Option(None, "--ocr-gpus-per-actor", max=1.0),
    page_elements_actors: Optional[int] = typer.Option(0, "--page-elements-actors"),
    page_elements_batch_size: Optional[int] = typer.Option(0, "--page-elements-batch-size"),
    page_elements_cpus_per_actor: Optional[float] = typer.Option(0.0, "--page-elements-cpus-per-actor"),
    page_elements_gpus_per_actor: Optional[float] = typer.Option(None, "--page-elements-gpus-per-actor", max=1.0),
    embed_actors: Optional[int] = typer.Option(0, "--embed-actors"),
    embed_batch_size: Optional[int] = typer.Option(0, "--embed-batch-size"),
    embed_cpus_per_actor: Optional[float] = typer.Option(0.0, "--embed-cpus-per-actor"),
    embed_gpus_per_actor: Optional[float] = typer.Option(None, "--embed-gpus-per-actor", max=1.0),
    pdf_split_batch_size: int = typer.Option(1, "--pdf-split-batch-size", min=1),
    pdf_extract_batch_size: Optional[int] = typer.Option(0, "--pdf-extract-batch-size"),
    pdf_extract_tasks: Optional[int] = typer.Option(0, "--pdf-extract-tasks"),
    pdf_extract_cpus_per_task: Optional[float] = typer.Option(0.0, "--pdf-extract-cpus-per-task"),
    nemotron_parse_actors: Optional[int] = typer.Option(0, "--nemotron-parse-actors"),
    nemotron_parse_gpus_per_actor: Optional[float] = typer.Option(
        None, "--nemotron-parse-gpus-per-actor", min=0.0, max=1.0
    ),
    nemotron_parse_batch_size: Optional[int] = typer.Option(0, "--nemotron-parse-batch-size"),
    # LanceDB / evaluation
    lancedb_uri: str = typer.Option(LANCEDB_URI, "--lancedb-uri"),
    save_intermediate: Optional[Path] = typer.Option(
        None,
        "--save-intermediate",
        help="Directory to write extraction results as Parquet (for full-page markdown / page index).",
        path_type=Path,
        file_okay=False,
        dir_okay=True,
    ),
    hybrid: bool = typer.Option(False, "--hybrid/--no-hybrid"),
    query_csv: Path = typer.Option("./data/bo767_query_gt.csv", "--query-csv", path_type=Path),
    recall_match_mode: str = typer.Option("pdf_page", "--recall-match-mode"),
    audio_match_tolerance_secs: float = typer.Option(2.0, "--audio-match-tolerance-secs", min=0.0),
    segment_audio: bool = typer.Option(False, "--segment-audio/--no-segment-audio"),
    audio_split_type: str = typer.Option("size", "--audio-split-type"),
    audio_split_interval: int = typer.Option(500000, "--audio-split-interval", min=1),
    evaluation_mode: str = typer.Option("recall", "--evaluation-mode"),
    reranker: Optional[bool] = typer.Option(False, "--reranker/--no-reranker"),
    reranker_model_name: str = typer.Option(VL_RERANK_MODEL, "--reranker-model-name"),
    reranker_invoke_url: Optional[str] = typer.Option(None, "--reranker-invoke-url"),
    beir_loader: Optional[str] = typer.Option(None, "--beir-loader"),
    beir_dataset_name: Optional[str] = typer.Option(None, "--beir-dataset-name"),
    beir_split: str = typer.Option("test", "--beir-split"),
    beir_query_language: Optional[str] = typer.Option(None, "--beir-query-language"),
    beir_doc_id_field: str = typer.Option("pdf_basename", "--beir-doc-id-field"),
    beir_k: list[int] = typer.Option([], "--beir-k"),
    recall_details: bool = typer.Option(True, "--recall-details/--no-recall-details"),
    runtime_metrics_dir: Optional[Path] = typer.Option(None, "--runtime-metrics-dir", path_type=Path),
    runtime_metrics_prefix: Optional[str] = typer.Option(None, "--runtime-metrics-prefix"),
    detection_summary_file: Optional[Path] = typer.Option(None, "--detection-summary-file", path_type=Path),
    log_file: Optional[Path] = typer.Option(None, "--log-file", path_type=Path, dir_okay=False),
) -> None:
    _ = ctx
    log_handle, original_stdout, original_stderr = _configure_logging(log_file, debug=bool(debug))
    try:
        if run_mode not in {"batch", "inprocess"}:
            raise ValueError(f"Unsupported --run-mode: {run_mode!r}")
        if recall_match_mode not in {"pdf_page", "pdf_only", "audio_segment"}:
            raise ValueError(f"Unsupported --recall-match-mode: {recall_match_mode!r}")
        if audio_split_type not in {"size", "time", "frame"}:
            raise ValueError(f"Unsupported --audio-split-type: {audio_split_type!r}")
        if evaluation_mode not in {"recall", "beir"}:
            raise ValueError(f"Unsupported --evaluation-mode: {evaluation_mode!r}")

        if run_mode == "batch":
            os.environ["RAY_LOG_TO_DRIVER"] = "1" if ray_log_to_driver else "0"

        lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
        _ensure_lancedb_table(lancedb_uri, LANCEDB_TABLE)

        remote_api_key = resolve_remote_api_key(api_key)
        extract_remote_api_key = remote_api_key
        embed_remote_api_key = remote_api_key
        caption_remote_api_key = remote_api_key
        reranker_remote_api_key = remote_api_key

        # Warn if remote URLs configured without an API key
        if (
            any(
                (
                    page_elements_invoke_url,
                    ocr_invoke_url,
                    graphic_elements_invoke_url,
                    table_structure_invoke_url,
                    embed_invoke_url,
                    reranker_invoke_url,
                )
            )
            and remote_api_key is None
        ):
            logger.warning("Remote endpoint URL(s) were configured without an API key.")

        # Zero out GPU fractions when a remote URL replaces the local model
        if page_elements_invoke_url and float(page_elements_gpus_per_actor or 0.0) != 0.0:
            logger.warning("Forcing page-elements GPUs to 0.0 because --page-elements-invoke-url is set.")
            page_elements_gpus_per_actor = 0.0
        if ocr_invoke_url and float(ocr_gpus_per_actor or 0.0) != 0.0:
            logger.warning("Forcing OCR GPUs to 0.0 because --ocr-invoke-url is set.")
            ocr_gpus_per_actor = 0.0
        if embed_invoke_url and float(embed_gpus_per_actor or 0.0) != 0.0:
            logger.warning("Forcing embed GPUs to 0.0 because --embed-invoke-url is set.")
            embed_gpus_per_actor = 0.0

        file_patterns = _resolve_file_patterns(Path(input_path), input_type)

        # ------------------------------------------------------------------
        # Build extraction params
        # ------------------------------------------------------------------
        extract_batch_tuning = BatchTuningParams(
            **{
                k: v
                for k, v in {
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
                        else (page_elements_gpus_per_actor if page_elements_gpus_per_actor is not None else None)
                    ),
                    "ocr_inference_batch_size": ocr_batch_size or None,
                    "ocr_workers": ocr_actors or None,
                    "ocr_cpus_per_actor": ocr_cpus_per_actor or None,
                    "gpu_ocr": (
                        0.0 if ocr_invoke_url else (ocr_gpus_per_actor if ocr_gpus_per_actor is not None else None)
                    ),
                    "nemotron_parse_batch_size": nemotron_parse_batch_size or None,
                    "nemotron_parse_workers": nemotron_parse_actors or None,
                    "gpu_nemotron_parse": (
                        nemotron_parse_gpus_per_actor if nemotron_parse_gpus_per_actor is not None else None
                    ),
                }.items()
                if v is not None
            }
        )
        extract_params = ExtractParams(
            **{
                k: v
                for k, v in {
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
                }.items()
                if v is not None
            }
        )

        # ------------------------------------------------------------------
        # Build embedding params
        # ------------------------------------------------------------------
        embed_batch_tuning = BatchTuningParams(
            **{
                k: v
                for k, v in {
                    "embed_batch_size": embed_batch_size or None,
                    "embed_workers": embed_actors or None,
                    "embed_cpus_per_actor": embed_cpus_per_actor or None,
                    "gpu_embed": (
                        0.0
                        if embed_invoke_url
                        else (embed_gpus_per_actor if embed_gpus_per_actor is not None else None)
                    ),
                }.items()
                if v is not None
            }
        )
        embed_params = EmbedParams(
            **{
                k: v
                for k, v in {
                    "model_name": embed_model_name,
                    "embed_invoke_url": embed_invoke_url,
                    "api_key": embed_remote_api_key,
                    "embed_modality": embed_modality,
                    "text_elements_modality": text_elements_modality,
                    "structured_elements_modality": structured_elements_modality,
                    "embed_granularity": embed_granularity,
                    "batch_tuning": embed_batch_tuning,
                    "inference_batch_size": embed_batch_size or None,
                }.items()
                if v is not None
            }
        )
        text_chunk_params = TextChunkParams(
            max_tokens=text_chunk_max_tokens or 1024,
            overlap_tokens=text_chunk_overlap_tokens if text_chunk_overlap_tokens is not None else 150,
        )

        # ------------------------------------------------------------------
        # Build GraphIngestor and configure pipeline stages
        # ------------------------------------------------------------------
        logger.info("Building graph pipeline (run_mode=%s) for %s ...", run_mode, input_path)

        node_overrides = {}
        if caption_gpus_per_actor is not None:
            node_overrides["CaptionActor"] = {"num_gpus": caption_gpus_per_actor}

        ingestor = GraphIngestor(run_mode=run_mode, ray_address=ray_address, node_overrides=node_overrides or None)
        ingestor = ingestor.files(file_patterns)

        # Extraction stage
        if input_type == "txt":
            ingestor = ingestor.extract_txt(text_chunk_params)
        elif input_type == "html":
            ingestor = ingestor.extract_html(text_chunk_params)
        elif input_type == "image":
            ingestor = ingestor.extract_image_files(extract_params)
        elif input_type == "audio":
            asr_params = asr_params_from_env().model_copy(update={"segment_audio": bool(segment_audio)})
            ingestor = ingestor.extract_audio(
                params=AudioChunkParams(split_type=audio_split_type, split_interval=int(audio_split_interval)),
                asr_params=asr_params,
            )
        else:
            # "pdf" or "doc"
            ingestor = ingestor.extract(extract_params)

        # Optional post-extraction stages
        enable_text_chunk = text_chunk or text_chunk_max_tokens is not None or text_chunk_overlap_tokens is not None
        if enable_text_chunk:
            ingestor = ingestor.split(text_chunk_params)

        enable_caption = caption or caption_invoke_url is not None
        enable_dedup = dedup if dedup is not None else enable_caption
        if enable_dedup:
            ingestor = ingestor.dedup(DedupParams(iou_threshold=dedup_iou_threshold))

        if enable_caption:
            ingestor = ingestor.caption(
                CaptionParams(
                    endpoint_url=caption_invoke_url,
                    api_key=caption_remote_api_key,
                    model_name=caption_model_name,
                    device=caption_device,
                    context_text_max_chars=caption_context_text_max_chars,
                    gpu_memory_utilization=caption_gpu_memory_utilization,
                    temperature=caption_temperature,
                    top_p=caption_top_p,
                    max_tokens=caption_max_tokens,
                )
            )

        if store_images_uri is not None:
            ingestor = ingestor.store(
                StoreParams(
                    storage_uri=store_images_uri,
                    store_text=store_text,
                    strip_base64=strip_base64,
                )
            )

        ingestor = ingestor.embed(embed_params)

        # ------------------------------------------------------------------
        # Execute the graph via the executor
        # ------------------------------------------------------------------
        logger.info("Starting ingestion of %s ...", input_path)
        ingest_start = time.perf_counter()

        # GraphIngestor.ingest() builds the Graph, creates the executor,
        # and calls executor.ingest(file_patterns) returning:
        #   batch mode     -> materialized ray.data.Dataset
        #   inprocess mode -> pandas.DataFrame
        result = ingestor.ingest()

        ingestion_only_total_time = time.perf_counter() - ingest_start

        # ------------------------------------------------------------------
        # Collect results
        # ------------------------------------------------------------------
        if run_mode == "batch":
            import ray

            ray_download_start = time.perf_counter()
            ingest_local_results = result.take_all()
            ray_download_time = time.perf_counter() - ray_download_start

            import pandas as pd

            result_df = pd.DataFrame(ingest_local_results)
            num_rows = _count_input_units(result_df)
        else:
            import pandas as pd

            result_df = result
            ingest_local_results = result_df.to_dict("records")
            ray_download_time = 0.0
            num_rows = _count_input_units(result_df)

        if save_intermediate is not None:
            out_dir = Path(save_intermediate).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "extraction.parquet"
            result_df.to_parquet(out_path, index=False)
            logger.info("Wrote extraction Parquet for intermediate use: %s", out_path)

        if detection_summary_file is not None:
            from nemo_retriever.utils.detection_summary import (
                collect_detection_summary_from_df,
                write_detection_summary,
            )

            write_detection_summary(
                Path(detection_summary_file),
                collect_detection_summary_from_df(result_df),
            )

        # ------------------------------------------------------------------
        # Write to LanceDB
        # ------------------------------------------------------------------
        lancedb_write_start = time.perf_counter()
        handle_lancedb(ingest_local_results, lancedb_uri, LANCEDB_TABLE, hybrid=hybrid, mode="overwrite")
        lancedb_write_time = time.perf_counter() - lancedb_write_start

        # ------------------------------------------------------------------
        # Recall / BEIR evaluation
        # ------------------------------------------------------------------
        import lancedb as _lancedb_mod

        db = _lancedb_mod.connect(lancedb_uri)
        table = db.open_table(LANCEDB_TABLE)

        if int(table.count_rows()) == 0:
            logger.warning("LanceDB table is empty; skipping %s evaluation.", evaluation_mode)
            _write_runtime_summary(
                runtime_metrics_dir,
                runtime_metrics_prefix,
                {
                    "run_mode": run_mode,
                    "input_path": str(Path(input_path).resolve()),
                    "input_pages": int(num_rows),
                    "num_pages": int(num_rows),
                    "num_rows": int(len(result_df.index)),
                    "ingestion_only_secs": float(ingestion_only_total_time),
                    "ray_download_secs": float(ray_download_time),
                    "lancedb_write_secs": float(lancedb_write_time),
                    "evaluation_secs": 0.0,
                    "total_secs": float(time.perf_counter() - ingest_start),
                    "evaluation_mode": evaluation_mode,
                    "evaluation_metrics": {},
                    "recall_details": bool(recall_details),
                    "lancedb_uri": str(lancedb_uri),
                    "lancedb_table": str(LANCEDB_TABLE),
                },
            )
            if run_mode == "batch":
                ray.shutdown()
            return

        from nemo_retriever.model import resolve_embed_model
        from nemo_retriever.utils.detection_summary import print_run_summary

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

            from nemo_retriever.recall.beir import BeirConfig, evaluate_lancedb_beir

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
                reranker_endpoint=reranker_invoke_url,
                reranker_api_key=reranker_remote_api_key or "",
            )
            evaluation_start = time.perf_counter()
            beir_dataset, _raw_hits, _run, evaluation_metrics = evaluate_lancedb_beir(cfg)
            evaluation_total_time = time.perf_counter() - evaluation_start
            evaluation_label = "BEIR"
            evaluation_query_count = len(beir_dataset.query_ids)
        else:
            query_csv_path = Path(query_csv)
            if not query_csv_path.exists():
                logger.warning("Query CSV not found at %s; skipping recall evaluation.", query_csv_path)
                _write_runtime_summary(
                    runtime_metrics_dir,
                    runtime_metrics_prefix,
                    {
                        "run_mode": run_mode,
                        "input_path": str(Path(input_path).resolve()),
                        "input_pages": int(num_rows),
                        "num_pages": int(num_rows),
                        "num_rows": int(len(result_df.index)),
                        "ingestion_only_secs": float(ingestion_only_total_time),
                        "ray_download_secs": float(ray_download_time),
                        "lancedb_write_secs": float(lancedb_write_time),
                        "evaluation_secs": 0.0,
                        "total_secs": float(time.perf_counter() - ingest_start),
                        "evaluation_mode": evaluation_mode,
                        "evaluation_metrics": {},
                        "recall_details": bool(recall_details),
                        "lancedb_uri": str(lancedb_uri),
                        "lancedb_table": str(LANCEDB_TABLE),
                    },
                )
                if run_mode == "batch":
                    ray.shutdown()
                return

            from nemo_retriever.recall.core import RecallConfig, retrieve_and_score

            recall_cfg = RecallConfig(
                lancedb_uri=str(lancedb_uri),
                lancedb_table=str(LANCEDB_TABLE),
                embedding_model=_recall_model,
                embedding_http_endpoint=embed_invoke_url,
                embedding_api_key=embed_remote_api_key or "",
                top_k=10,
                ks=(1, 5, 10),
                hybrid=hybrid,
                match_mode=recall_match_mode,
                audio_match_tolerance_secs=float(audio_match_tolerance_secs),
                reranker=reranker_model_name if reranker else None,
                reranker_endpoint=reranker_invoke_url,
                reranker_api_key=reranker_remote_api_key or "",
                embed_modality=embed_modality,
            )
            evaluation_start = time.perf_counter()
            _df_query, _gold, _raw_hits, _retrieved_keys, evaluation_metrics = retrieve_and_score(
                query_csv=query_csv_path, cfg=recall_cfg
            )
            evaluation_total_time = time.perf_counter() - evaluation_start
            evaluation_query_count = len(_df_query.index)

        total_time = time.perf_counter() - ingest_start

        _write_runtime_summary(
            runtime_metrics_dir,
            runtime_metrics_prefix,
            {
                "run_mode": run_mode,
                "input_path": str(Path(input_path).resolve()),
                "input_pages": int(num_rows),
                "num_pages": int(num_rows),
                "num_rows": int(len(result_df.index)),
                "ingestion_only_secs": float(ingestion_only_total_time),
                "ray_download_secs": float(ray_download_time),
                "lancedb_write_secs": float(lancedb_write_time),
                "evaluation_secs": float(evaluation_total_time),
                "total_secs": float(total_time),
                "evaluation_mode": evaluation_mode,
                "evaluation_metrics": dict(evaluation_metrics),
                "evaluation_count": evaluation_query_count,
                "recall_details": bool(recall_details),
                "lancedb_uri": str(lancedb_uri),
                "lancedb_table": str(LANCEDB_TABLE),
            },
        )

        if run_mode == "batch":
            ray.shutdown()

        print_run_summary(
            num_rows,
            Path(input_path),
            hybrid,
            lancedb_uri,
            LANCEDB_TABLE,
            total_time,
            ingestion_only_total_time,
            ray_download_time,
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
