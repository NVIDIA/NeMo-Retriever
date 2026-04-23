# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer CLI for the end-to-end graph ingestion pipeline.

Registered on the ``retriever`` CLI as the ``pipeline`` subcommand.

Examples::

    # Batch mode (Ray) with PDF extraction + embedding
    retriever pipeline run /data/pdfs \\
        --run-mode batch \\
        --embed-invoke-url http://localhost:8000/v1

    # In-process mode (no Ray) for quick local testing
    retriever pipeline run /data/pdfs \\
        --run-mode inprocess \\
        --ocr-invoke-url http://localhost:9000/v1

    # Save extraction Parquet for full-page markdown (page index / export)
    retriever pipeline run /data/pdfs \\
        --lancedb-uri lancedb \\
        --save-intermediate /path/to/extracted_parquet_dir
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, TextIO

import typer

from nemo_retriever.audio import asr_params_from_env
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.model import VL_EMBED_MODEL, VL_RERANK_MODEL
from nemo_retriever.params import (
    AudioChunkParams,
    CaptionParams,
    DedupParams,
    EmbedParams,
    ExtractParams,
    StoreParams,
    TextChunkParams,
)
from nemo_retriever.params.models import BatchTuningParams
from nemo_retriever.utils.input_files import resolve_input_patterns
from nemo_retriever.utils.remote_auth import resolve_remote_api_key
from nemo_retriever.vector_store.lancedb_store import handle_lancedb

logger = logging.getLogger(__name__)

app = typer.Typer(help="End-to-end graph-based ingestion pipeline (extract -> embed -> LanceDB).")

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"

# Help panel labels (keep stable so --help groupings read consistently).
_PANEL_IO = "I/O and Execution"
_PANEL_EXTRACT = "PDF / Document Extraction"
_PANEL_REMOTE = "Remote NIM Endpoints"
_PANEL_EMBED = "Embedding"
_PANEL_DEDUP_CAPTION = "Dedup and Caption"
_PANEL_STORE_CHUNK = "Storage and Text Chunking"
_PANEL_AUDIO = "Audio"
_PANEL_RAY = "Ray / Batch Tuning"
_PANEL_LANCEDB = "LanceDB and Outputs"
_PANEL_EVAL = "Evaluation (Recall / BEIR)"
_PANEL_OBS = "Observability"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


class _TeeStream:
    """Mirror stdout/stderr writes into a second stream (e.g. a log file)."""

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


# ---------------------------------------------------------------------------
# Small utilities (LanceDB, summaries, file patterns)
# ---------------------------------------------------------------------------


def _ensure_lancedb_table(uri: str, table_name: str) -> None:
    from nemo_retriever.vector_store.lancedb_utils import lancedb_schema
    import lancedb
    import pyarrow as pa

    Path(uri).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(uri)
    try:
        db.open_table(table_name)
        return
    except ValueError as e:
        # lancedb has no TableNotFoundError; missing tables raise ValueError, same
        # substring LanceDB uses internally in db.py for this case.
        if f"Table '{table_name}' was not found" not in str(e):
            raise
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
    """Resolve input paths to glob patterns, recursing into subdirectories.

    Uses :func:`~nemo_retriever.utils.input_files.resolve_input_patterns` (``**``
    segments) and keeps only patterns that match at least one file, matching the
    historical ``graph_pipeline`` / main-branch behavior.
    """

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
    logger.debug("Using recursive input globs: %s", matched)
    return matched


# ---------------------------------------------------------------------------
# Parameter builders (split out from the old monolithic main())
# ---------------------------------------------------------------------------


def _build_extract_params(
    *,
    method: str,
    dpi: int,
    extract_text: bool,
    extract_tables: bool,
    extract_charts: bool,
    extract_infographics: bool,
    extract_page_as_image: bool,
    use_graphic_elements: bool,
    use_table_structure: bool,
    table_output_format: Optional[str],
    extract_remote_api_key: Optional[str],
    page_elements_invoke_url: Optional[str],
    ocr_invoke_url: Optional[str],
    graphic_elements_invoke_url: Optional[str],
    table_structure_invoke_url: Optional[str],
    pdf_split_batch_size: int,
    pdf_extract_batch_size: Optional[int],
    pdf_extract_tasks: Optional[int],
    pdf_extract_cpus_per_task: Optional[float],
    page_elements_actors: Optional[int],
    page_elements_batch_size: Optional[int],
    page_elements_cpus_per_actor: Optional[float],
    page_elements_gpus_per_actor: Optional[float],
    ocr_actors: Optional[int],
    ocr_batch_size: Optional[int],
    ocr_cpus_per_actor: Optional[float],
    ocr_gpus_per_actor: Optional[float],
    nemotron_parse_actors: Optional[int],
    nemotron_parse_batch_size: Optional[int],
    nemotron_parse_gpus_per_actor: Optional[float],
) -> ExtractParams:
    """Assemble :class:`ExtractParams` plus its :class:`BatchTuningParams`."""

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
    return ExtractParams(
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


def _build_embed_params(
    *,
    embed_model_name: str,
    embed_invoke_url: Optional[str],
    embed_remote_api_key: Optional[str],
    embed_modality: str,
    text_elements_modality: Optional[str],
    structured_elements_modality: Optional[str],
    embed_granularity: str,
    embed_actors: Optional[int],
    embed_batch_size: Optional[int],
    embed_cpus_per_actor: Optional[float],
    embed_gpus_per_actor: Optional[float],
    local_ingest_backend: str = "vllm",
) -> EmbedParams:
    """Assemble :class:`EmbedParams` plus its :class:`BatchTuningParams`."""

    embed_batch_tuning = BatchTuningParams(
        **{
            k: v
            for k, v in {
                "embed_batch_size": embed_batch_size or None,
                "embed_workers": embed_actors or None,
                "embed_cpus_per_actor": embed_cpus_per_actor or None,
                "gpu_embed": (
                    0.0 if embed_invoke_url else (embed_gpus_per_actor if embed_gpus_per_actor is not None else None)
                ),
            }.items()
            if v is not None
        }
    )
    return EmbedParams(
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
                "local_ingest_backend": local_ingest_backend,
                "batch_tuning": embed_batch_tuning,
                "inference_batch_size": embed_batch_size or None,
            }.items()
            if v is not None
        }
    )


def _build_ingestor(
    *,
    run_mode: str,
    ray_address: Optional[str],
    file_patterns: list[str],
    input_type: str,
    extract_params: ExtractParams,
    embed_params: EmbedParams,
    text_chunk_params: TextChunkParams,
    enable_text_chunk: bool,
    enable_dedup: bool,
    enable_caption: bool,
    dedup_iou_threshold: float,
    caption_invoke_url: Optional[str],
    caption_remote_api_key: Optional[str],
    caption_model_name: str,
    caption_device: Optional[str],
    caption_context_text_max_chars: int,
    caption_gpu_memory_utilization: float,
    caption_gpus_per_actor: Optional[float],
    caption_temperature: float,
    caption_top_p: Optional[float],
    caption_max_tokens: int,
    store_images_uri: Optional[str],
    store_text: bool,
    strip_base64: bool,
    segment_audio: bool,
    audio_split_type: str,
    audio_split_interval: int,
) -> GraphIngestor:
    """Construct a :class:`GraphIngestor` with all requested stages attached."""

    node_overrides: dict[str, dict[str, Any]] = {}
    if caption_gpus_per_actor is not None:
        node_overrides["CaptionActor"] = {"num_gpus": caption_gpus_per_actor}

    ingestor = GraphIngestor(
        run_mode=run_mode,
        ray_address=ray_address,
        node_overrides=node_overrides or None,
    )
    ingestor = ingestor.files(file_patterns)

    # Extraction stage is selected by input type.
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

    if enable_text_chunk:
        ingestor = ingestor.split(text_chunk_params)

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

    return ingestor.embed(embed_params)


def _collect_results(run_mode: str, result: Any) -> tuple[list[dict[str, Any]], Any, float, int]:
    """Materialize the graph result into a list of records + DataFrame.

    Returns ``(records, result_df, ray_download_secs, num_input_units)``.
    """

    import pandas as pd

    if run_mode == "batch":
        ray_download_start = time.perf_counter()
        records = result.take_all()
        ray_download_time = time.perf_counter() - ray_download_start
        result_df = pd.DataFrame(records)
    else:
        result_df = result
        records = result_df.to_dict("records")
        ray_download_time = 0.0

    return records, result_df, float(ray_download_time), _count_input_units(result_df)


def _run_evaluation(
    *,
    evaluation_mode: str,
    lancedb_uri: str,
    embed_model_name: str,
    embed_invoke_url: Optional[str],
    embed_remote_api_key: Optional[str],
    embed_modality: str,
    query_csv: Path,
    recall_match_mode: str,
    audio_match_tolerance_secs: float,
    hybrid: bool,
    reranker: Optional[bool],
    reranker_model_name: str,
    reranker_invoke_url: Optional[str],
    reranker_api_key: str,
    beir_loader: Optional[str],
    beir_dataset_name: Optional[str],
    beir_split: str,
    beir_query_language: Optional[str],
    beir_doc_id_field: str,
    beir_k: list[int],
    local_query_embed_backend: str = "hf",
) -> tuple[str, float, dict[str, float], Optional[int], bool]:
    """Run recall or BEIR evaluation.

    Returns ``(label, elapsed_secs, metrics, query_count, ran)``.  When the
    query CSV is missing in recall mode, ``ran`` is ``False`` and the caller
    should skip metric recording.
    """

    from nemo_retriever.model import resolve_embed_model

    embed_model = resolve_embed_model(str(embed_model_name))

    if evaluation_mode == "beir":
        if not beir_loader:
            raise ValueError("--beir-loader is required when --evaluation-mode=beir")
        if not beir_dataset_name:
            raise ValueError("--beir-dataset-name is required when --evaluation-mode=beir")

        from nemo_retriever.recall.beir import BeirConfig, evaluate_lancedb_beir

        cfg = BeirConfig(
            lancedb_uri=str(lancedb_uri),
            lancedb_table=str(LANCEDB_TABLE),
            embedding_model=embed_model,
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
            reranker_api_key=reranker_api_key,
            local_query_embed_backend=local_query_embed_backend,
        )
        evaluation_start = time.perf_counter()
        beir_dataset, _raw_hits, _run, metrics = evaluate_lancedb_beir(cfg)
        return "BEIR", time.perf_counter() - evaluation_start, metrics, len(beir_dataset.query_ids), True

    # Default: recall eval against a query CSV.
    query_csv_path = Path(query_csv)
    if not query_csv_path.exists():
        logger.warning("Query CSV not found at %s; skipping recall evaluation.", query_csv_path)
        return "Recall", 0.0, {}, None, False

    from nemo_retriever.recall.core import RecallConfig, retrieve_and_score

    recall_cfg = RecallConfig(
        lancedb_uri=str(lancedb_uri),
        lancedb_table=str(LANCEDB_TABLE),
        embedding_model=embed_model,
        embedding_http_endpoint=embed_invoke_url,
        embedding_api_key=embed_remote_api_key or "",
        top_k=10,
        ks=(1, 5, 10),
        hybrid=hybrid,
        match_mode=recall_match_mode,
        audio_match_tolerance_secs=float(audio_match_tolerance_secs),
        reranker=reranker_model_name if reranker else None,
        reranker_endpoint=reranker_invoke_url,
        reranker_api_key=reranker_api_key,
        embed_modality=embed_modality,
        local_query_embed_backend=local_query_embed_backend,
    )
    evaluation_start = time.perf_counter()
    df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv_path, cfg=recall_cfg)
    return "Recall", time.perf_counter() - evaluation_start, metrics, len(df_query.index), True


# ---------------------------------------------------------------------------
# Typer command: `retriever pipeline run`
# ---------------------------------------------------------------------------


@app.command("run")
def run(
    ctx: typer.Context,
    input_path: Path = typer.Argument(
        ...,
        help="File or directory of documents to ingest.",
        path_type=Path,
    ),
    # --- I/O and execution ------------------------------------------------
    run_mode: str = typer.Option(
        "batch",
        "--run-mode",
        help="Execution mode: 'batch' (Ray Data) or 'inprocess' (pandas, no Ray).",
        rich_help_panel=_PANEL_IO,
    ),
    input_type: str = typer.Option(
        "pdf",
        "--input-type",
        help="Input type: 'pdf', 'doc', 'txt', 'html', 'image', or 'audio'.",
        rich_help_panel=_PANEL_IO,
    ),
    debug: bool = typer.Option(
        False, "--debug/--no-debug", help="Enable debug-level logging.", rich_help_panel=_PANEL_IO
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", path_type=Path, dir_okay=False, rich_help_panel=_PANEL_IO
    ),
    # --- PDF / document extraction ---------------------------------------
    method: str = typer.Option(
        "pdfium", "--method", help="PDF text extraction method.", rich_help_panel=_PANEL_EXTRACT
    ),
    dpi: int = typer.Option(
        300, "--dpi", min=72, help="Render DPI for PDF page images.", rich_help_panel=_PANEL_EXTRACT
    ),
    extract_text: bool = typer.Option(True, "--extract-text/--no-extract-text", rich_help_panel=_PANEL_EXTRACT),
    extract_tables: bool = typer.Option(True, "--extract-tables/--no-extract-tables", rich_help_panel=_PANEL_EXTRACT),
    extract_charts: bool = typer.Option(True, "--extract-charts/--no-extract-charts", rich_help_panel=_PANEL_EXTRACT),
    extract_infographics: bool = typer.Option(
        False, "--extract-infographics/--no-extract-infographics", rich_help_panel=_PANEL_EXTRACT
    ),
    extract_page_as_image: bool = typer.Option(
        True,
        "--extract-page-as-image/--no-extract-page-as-image",
        rich_help_panel=_PANEL_EXTRACT,
    ),
    use_graphic_elements: bool = typer.Option(False, "--use-graphic-elements", rich_help_panel=_PANEL_EXTRACT),
    use_table_structure: bool = typer.Option(False, "--use-table-structure", rich_help_panel=_PANEL_EXTRACT),
    table_output_format: Optional[str] = typer.Option(None, "--table-output-format", rich_help_panel=_PANEL_EXTRACT),
    # --- Remote NIM endpoints --------------------------------------------
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Bearer token for remote NIM endpoints.",
        rich_help_panel=_PANEL_REMOTE,
    ),
    page_elements_invoke_url: Optional[str] = typer.Option(
        None, "--page-elements-invoke-url", rich_help_panel=_PANEL_REMOTE
    ),
    ocr_invoke_url: Optional[str] = typer.Option(None, "--ocr-invoke-url", rich_help_panel=_PANEL_REMOTE),
    graphic_elements_invoke_url: Optional[str] = typer.Option(
        None, "--graphic-elements-invoke-url", rich_help_panel=_PANEL_REMOTE
    ),
    table_structure_invoke_url: Optional[str] = typer.Option(
        None, "--table-structure-invoke-url", rich_help_panel=_PANEL_REMOTE
    ),
    embed_invoke_url: Optional[str] = typer.Option(None, "--embed-invoke-url", rich_help_panel=_PANEL_REMOTE),
    # --- Embedding --------------------------------------------------------
    embed_model_name: str = typer.Option(VL_EMBED_MODEL, "--embed-model-name", rich_help_panel=_PANEL_EMBED),
    embed_modality: str = typer.Option("text", "--embed-modality", rich_help_panel=_PANEL_EMBED),
    embed_granularity: str = typer.Option("element", "--embed-granularity", rich_help_panel=_PANEL_EMBED),
    embed_local_ingest_backend: str = typer.Option(
        "vllm",
        "--embed-local-ingest-backend",
        help="Local ingest-time text embedder when --embed-invoke-url is unset: vllm or hf.",
        rich_help_panel=_PANEL_EMBED,
    ),
    text_elements_modality: Optional[str] = typer.Option(
        None, "--text-elements-modality", rich_help_panel=_PANEL_EMBED
    ),
    structured_elements_modality: Optional[str] = typer.Option(
        None, "--structured-elements-modality", rich_help_panel=_PANEL_EMBED
    ),
    # --- Dedup / caption -------------------------------------------------
    dedup: Optional[bool] = typer.Option(None, "--dedup/--no-dedup", rich_help_panel=_PANEL_DEDUP_CAPTION),
    dedup_iou_threshold: float = typer.Option(0.45, "--dedup-iou-threshold", rich_help_panel=_PANEL_DEDUP_CAPTION),
    caption: bool = typer.Option(False, "--caption/--no-caption", rich_help_panel=_PANEL_DEDUP_CAPTION),
    caption_invoke_url: Optional[str] = typer.Option(
        None, "--caption-invoke-url", rich_help_panel=_PANEL_DEDUP_CAPTION
    ),
    caption_model_name: str = typer.Option(
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        "--caption-model-name",
        rich_help_panel=_PANEL_DEDUP_CAPTION,
    ),
    caption_device: Optional[str] = typer.Option(None, "--caption-device", rich_help_panel=_PANEL_DEDUP_CAPTION),
    caption_context_text_max_chars: int = typer.Option(
        0, "--caption-context-text-max-chars", rich_help_panel=_PANEL_DEDUP_CAPTION
    ),
    caption_gpu_memory_utilization: float = typer.Option(
        0.5, "--caption-gpu-memory-utilization", rich_help_panel=_PANEL_DEDUP_CAPTION
    ),
    caption_gpus_per_actor: Optional[float] = typer.Option(
        None, "--caption-gpus-per-actor", max=1.0, rich_help_panel=_PANEL_DEDUP_CAPTION
    ),
    caption_temperature: float = typer.Option(
        1.0, "--caption-temperature", min=0.0, max=2.0, rich_help_panel=_PANEL_DEDUP_CAPTION
    ),
    caption_top_p: Optional[float] = typer.Option(
        None, "--caption-top-p", min=0.0, max=1.0, rich_help_panel=_PANEL_DEDUP_CAPTION
    ),
    caption_max_tokens: int = typer.Option(1024, "--caption-max-tokens", min=1, rich_help_panel=_PANEL_DEDUP_CAPTION),
    # --- Storage and text chunking --------------------------------------
    store_images_uri: Optional[str] = typer.Option(
        None,
        "--store-images-uri",
        help="Store extracted images to this URI.",
        rich_help_panel=_PANEL_STORE_CHUNK,
    ),
    store_text: bool = typer.Option(
        False,
        "--store-text/--no-store-text",
        help="Also store extracted text.",
        rich_help_panel=_PANEL_STORE_CHUNK,
    ),
    strip_base64: bool = typer.Option(
        True,
        "--strip-base64/--no-strip-base64",
        help="Strip base64 after storing.",
        rich_help_panel=_PANEL_STORE_CHUNK,
    ),
    text_chunk: bool = typer.Option(False, "--text-chunk", rich_help_panel=_PANEL_STORE_CHUNK),
    text_chunk_max_tokens: Optional[int] = typer.Option(
        None, "--text-chunk-max-tokens", rich_help_panel=_PANEL_STORE_CHUNK
    ),
    text_chunk_overlap_tokens: Optional[int] = typer.Option(
        None, "--text-chunk-overlap-tokens", rich_help_panel=_PANEL_STORE_CHUNK
    ),
    # --- Ray / batch tuning ---------------------------------------------
    # *_gpus_per_actor defaults are None (not 0.0) so we can distinguish
    # "not set -> use heuristic" from "explicitly 0 -> no GPU".  Other tuning
    # defaults use 0/0.0 because those values are never valid explicit choices.
    ray_address: Optional[str] = typer.Option(None, "--ray-address", rich_help_panel=_PANEL_RAY),
    ray_log_to_driver: bool = typer.Option(
        True, "--ray-log-to-driver/--no-ray-log-to-driver", rich_help_panel=_PANEL_RAY
    ),
    ocr_actors: Optional[int] = typer.Option(0, "--ocr-actors", rich_help_panel=_PANEL_RAY),
    ocr_batch_size: Optional[int] = typer.Option(0, "--ocr-batch-size", rich_help_panel=_PANEL_RAY),
    ocr_cpus_per_actor: Optional[float] = typer.Option(0.0, "--ocr-cpus-per-actor", rich_help_panel=_PANEL_RAY),
    ocr_gpus_per_actor: Optional[float] = typer.Option(
        None, "--ocr-gpus-per-actor", max=1.0, rich_help_panel=_PANEL_RAY
    ),
    page_elements_actors: Optional[int] = typer.Option(0, "--page-elements-actors", rich_help_panel=_PANEL_RAY),
    page_elements_batch_size: Optional[int] = typer.Option(0, "--page-elements-batch-size", rich_help_panel=_PANEL_RAY),
    page_elements_cpus_per_actor: Optional[float] = typer.Option(
        0.0, "--page-elements-cpus-per-actor", rich_help_panel=_PANEL_RAY
    ),
    page_elements_gpus_per_actor: Optional[float] = typer.Option(
        None, "--page-elements-gpus-per-actor", max=1.0, rich_help_panel=_PANEL_RAY
    ),
    embed_actors: Optional[int] = typer.Option(0, "--embed-actors", rich_help_panel=_PANEL_RAY),
    embed_batch_size: Optional[int] = typer.Option(0, "--embed-batch-size", rich_help_panel=_PANEL_RAY),
    embed_cpus_per_actor: Optional[float] = typer.Option(0.0, "--embed-cpus-per-actor", rich_help_panel=_PANEL_RAY),
    embed_gpus_per_actor: Optional[float] = typer.Option(
        None, "--embed-gpus-per-actor", max=1.0, rich_help_panel=_PANEL_RAY
    ),
    pdf_split_batch_size: int = typer.Option(1, "--pdf-split-batch-size", min=1, rich_help_panel=_PANEL_RAY),
    pdf_extract_batch_size: Optional[int] = typer.Option(0, "--pdf-extract-batch-size", rich_help_panel=_PANEL_RAY),
    pdf_extract_tasks: Optional[int] = typer.Option(0, "--pdf-extract-tasks", rich_help_panel=_PANEL_RAY),
    pdf_extract_cpus_per_task: Optional[float] = typer.Option(
        0.0, "--pdf-extract-cpus-per-task", rich_help_panel=_PANEL_RAY
    ),
    nemotron_parse_actors: Optional[int] = typer.Option(0, "--nemotron-parse-actors", rich_help_panel=_PANEL_RAY),
    nemotron_parse_gpus_per_actor: Optional[float] = typer.Option(
        None,
        "--nemotron-parse-gpus-per-actor",
        min=0.0,
        max=1.0,
        rich_help_panel=_PANEL_RAY,
    ),
    nemotron_parse_batch_size: Optional[int] = typer.Option(
        0, "--nemotron-parse-batch-size", rich_help_panel=_PANEL_RAY
    ),
    # --- Audio ----------------------------------------------------------
    segment_audio: bool = typer.Option(False, "--segment-audio/--no-segment-audio", rich_help_panel=_PANEL_AUDIO),
    audio_split_type: str = typer.Option("size", "--audio-split-type", rich_help_panel=_PANEL_AUDIO),
    audio_split_interval: int = typer.Option(500000, "--audio-split-interval", min=1, rich_help_panel=_PANEL_AUDIO),
    audio_match_tolerance_secs: float = typer.Option(
        2.0, "--audio-match-tolerance-secs", min=0.0, rich_help_panel=_PANEL_AUDIO
    ),
    # --- LanceDB / outputs ---------------------------------------------
    lancedb_uri: str = typer.Option(LANCEDB_URI, "--lancedb-uri", rich_help_panel=_PANEL_LANCEDB),
    save_intermediate: Optional[Path] = typer.Option(
        None,
        "--save-intermediate",
        help="Directory to write extraction results as Parquet (for full-page markdown / page index).",
        path_type=Path,
        file_okay=False,
        dir_okay=True,
        rich_help_panel=_PANEL_LANCEDB,
    ),
    hybrid: bool = typer.Option(False, "--hybrid/--no-hybrid", rich_help_panel=_PANEL_LANCEDB),
    detection_summary_file: Optional[Path] = typer.Option(
        None, "--detection-summary-file", path_type=Path, rich_help_panel=_PANEL_LANCEDB
    ),
    runtime_metrics_dir: Optional[Path] = typer.Option(
        None, "--runtime-metrics-dir", path_type=Path, rich_help_panel=_PANEL_OBS
    ),
    runtime_metrics_prefix: Optional[str] = typer.Option(None, "--runtime-metrics-prefix", rich_help_panel=_PANEL_OBS),
    # --- Evaluation -----------------------------------------------------
    evaluation_mode: str = typer.Option("recall", "--evaluation-mode", rich_help_panel=_PANEL_EVAL),
    query_csv: Path = typer.Option(
        "./data/bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        rich_help_panel=_PANEL_EVAL,
    ),
    recall_match_mode: str = typer.Option("pdf_page", "--recall-match-mode", rich_help_panel=_PANEL_EVAL),
    recall_details: bool = typer.Option(True, "--recall-details/--no-recall-details", rich_help_panel=_PANEL_EVAL),
    recall_local_query_embed_backend: str = typer.Option(
        "hf",
        "--recall-local-query-embed-backend",
        help="Local query embedding backend when --embed-invoke-url is unset: hf (default) or vllm.",
        rich_help_panel=_PANEL_EVAL,
    ),
    reranker: Optional[bool] = typer.Option(False, "--reranker/--no-reranker", rich_help_panel=_PANEL_EVAL),
    reranker_model_name: str = typer.Option(VL_RERANK_MODEL, "--reranker-model-name", rich_help_panel=_PANEL_EVAL),
    reranker_invoke_url: Optional[str] = typer.Option(
        None,
        "--reranker-invoke-url",
        help="OpenAI-compatible reranker NIM HTTP endpoint (recall and BEIR evaluation).",
        rich_help_panel=_PANEL_EVAL,
    ),
    reranker_api_key: Optional[str] = typer.Option(
        None,
        "--reranker-api-key",
        help="Bearer token for the reranker NIM; defaults to --api-key / NVIDIA_API_KEY when omitted.",
        rich_help_panel=_PANEL_EVAL,
    ),
    beir_loader: Optional[str] = typer.Option(None, "--beir-loader", rich_help_panel=_PANEL_EVAL),
    beir_dataset_name: Optional[str] = typer.Option(None, "--beir-dataset-name", rich_help_panel=_PANEL_EVAL),
    beir_split: str = typer.Option("test", "--beir-split", rich_help_panel=_PANEL_EVAL),
    beir_query_language: Optional[str] = typer.Option(None, "--beir-query-language", rich_help_panel=_PANEL_EVAL),
    beir_doc_id_field: str = typer.Option("pdf_basename", "--beir-doc-id-field", rich_help_panel=_PANEL_EVAL),
    beir_k: list[int] = typer.Option([], "--beir-k", rich_help_panel=_PANEL_EVAL),
    eval_config: Optional[Path] = typer.Option(
        None,
        "--eval-config",
        help="Path to QA sweep YAML/JSON (required when --evaluation-mode=qa; same as `retriever eval run --config`).",
        path_type=Path,
        dir_okay=False,
        rich_help_panel=_PANEL_EVAL,
    ),
    retrieval_save_path: Optional[Path] = typer.Option(
        None,
        "--retrieval-save-path",
        help="Override retrieval.save_path in the QA config (page-index / export JSON, optional).",
        path_type=Path,
        rich_help_panel=_PANEL_EVAL,
    ),
    eval_page_index: Optional[Path] = typer.Option(
        None,
        "--page-index",
        help="Override retrieval.page_index in the QA config (optional).",
        path_type=Path,
        dir_okay=False,
        file_okay=True,
        exists=True,
        rich_help_panel=_PANEL_EVAL,
    ),
) -> None:
    """Run the end-to-end graph ingestion pipeline against ``INPUT_PATH``."""

    _ = ctx
    log_handle, original_stdout, original_stderr = _configure_logging(log_file, debug=bool(debug))
    try:
        if run_mode not in {"batch", "inprocess"}:
            raise ValueError(f"Unsupported --run-mode: {run_mode!r}")
        if recall_match_mode not in {"pdf_page", "pdf_only", "audio_segment"}:
            raise ValueError(f"Unsupported --recall-match-mode: {recall_match_mode!r}")
        if audio_split_type not in {"size", "time", "frame"}:
            raise ValueError(f"Unsupported --audio-split-type: {audio_split_type!r}")
        if evaluation_mode not in {"recall", "beir", "qa"}:
            raise ValueError(f"Unsupported --evaluation-mode: {evaluation_mode!r}")
        if evaluation_mode == "qa" and eval_config is None:
            raise typer.BadParameter(
                "--evaluation-mode=qa requires --eval-config (QA sweep YAML/JSON). "
                "Use the same file format as `retriever eval run --config` (dataset, retrieval, models, ...)."
            )

        if run_mode == "batch":
            os.environ["RAY_LOG_TO_DRIVER"] = "1" if ray_log_to_driver else "0"

        lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
        _ensure_lancedb_table(lancedb_uri, LANCEDB_TABLE)

        remote_api_key = resolve_remote_api_key(api_key)
        extract_remote_api_key = remote_api_key
        embed_remote_api_key = remote_api_key
        caption_remote_api_key = remote_api_key
        reranker_bearer = (
            resolve_remote_api_key(reranker_api_key) if reranker_api_key is not None else remote_api_key
        ) or ""

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
        if reranker_invoke_url and not reranker_bearer.strip():
            logger.warning(
                "Reranker invoke URL is set but no bearer token was resolved; "
                "set --reranker-api-key or --api-key / NVIDIA_API_KEY."
            )

        # Zero out GPU fractions when a remote URL replaces the local model.
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

        extract_params = _build_extract_params(
            method=method,
            dpi=dpi,
            extract_text=extract_text,
            extract_tables=extract_tables,
            extract_charts=extract_charts,
            extract_infographics=extract_infographics,
            extract_page_as_image=extract_page_as_image,
            use_graphic_elements=use_graphic_elements,
            use_table_structure=use_table_structure,
            table_output_format=table_output_format,
            extract_remote_api_key=extract_remote_api_key,
            page_elements_invoke_url=page_elements_invoke_url,
            ocr_invoke_url=ocr_invoke_url,
            graphic_elements_invoke_url=graphic_elements_invoke_url,
            table_structure_invoke_url=table_structure_invoke_url,
            pdf_split_batch_size=pdf_split_batch_size,
            pdf_extract_batch_size=pdf_extract_batch_size,
            pdf_extract_tasks=pdf_extract_tasks,
            pdf_extract_cpus_per_task=pdf_extract_cpus_per_task,
            page_elements_actors=page_elements_actors,
            page_elements_batch_size=page_elements_batch_size,
            page_elements_cpus_per_actor=page_elements_cpus_per_actor,
            page_elements_gpus_per_actor=page_elements_gpus_per_actor,
            ocr_actors=ocr_actors,
            ocr_batch_size=ocr_batch_size,
            ocr_cpus_per_actor=ocr_cpus_per_actor,
            ocr_gpus_per_actor=ocr_gpus_per_actor,
            nemotron_parse_actors=nemotron_parse_actors,
            nemotron_parse_batch_size=nemotron_parse_batch_size,
            nemotron_parse_gpus_per_actor=nemotron_parse_gpus_per_actor,
        )

        embed_params = _build_embed_params(
            embed_model_name=embed_model_name,
            embed_invoke_url=embed_invoke_url,
            embed_remote_api_key=embed_remote_api_key,
            embed_modality=embed_modality,
            text_elements_modality=text_elements_modality,
            structured_elements_modality=structured_elements_modality,
            embed_granularity=embed_granularity,
            embed_actors=embed_actors,
            embed_batch_size=embed_batch_size,
            embed_cpus_per_actor=embed_cpus_per_actor,
            embed_gpus_per_actor=embed_gpus_per_actor,
            local_ingest_backend=embed_local_ingest_backend,
        )

        text_chunk_params = TextChunkParams(
            max_tokens=text_chunk_max_tokens or 1024,
            overlap_tokens=text_chunk_overlap_tokens if text_chunk_overlap_tokens is not None else 150,
        )

        enable_text_chunk = text_chunk or text_chunk_max_tokens is not None or text_chunk_overlap_tokens is not None
        enable_caption = caption or caption_invoke_url is not None
        enable_dedup = dedup if dedup is not None else enable_caption

        logger.info("Building graph pipeline (run_mode=%s) for %s ...", run_mode, input_path)
        ingestor = _build_ingestor(
            run_mode=run_mode,
            ray_address=ray_address,
            file_patterns=file_patterns,
            input_type=input_type,
            extract_params=extract_params,
            embed_params=embed_params,
            text_chunk_params=text_chunk_params,
            enable_text_chunk=enable_text_chunk,
            enable_dedup=enable_dedup,
            enable_caption=enable_caption,
            dedup_iou_threshold=dedup_iou_threshold,
            caption_invoke_url=caption_invoke_url,
            caption_remote_api_key=caption_remote_api_key,
            caption_model_name=caption_model_name,
            caption_device=caption_device,
            caption_context_text_max_chars=caption_context_text_max_chars,
            caption_gpu_memory_utilization=caption_gpu_memory_utilization,
            caption_gpus_per_actor=caption_gpus_per_actor,
            caption_temperature=caption_temperature,
            caption_top_p=caption_top_p,
            caption_max_tokens=caption_max_tokens,
            store_images_uri=store_images_uri,
            store_text=store_text,
            strip_base64=strip_base64,
            segment_audio=segment_audio,
            audio_split_type=audio_split_type,
            audio_split_interval=audio_split_interval,
        )

        # --- Execute ---------------------------------------------------
        logger.info("Starting ingestion of %s ...", input_path)
        ingest_start = time.perf_counter()
        raw_result = ingestor.ingest()
        ingestion_only_total_time = time.perf_counter() - ingest_start

        ingest_local_results, result_df, ray_download_time, num_rows = _collect_results(run_mode, raw_result)

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

        # --- Write to LanceDB ----------------------------------------
        lancedb_write_start = time.perf_counter()
        handle_lancedb(ingest_local_results, lancedb_uri, LANCEDB_TABLE, hybrid=hybrid, mode="overwrite")
        lancedb_write_time = time.perf_counter() - lancedb_write_start

        # --- Evaluation ---------------------------------------------
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
                import ray

                ray.shutdown()
            return

        if evaluation_mode == "qa":
            from nemo_retriever.evaluation.cli import run_qa_sweep_from_config_dict
            from nemo_retriever.evaluation.config import load_eval_config

            assert eval_config is not None
            cfg = load_eval_config(str(eval_config))
            r = cfg.setdefault("retrieval", {})
            if r.get("type", "lancedb") == "lancedb":
                r["type"] = "lancedb"
                r["lancedb_uri"] = lancedb_uri
                r["lancedb_table"] = LANCEDB_TABLE
            if retrieval_save_path is not None:
                r["save_path"] = str(Path(retrieval_save_path).resolve())
            if eval_page_index is not None:
                r["page_index"] = str(Path(eval_page_index).resolve())

            qa_t0 = time.perf_counter()
            qa_code = run_qa_sweep_from_config_dict(cfg)
            evaluation_total_time = time.perf_counter() - qa_t0
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
                    "evaluation_mode": "qa",
                    "evaluation_metrics": {},
                    "evaluation_count": None,
                    "recall_details": bool(recall_details),
                    "lancedb_uri": str(lancedb_uri),
                    "lancedb_table": str(LANCEDB_TABLE),
                    "qa_sweep_exit_code": qa_code,
                },
            )
            if run_mode == "batch":
                import ray

                ray.shutdown()

            from nemo_retriever.utils.detection_summary import print_run_summary

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
                {},
                evaluation_label="QA",
                evaluation_count=None,
            )
            if qa_code != 0:
                raise typer.Exit(code=qa_code)
            return

        evaluation_label, evaluation_total_time, evaluation_metrics, evaluation_query_count, ran = _run_evaluation(
            evaluation_mode=evaluation_mode,
            lancedb_uri=lancedb_uri,
            embed_model_name=embed_model_name,
            embed_invoke_url=embed_invoke_url,
            embed_remote_api_key=embed_remote_api_key,
            embed_modality=embed_modality,
            query_csv=query_csv,
            recall_match_mode=recall_match_mode,
            audio_match_tolerance_secs=audio_match_tolerance_secs,
            hybrid=hybrid,
            reranker=reranker,
            reranker_model_name=reranker_model_name,
            reranker_invoke_url=reranker_invoke_url,
            reranker_api_key=reranker_bearer,
            beir_loader=beir_loader,
            beir_dataset_name=beir_dataset_name,
            beir_split=beir_split,
            beir_query_language=beir_query_language,
            beir_doc_id_field=beir_doc_id_field,
            beir_k=beir_k,
            local_query_embed_backend=recall_local_query_embed_backend,
        )

        if not ran:
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
                import ray

                ray.shutdown()
            return

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
            import ray

            ray.shutdown()

        from nemo_retriever.utils.detection_summary import print_run_summary

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


def main() -> None:
    """Entrypoint for ``python -m nemo_retriever.pipeline``."""
    app()


if __name__ == "__main__":
    main()
