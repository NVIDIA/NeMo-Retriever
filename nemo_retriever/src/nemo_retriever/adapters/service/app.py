# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ray Serve application integrated with FastAPI.

Exposes /health, /version, /ingest, /stream-pdf, and OpenAI-compatible
/v1/embeddings, /v1/models, /v1/chat/completions, /v1/files,
/v1/audio/*, /v1/images/*, /v1/videos/*, /v1/pdf/*, plus /retrieve.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from ray import serve
except ImportError:
    serve = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NeMo Retriever",
    description=(
        "NeMo Retriever remote service.\n\n"
        "Provides document ingestion, text embedding, vector retrieval, "
        "and PDF streaming via a Ray Serve deployment.\n\n"
        "The `/v1/*` endpoints follow the OpenAI API specification."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "health", "description": "Service health and version endpoints."},
        {"name": "ingest", "description": "Document ingestion pipeline."},
        {"name": "openai", "description": "OpenAI-compatible API endpoints (/v1/*)."},
        {"name": "retrieval", "description": "Vector store query and retrieval."},
        {"name": "pdf", "description": "PDF management and processing (/v1/pdf/*)."},
    ],
    license_info={"name": "Apache-2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    contact={"name": "NVIDIA Corporation", "url": "https://www.nvidia.com/"},
)

STREAM_PDF_MAX_BYTES = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Request / response schemas — OpenAI-compatible
# ---------------------------------------------------------------------------


class CreateEmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text(s) to embed.")
    model: str = Field(..., description="Model ID to use for embedding.")
    encoding_format: Optional[Literal["float", "base64"]] = Field(
        None, description="Encoding format for the embeddings."
    )
    dimensions: Optional[int] = Field(None, description="Desired embedding dimensionality (model-dependent).")
    user: Optional[str] = Field(None, description="End-user identifier for abuse monitoring.")


class EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class CreateEmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingObject]
    model: str
    usage: EmbeddingUsage


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    url: Optional[str] = None


class ListModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelObject]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model ID to use.")
    messages: List[ChatMessage] = Field(..., description="Conversation messages.")
    temperature: Optional[float] = Field(None, description="Sampling temperature.")
    top_p: Optional[float] = Field(None, description="Nucleus sampling probability.")
    n: int = Field(1, description="Number of completions to generate.")
    stream: bool = Field(False, description="Stream responses via SSE.")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in the completion.")
    user: Optional[str] = Field(None, description="End-user identifier.")


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: CompletionUsage


class ChatDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatChunkChoice(BaseModel):
    index: int
    delta: ChatDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatChunkChoice]


class FileObject(BaseModel):
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: Literal["uploaded", "processed", "error"] = "processed"
    status_details: Optional[str] = None
    expires_at: Optional[int] = None


class ListFilesResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[FileObject]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class DeleteFileResponse(BaseModel):
    id: str
    object: Literal["file"] = "file"
    deleted: bool


# --- Audio schemas --------------------------------------------------------


class TranscriptionResponse(BaseModel):
    text: str


class TranslationResponse(BaseModel):
    text: str


# --- Images schemas -------------------------------------------------------


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None
    url: Optional[str] = None


class ImagesResponse(BaseModel):
    created: int
    data: List[ImageData]


# --- Videos schemas -------------------------------------------------------


class VideoObject(BaseModel):
    id: str
    object: Literal["video"] = "video"
    created_at: int
    status: Literal["queued", "in_progress", "completed", "failed"] = "queued"
    model: str = ""
    prompt: str = ""
    seconds: Optional[int] = None
    size: Optional[str] = None


class DeleteVideoResponse(BaseModel):
    id: str
    object: Literal["video"] = "video"
    deleted: bool


# --- PDF schemas ----------------------------------------------------------


class PdfObject(BaseModel):
    id: str
    object: Literal["pdf"] = "pdf"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: Literal["uploaded", "processed", "error"] = "uploaded"
    status_details: Optional[str] = None
    num_pages: Optional[int] = None


class ListPdfResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[PdfObject]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class DeletePdfResponse(BaseModel):
    id: str
    object: Literal["pdf"] = "pdf"
    deleted: bool


class PdfPageExtraction(BaseModel):
    page: int = Field(..., description="1-indexed page number.")
    text: str = Field("", description="Extracted text for this page.")
    tables: List[BoundingBoxDetection] = Field(default_factory=list, description="Detected tables.")
    charts: List[BoundingBoxDetection] = Field(default_factory=list, description="Detected charts.")
    infographics: List[BoundingBoxDetection] = Field(default_factory=list, description="Detected infographics.")
    elapsed_seconds: float = Field(0.0, description="Processing time for this page in seconds.")


class PdfExtractionResponse(BaseModel):
    object: Literal["pdf.extraction"] = "pdf.extraction"
    pdf_id: Optional[str] = Field(None, description="ID of a previously uploaded PDF, if used.")
    model: str = Field("", description="Model used for extraction.")
    filename: str = Field("", description="Original filename.")
    num_pages: int = Field(0, description="Total number of pages processed.")
    elapsed_seconds: float = Field(0.0, description="Total processing time in seconds.")
    pages: List[PdfPageExtraction] = Field(default_factory=list, description="Per-page extraction results.")


class PdfParseResponse(BaseModel):
    object: Literal["pdf.parse"] = "pdf.parse"
    pdf_id: Optional[str] = Field(None, description="ID of a previously uploaded PDF, if used.")
    model: str = Field("", description="Model used for parsing.")
    filename: str = Field("", description="Original filename.")
    text: str = Field("", description="Structured markdown / HTML output.")


# ---------------------------------------------------------------------------
# Request / response schemas — custom endpoints
# ---------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Query string for retrieval.")
    top_k: int = Field(10, description="Number of results to return.")
    lancedb_uri: Optional[str] = Field(None, description="Override LanceDB URI.")
    lancedb_table: Optional[str] = Field(None, description="Override LanceDB table name.")
    hybrid: bool = Field(False, description="Use hybrid search (vector + FTS).")


class RetrieveResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]


class IngestRequest(BaseModel):
    run_mode: str = Field("inprocess", description="Ingestor run mode (inprocess, batch, fused, remote).")


class BoundingBoxDetection(BaseModel):
    bbox_xyxy_norm: List[float] = Field(..., description="Normalized [x1, y1, x2, y2] bounding box.")
    label_name: str = Field("", description="Detection class (e.g. table, chart, cell, row, column).")
    score: Optional[float] = Field(None, description="Confidence score.")
    text: str = Field("", description="OCR / parsed text inside the region.")


class IngestPageResult(BaseModel):
    page_number: int = Field(..., description="1-indexed page number.")
    filename: str = Field("", description="Original uploaded filename.")
    text: str = Field("", description="Extracted page text.")
    elapsed_seconds: float = Field(0.0, description="Processing time for this page in seconds.")
    tables: List[BoundingBoxDetection] = Field(default_factory=list, description="Detected tables.")
    charts: List[BoundingBoxDetection] = Field(default_factory=list, description="Detected charts.")
    infographics: List[BoundingBoxDetection] = Field(default_factory=list, description="Detected infographics.")


class IngestResponse(BaseModel):
    num_documents: int = Field(..., description="Number of documents submitted.")
    elapsed_seconds: float = Field(0.0, description="Total server-side processing time in seconds.")
    pages: List[IngestPageResult] = Field(default_factory=list, description="Per-page results.")


def _parse_detections(raw: Any) -> list["BoundingBoxDetection"]:
    """Convert a list of detection dicts into BoundingBoxDetection models."""
    if not isinstance(raw, list):
        return []
    out: list[BoundingBoxDetection] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_xyxy_norm")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        out.append(
            BoundingBoxDetection(
                bbox_xyxy_norm=[float(c) for c in bbox],
                label_name=str(item.get("label_name", "")),
                score=float(item["score"]) if item.get("score") is not None else None,
                text=str(item.get("text", "")),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Pipeline stage configuration models (/api/v1/ingest)
# ---------------------------------------------------------------------------


class ExtractStageConfig(BaseModel):
    enabled: bool = Field(True, description="Enable extraction stage.")
    extract_text: bool = Field(True, description="Extract text from pages.")
    extract_images: bool = Field(True, description="Extract images.")
    extract_tables: bool = Field(True, description="Detect and extract tables.")
    extract_charts: bool = Field(True, description="Detect and extract charts.")
    extract_infographics: bool = Field(True, description="Detect and extract infographics.")
    method: str = Field("pdfium", description="Extraction method: pdfium, pdfium_hybrid, nemotron_parse.")
    dpi: int = Field(200, description="Render DPI for detection models.")
    use_table_structure: bool = Field(False, description="Use table structure detection model.")
    table_output_format: Optional[str] = Field(None, description="pseudo_markdown or markdown.")
    use_graphic_elements: bool = Field(False, description="Use graphic elements detection model.")
    inference_batch_size: int = Field(8, description="Inference batch size for detection.")
    page_elements_invoke_url: Optional[str] = Field(None, description="NIM endpoint for page element detection.")
    ocr_invoke_url: Optional[str] = Field(None, description="NIM endpoint for OCR.")
    table_structure_invoke_url: Optional[str] = Field(None, description="NIM endpoint for table structure.")
    graphic_elements_invoke_url: Optional[str] = Field(None, description="NIM endpoint for graphic elements.")
    api_key: Optional[str] = Field(None, description="API key for NIM endpoints.")


class DedupStageConfig(BaseModel):
    enabled: bool = Field(False, description="Enable image deduplication.")
    content_hash: bool = Field(True, description="Deduplicate by content hash.")
    bbox_iou: bool = Field(True, description="Deduplicate by bounding box IoU.")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold for dedup.")


class CaptionStageConfig(BaseModel):
    enabled: bool = Field(False, description="Enable VLM captioning.")
    endpoint_url: Optional[str] = Field(None, description="Remote VLM endpoint URL.")
    model_name: str = Field("nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", description="Caption model name.")
    api_key: Optional[str] = Field(None, description="API key for remote captioning endpoint.")
    prompt: str = Field("Caption the content of this image:", description="Captioning prompt.")
    temperature: float = Field(1.0, description="Sampling temperature.")
    batch_size: int = Field(8, description="Captioning batch size.")


class SplitStageConfig(BaseModel):
    enabled: bool = Field(False, description="Enable text chunking.")
    max_tokens: int = Field(1024, description="Maximum tokens per chunk.")
    overlap_tokens: int = Field(0, description="Token overlap between chunks.")
    tokenizer_model_id: Optional[str] = Field(None, description="HuggingFace tokenizer model ID.")


class EmbedStageConfig(BaseModel):
    enabled: bool = Field(True, description="Enable text/image embedding.")
    model_name: Optional[str] = Field(None, description="Embedding model name (defaults to server config).")
    embedding_endpoint: Optional[str] = Field(None, description="Remote NIM embedding endpoint.")
    api_key: Optional[str] = Field(None, description="API key for remote embedding endpoint.")
    embed_modality: str = Field("text", description="Modality: text, image, or text_image.")
    embed_granularity: Literal["element", "page"] = Field(
        "element", description="Granularity: element (per-element) or page (per-page)."
    )
    inference_batch_size: int = Field(32, description="Embedding inference batch size.")


class VdbUploadStageConfig(BaseModel):
    enabled: bool = Field(True, description="Enable LanceDB upload.")
    lancedb_uri: Optional[str] = Field(None, description="LanceDB URI (defaults to server config).")
    table_name: Optional[str] = Field(None, description="LanceDB table name (defaults to server config).")
    overwrite: bool = Field(True, description="Overwrite existing table data.")
    create_index: bool = Field(True, description="Create vector index after upload.")
    hybrid: bool = Field(False, description="Create FTS index for hybrid search.")


class IngestPipelineConfig(BaseModel):
    extract: Optional[ExtractStageConfig] = Field(default_factory=ExtractStageConfig)
    dedup: Optional[DedupStageConfig] = None
    caption: Optional[CaptionStageConfig] = None
    split: Optional[SplitStageConfig] = None
    embed: Optional[EmbedStageConfig] = Field(default_factory=EmbedStageConfig)
    vdb_upload: Optional[VdbUploadStageConfig] = Field(default_factory=VdbUploadStageConfig)


class IngestJobResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier.")
    status: str = Field(..., description="Job status.")
    num_documents: int = Field(..., description="Number of documents submitted.")
    created_at: int = Field(..., description="Unix timestamp of job creation.")


class IngestJobProgress(BaseModel):
    documents_total: int = 0
    documents_completed: int = 0
    current_stage: Optional[str] = None
    elapsed_seconds: float = 0.0


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: str
    num_documents: int
    created_at: int
    progress: Optional[IngestJobProgress] = None
    error: Optional[str] = None


class IngestJobResultResponse(BaseModel):
    job_id: str
    status: str
    num_documents: int
    elapsed_seconds: float = 0.0
    records: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted records.")


# ---------------------------------------------------------------------------
# Pipeline graph builder
# ---------------------------------------------------------------------------


def _build_serve_graph(
    config: IngestPipelineConfig,
    *,
    default_embed_model: str = "nvidia/llama-nemotron-embed-1b-v2",
    default_embed_endpoint: Optional[str] = None,
    default_embed_api_key: str = "",
    default_lancedb_uri: str = "lancedb",
    default_lancedb_table: str = "nv-ingest",
):
    """Build a pipeline Graph from API stage configuration.

    Returns a linear graph:
    MultiTypeExtract -> [Dedup] -> [Caption] -> [Reshape] -> [Split] -> [Embed] -> [VDBUpload]
    """
    from functools import partial

    from nemo_retriever.graph import Graph, UDFOperator
    from nemo_retriever.graph.content_transforms import (
        _CONTENT_COLUMNS,
        collapse_content_to_page_rows,
        explode_content_to_rows,
    )
    from nemo_retriever.graph.multi_type_extract_operator import MultiTypeExtractOperator
    from nemo_retriever.params import (
        CaptionParams,
        EmbedParams,
        ExtractParams,
        TextChunkParams,
    )

    extract_cfg = config.extract or ExtractStageConfig()
    extract_params = ExtractParams(
        extract_text=extract_cfg.extract_text,
        extract_images=extract_cfg.extract_images,
        extract_tables=extract_cfg.extract_tables,
        extract_charts=extract_cfg.extract_charts,
        extract_infographics=extract_cfg.extract_infographics,
        method=extract_cfg.method,
        dpi=extract_cfg.dpi,
        use_table_structure=extract_cfg.use_table_structure,
        table_output_format=extract_cfg.table_output_format,
        use_graphic_elements=extract_cfg.use_graphic_elements,
        inference_batch_size=extract_cfg.inference_batch_size,
        page_elements_invoke_url=extract_cfg.page_elements_invoke_url,
        ocr_invoke_url=extract_cfg.ocr_invoke_url,
        table_structure_invoke_url=extract_cfg.table_structure_invoke_url,
        graphic_elements_invoke_url=extract_cfg.graphic_elements_invoke_url,
        api_key=extract_cfg.api_key,
    )

    graph = Graph() >> MultiTypeExtractOperator(
        extraction_mode="auto",
        extract_params=extract_params,
    )

    dedup_cfg = config.dedup
    if dedup_cfg is not None and dedup_cfg.enabled:
        from nemo_retriever.dedup.dedup import dedup_images

        graph = graph >> UDFOperator(
            partial(
                dedup_images,
                content_hash=dedup_cfg.content_hash,
                bbox_iou=dedup_cfg.bbox_iou,
                iou_threshold=dedup_cfg.iou_threshold,
            ),
            name="Dedup",
        )

    caption_params = None
    caption_cfg = config.caption
    if caption_cfg is not None and caption_cfg.enabled:
        caption_params = CaptionParams(
            endpoint_url=caption_cfg.endpoint_url,
            model_name=caption_cfg.model_name,
            api_key=caption_cfg.api_key,
            prompt=caption_cfg.prompt,
            temperature=caption_cfg.temperature,
            batch_size=caption_cfg.batch_size,
        )
        from nemo_retriever.caption.caption import CaptionActor

        graph = graph >> CaptionActor(caption_params)

    embed_params = None
    embed_cfg = config.embed
    if embed_cfg is not None and embed_cfg.enabled:
        embed_params = EmbedParams(
            model_name=embed_cfg.model_name or default_embed_model,
            embedding_endpoint=embed_cfg.embedding_endpoint or default_embed_endpoint,
            api_key=embed_cfg.api_key or default_embed_api_key,
            embed_modality=embed_cfg.embed_modality,
            embed_granularity=embed_cfg.embed_granularity,
            inference_batch_size=embed_cfg.inference_batch_size,
        )

        content_columns = (_CONTENT_COLUMNS + ("images",)) if caption_params is not None else _CONTENT_COLUMNS
        if embed_params.embed_granularity == "page":
            graph = graph >> UDFOperator(
                partial(
                    collapse_content_to_page_rows,
                    modality=embed_params.embed_modality,
                    content_columns=content_columns,
                ),
                name="CollapseContentToPageRows",
            )
        else:
            graph = graph >> UDFOperator(
                partial(
                    explode_content_to_rows,
                    modality=embed_params.embed_modality,
                    text_elements_modality=(embed_params.text_elements_modality or embed_params.embed_modality),
                    structured_elements_modality=(
                        embed_params.structured_elements_modality or embed_params.embed_modality
                    ),
                    content_columns=content_columns,
                ),
                name="ExplodeContentToRows",
            )

    split_cfg = config.split
    if split_cfg is not None and split_cfg.enabled:
        split_params = TextChunkParams(
            max_tokens=split_cfg.max_tokens,
            overlap_tokens=split_cfg.overlap_tokens,
            tokenizer_model_id=split_cfg.tokenizer_model_id,
        )
        from nemo_retriever.txt.ray_data import TextChunkActor

        graph = graph >> TextChunkActor(split_params)

    if embed_params is not None:
        from nemo_retriever.text_embed.operators import _BatchEmbedActor

        graph = graph >> _BatchEmbedActor(params=embed_params)

    vdb_cfg = config.vdb_upload
    if vdb_cfg is not None and vdb_cfg.enabled:
        from nemo_retriever.ingest_modes.inprocess import upload_embeddings_to_lancedb_inprocess

        graph = graph >> UDFOperator(
            partial(
                upload_embeddings_to_lancedb_inprocess,
                lancedb_uri=vdb_cfg.lancedb_uri or default_lancedb_uri,
                table_name=vdb_cfg.table_name or default_lancedb_table,
                overwrite=vdb_cfg.overwrite,
                create_index=vdb_cfg.create_index,
                hybrid=vdb_cfg.hybrid,
            ),
            name="VDBUpload",
        )

    return graph


def _load_files_to_df(paths: list[str]):
    """Read files as raw bytes into a DataFrame with ``bytes`` and ``path`` columns."""
    import pandas as pd

    rows = []
    for p in paths:
        fp = Path(p)
        if fp.is_file():
            rows.append({"bytes": fp.read_bytes(), "path": str(fp.resolve())})
    if not rows:
        return pd.DataFrame(columns=["bytes", "path"])
    return pd.DataFrame(rows)


_BINARY_COLUMNS = frozenset(
    {
        "bytes",
        "page_image",
        "_image_b64",
        "text_embeddings_1b_v2",
        "text_embeddings_1b_v2_dim",
        "text_embeddings_1b_v2_has_embedding",
    }
)


def _serialize_df_records(df) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records, omitting large binary columns."""
    import numpy as np

    records = df.to_dict(orient="records")
    safe: list[dict[str, Any]] = []
    for record in records:
        cleaned: dict[str, Any] = {}
        for k, v in record.items():
            if k in _BINARY_COLUMNS:
                continue
            if isinstance(v, (np.integer,)):
                cleaned[k] = int(v)
            elif isinstance(v, (np.floating,)):
                cleaned[k] = float(v)
            elif isinstance(v, np.ndarray):
                continue
            elif isinstance(v, bytes):
                continue
            else:
                try:
                    json.dumps(v)
                    cleaned[k] = v
                except (TypeError, ValueError):
                    cleaned[k] = str(v)
        safe.append(cleaned)
    return safe


def _ndjson_line(obj: dict) -> bytes:
    """Encode a dict as a single NDJSON line (UTF-8 bytes with trailing newline)."""
    return (json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# PDF streaming helper (carried forward)
# ---------------------------------------------------------------------------


def _pdf_page_text_stream(pdf_bytes: bytes):
    """Open PDF with pypdfium2 and yield (page_number_1based, text) per page."""
    import pypdfium2 as pdfium

    try:
        doc = pdfium.PdfDocument(pdf_bytes)
    except Exception:
        doc = pdfium.PdfDocument(BytesIO(pdf_bytes))
    try:
        for i in range(len(doc)):
            page = doc.get_page(i)
            tp = page.get_textpage()
            text = tp.get_text_bounded()
            if text is None:
                text = ""
            yield (i + 1, text)
    finally:
        try:
            doc.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
def health() -> dict:
    """Health check for Kubernetes liveness / readiness probes."""
    return {"status": "ok"}


@app.get("/version", tags=["health"])
def version() -> dict:
    """Return the running application version and build metadata."""
    try:
        from nemo_retriever.version import get_version_info

        return get_version_info()
    except Exception:
        return {"version": "unknown", "git_sha": "unknown", "build_date": "unknown", "full_version": "unknown"}


@app.post("/stream-pdf", tags=["pdf"])
def stream_pdf(
    file: UploadFile = File(..., description="PDF file to extract text from (page-by-page stream)."),
):
    """Stream back pdfium get_text (per page) as NDJSON."""
    if file.content_type and file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(400, detail="Expected a PDF file (application/pdf).")
    try:
        import pypdfium2 as pdfium  # noqa: F401
    except ImportError as e:
        raise HTTPException(503, detail="PDF processing (pypdfium2) is not available.") from e

    raw = file.file.read()
    if len(raw) > STREAM_PDF_MAX_BYTES:
        raise HTTPException(413, detail=f"PDF larger than {STREAM_PDF_MAX_BYTES // (1024 * 1024)} MiB not allowed.")

    def ndjson_stream():
        for page_num, text in _pdf_page_text_stream(raw):
            line = json.dumps({"page": page_num, "text": text}, ensure_ascii=False) + "\n"
            yield line.encode("utf-8")

    return StreamingResponse(
        ndjson_stream(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "inline; filename=pages.ndjson"},
    )


# ---------------------------------------------------------------------------
# Ray Serve deployment
# ---------------------------------------------------------------------------

if serve is not None:

    @serve.deployment(
        name="retriever_api",
        num_replicas=1,
        ray_actor_options={"num_cpus": 1},
    )
    @serve.ingress(app)
    class RetrieverAPIDeployment:
        """Ray Serve deployment that serves the NeMo Retriever FastAPI app."""

        def __init__(
            self,
            lancedb_uri: str = "lancedb",
            lancedb_table: str = "nv-ingest",
            embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2",
            embedding_endpoint: Optional[str] = None,
            embedding_api_key: str = "",
            top_k: int = 10,
            reranker: bool = False,
            reranker_endpoint: Optional[str] = None,
        ):
            self._lancedb_uri = lancedb_uri
            self._lancedb_table = lancedb_table
            self._embedding_model = embedding_model
            self._embedding_endpoint = embedding_endpoint
            self._embedding_api_key = embedding_api_key
            self._top_k = top_k
            self._reranker = reranker
            self._reranker_endpoint = reranker_endpoint

            self._retriever: Any = None
            self._embed_actor: Any = None

            self._files_dir = Path(tempfile.mkdtemp(prefix="retriever_files_"))
            self._file_store: dict[str, FileObject] = {}
            self._video_store: dict[str, VideoObject] = {}

            self._pdf_dir = Path(tempfile.mkdtemp(prefix="retriever_pdf_"))
            self._pdf_store: dict[str, PdfObject] = {}

            self._ingest_dir = Path(tempfile.mkdtemp(prefix="retriever_ingest_"))
            self._jobs: dict[str, dict[str, Any]] = {}
            self._job_lock = threading.Lock()

        def _create_job(self, num_documents: int, config: dict) -> str:
            job_id = f"job-{uuid.uuid4().hex[:16]}"
            with self._job_lock:
                self._jobs[job_id] = {
                    "job_id": job_id,
                    "status": "queued",
                    "num_documents": num_documents,
                    "created_at": time.time(),
                    "started_at": None,
                    "completed_at": None,
                    "error": None,
                    "results": None,
                    "temp_dir": None,
                    "config": config,
                    "current_stage": None,
                    "documents_completed": 0,
                }
            return job_id

        def _run_ingest_job(self, job_id: str, file_paths: list[str], config: IngestPipelineConfig) -> None:
            """Background worker: build the pipeline graph, execute it, store results."""
            with self._job_lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job["status"] = "running"
                job["started_at"] = time.time()

            try:
                graph = _build_serve_graph(
                    config,
                    default_embed_model=self._embedding_model,
                    default_embed_endpoint=self._embedding_endpoint,
                    default_embed_api_key=self._embedding_api_key,
                    default_lancedb_uri=self._lancedb_uri,
                    default_lancedb_table=self._lancedb_table,
                )

                df = _load_files_to_df(file_paths)
                if df.empty:
                    raise ValueError("No valid files to process.")

                results = graph.execute(df)
                result_df = results[0] if results else df

                import pandas as pd

                if isinstance(result_df, pd.DataFrame):
                    records = _serialize_df_records(result_df)
                elif isinstance(result_df, list):
                    records = result_df
                else:
                    records = []

                with self._job_lock:
                    job = self._jobs[job_id]
                    job["status"] = "completed"
                    job["completed_at"] = time.time()
                    job["results"] = records
                    job["documents_completed"] = job["num_documents"]

            except Exception as exc:
                logger.exception("Ingest job %s failed", job_id)
                with self._job_lock:
                    job = self._jobs[job_id]
                    job["status"] = "failed"
                    job["completed_at"] = time.time()
                    job["error"] = str(exc)

        def _get_retriever(self):
            if self._retriever is None:
                from nemo_retriever.retriever import Retriever

                self._retriever = Retriever(
                    lancedb_uri=self._lancedb_uri,
                    lancedb_table=self._lancedb_table,
                    embedder=self._embedding_model,
                    embedding_endpoint=self._embedding_endpoint,
                    embedding_api_key=self._embedding_api_key,
                    top_k=self._top_k,
                    reranker=self._reranker,
                    reranker_endpoint=self._reranker_endpoint,
                )
            return self._retriever

        def _get_embed_actor(self):
            if self._embed_actor is None:
                from nemo_retriever.text_embed import TextEmbedActor

                self._embed_actor = TextEmbedActor(model_name=self._embedding_model)
            return self._embed_actor

        @app.post("/ingest", response_model=IngestResponse, tags=["ingest"])
        async def ingest(
            self,
            files: List[UploadFile] = File(..., description="Documents to ingest."),
            run_mode: str = "inprocess",
        ) -> IngestResponse:
            """Accept file uploads, run the ingestion pipeline, and return results."""
            import pandas as pd

            from nemo_retriever.pdf.extract import pdf_extraction
            from nemo_retriever.pdf.split import pdf_path_to_pages_df

            t_start = time.perf_counter()

            with tempfile.TemporaryDirectory(prefix="retriever_ingest_") as tmp_dir:
                saved_paths: list[str] = []
                for upload in files:
                    dest = Path(tmp_dir) / (upload.filename or "upload")
                    dest.write_bytes(await upload.read())
                    saved_paths.append(str(dest))

                pages: list[IngestPageResult] = []
                try:
                    for doc_path in saved_paths:
                        pages_df = pdf_path_to_pages_df(doc_path)
                        for row_idx in range(len(pages_df)):
                            single = pages_df.iloc[[row_idx]]
                            t_page = time.perf_counter()
                            extracted = pdf_extraction(single, extract_text=True)
                            page_elapsed = time.perf_counter() - t_page

                            if isinstance(extracted, pd.DataFrame) and not extracted.empty:
                                r = extracted.iloc[0]
                            elif isinstance(extracted, list) and extracted:
                                r = extracted[0]
                            else:
                                continue

                            row = r if isinstance(r, dict) else r.to_dict()
                            pages.append(
                                IngestPageResult(
                                    page_number=int(row.get("page_number", 0)),
                                    filename=Path(doc_path).name,
                                    text=str(row.get("text", "")),
                                    elapsed_seconds=round(page_elapsed, 4),
                                    tables=_parse_detections(row.get("table")),
                                    charts=_parse_detections(row.get("chart")),
                                    infographics=_parse_detections(row.get("infographic")),
                                )
                            )
                except Exception as exc:
                    logger.exception("Ingestion failed")
                    raise HTTPException(500, detail=f"Ingestion failed: {exc}") from exc

            total_elapsed = round(time.perf_counter() - t_start, 4)
            return IngestResponse(
                num_documents=len(saved_paths),
                elapsed_seconds=total_elapsed,
                pages=pages,
            )

        # ---------------------------------------------------------------
        # OpenAI-compatible: POST /v1/embeddings
        # ---------------------------------------------------------------

        @app.post("/v1/embeddings", response_model=CreateEmbeddingResponse, tags=["openai"])
        async def create_embedding(self, request: CreateEmbeddingRequest) -> CreateEmbeddingResponse:
            """Create embeddings (OpenAI-compatible)."""
            import pandas as pd

            from nemo_retriever.text_embed import embed_text_1b_v2

            texts = [request.input] if isinstance(request.input, str) else list(request.input)
            if not texts:
                raise HTTPException(400, detail="input must not be empty.")

            actor = self._get_embed_actor()
            df = pd.DataFrame({"text": texts})

            try:
                result_df = embed_text_1b_v2(
                    df,
                    model=actor._model,
                    model_name=request.model,
                    input_type="passage",
                )
            except Exception as exc:
                logger.exception("Embedding failed")
                raise HTTPException(500, detail=f"Embedding failed: {exc}") from exc

            data: list[EmbeddingObject] = []
            col = "text_embeddings_1b_v2"
            for idx, row in result_df.iterrows():
                payload = row.get(col, {})
                emb = payload.get("embedding") if isinstance(payload, dict) else None
                data.append(EmbeddingObject(index=int(idx), embedding=emb or []))

            prompt_tokens = sum(len(t.split()) for t in texts)
            return CreateEmbeddingResponse(
                data=data,
                model=request.model,
                usage=EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
            )

        # ---------------------------------------------------------------
        # OpenAI-compatible: GET /v1/models
        # ---------------------------------------------------------------

        @app.get("/v1/models", response_model=ListModelsResponse, tags=["openai"])
        async def list_models(self) -> ListModelsResponse:
            """List available models (OpenAI-compatible)."""
            from nemo_retriever.utils.hf_model_registry import HF_MODEL_REGISTRY

            models: list[ModelObject] = []
            for model_id, info in HF_MODEL_REGISTRY.items():
                owner = model_id.split("/")[0] if "/" in model_id else "unknown"
                models.append(
                    ModelObject(
                        id=model_id,
                        created=info.created_at,
                        owned_by=owner,
                        url=info.url,
                    )
                )
            return ListModelsResponse(data=models)

        # ---------------------------------------------------------------
        # OpenAI-compatible: /v1/files
        # ---------------------------------------------------------------

        @app.post("/v1/files", response_model=FileObject, tags=["openai"])
        async def upload_file(
            self,
            file: UploadFile = File(..., description="The file to upload."),
            purpose: str = "user_data",
        ) -> FileObject:
            """Upload a file (OpenAI-compatible)."""
            content = await file.read()
            file_id = f"file-{uuid.uuid4().hex[:24]}"
            dest = self._files_dir / file_id
            dest.write_bytes(content)

            obj = FileObject(
                id=file_id,
                bytes=len(content),
                created_at=int(time.time()),
                filename=file.filename or "upload",
                purpose=purpose,
            )
            self._file_store[file_id] = obj
            return obj

        @app.get("/v1/files", response_model=ListFilesResponse, tags=["openai"])
        async def list_files(
            self,
            purpose: Optional[str] = None,
            limit: int = 10000,
            order: str = "desc",
            after: Optional[str] = None,
        ) -> ListFilesResponse:
            """List uploaded files (OpenAI-compatible)."""
            files = list(self._file_store.values())
            if purpose:
                files = [f for f in files if f.purpose == purpose]

            reverse = order != "asc"
            files.sort(key=lambda f: f.created_at, reverse=reverse)

            if after:
                skip = next((i for i, f in enumerate(files) if f.id == after), -1)
                if skip >= 0:
                    files = files[skip + 1 :]

            has_more = len(files) > limit
            files = files[:limit]

            return ListFilesResponse(
                data=files,
                first_id=files[0].id if files else None,
                last_id=files[-1].id if files else None,
                has_more=has_more,
            )

        @app.get("/v1/files/{file_id}", response_model=FileObject, tags=["openai"])
        async def retrieve_file(self, file_id: str) -> FileObject:
            """Retrieve file metadata (OpenAI-compatible)."""
            obj = self._file_store.get(file_id)
            if obj is None:
                raise HTTPException(404, detail=f"No file with id '{file_id}'.")
            return obj

        @app.delete("/v1/files/{file_id}", response_model=DeleteFileResponse, tags=["openai"])
        async def delete_file(self, file_id: str) -> DeleteFileResponse:
            """Delete a file (OpenAI-compatible)."""
            obj = self._file_store.pop(file_id, None)
            if obj is None:
                raise HTTPException(404, detail=f"No file with id '{file_id}'.")
            disk_path = self._files_dir / file_id
            if disk_path.exists():
                os.remove(disk_path)
            return DeleteFileResponse(id=file_id, deleted=True)

        @app.get("/v1/files/{file_id}/content", tags=["openai"])
        async def retrieve_file_content(self, file_id: str):
            """Retrieve file content (OpenAI-compatible)."""
            obj = self._file_store.get(file_id)
            if obj is None:
                raise HTTPException(404, detail=f"No file with id '{file_id}'.")
            disk_path = self._files_dir / file_id
            if not disk_path.exists():
                raise HTTPException(404, detail="File content not found on disk.")
            content = disk_path.read_bytes()
            return StreamingResponse(
                BytesIO(content),
                media_type="application/octet-stream",
                headers={"Content-Disposition": f'attachment; filename="{obj.filename}"'},
            )

        # ---------------------------------------------------------------
        # OpenAI-compatible: /v1/audio (stubs)
        # ---------------------------------------------------------------

        @app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse, tags=["openai"])
        async def create_transcription(
            self,
            file: UploadFile = File(..., description="Audio file to transcribe."),
            model: str = "whisper-1",
            language: Optional[str] = None,
            prompt: Optional[str] = None,
            response_format: Optional[str] = "json",
            temperature: Optional[float] = None,
        ) -> TranscriptionResponse:
            """Transcribe audio to text (OpenAI-compatible, stub)."""
            _ = await file.read()
            return TranscriptionResponse(
                text="[Transcription backend not configured. Audio received.]",
            )

        @app.post("/v1/audio/translations", response_model=TranslationResponse, tags=["openai"])
        async def create_translation(
            self,
            file: UploadFile = File(..., description="Audio file to translate."),
            model: str = "whisper-1",
            prompt: Optional[str] = None,
            response_format: Optional[str] = "json",
            temperature: Optional[float] = None,
        ) -> TranslationResponse:
            """Translate audio to English text (OpenAI-compatible, stub)."""
            _ = await file.read()
            return TranslationResponse(
                text="[Translation backend not configured. Audio received.]",
            )

        @app.post("/v1/audio/speech", tags=["openai"])
        async def create_speech(
            self,
            request: dict = {},
        ):
            """Generate speech from text (OpenAI-compatible, stub).

            Expects JSON body with ``model``, ``input``, ``voice``.
            Returns a placeholder WAV header.
            """
            raise HTTPException(
                501,
                detail="Speech synthesis backend not configured.",
            )

        # ---------------------------------------------------------------
        # OpenAI-compatible: /v1/images (stubs)
        # ---------------------------------------------------------------

        @app.post("/v1/images/generations", response_model=ImagesResponse, tags=["openai"])
        async def create_image(
            self,
            request: dict = {},
        ) -> ImagesResponse:
            """Generate images from a text prompt (OpenAI-compatible, stub).

            Expects JSON body with ``prompt``, ``model``, ``n``, ``size``.
            """
            return ImagesResponse(
                created=int(time.time()),
                data=[ImageData(b64_json=None, revised_prompt=None, url=None)],
            )

        @app.post("/v1/images/edits", response_model=ImagesResponse, tags=["openai"])
        async def create_image_edit(
            self,
            image: UploadFile = File(..., description="Image to edit."),
            prompt: str = "",
            model: Optional[str] = None,
            n: int = 1,
            size: Optional[str] = None,
        ) -> ImagesResponse:
            """Edit an image (OpenAI-compatible, stub)."""
            _ = await image.read()
            return ImagesResponse(
                created=int(time.time()),
                data=[ImageData(b64_json=None, revised_prompt=None, url=None)],
            )

        @app.post("/v1/images/variations", response_model=ImagesResponse, tags=["openai"])
        async def create_image_variation(
            self,
            image: UploadFile = File(..., description="Image to create variations of."),
            model: Optional[str] = None,
            n: int = 1,
            size: Optional[str] = None,
        ) -> ImagesResponse:
            """Create image variations (OpenAI-compatible, stub)."""
            _ = await image.read()
            return ImagesResponse(
                created=int(time.time()),
                data=[ImageData(b64_json=None, revised_prompt=None, url=None)],
            )

        # ---------------------------------------------------------------
        # OpenAI-compatible: /v1/videos (stubs)
        # ---------------------------------------------------------------

        @app.post("/v1/videos", response_model=VideoObject, tags=["openai"])
        async def create_video(
            self,
            request: dict = {},
        ) -> VideoObject:
            """Create a video generation job (OpenAI-compatible, stub).

            Expects JSON body with ``prompt``, ``model``, ``seconds``, ``size``.
            """
            video_id = f"video-{uuid.uuid4().hex[:24]}"
            obj = VideoObject(
                id=video_id,
                created_at=int(time.time()),
                status="failed",
                model=request.get("model", "sora-2"),
                prompt=request.get("prompt", ""),
                seconds=request.get("seconds"),
                size=request.get("size"),
            )
            self._video_store[video_id] = obj
            return obj

        @app.get("/v1/videos/{video_id}", response_model=VideoObject, tags=["openai"])
        async def retrieve_video(self, video_id: str) -> VideoObject:
            """Get video generation job status (OpenAI-compatible, stub)."""
            obj = self._video_store.get(video_id)
            if obj is None:
                raise HTTPException(404, detail=f"No video with id '{video_id}'.")
            return obj

        @app.delete("/v1/videos/{video_id}", response_model=DeleteVideoResponse, tags=["openai"])
        async def delete_video(self, video_id: str) -> DeleteVideoResponse:
            """Delete a video job (OpenAI-compatible, stub)."""
            obj = self._video_store.pop(video_id, None)
            if obj is None:
                raise HTTPException(404, detail=f"No video with id '{video_id}'.")
            return DeleteVideoResponse(id=video_id, deleted=True)

        @app.get("/v1/videos/{video_id}/content", tags=["openai"])
        async def retrieve_video_content(self, video_id: str):
            """Download generated video content (OpenAI-compatible, stub)."""
            obj = self._video_store.get(video_id)
            if obj is None:
                raise HTTPException(404, detail=f"No video with id '{video_id}'.")
            raise HTTPException(
                501,
                detail="Video generation backend not configured.",
            )

        # ---------------------------------------------------------------
        # PDF endpoints (/v1/pdf) — modeled after OpenAI /v1/files
        # ---------------------------------------------------------------

        @app.post("/v1/pdf", response_model=PdfObject, tags=["pdf"])
        async def upload_pdf(
            self,
            file: UploadFile = File(..., description="PDF file to upload."),
            purpose: str = "extraction",
        ) -> PdfObject:
            """Upload a PDF for later processing."""
            content = await file.read()
            if not content:
                raise HTTPException(400, detail="Uploaded file is empty.")

            pdf_id = f"pdf-{uuid.uuid4().hex[:24]}"
            dest = self._pdf_dir / pdf_id
            dest.write_bytes(content)

            num_pages: int | None = None
            try:
                import pypdfium2 as pdfium

                doc = pdfium.PdfDocument(content)
                num_pages = len(doc)
                doc.close()
            except Exception:
                pass

            obj = PdfObject(
                id=pdf_id,
                bytes=len(content),
                created_at=int(time.time()),
                filename=file.filename or "upload.pdf",
                purpose=purpose,
                status="uploaded",
                num_pages=num_pages,
            )
            self._pdf_store[pdf_id] = obj
            return obj

        @app.get("/v1/pdf", response_model=ListPdfResponse, tags=["pdf"])
        async def list_pdfs(
            self,
            purpose: Optional[str] = None,
            limit: int = 10000,
            order: str = "desc",
            after: Optional[str] = None,
        ) -> ListPdfResponse:
            """List uploaded PDFs."""
            pdfs = list(self._pdf_store.values())
            if purpose:
                pdfs = [p for p in pdfs if p.purpose == purpose]

            reverse = order != "asc"
            pdfs.sort(key=lambda p: p.created_at, reverse=reverse)

            if after:
                skip = next((i for i, p in enumerate(pdfs) if p.id == after), -1)
                if skip >= 0:
                    pdfs = pdfs[skip + 1 :]

            has_more = len(pdfs) > limit
            pdfs = pdfs[:limit]

            return ListPdfResponse(
                data=pdfs,
                first_id=pdfs[0].id if pdfs else None,
                last_id=pdfs[-1].id if pdfs else None,
                has_more=has_more,
            )

        @app.get("/v1/pdf/{pdf_id}", response_model=PdfObject, tags=["pdf"])
        async def retrieve_pdf(self, pdf_id: str) -> PdfObject:
            """Retrieve PDF metadata."""
            obj = self._pdf_store.get(pdf_id)
            if obj is None:
                raise HTTPException(404, detail=f"No PDF with id '{pdf_id}'.")
            return obj

        @app.delete("/v1/pdf/{pdf_id}", response_model=DeletePdfResponse, tags=["pdf"])
        async def delete_pdf(self, pdf_id: str) -> DeletePdfResponse:
            """Delete a previously uploaded PDF."""
            obj = self._pdf_store.pop(pdf_id, None)
            if obj is None:
                raise HTTPException(404, detail=f"No PDF with id '{pdf_id}'.")
            disk_path = self._pdf_dir / pdf_id
            if disk_path.exists():
                os.remove(disk_path)
            return DeletePdfResponse(id=pdf_id, deleted=True)

        @app.get("/v1/pdf/{pdf_id}/content", tags=["pdf"])
        async def retrieve_pdf_content(self, pdf_id: str):
            """Download a previously uploaded PDF."""
            obj = self._pdf_store.get(pdf_id)
            if obj is None:
                raise HTTPException(404, detail=f"No PDF with id '{pdf_id}'.")
            disk_path = self._pdf_dir / pdf_id
            if not disk_path.exists():
                raise HTTPException(404, detail="PDF content not found on disk.")
            content = disk_path.read_bytes()
            return StreamingResponse(
                BytesIO(content),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{obj.filename}"'},
            )

        # ---------------------------------------------------------------
        # PDF processing — modeled after OpenAI /v1/audio/*
        # ---------------------------------------------------------------

        @app.post("/v1/pdf/extractions", response_model=PdfExtractionResponse, tags=["pdf"])
        async def create_pdf_extraction(
            self,
            file: Optional[UploadFile] = File(None, description="PDF file to extract (or use pdf_id)."),
            pdf_id: Optional[str] = None,
            model: str = "pdfium",
        ) -> PdfExtractionResponse:
            """Extract text and structure from a PDF.

            Mirrors ``POST /v1/audio/transcriptions``.  Supply either a
            ``file`` upload or a ``pdf_id`` referencing a previously uploaded
            PDF.
            """
            import pandas as pd

            from nemo_retriever.pdf.extract import pdf_extraction
            from nemo_retriever.pdf.split import pdf_path_to_pages_df

            pdf_bytes: bytes | None = None
            filename = "upload.pdf"
            resolved_pdf_id: str | None = pdf_id

            if pdf_id:
                obj = self._pdf_store.get(pdf_id)
                if obj is None:
                    raise HTTPException(404, detail=f"No PDF with id '{pdf_id}'.")
                disk_path = self._pdf_dir / pdf_id
                if not disk_path.exists():
                    raise HTTPException(404, detail="PDF content not found on disk.")
                pdf_bytes = disk_path.read_bytes()
                filename = obj.filename
            elif file is not None:
                pdf_bytes = await file.read()
                filename = file.filename or "upload.pdf"
            else:
                raise HTTPException(400, detail="Provide either a file upload or a pdf_id.")

            if not pdf_bytes:
                raise HTTPException(400, detail="PDF content is empty.")

            t_start = time.perf_counter()
            pages: list[PdfPageExtraction] = []

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            try:
                pages_df = pdf_path_to_pages_df(tmp_path)
                for row_idx in range(len(pages_df)):
                    single = pages_df.iloc[[row_idx]]
                    t_page = time.perf_counter()
                    extracted = pdf_extraction(single, extract_text=True)
                    page_elapsed = time.perf_counter() - t_page

                    if isinstance(extracted, pd.DataFrame) and not extracted.empty:
                        r = extracted.iloc[0]
                    elif isinstance(extracted, list) and extracted:
                        r = extracted[0]
                    else:
                        continue

                    row = r if isinstance(r, dict) else r.to_dict()
                    pages.append(
                        PdfPageExtraction(
                            page=int(row.get("page_number", row_idx + 1)),
                            text=str(row.get("text", "")),
                            tables=_parse_detections(row.get("table")),
                            charts=_parse_detections(row.get("chart")),
                            infographics=_parse_detections(row.get("infographic")),
                            elapsed_seconds=round(page_elapsed, 4),
                        )
                    )
            except Exception as exc:
                logger.exception("PDF extraction failed")
                raise HTTPException(500, detail=f"PDF extraction failed: {exc}") from exc
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            if pdf_id:
                obj = self._pdf_store.get(pdf_id)
                if obj is not None:
                    self._pdf_store[pdf_id] = obj.model_copy(update={"status": "processed"})

            total_elapsed = round(time.perf_counter() - t_start, 4)
            return PdfExtractionResponse(
                pdf_id=resolved_pdf_id,
                model=model,
                filename=filename,
                num_pages=len(pages),
                elapsed_seconds=total_elapsed,
                pages=pages,
            )

        @app.post("/v1/pdf/parse", response_model=PdfParseResponse, tags=["pdf"])
        async def create_pdf_parse(
            self,
            file: Optional[UploadFile] = File(None, description="PDF file to parse (or use pdf_id)."),
            pdf_id: Optional[str] = None,
            model: str = "nemotron-parse",
        ) -> PdfParseResponse:
            """Parse a PDF into structured markdown.

            Mirrors ``POST /v1/audio/translations``.  Supply either a
            ``file`` upload or a ``pdf_id``.
            """
            filename = "upload.pdf"
            if pdf_id:
                obj = self._pdf_store.get(pdf_id)
                if obj is None:
                    raise HTTPException(404, detail=f"No PDF with id '{pdf_id}'.")
                filename = obj.filename
            elif file is not None:
                _ = await file.read()
                filename = file.filename or "upload.pdf"
            else:
                raise HTTPException(400, detail="Provide either a file upload or a pdf_id.")

            return PdfParseResponse(
                pdf_id=pdf_id,
                model=model,
                filename=filename,
                text="[PDF parsing backend (Nemotron-Parse) not configured. PDF received.]",
            )

        @app.post("/v1/pdf/render", tags=["pdf"])
        async def create_pdf_render(
            self,
            request: dict = {},
        ):
            """Render a PDF from text or markdown input.

            Mirrors ``POST /v1/audio/speech``.
            Expects JSON body with ``input`` and optional ``model``.
            """
            raise HTTPException(
                501,
                detail="PDF rendering backend not configured.",
            )

        # ---------------------------------------------------------------
        # OpenAI-compatible: POST /v1/chat/completions (stub)
        # ---------------------------------------------------------------

        @app.post("/v1/chat/completions", response_model=None, tags=["openai"])
        async def chat_completions(self, request: ChatCompletionRequest):
            """Chat completion with RAG context (OpenAI-compatible, stub).

            The LLM generation backend is not yet wired. This endpoint
            retrieves relevant context from the vector store and returns a
            placeholder response in the correct OpenAI shape, including SSE
            streaming when ``stream: true``.
            """
            user_query = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    user_query = msg.content
                    break

            context_snippets: list[str] = []
            try:
                retriever = self._get_retriever()
                hits = retriever.query(user_query)
                for hit in hits[: self._top_k]:
                    text = hit.get("text", "")
                    if text:
                        context_snippets.append(str(text))
            except Exception:
                logger.debug("Retrieval unavailable for chat context", exc_info=True)

            if context_snippets:
                stub_text = (
                    f"[LLM backend not configured. Retrieved {len(context_snippets)} "
                    f"chunk(s) for context.]\n\n" + "\n---\n".join(context_snippets)
                )
            else:
                stub_text = "[LLM backend not configured. No context retrieved.]"

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            prompt_tokens = sum(len(m.content.split()) for m in request.messages)
            completion_tokens = len(stub_text.split())

            if request.stream:

                async def _sse_stream():
                    # First chunk: role
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[ChatChunkChoice(index=0, delta=ChatDelta(role="assistant"), finish_reason=None)],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                    # Content chunk
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[ChatChunkChoice(index=0, delta=ChatDelta(content=stub_text), finish_reason=None)],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                    # Final chunk: finish_reason
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[ChatChunkChoice(index=0, delta=ChatDelta(), finish_reason="stop")],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(_sse_stream(), media_type="text/event-stream")

            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=stub_text),
                        finish_reason="stop",
                    ),
                ],
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

        # ---------------------------------------------------------------
        # Custom: POST /retrieve
        # ---------------------------------------------------------------

        @app.post("/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
        async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
            """Query the vector store and return ranked results."""
            retriever = self._get_retriever()

            try:
                hits = retriever.query(
                    request.query,
                    lancedb_uri=request.lancedb_uri,
                    lancedb_table=request.lancedb_table,
                )
            except Exception as exc:
                logger.exception("Retrieval failed")
                raise HTTPException(500, detail=f"Retrieval failed: {exc}") from exc

            safe_hits: list[dict] = []
            for hit in hits[: request.top_k]:
                cleaned: dict[str, Any] = {}
                for k, v in hit.items():
                    if k == "vector":
                        continue
                    try:
                        json.dumps(v)
                        cleaned[k] = v
                    except (TypeError, ValueError):
                        cleaned[k] = str(v)
                safe_hits.append(cleaned)

            return RetrieveResponse(query=request.query, results=safe_hits)

        # ---------------------------------------------------------------
        # Pipeline: POST /api/v1/ingest
        # ---------------------------------------------------------------

        @app.post("/api/v1/ingest", response_model=IngestJobResponse, tags=["ingest"], status_code=202)
        async def ingest_pipeline(
            self,
            files: List[UploadFile] = File(..., description="Documents to ingest."),
            config: Optional[str] = Form(None, description="Pipeline config as JSON string."),
        ) -> IngestJobResponse:
            """Submit documents for async pipeline ingestion.

            Upload one or more files and optionally provide a JSON pipeline
            configuration via the ``config`` form field. The server builds a
            processing graph (extract -> dedup -> caption -> split -> embed ->
            vdb_upload) and executes it in the background.  Returns a job ID
            that can be polled via ``GET /api/v1/jobs/{job_id}``.
            """
            pipeline_config = IngestPipelineConfig()
            if config:
                try:
                    pipeline_config = IngestPipelineConfig(**json.loads(config))
                except (json.JSONDecodeError, Exception) as exc:
                    raise HTTPException(400, detail=f"Invalid pipeline config JSON: {exc}") from exc

            job_dir = self._ingest_dir / f"job-{uuid.uuid4().hex[:8]}"
            job_dir.mkdir(parents=True, exist_ok=True)
            saved_paths: list[str] = []
            for upload in files:
                dest = job_dir / (upload.filename or f"upload-{uuid.uuid4().hex[:8]}")
                dest.write_bytes(await upload.read())
                saved_paths.append(str(dest))

            job_id = self._create_job(
                num_documents=len(saved_paths),
                config=pipeline_config.model_dump(mode="json"),
            )
            with self._job_lock:
                self._jobs[job_id]["temp_dir"] = str(job_dir)

            thread = threading.Thread(
                target=self._run_ingest_job,
                args=(job_id, saved_paths, pipeline_config),
                daemon=True,
            )
            thread.start()

            return IngestJobResponse(
                job_id=job_id,
                status="queued",
                num_documents=len(saved_paths),
                created_at=int(self._jobs[job_id]["created_at"]),
            )

        # ---------------------------------------------------------------
        # Pipeline: GET /api/v1/jobs
        # ---------------------------------------------------------------

        @app.get("/api/v1/jobs", tags=["ingest"])
        async def list_jobs(
            self,
            status: Optional[str] = None,
            limit: int = 100,
            offset: int = 0,
        ) -> dict:
            """List all ingest jobs, optionally filtered by status."""
            with self._job_lock:
                jobs = list(self._jobs.values())
            if status:
                jobs = [j for j in jobs if j["status"] == status]
            jobs.sort(key=lambda j: j["created_at"], reverse=True)
            total = len(jobs)
            page = jobs[offset : offset + limit]
            return {
                "jobs": [
                    IngestJobStatusResponse(
                        job_id=j["job_id"],
                        status=j["status"],
                        num_documents=j["num_documents"],
                        created_at=int(j["created_at"]),
                        progress=(
                            IngestJobProgress(
                                documents_total=j["num_documents"],
                                documents_completed=j.get("documents_completed", 0),
                                current_stage=j.get("current_stage"),
                                elapsed_seconds=round(
                                    (j.get("completed_at") or time.time()) - (j.get("started_at") or j["created_at"]),
                                    2,
                                ),
                            )
                            if j["status"] in ("running", "completed")
                            else None
                        ),
                        error=j.get("error"),
                    ).model_dump(mode="json")
                    for j in page
                ],
                "total": total,
                "limit": limit,
                "offset": offset,
            }

        # ---------------------------------------------------------------
        # Pipeline: GET /api/v1/jobs/{job_id}
        # ---------------------------------------------------------------

        @app.get("/api/v1/jobs/{job_id}", response_model=IngestJobStatusResponse, tags=["ingest"])
        async def get_job_status(self, job_id: str) -> IngestJobStatusResponse:
            """Get the status and progress of an ingest job."""
            with self._job_lock:
                job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(404, detail=f"No job with id '{job_id}'.")

            progress = None
            if job["status"] in ("running", "completed"):
                progress = IngestJobProgress(
                    documents_total=job["num_documents"],
                    documents_completed=job.get("documents_completed", 0),
                    current_stage=job.get("current_stage"),
                    elapsed_seconds=round(
                        (job.get("completed_at") or time.time()) - (job.get("started_at") or job["created_at"]),
                        2,
                    ),
                )

            return IngestJobStatusResponse(
                job_id=job["job_id"],
                status=job["status"],
                num_documents=job["num_documents"],
                created_at=int(job["created_at"]),
                progress=progress,
                error=job.get("error"),
            )

        # ---------------------------------------------------------------
        # Pipeline: GET /api/v1/jobs/{job_id}/results
        # ---------------------------------------------------------------

        @app.get("/api/v1/jobs/{job_id}/results", response_model=IngestJobResultResponse, tags=["ingest"])
        async def get_job_results(self, job_id: str) -> IngestJobResultResponse:
            """Retrieve results from a completed ingest job."""
            with self._job_lock:
                job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(404, detail=f"No job with id '{job_id}'.")
            if job["status"] not in ("completed", "failed"):
                raise HTTPException(409, detail=f"Job '{job_id}' is not yet completed (status={job['status']}).")

            elapsed = 0.0
            if job.get("started_at") and job.get("completed_at"):
                elapsed = round(job["completed_at"] - job["started_at"], 4)

            return IngestJobResultResponse(
                job_id=job["job_id"],
                status=job["status"],
                num_documents=job["num_documents"],
                elapsed_seconds=elapsed,
                records=job.get("results") or [],
            )

        # ---------------------------------------------------------------
        # Pipeline: DELETE /api/v1/jobs/{job_id}
        # ---------------------------------------------------------------

        @app.delete("/api/v1/jobs/{job_id}", tags=["ingest"])
        async def delete_job(self, job_id: str) -> dict:
            """Cancel a running job or delete a completed job and its temp files."""
            with self._job_lock:
                job = self._jobs.pop(job_id, None)
            if job is None:
                raise HTTPException(404, detail=f"No job with id '{job_id}'.")

            temp_dir = job.get("temp_dir")
            if temp_dir:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)

            return {"job_id": job_id, "deleted": True}

        # ---------------------------------------------------------------
        # Pipeline: POST /api/v1/ingest/stream
        # ---------------------------------------------------------------

        _SPLITTABLE_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx"})

        @app.post("/api/v1/ingest/stream", tags=["ingest"])
        async def ingest_pipeline_stream(
            self,
            files: List[UploadFile] = File(..., description="Documents to ingest."),
            config: Optional[str] = Form(None, description="Pipeline config as JSON string."),
        ) -> StreamingResponse:
            """Stream ingestion results page-by-page as NDJSON.

            Each output line is a JSON object for one processed page or chunk.
            PDF/DOCX/PPTX documents are split into individual pages and each
            page is run through the full pipeline graph independently so that
            results arrive as soon as a page finishes.

            Non-PDF files (text, HTML, images, audio) are processed as a
            single unit and streamed once complete.

            If VDB upload is enabled, it runs as a batch operation after all
            pages have been streamed back, and a final ``_summary`` line is
            emitted.

            Error lines have ``"_error": true``.  The last line has
            ``"_summary": true`` with aggregate timing and VDB status.
            """
            pipeline_config = IngestPipelineConfig()
            if config:
                try:
                    pipeline_config = IngestPipelineConfig(**json.loads(config))
                except (json.JSONDecodeError, Exception) as exc:
                    raise HTTPException(400, detail=f"Invalid pipeline config: {exc}") from exc

            job_dir = self._ingest_dir / f"stream-{uuid.uuid4().hex[:8]}"
            job_dir.mkdir(parents=True, exist_ok=True)
            saved: list[tuple[str, str]] = []
            for upload in files:
                filename = upload.filename or f"upload-{uuid.uuid4().hex[:8]}"
                dest = job_dir / filename
                dest.write_bytes(await upload.read())
                saved.append((str(dest), filename))

            vdb_cfg = pipeline_config.vdb_upload
            stream_config = pipeline_config.model_copy(update={"vdb_upload": None})

            embed_model = self._embedding_model
            embed_endpoint = self._embedding_endpoint
            embed_api_key = self._embedding_api_key
            lancedb_uri = self._lancedb_uri
            lancedb_table = self._lancedb_table

            def _generate():
                import shutil

                import pandas as pd

                from nemo_retriever.ingest_modes.inprocess import pdf_to_pages_df

                graph = _build_serve_graph(
                    stream_config,
                    default_embed_model=embed_model,
                    default_embed_endpoint=embed_endpoint,
                    default_embed_api_key=embed_api_key,
                    default_lancedb_uri=lancedb_uri,
                    default_lancedb_table=lancedb_table,
                )

                accumulated_dfs: list[pd.DataFrame] = []
                t_start = time.perf_counter()
                page_count = 0

                for file_path, filename in saved:
                    ext = Path(file_path).suffix.lower()

                    if ext in self._SPLITTABLE_EXTENSIONS:
                        try:
                            pages_df = pdf_to_pages_df(file_path)
                        except Exception as exc:
                            yield _ndjson_line({"_error": True, "filename": filename, "detail": str(exc)})
                            continue

                        for row_idx in range(len(pages_df)):
                            single = pages_df.iloc[[row_idx]].copy()
                            original_page = int(single.iloc[0].get("page_number", row_idx + 1))

                            try:
                                t_page = time.perf_counter()
                                results = graph.execute(single)
                                result_df = results[0] if results else single
                                elapsed = time.perf_counter() - t_page

                                if isinstance(result_df, pd.DataFrame):
                                    accumulated_dfs.append(result_df)
                                    records = _serialize_df_records(result_df)
                                else:
                                    records = []

                                for record in records:
                                    record["page_number"] = original_page
                                    record["_filename"] = filename
                                    record["_elapsed_seconds"] = round(elapsed, 4)
                                    page_count += 1
                                    yield _ndjson_line(record)

                            except Exception as exc:
                                logger.exception("Page %d of %s failed", original_page, filename)
                                yield _ndjson_line(
                                    {
                                        "_error": True,
                                        "filename": filename,
                                        "page_number": original_page,
                                        "detail": str(exc),
                                    }
                                )
                    else:
                        try:
                            df = _load_files_to_df([file_path])
                            t_file = time.perf_counter()
                            results = graph.execute(df)
                            result_df = results[0] if results else df
                            elapsed = time.perf_counter() - t_file

                            if isinstance(result_df, pd.DataFrame):
                                accumulated_dfs.append(result_df)
                                records = _serialize_df_records(result_df)
                            else:
                                records = []

                            for record in records:
                                record["_filename"] = filename
                                record["_elapsed_seconds"] = round(elapsed, 4)
                                page_count += 1
                                yield _ndjson_line(record)

                        except Exception as exc:
                            logger.exception("File %s failed", filename)
                            yield _ndjson_line(
                                {
                                    "_error": True,
                                    "filename": filename,
                                    "detail": str(exc),
                                }
                            )

                vdb_uploaded = False
                if vdb_cfg is not None and vdb_cfg.enabled and accumulated_dfs:
                    try:
                        from nemo_retriever.ingest_modes.inprocess import (
                            upload_embeddings_to_lancedb_inprocess,
                        )

                        combined = pd.concat(accumulated_dfs, ignore_index=True)
                        upload_embeddings_to_lancedb_inprocess(
                            combined,
                            lancedb_uri=vdb_cfg.lancedb_uri or lancedb_uri,
                            table_name=vdb_cfg.table_name or lancedb_table,
                            overwrite=vdb_cfg.overwrite,
                            create_index=vdb_cfg.create_index,
                            hybrid=vdb_cfg.hybrid,
                        )
                        vdb_uploaded = True
                    except Exception as exc:
                        logger.exception("VDB upload failed during streaming ingest")
                        yield _ndjson_line(
                            {
                                "_error": True,
                                "stage": "vdb_upload",
                                "detail": str(exc),
                            }
                        )

                total_elapsed = round(time.perf_counter() - t_start, 4)
                yield _ndjson_line(
                    {
                        "_summary": True,
                        "num_documents": len(saved),
                        "num_pages": page_count,
                        "elapsed_seconds": total_elapsed,
                        "vdb_uploaded": vdb_uploaded,
                    }
                )

                shutil.rmtree(job_dir, ignore_errors=True)

            return StreamingResponse(
                _generate(),
                media_type="application/x-ndjson",
                headers={"Content-Disposition": "inline; filename=ingest.ndjson"},
            )
