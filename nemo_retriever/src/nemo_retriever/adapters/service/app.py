# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ray Serve application integrated with FastAPI.

Exposes /health, /version, /ingest, /stream-pdf, /embeddings, and /retrieve.
"""

from __future__ import annotations

import json
import logging
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
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
        "and PDF streaming via a Ray Serve deployment."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "health", "description": "Service health and version endpoints."},
        {"name": "ingest", "description": "Document ingestion pipeline."},
        {"name": "embeddings", "description": "Text embedding generation."},
        {"name": "retrieval", "description": "Vector store query and retrieval."},
        {"name": "pdf", "description": "PDF text extraction and streaming."},
    ],
    license_info={"name": "Apache-2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    contact={"name": "NVIDIA Corporation", "url": "https://www.nvidia.com/"},
)

STREAM_PDF_MAX_BYTES = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class EmbeddingsRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text strings to embed.")
    model: Optional[str] = Field(None, description="Override embedding model name.")
    input_type: str = Field("passage", description="Input type hint (passage or query).")


class EmbeddingItem(BaseModel):
    index: int
    embedding: List[float]


class EmbeddingsResponse(BaseModel):
    model: str
    embeddings: List[EmbeddingItem]


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


class IngestPageResult(BaseModel):
    page_number: int = Field(..., description="1-indexed page number.")
    filename: str = Field("", description="Original uploaded filename.")
    text: str = Field("", description="Extracted page text.")


class IngestResponse(BaseModel):
    num_documents: int = Field(..., description="Number of documents submitted.")
    pages: List[IngestPageResult] = Field(default_factory=list, description="Per-page results.")


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


def _sanitize_value(v: object) -> object:
    """Coerce a single value into something JSON-safe."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, str):
        return v.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(v, dict):
        return _sanitize_for_json(v)
    if isinstance(v, (list, tuple)):
        return [_sanitize_value(i) for i in v]
    return v


def _sanitize_for_json(d: dict) -> dict:
    """Recursively ensure every value in *d* is valid UTF-8 JSON-serializable."""
    return {k: _sanitize_value(v) for k, v in d.items()}


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
            from nemo_retriever.ingestor import create_ingestor

            with tempfile.TemporaryDirectory(prefix="retriever_ingest_") as tmp_dir:
                saved_paths: list[str] = []
                for upload in files:
                    dest = Path(tmp_dir) / (upload.filename or "upload")
                    dest.write_bytes(await upload.read())
                    saved_paths.append(str(dest))

                try:
                    from nemo_retriever.pdf.extract import pdf_extraction

                    ing = create_ingestor(run_mode=run_mode)  # type: ignore[arg-type]
                    ing.files(saved_paths)
                    ing._pipeline_type = "pdf"
                    ing._tasks.append((pdf_extraction, {"extract_text": True}))
                    raw_results = ing.ingest()
                except Exception as exc:
                    logger.exception("Ingestion failed")
                    raise HTTPException(500, detail=f"Ingestion failed: {exc}") from exc

                if isinstance(raw_results, tuple):
                    raw_results = list(raw_results[0]) if raw_results else []
                if not isinstance(raw_results, list):
                    raw_results = [raw_results]

                pages: list[IngestPageResult] = []
                for item in raw_results:
                    df_dict: dict = {}
                    if hasattr(item, "to_dict"):
                        df_dict = item.to_dict()
                    elif isinstance(item, dict):
                        df_dict = item

                    page_nums = df_dict.get("page_number", {})
                    filenames = df_dict.get("path", {})
                    texts = df_dict.get("text", {})
                    for idx in sorted(page_nums.keys(), key=lambda k: int(k)):
                        raw_path = filenames.get(idx, "")
                        fname = Path(raw_path).name if raw_path else ""
                        page_text = texts.get(idx, "")
                        pages.append(
                            IngestPageResult(
                                page_number=int(page_nums[idx]),
                                filename=fname,
                                text=str(page_text) if page_text else "",
                            )
                        )

            return IngestResponse(num_documents=len(saved_paths), pages=pages)

        @app.post("/embeddings", response_model=EmbeddingsResponse, tags=["embeddings"])
        async def embeddings(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
            """Generate embeddings for a list of text strings."""
            import pandas as pd

            from nemo_retriever.text_embed import embed_text_1b_v2

            if not request.texts:
                raise HTTPException(400, detail="texts list must not be empty.")

            actor = self._get_embed_actor()
            df = pd.DataFrame({"text": request.texts})

            try:
                result_df = embed_text_1b_v2(
                    df,
                    model=actor._model,
                    model_name=request.model or self._embedding_model,
                    input_type=request.input_type,
                )
            except Exception as exc:
                logger.exception("Embedding failed")
                raise HTTPException(500, detail=f"Embedding failed: {exc}") from exc

            items: list[EmbeddingItem] = []
            col = "text_embeddings_1b_v2"
            for idx, row in result_df.iterrows():
                payload = row.get(col, {})
                emb = payload.get("embedding") if isinstance(payload, dict) else None
                items.append(EmbeddingItem(index=int(idx), embedding=emb or []))

            return EmbeddingsResponse(
                model=request.model or self._embedding_model,
                embeddings=items,
            )

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
