# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Tuple
from urllib.parse import urlparse

import warnings


from upath import UPath

from nemo_retriever.tabular_data.sql_database import SQLDatabase
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_retriever.utils.remote_auth import resolve_remote_api_key

RunMode = Literal["inprocess", "batch", "fused", "online"]

# Pass as an api_key value to suppress auto-resolution from environment variables.
# Example: EmbedParams(api_key=NO_API_KEY)
NO_API_KEY = ""


class _ParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _resolve_api_keys(self) -> "_ParamsModel":
        for field_name in type(self).model_fields:
            if field_name == "api_key" or field_name.endswith("_api_key"):
                value = getattr(self, field_name, None)
                if value is None:
                    setattr(self, field_name, resolve_remote_api_key())
                elif value == NO_API_KEY:
                    setattr(self, field_name, None)
        return self


class RemoteRetryParams(_ParamsModel):
    remote_max_pool_workers: int = 8
    remote_max_retries: int = 5
    remote_max_429_retries: int = 3


class RemoteInvokeParams(_ParamsModel):
    invoke_url: Optional[str] = None
    api_key: Optional[str] = None
    request_timeout_s: float = 60.0


class ModelRuntimeParams(_ParamsModel):
    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    normalize: bool = True
    max_length: int = 8192
    model_name: Optional[str] = None


class IngestorCreateParams(_ParamsModel):
    documents: list[str] = Field(default_factory=list)
    ray_address: Optional[str] = None
    ray_log_to_driver: bool = True
    debug: bool = False
    base_url: str = "http://localhost:7670"
    allow_no_gpu: bool = False


class IngestExecuteParams(_ParamsModel):
    show_progress: bool = False
    return_failures: bool = False
    save_to_disk: bool = False
    return_traces: bool = False
    parallel: bool = False
    max_workers: Optional[int] = None
    gpu_devices: list[str] = Field(default_factory=list)
    page_chunk_size: int = 32
    runtime_metrics_dir: Optional[str] = None
    runtime_metrics_prefix: Optional[str] = None


class PdfSplitParams(_ParamsModel):
    start_page: Optional[int] = None
    end_page: Optional[int] = None


class TextChunkParams(_ParamsModel):
    max_tokens: int = 1024
    overlap_tokens: int = 0
    tokenizer_model_id: Optional[str] = None
    encoding: str = "utf-8"
    tokenizer_cache_dir: Optional[str] = None


class HtmlChunkParams(TextChunkParams):
    pass


class AudioChunkParams(_ParamsModel):
    """Params for media chunking (audio/video split). Aligned with nv-ingest-api dataloader."""

    split_type: Literal["size", "time", "frame"] = "size"
    split_interval: int = 450
    audio_only: bool = False
    video_audio_separate: bool = False


class ASRParams(_ParamsModel):
    """Params for ASR (Parakeet/Riva gRPC or local transformers backend)."""

    audio_endpoints: Tuple[Optional[str], Optional[str]] = (None, None)
    audio_infer_protocol: str = "grpc"
    function_id: Optional[str] = None
    auth_token: Optional[str] = None
    segment_audio: bool = False


class LanceDbParams(_ParamsModel):
    lancedb_uri: str = "lancedb"
    table_name: str = "nv-ingest"
    overwrite: bool = True
    create_index: bool = True
    index_type: str = "IVF_HNSW_SQ"
    metric: str = "l2"
    num_partitions: int = 16
    num_sub_vectors: int = 256
    embedding_column: str = "text_embeddings_1b_v2"
    embedding_key: str = "embedding"
    include_text: bool = True
    text_column: str = "text"
    hybrid: bool = False
    fts_language: str = "English"


class BatchTuningParams(_ParamsModel):
    debug_run_id: str = "unknown"
    pdf_split_batch_size: int = 1
    pdf_extract_batch_size: int = 4
    pdf_extract_num_cpus: float = 2
    pdf_extract_workers: Optional[int] = None
    page_elements_batch_size: int = 24
    detect_batch_size: int = 24
    ocr_inference_batch_size: Optional[int] = None
    page_elements_workers: Optional[int] = None
    ocr_workers: Optional[int] = None
    detect_workers: Optional[int] = None
    page_elements_cpus_per_actor: float = 1
    ocr_cpus_per_actor: float = 1
    embed_workers: Optional[int] = None
    embed_batch_size: int = 256
    embed_cpus_per_actor: float = 1
    gpu_page_elements: Optional[float] = None
    gpu_ocr: Optional[float] = None
    gpu_embed: Optional[float] = None
    nemotron_parse_workers: float = 0.0
    gpu_nemotron_parse: float = 0.0
    nemotron_parse_batch_size: float = 0.0
    inference_batch_size: int = 8


class FusedTuningParams(_ParamsModel):
    fused_workers: int = 1
    fused_batch_size: int = 64
    fused_cpus_per_actor: float = 1
    fused_gpus_per_actor: float = 1.0


class GpuAllocationParams(_ParamsModel):
    gpu_devices: list[str] = Field(default_factory=list)
    startup_timeout: float = 600.0


class ExtractParams(_ParamsModel):
    # Extraction flags
    extract_text: bool = True
    extract_images: bool = True
    extract_tables: bool = True
    extract_charts: bool = True
    extract_infographics: bool = True
    extract_page_as_image: Optional[bool] = None

    # Extraction options
    method: str = "pdfium"
    use_table_structure: bool = False
    table_output_format: Optional[Literal["pseudo_markdown", "markdown"]] = None
    use_graphic_elements: bool = False
    dpi: int = 200
    image_format: str = "jpeg"
    jpeg_quality: int = 100
    render_mode: Literal["full_dpi", "fit_to_model"] = "fit_to_model"
    inference_batch_size: int = 8
    ocr_model_dir: Optional[str] = None

    # Service endpoints
    invoke_url: Optional[str] = None
    api_key: Optional[str] = None
    request_timeout_s: float = 60.0
    page_elements_invoke_url: Optional[str] = None
    page_elements_api_key: Optional[str] = None
    page_elements_request_timeout_s: Optional[float] = None
    ocr_invoke_url: Optional[str] = None
    ocr_api_key: Optional[str] = None
    ocr_request_timeout_s: Optional[float] = None
    graphic_elements_invoke_url: Optional[str] = None
    table_structure_invoke_url: Optional[str] = None
    nemotron_parse_invoke_url: Optional[str] = None
    nemotron_parse_model: Optional[str] = None

    # Output columns
    output_column: str = "page_elements_v3"
    num_detections_column: str = "page_elements_v3_num_detections"
    counts_by_label_column: str = "page_elements_v3_counts_by_label"

    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    batch_tuning: BatchTuningParams = Field(default_factory=BatchTuningParams)

    @model_validator(mode="after")
    def _auto_enable_features(self) -> "ExtractParams":
        """Auto-configure feature flags from remote endpoints.

        * Enable ``use_graphic_elements`` when ``graphic_elements_invoke_url``
          is provided.
        * Enable ``use_table_structure`` when ``table_structure_invoke_url``
          is provided.
        * Default ``table_output_format`` to ``"markdown"`` when the stage is
          enabled and the caller did not explicitly choose a format.
        """
        if self.graphic_elements_invoke_url and not self.use_graphic_elements:
            self.use_graphic_elements = True
        if self.table_structure_invoke_url and not self.use_table_structure:
            self.use_table_structure = True
        if self.table_output_format is None:
            self.table_output_format = "markdown" if self.use_table_structure else "pseudo_markdown"
        return self


VALID_EMBED_MODALITIES: frozenset[str] = frozenset({"text", "image", "text_image"})
IMAGE_MODALITIES: frozenset[str] = frozenset({"image", "text_image"})


class EmbedParams(_ParamsModel):
    model_name: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    embed_invoke_url: Optional[str] = None
    api_key: Optional[str] = None
    input_type: str = "passage"
    embed_modality: str = "text"  # "text", "image", or "text_image" — default for all element types
    embed_granularity: Literal["element", "page"] = "element"  # "element" = per-element rows, "page" = one row per page
    text_elements_modality: Optional[str] = None  # per-type override for page-text rows
    structured_elements_modality: Optional[str] = None  # per-type override for table/chart/infographic rows
    text_column: str = "text"
    inference_batch_size: int = 32
    output_column: str = "text_embeddings_1b_v2"
    embedding_dim_column: str = "text_embeddings_1b_v2_dim"
    has_embedding_column: str = "text_embeddings_1b_v2_has_embedding"
    embed_output_column: str = "text_embeddings_1b_v2"
    embed_inference_batch_size: int = 16

    runtime: ModelRuntimeParams = Field(default_factory=ModelRuntimeParams)
    batch_tuning: BatchTuningParams = Field(default_factory=BatchTuningParams)
    fused_tuning: FusedTuningParams = Field(default_factory=FusedTuningParams)

    @field_validator("embed_modality", "text_elements_modality", "structured_elements_modality", mode="before")
    @classmethod
    def _validate_modality(cls, v: str | None) -> str | None:
        if v is None:
            return None
        modality = str(v).strip()
        if modality == "image_text":
            raise ValueError("Use 'text_image' instead of 'image_text'.")
        if modality not in VALID_EMBED_MODALITIES:
            raise ValueError(f"Modality must be one of {sorted(VALID_EMBED_MODALITIES)}")
        return modality

    @model_validator(mode="after")
    def _warn_page_granularity_overrides(self) -> "EmbedParams":
        if self.embed_granularity == "page" and (
            self.text_elements_modality is not None or self.structured_elements_modality is not None
        ):
            warnings.warn(
                "text_elements_modality and structured_elements_modality are ignored when "
                "embed_granularity='page' (only embed_modality is used).",
                UserWarning,
                stacklevel=2,
            )
        return self


class VdbUploadParams(_ParamsModel):
    purge_results_after_upload: bool = True
    lancedb: LanceDbParams = Field(default_factory=LanceDbParams)


class StoreParams(_ParamsModel):
    storage_uri: str = "stored_images"
    storage_options: dict[str, Any] = Field(default_factory=dict)
    public_base_url: Optional[str] = None
    store_page_images: bool = True
    store_tables: bool = True
    store_charts: bool = True
    store_infographics: bool = True
    store_images: bool = True
    store_text: bool = False
    image_format: str = "png"
    strip_base64: bool = True

    @model_validator(mode="after")
    def _resolve_local_storage_uri(self) -> "StoreParams":
        """Resolve relative local paths to absolute so they survive Ray serialization."""
        if not urlparse(self.storage_uri).scheme:
            self.storage_uri = str(UPath(self.storage_uri).resolve())
        return self


class PageElementsParams(_ParamsModel):
    remote: RemoteInvokeParams = Field(default_factory=RemoteInvokeParams)
    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    inference_batch_size: int = 8
    output_column: str = "page_elements_v3"
    num_detections_column: str = "page_elements_v3_num_detections"
    counts_by_label_column: str = "page_elements_v3_counts_by_label"


class OcrParams(_ParamsModel):
    remote: RemoteInvokeParams = Field(default_factory=RemoteInvokeParams)
    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    inference_batch_size: int = 8
    extract_tables: bool = False
    extract_charts: bool = False
    extract_infographics: bool = False


class TableParams(_ParamsModel):
    remote: RemoteInvokeParams = Field(default_factory=RemoteInvokeParams)
    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    inference_batch_size: int = 8
    output_column: str = "table_structure_v1"
    num_detections_column: str = "table_structure_v1_num_detections"
    counts_by_label_column: str = "table_structure_v1_counts_by_label"


class ChartParams(_ParamsModel):
    remote: RemoteInvokeParams = Field(default_factory=RemoteInvokeParams)
    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    inference_batch_size: int = 8


class LLMInferenceParams(_ParamsModel):
    """Reusable LLM sampling / generation parameters.

    Inherit from this model to add temperature, top_p, and max_tokens
    to any task that invokes an LLM (captioning, summarization, etc.).
    """

    temperature: float = 1.0
    top_p: Optional[float] = None
    max_tokens: int = 1024

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _check_max_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be > 0")
        return v

    def to_sampling_kwargs(self) -> dict[str, Any]:
        """Build a dict of sampling parameters suitable for LLM inference calls.

        ``top_p`` is only included when explicitly set (not ``None``), because
        many backends (vLLM, OpenAI, NIM) change behaviour when the key is
        present vs. absent.
        """
        kw: dict[str, Any] = {"temperature": self.temperature, "max_tokens": self.max_tokens}
        if self.top_p is not None:
            kw["top_p"] = self.top_p
        return kw


class CaptionParams(LLMInferenceParams):
    endpoint_url: Optional[str] = None
    model_name: str = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
    api_key: Optional[str] = None
    prompt: str = "Caption the content of this image:"
    system_prompt: Optional[str] = "/no_think"
    batch_size: int = 8
    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    context_text_max_chars: int = 0
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.5
    caption_infographics: bool = False


class DedupParams(_ParamsModel):
    content_hash: bool = True
    bbox_iou: bool = True
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)


class InfographicParams(_ParamsModel):
    remote: RemoteInvokeParams = Field(default_factory=RemoteInvokeParams)
    remote_retry: RemoteRetryParams = Field(default_factory=RemoteRetryParams)
    inference_batch_size: int = 8
    allowed_page_element_labels: Sequence[str] = ("infographic", "title")
    output_column: str = "infographic_elements_v1"
    num_detections_column: str = "infographic_elements_v1_num_detections"
    counts_by_label_column: str = "infographic_elements_v1_counts_by_label"


# ---------------------------------------------------------------------------
# Structured (database) ingestion params
# ---------------------------------------------------------------------------


class TabularExtractParams(_ParamsModel):
    """Params for step 1: extract schema metadata and write to Neo4j.

    Covers SQLAlchemy reflection of a live database and/or parsing of
    pre-existing SQL DDL/query files.  Produces Database, Schema, Table,
    Column, View and Query nodes together with their relationships.
    The Neo4j connection is provided by get_neo4j_conn() (see
    tabular_data.neo4j) and is not configured here.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    connector: Optional[SQLDatabase] = None
