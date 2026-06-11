# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob as _glob
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from nemo_retriever.params import CaptionParams, DedupParams, EmbedParams, ExtractParams, StoreParams, TextChunkParams
from nemo_retriever.utils.input_files import (
    AUTO_INPUT_EXTENSIONS,
    INPUT_TYPE_EXTENSIONS,
    expand_input_file_patterns,
    input_type_for_path,
    resolve_input_files,
)

logger = logging.getLogger(__name__)

ServiceIngestInputTypeValue = Literal["auto", "pdf", "doc", "txt", "html", "image", "audio", "video"]
ServiceIngestProfileValue = Literal["auto", "fast-text"]
_SUPPORTED_SERVICE_INPUT_TYPES: tuple[ServiceIngestInputTypeValue, ...] = (
    "auto",
    "pdf",
    "doc",
    "txt",
    "html",
    "image",
    "audio",
    "video",
)
_SUPPORTED_SERVICE_PROFILES: tuple[ServiceIngestProfileValue, ...] = ("auto", "fast-text")
_DEFAULT_TEXT_CHUNK_MAX_TOKENS = 1024
_DEFAULT_TEXT_CHUNK_OVERLAP_TOKENS = 150


@dataclass(frozen=True)
class ServiceIngestSourceOptions:
    documents: Sequence[str]
    profile: ServiceIngestProfileValue = "auto"
    input_type: ServiceIngestInputTypeValue = "auto"


@dataclass(frozen=True)
class ServiceIngestConnectionOptions:
    service_url: str = "http://localhost:7670"
    service_concurrency: int = 8
    service_api_token: str | None = None


@dataclass(frozen=True)
class ServiceIngestExtractOptions:
    method: str | None = None
    dpi: int | None = None
    extract_text: bool | None = None
    extract_images: bool | None = None
    extract_tables: bool | None = None
    extract_charts: bool | None = None
    extract_infographics: bool | None = None
    extract_page_as_image: bool | None = None
    use_page_elements: bool | None = None
    use_graphic_elements: bool | None = None
    use_table_structure: bool | None = None
    table_output_format: str | None = None
    ocr_version: str | None = None


@dataclass(frozen=True)
class ServiceIngestDedupOptions:
    enabled: bool = False
    iou_threshold: float | None = None


@dataclass(frozen=True)
class ServiceIngestCaptionOptions:
    enabled: bool = False
    context_text_max_chars: int | None = None
    caption_infographics: bool | None = None


@dataclass(frozen=True)
class ServiceIngestChunkOptions:
    enabled: bool = False
    text_chunk_max_tokens: int | None = None
    text_chunk_overlap_tokens: int | None = None


@dataclass(frozen=True)
class ServiceIngestEmbedOptions:
    embed_modality: str | None = None
    text_elements_modality: str | None = None
    structured_elements_modality: str | None = None
    embed_granularity: str | None = None


@dataclass(frozen=True)
class ServiceIngestImageStoreOptions:
    images_uri: str | None = None


@dataclass(frozen=True)
class ServiceIngestPlanRequest:
    source: ServiceIngestSourceOptions
    connection: ServiceIngestConnectionOptions = field(default_factory=ServiceIngestConnectionOptions)
    extract: ServiceIngestExtractOptions = field(default_factory=ServiceIngestExtractOptions)
    dedup: ServiceIngestDedupOptions = field(default_factory=ServiceIngestDedupOptions)
    caption: ServiceIngestCaptionOptions = field(default_factory=ServiceIngestCaptionOptions)
    chunk: ServiceIngestChunkOptions = field(default_factory=ServiceIngestChunkOptions)
    embed: ServiceIngestEmbedOptions = field(default_factory=ServiceIngestEmbedOptions)
    image_store: ServiceIngestImageStoreOptions = field(default_factory=ServiceIngestImageStoreOptions)


@dataclass(frozen=True)
class ServiceIngestRequest:
    documents: Sequence[str]
    input_type: ServiceIngestInputTypeValue
    extract_params: ExtractParams = field(default_factory=ExtractParams)
    embed_params: EmbedParams = field(default_factory=EmbedParams)
    text_chunk_params: TextChunkParams = field(default_factory=TextChunkParams)
    enable_text_chunk: bool = False
    dedup_params: DedupParams | None = None
    caption_params: CaptionParams | None = None
    store_params: StoreParams | None = None
    connection: ServiceIngestConnectionOptions = field(default_factory=ServiceIngestConnectionOptions)


@dataclass(frozen=True)
class ServiceIngestExecutionResult:
    """Structured result from executing a resolved service ingest request.

    Service mode does not locally verify the remote vector database after
    ingest. ``result_n_rows`` counts rows from the service ingest result when
    available, and ``n_rows`` mirrors that value so root CLI summaries keep the
    same top-level row-count contract as local ingest results.
    """

    request: ServiceIngestRequest
    result: object
    n_rows: int | None
    result_n_rows: int | None
    metadata: dict[str, Any]

    @property
    def documents(self) -> list[str]:
        return list(self.request.documents)

    @property
    def service_url(self) -> str:
        return self.request.connection.service_url

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "run_mode": "service",
            "documents": self.documents,
            "service_url": self.service_url,
            "result": self.result,
            "n_rows": self.n_rows,
            "result_n_rows": self.result_n_rows,
        }


def resolve_service_ingest_request(request: ServiceIngestPlanRequest) -> ServiceIngestRequest:
    """Resolve first-class root service ingest options into service params."""

    source = request.source
    input_type = _validate_service_input_type(source.input_type)
    profile = _validate_service_profile(source.profile)
    documents = resolve_service_documents(source.documents, input_type=input_type)
    _validate_service_profile_documents(profile, documents)

    extract_kwargs = _service_profile_extract_defaults(profile)
    extract_kwargs.update(
        {
            key: value
            for key, value in {
                "method": request.extract.method,
                "dpi": request.extract.dpi,
                "extract_text": request.extract.extract_text,
                "extract_images": request.extract.extract_images,
                "extract_tables": request.extract.extract_tables,
                "extract_charts": request.extract.extract_charts,
                "extract_infographics": request.extract.extract_infographics,
                "extract_page_as_image": request.extract.extract_page_as_image,
                "use_page_elements": request.extract.use_page_elements,
                "use_graphic_elements": request.extract.use_graphic_elements,
                "use_table_structure": request.extract.use_table_structure,
                "table_output_format": request.extract.table_output_format,
                "ocr_version": request.extract.ocr_version,
            }.items()
            if value is not None
        }
    )
    if request.extract.table_output_format == "markdown":
        extract_kwargs["use_table_structure"] = True

    embed_kwargs = {
        key: value
        for key, value in {
            "embed_modality": request.embed.embed_modality,
            "text_elements_modality": request.embed.text_elements_modality,
            "structured_elements_modality": request.embed.structured_elements_modality,
            "embed_granularity": request.embed.embed_granularity,
        }.items()
        if value is not None
    }
    enable_text_chunk, text_chunk_params = _build_service_text_chunk_params(request.chunk)

    return ServiceIngestRequest(
        documents=documents,
        input_type=input_type,
        extract_params=ExtractParams(**extract_kwargs),
        embed_params=EmbedParams(**embed_kwargs) if embed_kwargs else EmbedParams(),
        text_chunk_params=text_chunk_params,
        enable_text_chunk=enable_text_chunk,
        dedup_params=_build_service_dedup_params(request.dedup),
        caption_params=_build_service_caption_params(request.caption),
        store_params=_build_service_store_params(request.image_store),
        connection=request.connection,
    )


def build_service_ingestor(request: ServiceIngestRequest) -> Any:
    """Construct a remote-service ingestor with service-compatible stages."""

    from nemo_retriever.service_ingestor import ServiceIngestor

    resolved_files = expand_service_file_patterns(request.documents)
    if not resolved_files:
        raise ValueError("No files matched the input patterns for service mode.")

    ingestor = ServiceIngestor(
        base_url=request.connection.service_url,
        max_concurrency=request.connection.service_concurrency,
        api_token=request.connection.service_api_token,
    ).files(resolved_files)

    ingestor = _attach_service_extract_stage(
        ingestor,
        input_type=request.input_type,
        documents=resolved_files,
        extract_params=request.extract_params,
        enable_text_chunk=request.enable_text_chunk,
        text_chunk_params=request.text_chunk_params,
    )

    if request.dedup_params is not None:
        ingestor = ingestor.dedup(request.dedup_params)

    if request.caption_params is not None:
        ingestor = ingestor.caption(_sanitize_service_caption_params(request.caption_params))

    ingestor = ingestor.embed(request.embed_params)

    if request.store_params is not None:
        ingestor = ingestor.store(request.store_params)

    return ingestor


def execute_service_ingest_request(request: ServiceIngestRequest) -> ServiceIngestExecutionResult:
    """Execute a service ingest request and return its structured result."""

    result = build_service_ingestor(request).ingest()
    result_n_rows = _count_service_result_rows(result)
    return ServiceIngestExecutionResult(
        request=request,
        result=result,
        n_rows=result_n_rows,
        result_n_rows=result_n_rows,
        metadata={
            "service_url": request.connection.service_url,
            "input_type": request.input_type,
        },
    )


def resolve_service_documents(
    documents: Sequence[str],
    *,
    input_type: ServiceIngestInputTypeValue = "auto",
) -> list[str]:
    """Expand root service ingest files/directories/globs without creating an ingestor."""

    inputs: list[str] = []
    for document in documents:
        raw_document = str(document)
        path = Path(raw_document).expanduser()
        if path.is_dir():
            directory_files = resolve_input_files(path, input_type)
            if not directory_files:
                raise FileNotFoundError(f"No supported ingest files found under directory: {path}")
            inputs.extend(str(file) for file in directory_files)
        else:
            inputs.append(raw_document)

    document_list = expand_input_file_patterns(inputs)
    _validate_service_document_types(document_list, input_type=input_type)
    return document_list


def expand_service_file_patterns(documents: Sequence[str]) -> list[str]:
    """Expand recursive file patterns for service ingest construction."""

    resolved_files: list[str] = []
    for pattern in documents:
        resolved_files.extend(sorted(_glob.glob(str(pattern), recursive=True)))
    return resolved_files


def service_split_config_for_request(request: ServiceIngestRequest) -> dict[str, Any] | None:
    """Build the service split configuration for a resolved ingest request."""

    chunk_dict = _service_text_chunk_dict(request.text_chunk_params) if request.enable_text_chunk else None
    return _split_config_for_input_type(request.input_type, chunk_dict, documents=request.documents)


def _validate_service_input_type(input_type: str) -> ServiceIngestInputTypeValue:
    if input_type not in _SUPPORTED_SERVICE_INPUT_TYPES:
        raise ValueError(f"input_type must be one of {', '.join(_SUPPORTED_SERVICE_INPUT_TYPES)}, got {input_type!r}.")
    return cast(ServiceIngestInputTypeValue, input_type)


def _validate_service_profile(profile: str) -> ServiceIngestProfileValue:
    if profile not in _SUPPORTED_SERVICE_PROFILES:
        raise ValueError(f"profile must be one of {', '.join(_SUPPORTED_SERVICE_PROFILES)}, got {profile!r}.")
    return cast(ServiceIngestProfileValue, profile)


def _validate_service_document_types(
    documents: Sequence[str],
    *,
    input_type: ServiceIngestInputTypeValue,
) -> None:
    allowed_extensions = AUTO_INPUT_EXTENSIONS if input_type == "auto" else INPUT_TYPE_EXTENSIONS[input_type]
    unsupported = [
        document
        for document in documents
        if not _glob.has_magic(str(document)) and Path(document).suffix.lower() not in allowed_extensions
    ]
    if unsupported:
        examples = ", ".join(unsupported[:3])
        raise ValueError(f"Unsupported input file type(s) for retriever ingest: {examples}")


def _validate_service_profile_documents(profile: ServiceIngestProfileValue, documents: Sequence[str]) -> None:
    if profile != "fast-text":
        return
    disallowed = sorted(
        {
            family
            for document in documents
            if not _glob.has_magic(str(document))
            for family in [input_type_for_path(document)]
            if family not in {"pdf", "doc"}
        }
    )
    if disallowed:
        observed = ", ".join(disallowed)
        raise ValueError(f"--profile {profile} only supports PDF/document inputs; observed {observed}.")


def _service_profile_extract_defaults(profile: ServiceIngestProfileValue) -> dict[str, Any]:
    if profile == "fast-text":
        return {
            "method": "pdfium",
            "extract_text": True,
            "extract_images": False,
            "extract_tables": False,
            "extract_charts": False,
            "extract_infographics": False,
            "extract_page_as_image": False,
            "use_page_elements": False,
        }
    return {}


def _build_service_text_chunk_params(chunk: ServiceIngestChunkOptions) -> tuple[bool, TextChunkParams]:
    enabled = (
        bool(chunk.enabled) or chunk.text_chunk_max_tokens is not None or chunk.text_chunk_overlap_tokens is not None
    )
    if not enabled:
        return False, TextChunkParams()
    return True, TextChunkParams(
        max_tokens=(
            int(chunk.text_chunk_max_tokens)
            if chunk.text_chunk_max_tokens is not None
            else _DEFAULT_TEXT_CHUNK_MAX_TOKENS
        ),
        overlap_tokens=(
            int(chunk.text_chunk_overlap_tokens)
            if chunk.text_chunk_overlap_tokens is not None
            else _DEFAULT_TEXT_CHUNK_OVERLAP_TOKENS
        ),
    )


def _build_service_dedup_params(dedup: ServiceIngestDedupOptions) -> DedupParams | None:
    if not dedup.enabled:
        if dedup.iou_threshold is not None:
            raise ValueError("Dedup options require --dedup: dedup_iou_threshold.")
        return None
    dedup_kwargs = {}
    if dedup.iou_threshold is not None:
        dedup_kwargs["iou_threshold"] = dedup.iou_threshold
    return DedupParams(**dedup_kwargs)


def _build_service_caption_params(caption: ServiceIngestCaptionOptions) -> CaptionParams | None:
    overrides = {
        "caption_context_text_max_chars": caption.context_text_max_chars,
        "caption_infographics": caption.caption_infographics,
    }
    if not caption.enabled:
        provided = [name for name, value in overrides.items() if value is not None]
        if provided:
            raise ValueError(f"Caption options require --caption: {', '.join(provided)}.")
        return None
    if caption.context_text_max_chars is not None and caption.context_text_max_chars < 0:
        raise ValueError("caption_context_text_max_chars must be >= 0.")

    caption_kwargs = {
        key: value
        for key, value in {
            "context_text_max_chars": caption.context_text_max_chars,
            "caption_infographics": caption.caption_infographics,
        }.items()
        if value is not None
    }
    return CaptionParams(**caption_kwargs)


def _build_service_store_params(image_store: ServiceIngestImageStoreOptions) -> StoreParams | None:
    if image_store.images_uri is None:
        return None
    return StoreParams(storage_uri=image_store.images_uri)


def _service_extraction_mode(input_type: str) -> str:
    """Map ingest input type to :class:`PipelineSpec` ``extraction_mode``."""

    return {
        "pdf": "pdf",
        "doc": "pdf",
        "txt": "text",
        "html": "html",
        "audio": "audio",
        "video": "auto",
    }.get(input_type, "auto")


def _service_text_chunk_dict(text_chunk_params: TextChunkParams) -> dict[str, Any]:
    """Serialize text-chunk knobs allowed by the service split_config policy."""

    from nemo_retriever.service.policy import _DEFAULT_ALLOWED_SPLIT_KEYS

    raw = text_chunk_params.model_dump(exclude_none=True)
    return {key: value for key, value in raw.items() if key in _DEFAULT_ALLOWED_SPLIT_KEYS}


def _attach_service_extract_stage(
    ingestor: Any,
    *,
    input_type: str,
    documents: Sequence[str],
    extract_params: ExtractParams,
    enable_text_chunk: bool,
    text_chunk_params: TextChunkParams,
) -> Any:
    """Wire the extraction stage for the remote service ingestor."""

    chunk_dict = _service_text_chunk_dict(text_chunk_params) if enable_text_chunk else None
    if input_type == "image":
        return ingestor.extract_image_files(
            extract_params,
            split_config={"image": chunk_dict} if chunk_dict else None,
        )
    return ingestor.extract(
        extract_params,
        split_config=_split_config_for_input_type(input_type, chunk_dict, documents=documents),
        extraction_mode=_service_extraction_mode(input_type),
    )


def _split_config_for_input_type(
    input_type: str,
    chunk_dict: dict[str, Any] | None,
    *,
    documents: Sequence[str] = (),
) -> dict[str, Any] | None:
    if chunk_dict is None:
        return None
    if input_type == "auto":
        return _split_config_for_auto_documents(documents, chunk_dict)
    if input_type in {"pdf", "doc"}:
        return {"pdf": chunk_dict}
    if input_type == "txt":
        return {"text": chunk_dict}
    if input_type == "html":
        return {"html": chunk_dict}
    if input_type == "image":
        return {"image": chunk_dict}
    if input_type == "audio":
        return {"audio": chunk_dict}
    if input_type == "video":
        return {"video": chunk_dict, "audio": chunk_dict}
    return None


def _split_config_for_auto_documents(
    documents: Sequence[str],
    chunk_dict: dict[str, Any],
) -> dict[str, Any] | None:
    input_types = {input_type_for_path(document) for document in documents if not _glob.has_magic(str(document))}
    split_config: dict[str, Any] = {}
    if input_types & {"pdf", "doc"}:
        split_config["pdf"] = dict(chunk_dict)
    if "txt" in input_types:
        split_config["text"] = dict(chunk_dict)
    if "html" in input_types:
        split_config["html"] = dict(chunk_dict)
    if "image" in input_types:
        split_config["image"] = dict(chunk_dict)
    if "audio" in input_types:
        split_config["audio"] = dict(chunk_dict)
    if "video" in input_types:
        split_config["video"] = dict(chunk_dict)
        split_config["audio"] = dict(chunk_dict)
    return split_config or None


def _sanitize_service_caption_params(caption_params: CaptionParams) -> CaptionParams:
    params_dict = caption_params.model_dump(exclude_none=True, exclude_unset=True)
    return CaptionParams(
        **{
            key: value
            for key, value in params_dict.items()
            if key
            in {
                "prompt",
                "system_prompt",
                "batch_size",
                "context_text_max_chars",
                "caption_infographics",
                "temperature",
                "max_tokens",
                "top_p",
                "top_k",
            }
        }
    )


def _count_service_result_rows(result: object) -> int | None:
    dataframe = getattr(result, "dataframe", None)
    if dataframe is None:
        return None
    try:
        return len(dataframe)
    except TypeError:
        return None
