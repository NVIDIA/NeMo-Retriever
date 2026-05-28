# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast
import logging

from nemo_retriever.ingest_manifest import (
    ExtractionBranchPlan,
    build_input_manifest,
    format_branch_summary,
    plan_extraction_branches,
)
from nemo_retriever.ingestor import create_ingestor
from nemo_retriever.ocr.config import OCRLang, OCRVersion
from nemo_retriever.params import (
    ASRParams,
    AudioChunkParams,
    AudioVisualFuseParams,
    BatchTuningParams,
    CaptionParams,
    EmbedParams,
    ExtractParams,
    HtmlChunkParams,
    TextChunkParams,
    VdbUploadParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
)
from nemo_retriever.params.utils import normalize_embed_kwargs
from nemo_retriever.retriever import Retriever
from nemo_retriever.utils.input_files import (
    AUTO_INPUT_EXTENSIONS,
    INPUT_TYPE_EXTENSIONS,
    expand_input_file_patterns,
    resolve_input_files,
)
from nemo_retriever.utils.remote_auth import resolve_remote_api_key
from nemo_retriever.vdb.records import RetrievalHit

logger = logging.getLogger(__name__)

IngestRunModeValue = Literal["inprocess", "batch"]
IngestInputTypeValue = Literal["auto", "pdf", "doc", "txt", "html", "image", "audio", "video"]
IngestProfileValue = Literal["auto", "fast-text"]
AudioSplitTypeValue = Literal["size", "time", "frame"]
LocalIngestEmbedBackendValue = Literal["vllm", "hf"]
OcrLangValue = OCRLang
OcrVersionValue = OCRVersion
TableOutputFormatValue = Literal["pseudo_markdown", "markdown"]
_SUPPORTED_RUN_MODES: tuple[IngestRunModeValue, ...] = ("inprocess", "batch")
_SUPPORTED_PROFILES: tuple[IngestProfileValue, ...] = ("auto", "fast-text")
_SUPPORTED_AUDIO_SPLIT_TYPES: tuple[AudioSplitTypeValue, ...] = ("size", "time", "frame")
_SUPPORTED_INPUT_TYPES: tuple[IngestInputTypeValue, ...] = (
    "auto",
    "pdf",
    "doc",
    "txt",
    "html",
    "image",
    "audio",
    "video",
)
_AUDIO_SPLIT_INTERVAL = 500000
_VIDEO_FRAME_FPS = 0.5
_VIDEO_TEXT_DEDUP_MAX_DROPPED_FRAMES = 2
DEFAULT_LANCEDB_URI = "lancedb"
DEFAULT_TABLE_NAME = "nemo-retriever"
_DRY_RUN_SECRET_FIELD_PATTERNS = ("api_key", "password", "secret", "credential", "bearer")


def _validate_run_mode(run_mode: str) -> IngestRunModeValue:
    if run_mode not in _SUPPORTED_RUN_MODES:
        raise ValueError(f"run_mode must be one of {', '.join(_SUPPORTED_RUN_MODES)}, got {run_mode!r}.")
    return cast(IngestRunModeValue, run_mode)


def _validate_input_type(input_type: str) -> IngestInputTypeValue:
    if input_type not in _SUPPORTED_INPUT_TYPES:
        raise ValueError(f"input_type must be one of {', '.join(_SUPPORTED_INPUT_TYPES)}, got {input_type!r}.")
    return cast(IngestInputTypeValue, input_type)


def _validate_profile(profile: str) -> IngestProfileValue:
    if profile not in _SUPPORTED_PROFILES:
        raise ValueError(f"profile must be one of {', '.join(_SUPPORTED_PROFILES)}, got {profile!r}.")
    return cast(IngestProfileValue, profile)


def _validate_audio_split_type(split_type: str) -> AudioSplitTypeValue:
    if split_type not in _SUPPORTED_AUDIO_SPLIT_TYPES:
        raise ValueError(
            f"audio_split_type must be one of {', '.join(_SUPPORTED_AUDIO_SPLIT_TYPES)}, got {split_type!r}."
        )
    return cast(AudioSplitTypeValue, split_type)


# The ingest command accepts bare dataset directories; expand those to supported
# files before passing file/glob inputs through the shared input normalizer.
def _validate_ingest_document_types(documents: Sequence[str], *, input_type: IngestInputTypeValue) -> None:
    allowed_extensions = AUTO_INPUT_EXTENSIONS if input_type == "auto" else INPUT_TYPE_EXTENSIONS[input_type]
    unsupported = [
        document
        for document in documents
        if not any(ch in str(document) for ch in "*?[") and Path(document).suffix.lower() not in allowed_extensions
    ]
    if unsupported:
        examples = ", ".join(unsupported[:3])
        raise ValueError(f"Unsupported input file type(s) for retriever ingest: {examples}")


def _expand_ingest_documents(documents: Sequence[str], *, input_type: IngestInputTypeValue = "auto") -> list[str]:
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
    _validate_ingest_document_types(document_list, input_type=input_type)
    return document_list


@dataclass(frozen=True)
class ResolvedIngestPlan:
    documents: list[str]
    profile: IngestProfileValue
    branches: tuple[ExtractionBranchPlan, ...]
    create_kwargs: dict[str, Any]
    extract_params: ExtractParams
    text_params: TextChunkParams | None
    html_params: HtmlChunkParams | None
    audio_chunk_params: AudioChunkParams | None
    asr_params: ASRParams | None
    video_frame_params: VideoFrameParams | None
    video_text_dedup_params: VideoFrameTextDedupParams | None
    av_fuse_params: AudioVisualFuseParams | None
    caption_params: CaptionParams | None
    embed_params: EmbedParams | None
    vdb_params: VdbUploadParams
    lancedb_uri: str
    table_name: str

    def extract_call_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for key, value in {
            "text_params": self.text_params,
            "html_params": self.html_params,
            "audio_chunk_params": self.audio_chunk_params,
            "asr_params": self.asr_params,
            "video_frame_params": self.video_frame_params,
            "video_text_dedup_params": self.video_text_dedup_params,
            "av_fuse_params": self.av_fuse_params,
        }.items():
            if value is not None:
                kwargs[key] = value
        return kwargs

    def dry_run_data(self) -> dict[str, Any]:
        return {
            "dry_run": True,
            "profile": self.profile,
            "documents": list(self.documents),
            "branches": [
                {
                    "family": branch.family,
                    "extraction_mode": branch.extraction_mode,
                    "count": len(branch.input_paths),
                    "input_paths": list(branch.input_paths),
                }
                for branch in self.branches
            ],
            "branch_summary": format_branch_summary(self.branches),
            "create_ingestor": dict(self.create_kwargs),
            "extract": _params_to_dry_run_dict(self.extract_params),
            "text": _params_to_dry_run_dict(self.text_params),
            "html": _params_to_dry_run_dict(self.html_params),
            "audio": _params_to_dry_run_dict(self.audio_chunk_params),
            "asr": _params_to_dry_run_dict(self.asr_params),
            "video_frames": _params_to_dry_run_dict(self.video_frame_params),
            "video_frame_text_dedup": _params_to_dry_run_dict(self.video_text_dedup_params),
            "audio_visual_fuse": _params_to_dry_run_dict(self.av_fuse_params),
            "caption": _params_to_dry_run_dict(self.caption_params),
            "embed": _params_to_dry_run_dict(self.embed_params),
            "vdb_upload": _params_to_dry_run_dict(self.vdb_params),
        }


def _params_to_dry_run_dict(params: Any | None) -> dict[str, Any] | None:
    if params is None:
        return None
    if hasattr(params, "model_dump"):
        data = params.model_dump(mode="json")
    elif isinstance(params, dict):
        data = dict(params)
    else:
        return {"value": str(params)}
    return _strip_secret_values(data)


def _is_dry_run_secret_field(key: Any) -> bool:
    normalized_key = str(key).lower().replace("-", "_")
    return normalized_key.endswith("token") or any(
        pattern in normalized_key for pattern in _DRY_RUN_SECRET_FIELD_PATTERNS
    )


def _strip_secret_values(value: Any) -> Any:
    """Redact secrets from dry-run reporting only."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            if _is_dry_run_secret_field(key):
                out[key] = "<redacted>" if nested else nested
            else:
                out[key] = _strip_secret_values(nested)
        return out
    if isinstance(value, list):
        return [_strip_secret_values(item) for item in value]
    return value


def _branch_families(branches: Sequence[ExtractionBranchPlan]) -> set[str]:
    return {branch.family for branch in branches}


def _require_branch_families(
    *,
    profile: IngestProfileValue,
    branches: Sequence[ExtractionBranchPlan],
    allowed: set[str],
    description: str,
) -> None:
    observed = _branch_families(branches)
    disallowed = observed - allowed
    if disallowed:
        allowed_text = ", ".join(sorted(allowed))
        observed_text = ", ".join(sorted(observed))
        raise ValueError(
            f"--profile {profile} only supports {description} inputs ({allowed_text}); observed {observed_text}."
        )


def _validate_profile_manifest(profile: IngestProfileValue, branches: Sequence[ExtractionBranchPlan]) -> None:
    if profile == "fast-text":
        _require_branch_families(profile=profile, branches=branches, allowed={"pdf"}, description="PDF/document")


def _profile_extract_defaults(profile: IngestProfileValue) -> dict[str, Any]:
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


def _build_asr_params(*, segment_audio: bool | None, needed: bool) -> ASRParams | None:
    if not needed and segment_audio is None:
        return None
    from nemo_retriever.audio.asr_actor import asr_params_from_env

    params = asr_params_from_env()
    if segment_audio is None:
        return params
    return params.model_copy(update={"segment_audio": bool(segment_audio)})


def _resolve_media_params(
    *,
    branches: Sequence[ExtractionBranchPlan],
    segment_audio: bool | None,
    audio_split_type: AudioSplitTypeValue,
    audio_split_interval: int | None,
    video_extract_audio: bool | None,
    video_extract_frames: bool | None,
    video_frame_fps: float | None,
    video_frame_dedup: bool | None,
    video_frame_text_dedup: bool | None,
    video_frame_text_dedup_max_dropped_frames: int | None,
    video_av_fuse: bool | None,
) -> tuple[
    AudioChunkParams | None,
    ASRParams | None,
    VideoFrameParams | None,
    VideoFrameTextDedupParams | None,
    AudioVisualFuseParams | None,
]:
    families = _branch_families(branches)
    needs_audio = bool(families & {"audio", "video"})
    needs_video = "video" in families
    if not needs_audio and not needs_video:
        return None, _build_asr_params(segment_audio=segment_audio, needed=False), None, None, None

    split_interval = int(audio_split_interval) if audio_split_interval is not None else _AUDIO_SPLIT_INTERVAL
    audio_chunk_params = AudioChunkParams(
        enabled=bool(video_extract_audio) if video_extract_audio is not None and needs_video else True,
        split_type=audio_split_type,
        split_interval=split_interval,
    )
    asr_params = _build_asr_params(segment_audio=segment_audio, needed=needs_audio)

    if not needs_video:
        return audio_chunk_params, asr_params, None, None, None

    video_frame_params = VideoFrameParams(
        enabled=bool(video_extract_frames) if video_extract_frames is not None else True,
        fps=float(video_frame_fps) if video_frame_fps is not None else _VIDEO_FRAME_FPS,
        dedup=bool(video_frame_dedup) if video_frame_dedup is not None else True,
    )
    video_text_dedup_params = VideoFrameTextDedupParams(
        enabled=bool(video_frame_text_dedup) if video_frame_text_dedup is not None else True,
        max_dropped_frames=(
            int(video_frame_text_dedup_max_dropped_frames)
            if video_frame_text_dedup_max_dropped_frames is not None
            else _VIDEO_TEXT_DEDUP_MAX_DROPPED_FRAMES
        ),
    )
    av_fuse_params = AudioVisualFuseParams(enabled=bool(video_av_fuse) if video_av_fuse is not None else True)
    return audio_chunk_params, asr_params, video_frame_params, video_text_dedup_params, av_fuse_params


def _build_caption_params(
    *,
    caption: bool,
    caption_invoke_url: str | None,
    caption_model_name: str | None,
    caption_context_text_max_chars: int | None,
    caption_infographics: bool | None,
) -> CaptionParams | None:
    overrides = {
        "caption_invoke_url": caption_invoke_url,
        "caption_model_name": caption_model_name,
        "caption_context_text_max_chars": caption_context_text_max_chars,
        "caption_infographics": caption_infographics,
    }
    if not caption:
        provided = [name for name, value in overrides.items() if value is not None]
        if provided:
            raise ValueError(f"Caption options require --caption: {', '.join(provided)}.")
        return None
    if caption_context_text_max_chars is not None and caption_context_text_max_chars < 0:
        raise ValueError("caption_context_text_max_chars must be >= 0.")

    caption_kwargs = {
        key: value
        for key, value in {
            "endpoint_url": caption_invoke_url,
            "model_name": caption_model_name,
            "context_text_max_chars": caption_context_text_max_chars,
            "caption_infographics": caption_infographics,
        }.items()
        if value is not None
    }
    return CaptionParams(**caption_kwargs)


def _build_embed_kwargs(
    embed_invoke_url: str | None,
    embed_model_name: str | None,
    local_ingest_embed_backend: LocalIngestEmbedBackendValue | None = None,
    embed_workers: int | None = None,
    embed_batch_size: int | None = None,
    embed_cpus_per_actor: float | None = None,
    embed_gpus_per_actor: float | None = None,
) -> dict[str, Any]:
    embed_kwargs: dict[str, Any] = {}
    if embed_invoke_url is not None:
        embed_kwargs["embed_invoke_url"] = embed_invoke_url
    if embed_model_name is not None:
        # Remote HTTP embedding reads model_name; local/GPU paths read embed_model_name.
        embed_kwargs["model_name"] = embed_model_name
        embed_kwargs["embed_model_name"] = embed_model_name
    if local_ingest_embed_backend is not None:
        embed_kwargs["local_ingest_embed_backend"] = local_ingest_embed_backend
    embed_tuning = _build_embed_batch_tuning(
        embed_workers=embed_workers,
        embed_batch_size=embed_batch_size,
        embed_cpus_per_actor=embed_cpus_per_actor,
        embed_gpus_per_actor=embed_gpus_per_actor,
    )
    if embed_tuning is not None:
        embed_kwargs["batch_tuning"] = embed_tuning
    return normalize_embed_kwargs(embed_kwargs)


def _build_extract_batch_tuning(
    *,
    pdf_extract_workers: int | None,
    pdf_extract_batch_size: int | None,
    pdf_extract_cpus_per_task: float | None,
    page_elements_workers: int | None,
    page_elements_batch_size: int | None,
    page_elements_cpus_per_actor: float | None,
    page_elements_gpus_per_actor: float | None,
    ocr_workers: int | None,
    ocr_batch_size: int | None,
    ocr_cpus_per_actor: float | None,
    ocr_gpus_per_actor: float | None,
    table_structure_workers: int | None,
    table_structure_batch_size: int | None,
    table_structure_cpus_per_actor: float | None,
    table_structure_gpus_per_actor: float | None,
) -> BatchTuningParams | None:
    tuning_kwargs = {
        key: value
        for key, value in {
            "pdf_extract_workers": pdf_extract_workers,
            "pdf_extract_batch_size": pdf_extract_batch_size,
            # BatchTuningParams names this per-Ray-task reservation num_cpus.
            "pdf_extract_num_cpus": pdf_extract_cpus_per_task,
            "page_elements_workers": page_elements_workers,
            "page_elements_batch_size": page_elements_batch_size,
            "page_elements_cpus_per_actor": page_elements_cpus_per_actor,
            "gpu_page_elements": page_elements_gpus_per_actor,
            "ocr_workers": ocr_workers,
            "ocr_inference_batch_size": ocr_batch_size,
            "ocr_cpus_per_actor": ocr_cpus_per_actor,
            "gpu_ocr": ocr_gpus_per_actor,
            "table_structure_workers": table_structure_workers,
            "table_structure_batch_size": table_structure_batch_size,
            "table_structure_cpus_per_actor": table_structure_cpus_per_actor,
            "gpu_table_structure": table_structure_gpus_per_actor,
        }.items()
        if value is not None
    }
    return BatchTuningParams(**tuning_kwargs) if tuning_kwargs else None


def _build_embed_batch_tuning(
    *,
    embed_workers: int | None,
    embed_batch_size: int | None,
    embed_cpus_per_actor: float | None,
    embed_gpus_per_actor: float | None,
) -> BatchTuningParams | None:
    tuning_kwargs = {
        key: value
        for key, value in {
            "embed_workers": embed_workers,
            "embed_batch_size": embed_batch_size,
            "embed_cpus_per_actor": embed_cpus_per_actor,
            "gpu_embed": embed_gpus_per_actor,
        }.items()
        if value is not None
    }
    return BatchTuningParams(**tuning_kwargs) if tuning_kwargs else None


_LOCAL_VL_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"


def _build_rerank_kwargs(
    reranker_invoke_url: str | None,
    reranker_model_name: str | None = None,
    reranker_backend: str | None = None,
) -> dict[str, str]:
    """Build kwargs for the rerank stage. Mirrors :func:`_build_embed_kwargs`:
    if ``reranker_invoke_url`` is given the remote NIM path is configured;
    otherwise the local GPU reranker runs with ``reranker_model_name`` (or the
    matching VL default to pair with the local VL embedder).

    ``reranker_backend`` only applies to the local path and selects the local
    inference backend (``"vllm"`` or ``"hf"``); ``None`` defers to the library
    default in ``_default_rerank_actor_kwargs``.
    """
    reranker_url = (reranker_invoke_url or "").strip()
    if reranker_url:
        rerank_kwargs: dict[str, str] = {"rerank_invoke_url": reranker_url}
        if reranker_model_name:
            rerank_kwargs["model_name"] = reranker_model_name
        api_key = resolve_remote_api_key()
        if api_key is not None:
            rerank_kwargs["api_key"] = api_key
        return rerank_kwargs

    # Local GPU reranker — VL by default to pair with the local VL embedder.
    # ``NemotronRerankGPUActor`` loads the model once per actor; the rerank
    # model is ~2 GB and coexists with the vLLM embedder (which respects
    # ``gpu_memory_utilization=0.45``).
    local: dict[str, str] = {"model_name": reranker_model_name or _LOCAL_VL_RERANK_MODEL}
    if reranker_backend:
        local["local_reranker_backend"] = reranker_backend
    return local


def resolve_ingest_plan(
    documents: Sequence[str],
    *,
    profile: IngestProfileValue = "auto",
    input_type: IngestInputTypeValue = "auto",
    run_mode: IngestRunModeValue = "batch",
    method: str | None = None,
    dpi: int | None = None,
    extract_text: bool | None = None,
    extract_images: bool | None = None,
    extract_tables: bool | None = None,
    extract_charts: bool | None = None,
    extract_infographics: bool | None = None,
    extract_page_as_image: bool | None = None,
    use_page_elements: bool | None = None,
    segment_audio: bool | None = None,
    audio_split_type: AudioSplitTypeValue = "size",
    audio_split_interval: int | None = None,
    video_extract_audio: bool | None = None,
    video_extract_frames: bool | None = None,
    video_frame_fps: float | None = None,
    video_frame_dedup: bool | None = None,
    video_frame_text_dedup: bool | None = None,
    video_frame_text_dedup_max_dropped_frames: int | None = None,
    video_av_fuse: bool | None = None,
    caption: bool = False,
    caption_invoke_url: str | None = None,
    caption_model_name: str | None = None,
    caption_context_text_max_chars: int | None = None,
    caption_infographics: bool | None = None,
    ray_address: str | None = None,
    ray_log_to_driver: bool | None = None,
    lancedb_uri: str = DEFAULT_LANCEDB_URI,
    table_name: str = DEFAULT_TABLE_NAME,
    overwrite: bool = True,
    page_elements_invoke_url: str | None = None,
    ocr_invoke_url: str | None = None,
    ocr_version: OcrVersionValue | None = None,
    ocr_lang: OcrLangValue | None = None,
    graphic_elements_invoke_url: str | None = None,
    table_structure_invoke_url: str | None = None,
    table_output_format: TableOutputFormatValue | None = None,
    embed_invoke_url: str | None = None,
    embed_model_name: str | None = None,
    local_ingest_embed_backend: LocalIngestEmbedBackendValue | None = None,
    pdf_extract_workers: int | None = None,
    pdf_extract_batch_size: int | None = None,
    pdf_extract_cpus_per_task: float | None = None,
    page_elements_workers: int | None = None,
    page_elements_batch_size: int | None = None,
    page_elements_cpus_per_actor: float | None = None,
    page_elements_gpus_per_actor: float | None = None,
    ocr_workers: int | None = None,
    ocr_batch_size: int | None = None,
    ocr_cpus_per_actor: float | None = None,
    ocr_gpus_per_actor: float | None = None,
    table_structure_workers: int | None = None,
    table_structure_batch_size: int | None = None,
    table_structure_cpus_per_actor: float | None = None,
    table_structure_gpus_per_actor: float | None = None,
    embed_workers: int | None = None,
    embed_batch_size: int | None = None,
    embed_cpus_per_actor: float | None = None,
    embed_gpus_per_actor: float | None = None,
) -> ResolvedIngestPlan:
    """Resolve root ingest options into ordinary params for one extract call.

    Root ``retriever ingest`` intentionally defaults to ``run_mode="batch"``.
    Programmatic callers that need Ray-free local execution should pass
    ``run_mode="inprocess"`` explicitly. ``input_type`` remains a private
    expansion/validation constraint; extraction still routes from the manifest.
    """

    validated_run_mode = _validate_run_mode(run_mode)
    validated_profile = _validate_profile(profile)
    validated_input_type = _validate_input_type(input_type)
    validated_audio_split_type = _validate_audio_split_type(audio_split_type)
    document_list = _expand_ingest_documents(documents, input_type=validated_input_type)
    branches = plan_extraction_branches(build_input_manifest(document_list))
    _validate_profile_manifest(validated_profile, branches)

    extract_kwargs = _profile_extract_defaults(validated_profile)
    extract_kwargs.update(
        {
            key: value
            for key, value in {
                "method": method,
                "dpi": dpi,
                "extract_text": extract_text,
                "extract_images": extract_images,
                "extract_tables": extract_tables,
                "extract_charts": extract_charts,
                "extract_infographics": extract_infographics,
                "extract_page_as_image": extract_page_as_image,
                "use_page_elements": use_page_elements,
                "page_elements_invoke_url": page_elements_invoke_url,
                "ocr_invoke_url": ocr_invoke_url,
                "ocr_version": ocr_version,
                "ocr_lang": ocr_lang,
                "graphic_elements_invoke_url": graphic_elements_invoke_url,
                "table_structure_invoke_url": table_structure_invoke_url,
                "table_output_format": table_output_format,
            }.items()
            if value is not None
        }
    )
    if table_output_format == "markdown":
        extract_kwargs["use_table_structure"] = True

    extract_tuning = _build_extract_batch_tuning(
        pdf_extract_workers=pdf_extract_workers,
        pdf_extract_batch_size=pdf_extract_batch_size,
        pdf_extract_cpus_per_task=pdf_extract_cpus_per_task,
        page_elements_workers=page_elements_workers,
        page_elements_batch_size=page_elements_batch_size,
        page_elements_cpus_per_actor=page_elements_cpus_per_actor,
        page_elements_gpus_per_actor=page_elements_gpus_per_actor,
        ocr_workers=ocr_workers,
        ocr_batch_size=ocr_batch_size,
        ocr_cpus_per_actor=ocr_cpus_per_actor,
        ocr_gpus_per_actor=ocr_gpus_per_actor,
        table_structure_workers=table_structure_workers,
        table_structure_batch_size=table_structure_batch_size,
        table_structure_cpus_per_actor=table_structure_cpus_per_actor,
        table_structure_gpus_per_actor=table_structure_gpus_per_actor,
    )
    if extract_tuning is not None:
        extract_kwargs["batch_tuning"] = extract_tuning

    embed_kwargs = _build_embed_kwargs(
        embed_invoke_url,
        embed_model_name,
        local_ingest_embed_backend=local_ingest_embed_backend,
        embed_workers=embed_workers,
        embed_batch_size=embed_batch_size,
        embed_cpus_per_actor=embed_cpus_per_actor,
        embed_gpus_per_actor=embed_gpus_per_actor,
    )
    extract_params = ExtractParams(**extract_kwargs)
    embed_params = EmbedParams(**embed_kwargs) if embed_kwargs else None
    vdb_params = VdbUploadParams(
        vdb_kwargs={"uri": lancedb_uri, "table_name": table_name, "overwrite": bool(overwrite)}
    )
    caption_params = _build_caption_params(
        caption=caption,
        caption_invoke_url=caption_invoke_url,
        caption_model_name=caption_model_name,
        caption_context_text_max_chars=caption_context_text_max_chars,
        caption_infographics=caption_infographics,
    )

    families = _branch_families(branches)
    text_params = TextChunkParams() if "txt" in families else None
    html_params = HtmlChunkParams() if "html" in families else None
    (
        audio_chunk_params,
        asr_params,
        video_frame_params,
        video_text_dedup_params,
        av_fuse_params,
    ) = _resolve_media_params(
        branches=branches,
        segment_audio=segment_audio,
        audio_split_type=validated_audio_split_type,
        audio_split_interval=audio_split_interval,
        video_extract_audio=video_extract_audio,
        video_extract_frames=video_extract_frames,
        video_frame_fps=video_frame_fps,
        video_frame_dedup=video_frame_dedup,
        video_frame_text_dedup=video_frame_text_dedup,
        video_frame_text_dedup_max_dropped_frames=video_frame_text_dedup_max_dropped_frames,
        video_av_fuse=video_av_fuse,
    )

    create_kwargs: dict[str, Any] = {"run_mode": validated_run_mode}
    if ray_address is not None:
        create_kwargs["ray_address"] = ray_address
    if ray_log_to_driver is not None:
        create_kwargs["ray_log_to_driver"] = ray_log_to_driver

    return ResolvedIngestPlan(
        documents=document_list,
        profile=validated_profile,
        branches=branches,
        create_kwargs=create_kwargs,
        extract_params=extract_params,
        text_params=text_params,
        html_params=html_params,
        audio_chunk_params=audio_chunk_params,
        asr_params=asr_params,
        video_frame_params=video_frame_params,
        video_text_dedup_params=video_text_dedup_params,
        av_fuse_params=av_fuse_params,
        caption_params=caption_params,
        embed_params=embed_params,
        vdb_params=vdb_params,
        lancedb_uri=lancedb_uri,
        table_name=table_name,
    )


def ingest_documents(
    documents: Sequence[str],
    *,
    profile: IngestProfileValue = "auto",
    input_type: IngestInputTypeValue = "auto",
    run_mode: IngestRunModeValue = "batch",
    dry_run: bool = False,
    method: str | None = None,
    dpi: int | None = None,
    extract_text: bool | None = None,
    extract_images: bool | None = None,
    extract_tables: bool | None = None,
    extract_charts: bool | None = None,
    extract_infographics: bool | None = None,
    extract_page_as_image: bool | None = None,
    use_page_elements: bool | None = None,
    segment_audio: bool | None = None,
    audio_split_type: AudioSplitTypeValue = "size",
    audio_split_interval: int | None = None,
    video_extract_audio: bool | None = None,
    video_extract_frames: bool | None = None,
    video_frame_fps: float | None = None,
    video_frame_dedup: bool | None = None,
    video_frame_text_dedup: bool | None = None,
    video_frame_text_dedup_max_dropped_frames: int | None = None,
    video_av_fuse: bool | None = None,
    caption: bool = False,
    caption_invoke_url: str | None = None,
    caption_model_name: str | None = None,
    caption_context_text_max_chars: int | None = None,
    caption_infographics: bool | None = None,
    ray_address: str | None = None,
    ray_log_to_driver: bool | None = None,
    lancedb_uri: str = DEFAULT_LANCEDB_URI,
    table_name: str = DEFAULT_TABLE_NAME,
    overwrite: bool = True,
    page_elements_invoke_url: str | None = None,
    ocr_invoke_url: str | None = None,
    ocr_version: OcrVersionValue | None = None,
    ocr_lang: OcrLangValue | None = None,
    graphic_elements_invoke_url: str | None = None,
    table_structure_invoke_url: str | None = None,
    table_output_format: TableOutputFormatValue | None = None,
    embed_invoke_url: str | None = None,
    embed_model_name: str | None = None,
    local_ingest_embed_backend: LocalIngestEmbedBackendValue | None = None,
    pdf_extract_workers: int | None = None,
    pdf_extract_batch_size: int | None = None,
    pdf_extract_cpus_per_task: float | None = None,
    page_elements_workers: int | None = None,
    page_elements_batch_size: int | None = None,
    page_elements_cpus_per_actor: float | None = None,
    page_elements_gpus_per_actor: float | None = None,
    ocr_workers: int | None = None,
    ocr_batch_size: int | None = None,
    ocr_cpus_per_actor: float | None = None,
    ocr_gpus_per_actor: float | None = None,
    table_structure_workers: int | None = None,
    table_structure_batch_size: int | None = None,
    table_structure_cpus_per_actor: float | None = None,
    table_structure_gpus_per_actor: float | None = None,
    embed_workers: int | None = None,
    embed_batch_size: int | None = None,
    embed_cpus_per_actor: float | None = None,
    embed_gpus_per_actor: float | None = None,
) -> dict[str, Any]:
    """Run the root CLI ingestion path through the SDK adapter.

    Input families are inferred from concrete file extensions and routed by
    the graph ingestor manifest planner; the root CLI intentionally has no
    user-facing input-type selector.

    ``ray_address`` and ``ray_log_to_driver`` are forwarded only when the
    caller sets them, preserving the default ``create_ingestor`` behavior.
    Batch tuning arguments are opt-in and are translated into
    ``BatchTuningParams`` for extraction or embedding; they are meaningful for
    ``run_mode="batch"`` and ignored by callers that leave them unset.
    Root ``retriever ingest`` intentionally defaults to ``run_mode="batch"``;
    pass ``run_mode="inprocess"`` explicitly for local debug or CI callers
    that need to skip Ray startup.
    The legacy ``input_type`` argument constrains directory expansion and file
    validation only; extraction routing remains manifest-planned.
    """
    plan = resolve_ingest_plan(
        documents,
        profile=profile,
        input_type=input_type,
        run_mode=run_mode,
        method=method,
        dpi=dpi,
        extract_text=extract_text,
        extract_images=extract_images,
        extract_tables=extract_tables,
        extract_charts=extract_charts,
        extract_infographics=extract_infographics,
        extract_page_as_image=extract_page_as_image,
        use_page_elements=use_page_elements,
        segment_audio=segment_audio,
        audio_split_type=audio_split_type,
        audio_split_interval=audio_split_interval,
        video_extract_audio=video_extract_audio,
        video_extract_frames=video_extract_frames,
        video_frame_fps=video_frame_fps,
        video_frame_dedup=video_frame_dedup,
        video_frame_text_dedup=video_frame_text_dedup,
        video_frame_text_dedup_max_dropped_frames=video_frame_text_dedup_max_dropped_frames,
        video_av_fuse=video_av_fuse,
        caption=caption,
        caption_invoke_url=caption_invoke_url,
        caption_model_name=caption_model_name,
        caption_context_text_max_chars=caption_context_text_max_chars,
        caption_infographics=caption_infographics,
        ray_address=ray_address,
        ray_log_to_driver=ray_log_to_driver,
        lancedb_uri=lancedb_uri,
        table_name=table_name,
        overwrite=overwrite,
        page_elements_invoke_url=page_elements_invoke_url,
        ocr_invoke_url=ocr_invoke_url,
        ocr_version=ocr_version,
        ocr_lang=ocr_lang,
        graphic_elements_invoke_url=graphic_elements_invoke_url,
        table_structure_invoke_url=table_structure_invoke_url,
        table_output_format=table_output_format,
        embed_invoke_url=embed_invoke_url,
        embed_model_name=embed_model_name,
        local_ingest_embed_backend=local_ingest_embed_backend,
        pdf_extract_workers=pdf_extract_workers,
        pdf_extract_batch_size=pdf_extract_batch_size,
        pdf_extract_cpus_per_task=pdf_extract_cpus_per_task,
        page_elements_workers=page_elements_workers,
        page_elements_batch_size=page_elements_batch_size,
        page_elements_cpus_per_actor=page_elements_cpus_per_actor,
        page_elements_gpus_per_actor=page_elements_gpus_per_actor,
        ocr_workers=ocr_workers,
        ocr_batch_size=ocr_batch_size,
        ocr_cpus_per_actor=ocr_cpus_per_actor,
        ocr_gpus_per_actor=ocr_gpus_per_actor,
        table_structure_workers=table_structure_workers,
        table_structure_batch_size=table_structure_batch_size,
        table_structure_cpus_per_actor=table_structure_cpus_per_actor,
        table_structure_gpus_per_actor=table_structure_gpus_per_actor,
        embed_workers=embed_workers,
        embed_batch_size=embed_batch_size,
        embed_cpus_per_actor=embed_cpus_per_actor,
        embed_gpus_per_actor=embed_gpus_per_actor,
    )
    if dry_run:
        return plan.dry_run_data()

    initial_n_rows = None if overwrite else _count_lancedb_rows(plan.lancedb_uri, plan.table_name)
    ingestor = create_ingestor(**plan.create_kwargs).files(plan.documents)
    ingestor = ingestor.extract(plan.extract_params, **plan.extract_call_kwargs())
    if plan.caption_params is not None:
        ingestor = ingestor.caption(plan.caption_params)
    ingestor = ingestor.embed(plan.embed_params) if plan.embed_params is not None else ingestor.embed()
    result = ingestor.vdb_upload(plan.vdb_params).ingest()
    n_rows = _count_lancedb_rows(plan.lancedb_uri, plan.table_name)
    _raise_for_empty_ingest(
        documents=plan.documents,
        lancedb_uri=plan.lancedb_uri,
        table_name=plan.table_name,
        n_rows=n_rows,
        initial_n_rows=initial_n_rows,
    )
    return {
        "documents": plan.documents,
        "lancedb_uri": plan.lancedb_uri,
        "result": result,
        "table_name": plan.table_name,
        "n_rows": n_rows,
    }


def _raise_for_empty_ingest(
    *,
    documents: Sequence[str],
    lancedb_uri: str,
    table_name: str,
    n_rows: int | None,
    initial_n_rows: int | None,
) -> None:
    target = f"{lancedb_uri}/{table_name}"
    if n_rows is None:
        raise RuntimeError(
            f"retriever ingest could not verify rows in LanceDB {target} for {len(documents)} input file(s). "
            "This usually means the LanceDB table was not created or could not be read after ingestion; check "
            "the captured stage logs above, and verify NVIDIA_API_KEY/NGC_API_KEY or the configured local/remote "
            "endpoints."
        )
    if n_rows > 0 and (initial_n_rows is None or n_rows > initial_n_rows):
        return

    if initial_n_rows is not None:
        raise RuntimeError(
            f"retriever ingest did not add rows to LanceDB {target}; row count stayed at {n_rows} "
            f"for {len(documents)} input file(s). This usually means extraction or embedding failed before "
            "any rows were written; check the captured stage logs above, and verify NVIDIA_API_KEY/NGC_API_KEY "
            "or the configured local/remote endpoints."
        )

    raise RuntimeError(
        f"retriever ingest produced 0 rows in LanceDB {target} for {len(documents)} input file(s). "
        "This usually means extraction or embedding failed before any rows were written; check the captured "
        "stage logs above, and verify NVIDIA_API_KEY/NGC_API_KEY or the configured local/remote endpoints."
    )


def _count_lancedb_rows(lancedb_uri: str, table_name: str) -> int | None:
    """Return the actual row count in ``<lancedb_uri>/<table_name>`` or ``None``.

    The low-level reader is best-effort so callers can decide whether an
    unknown count is acceptable. Root ingest treats an unknown final count as a
    failure because agents need proof that rows landed.
    """
    try:
        import lancedb  # local import — keeps the CLI startup snappy

        return int(lancedb.connect(lancedb_uri).open_table(table_name).count_rows())
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        logger.debug("could not count rows in %s/%s: %s", lancedb_uri, table_name, exc)
        return None


def query_documents(
    query: str,
    *,
    top_k: int = 10,
    candidate_k: int | None = None,
    page_dedup: bool = False,
    content_types: str | Sequence[str] | None = None,
    lancedb_uri: str = DEFAULT_LANCEDB_URI,
    table_name: str = DEFAULT_TABLE_NAME,
    embed_invoke_url: str | None = None,
    embed_model_name: str | None = None,
    reranker_invoke_url: str | None = None,
    reranker_model_name: str | None = None,
    reranker_backend: str | None = None,
    rerank: bool = False,
) -> list[RetrievalHit]:
    """Run the minimal SDK query path used by the root CLI.

    Reranking is opt-in: pass ``rerank=True`` (or any of the rerank-related
    args via the CLI, which implicitly set ``rerank=True``) to enable.
    """
    embed_kwargs = _build_embed_kwargs(embed_invoke_url, embed_model_name)
    retriever_kwargs: dict[str, Any] = {
        "top_k": top_k,
        "vdb_kwargs": {"uri": lancedb_uri, "table_name": table_name},
    }
    if embed_kwargs:
        retriever_kwargs["embed_kwargs"] = embed_kwargs
    if rerank:
        rerank_kwargs = _build_rerank_kwargs(reranker_invoke_url, reranker_model_name, reranker_backend)
        retriever_kwargs["rerank"] = True
        if rerank_kwargs:
            retriever_kwargs["rerank_kwargs"] = rerank_kwargs

    retriever = Retriever(**retriever_kwargs)
    return retriever.query(
        query,
        candidate_k=candidate_k,
        page_dedup=page_dedup,
        content_types=content_types,
    )
