"""Low-level ingestor execution plan used by graph executors and the fluent API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nemo_retriever.common.params import ASRParams
from nemo_retriever.common.params import AudioChunkParams
from nemo_retriever.common.params import CaptionParams
from nemo_retriever.common.params import DedupParams
from nemo_retriever.common.params import EmbedParams
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.common.params import HtmlChunkParams
from nemo_retriever.common.params import TextChunkParams
from nemo_retriever.common.params import VdbUploadParams


def resolve_effective_dedup_params(
    configured_params: DedupParams | None,
    *,
    caption_enabled: bool,
    image_only: bool,
) -> DedupParams | None:
    """Resolve caption-triggered image dedup without changing configured state.

    Explicit parameters are authoritative, including the all-disabled form
    used to suppress automatic dedup. When dedup is unspecified, captioning
    enables the default policy for document inputs but not standalone images.

    Parameters
    ----------
    configured_params
        Explicit deduplication parameters, or ``None`` when deduplication was
        not configured. Explicit parameters are returned unchanged.
    caption_enabled
        Whether image captioning is configured for the ingestion request.
    image_only
        Whether the request contains only standalone image inputs.

    Returns
    -------
    DedupParams | None
        The explicit parameters when provided, default parameters for a
        captioned non-image request, or ``None`` when deduplication should not
        run.
    """

    if configured_params is not None:
        return configured_params
    if caption_enabled and not image_only:
        return DedupParams()
    return None


def dedup_params_enabled(params: DedupParams | None) -> bool:
    """Return whether at least one image-deduplication pass is enabled.

    Parameters
    ----------
    params
        Effective deduplication parameters, or ``None`` when no deduplication
        stage is configured.

    Returns
    -------
    bool
        ``True`` when content-hash or bounding-box-overlap deduplication is
        enabled; otherwise ``False``.
    """

    return params is not None and (params.content_hash or params.bbox_iou)


@dataclass(frozen=True)
class PlannedStage:
    """A normalized transform stage ready for executor-specific translation."""

    name: str
    params: Any


@dataclass(frozen=True)
class PlannedSink:
    """A normalized output sink that runs after transform stages complete."""

    name: str
    params: Any


@dataclass(frozen=True)
class IngestExecutionPlan:
    """Ordered execution contract derived from the fluent ingestor API."""

    extraction_mode: str
    extract_params: ExtractParams | None = None
    text_params: TextChunkParams | None = None
    html_params: HtmlChunkParams | None = None
    audio_chunk_params: AudioChunkParams | None = None
    asr_params: ASRParams | None = None
    split_config: dict[str, Any] | None = None
    stages: tuple[PlannedStage, ...] = ()
    sinks: tuple[PlannedSink, ...] = ()

    def has_extraction(self) -> bool:
        return any(
            (
                self.extract_params is not None,
                self.text_params is not None,
                self.html_params is not None,
                self.audio_chunk_params is not None,
            )
        )


@dataclass
class BaseIngestPlan:
    """Shared ingestion plan state used by multiple execution frontends."""

    extraction_mode: str = "pdf"
    extract_params: ExtractParams | None = None
    text_params: TextChunkParams | None = None
    html_params: HtmlChunkParams | None = None
    audio_chunk_params: AudioChunkParams | None = None
    asr_params: ASRParams | None = None
    split_config: dict[str, Any] | None = None
    dedup_params: DedupParams | None = None
    caption_params: CaptionParams | None = None
    embed_params: EmbedParams | None = None
    vdb_upload_params: VdbUploadParams | None = None
    stage_order: list[str] = field(default_factory=list)
    sink_order: list[str] = field(default_factory=list)

    def set_extraction(
        self,
        *,
        mode: str,
        extract_params: ExtractParams | None = None,
        text_params: TextChunkParams | None = None,
        html_params: HtmlChunkParams | None = None,
        audio_chunk_params: AudioChunkParams | None = None,
        asr_params: ASRParams | None = None,
    ) -> None:
        self.extraction_mode = mode
        self.extract_params = extract_params
        self.text_params = text_params
        self.html_params = html_params
        self.audio_chunk_params = audio_chunk_params
        self.asr_params = asr_params

    def has_extraction(self) -> bool:
        return any(
            (
                self.extract_params is not None,
                self.text_params is not None,
                self.html_params is not None,
                self.audio_chunk_params is not None,
            )
        )

    def record_stage(self, stage_name: str) -> None:
        if stage_name in self.stage_order:
            self.stage_order = [stage for stage in self.stage_order if stage != stage_name]
        self.stage_order.append(stage_name)

    def record_sink(self, sink_name: str) -> None:
        if sink_name in self.sink_order:
            self.sink_order = [sink for sink in self.sink_order if sink != sink_name]
        self.sink_order.append(sink_name)

    def build_execution_plan(self) -> IngestExecutionPlan:
        """Collapse fluent plan state into an ordered execution-ready contract.

        Splitting is configured via ``split_config`` on the returned plan, not
        as a :class:`PlannedStage`. The ordered ``stages`` tuple covers only the
        post-extraction transforms (``dedup``, ``caption``, ``embed``).
        """

        stage_params = {
            "dedup": self.dedup_params,
            "caption": self.caption_params,
            "embed": self.embed_params,
        }
        sink_params = {
            "vdb_upload": self.vdb_upload_params,
        }

        stages = tuple(
            PlannedStage(name=stage_name, params=stage_params[stage_name])
            for stage_name in self.stage_order
            if stage_name in stage_params and stage_params[stage_name] is not None
        )
        sinks = tuple(
            PlannedSink(name=sink_name, params=sink_params[sink_name])
            for sink_name in self.sink_order
            if sink_name in sink_params and sink_params[sink_name] is not None
        )

        return IngestExecutionPlan(
            extraction_mode=self.extraction_mode,
            extract_params=self.extract_params,
            text_params=self.text_params,
            html_params=self.html_params,
            audio_chunk_params=self.audio_chunk_params,
            asr_params=self.asr_params,
            split_config=self.split_config,
            stages=stages,
            sinks=sinks,
        )
