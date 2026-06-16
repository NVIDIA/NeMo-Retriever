from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from nemo_retriever.graph import Graph
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.ingestor.branch_extraction import normalize_ray_branch_datasets
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor
from nemo_retriever.ingest.plan import (
    IngestCaptionOptions,
    IngestExtractOptions,
    IngestMediaOptions,
    IngestPlanRequest,
    IngestSourceOptions,
    resolve_ingest_plan,
)
from nemo_retriever.ingestor.manifest import (
    build_input_manifest,
    plan_extraction_branches,
    resolve_branch_extraction_inputs,
)
from nemo_retriever.common.params import ASRParams


def _resolve_plan(
    documents: list[str],
    *,
    profile: str = "auto",
    input_type: str = "auto",
    extract: IngestExtractOptions | None = None,
    media: IngestMediaOptions | None = None,
    caption: IngestCaptionOptions | None = None,
):
    return resolve_ingest_plan(
        IngestPlanRequest(
            source=IngestSourceOptions(
                documents=documents,
                profile=profile,  # type: ignore[arg-type]
                input_type=input_type,  # type: ignore[arg-type]
            ),
            extract=extract or IngestExtractOptions(),
            media=media or IngestMediaOptions(),
            caption=caption or IngestCaptionOptions(),
        )
    )


class _TagOperator(AbstractOperator):
    def __init__(self, *, tag: str) -> None:
        super().__init__(tag=tag)
        self.tag = tag

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return pd.DataFrame(
            {
                "path": list(data["path"]),
                f"{self.tag}_value": [self.tag] * len(data),
            }
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class _PostOperator(AbstractOperator):
    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data.assign(post_extract=True)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def _graph_with(operator: AbstractOperator) -> Graph:
    return Graph() >> operator


def test_manifest_planner_pdf_doc_share_dedicated_pdf_branch(tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    pptx = tmp_path / "deck.pptx"
    pdf.write_bytes(b"pdf")
    pptx.write_bytes(b"pptx")

    branches = plan_extraction_branches(build_input_manifest([str(pdf), str(pptx)]))

    assert [(branch.family, branch.extraction_mode, branch.input_paths) for branch in branches] == [
        ("pdf", "pdf", (str(pdf), str(pptx))),
    ]


def test_manifest_planner_mixed_inputs_use_stable_family_order(tmp_path) -> None:
    text = tmp_path / "notes.txt"
    image = tmp_path / "scan.png"
    pdf = tmp_path / "manual.pdf"
    text.write_text("notes", encoding="utf-8")
    image.write_bytes(b"png")
    pdf.write_bytes(b"pdf")

    branches = plan_extraction_branches(build_input_manifest([str(text), str(image), str(pdf)]))

    assert [branch.family for branch in branches] == ["pdf", "image", "txt"]


def test_manifest_branch_specs_resolve_default_params(monkeypatch, tmp_path) -> None:
    audio = tmp_path / "clip.wav"
    video = tmp_path / "scene.mp4"
    audio.write_bytes(b"audio")
    video.write_bytes(b"video")
    monkeypatch.setattr("nemo_retriever.ingestor.manifest._default_asr_params", lambda: ASRParams(segment_audio=False))

    branches = plan_extraction_branches(build_input_manifest([str(video), str(audio)]))
    by_family = {branch.family: branch for branch in branches}

    audio_inputs = resolve_branch_extraction_inputs(
        by_family["audio"],
        extract_params=None,
        text_params=None,
        html_params=None,
        audio_chunk_params=None,
        asr_params=None,
        video_frame_params=None,
        video_text_dedup_params=None,
        av_fuse_params=None,
    )
    video_inputs = resolve_branch_extraction_inputs(
        by_family["video"],
        extract_params=None,
        text_params=None,
        html_params=None,
        audio_chunk_params=None,
        asr_params=None,
        video_frame_params=None,
        video_text_dedup_params=None,
        av_fuse_params=None,
    )

    assert audio_inputs.extraction_mode == "audio"
    assert audio_inputs.audio_chunk_params.split_interval == 500000
    assert audio_inputs.asr_params.segment_audio is False
    assert video_inputs.extraction_mode == "auto"
    assert video_inputs.extract_params is not None
    assert video_inputs.audio_chunk_params.enabled is True
    assert video_inputs.video_frame_params.fps == 0.5
    assert video_inputs.video_frame_params.dedup is True
    assert video_inputs.video_text_dedup_params.enabled is True
    assert video_inputs.av_fuse_params.enabled is True


def test_manifest_planner_rejects_unsupported_concrete_extensions(tmp_path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"unknown")

    with pytest.raises(ValueError, match="payload.bin"):
        plan_extraction_branches(build_input_manifest([str(payload)]))


def test_manifest_planner_empty_glob_does_not_invent_modal_branches(tmp_path) -> None:
    branches = plan_extraction_branches(build_input_manifest([str(tmp_path / "*.wav")]))

    assert [(branch.family, branch.input_paths) for branch in branches] == [("pdf", (str(tmp_path / "*.wav"),))]


def test_ingest_plan_auto_profile_preserves_manifest_defaults(tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"pdf")

    plan = _resolve_plan([str(pdf)], profile="auto")

    assert plan.profile == "auto"
    assert [branch.family for branch in plan.branches] == ["pdf"]
    assert plan.extract_params.method == "pdfium"
    assert plan.extract_params.dpi == 200
    assert plan.extract_params.extract_images is True
    assert plan.extract_params.extract_tables is True
    assert plan.extract_params.extract_charts is True
    assert plan.extract_params.extract_infographics is False
    assert plan.extract_params.extract_page_as_image is True
    assert plan.extract_params.use_page_elements is True
    assert plan.create_kwargs == {"run_mode": "inprocess"}


def test_ingest_plan_fast_text_profile_is_pdf_text_only(tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"pdf")

    plan = _resolve_plan([str(pdf)], profile="fast-text")

    assert plan.extract_params.method == "pdfium"
    assert plan.extract_params.extract_text is True
    assert plan.extract_params.extract_images is False
    assert plan.extract_params.extract_tables is False
    assert plan.extract_params.extract_charts is False
    assert plan.extract_params.extract_infographics is False
    assert plan.extract_params.extract_page_as_image is False
    assert plan.extract_params.use_page_elements is False


def test_ingest_plan_fast_text_allows_extract_images_override(tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"pdf")

    plan = _resolve_plan([str(pdf)], profile="fast-text", extract=IngestExtractOptions(extract_images=True))

    assert plan.extract_params.extract_images is True
    assert plan.extract_params.extract_tables is False
    assert plan.extract_params.extract_charts is False
    assert plan.extract_params.extract_infographics is False
    assert plan.extract_params.use_page_elements is False


def test_ingest_plan_caption_is_absent_by_default_and_optional(tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"pdf")

    default_plan = _resolve_plan([str(pdf)])
    caption_plan = _resolve_plan(
        [str(pdf)],
        caption=IngestCaptionOptions(
            enabled=True,
            caption_invoke_url="http://vlm:8000/v1/chat/completions",
            caption_model_name="nvidia/test-vlm",
            caption_context_text_max_chars=256,
            caption_infographics=True,
        ),
    )

    assert default_plan.caption_params is None
    assert caption_plan.caption_params is not None
    assert caption_plan.caption_params.endpoint_url == "http://vlm:8000/v1/chat/completions"
    assert caption_plan.caption_params.model_name == "nvidia/test-vlm"
    assert caption_plan.caption_params.context_text_max_chars == 256
    assert caption_plan.caption_params.caption_infographics is True


def test_ingest_plan_caption_options_require_caption(tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"pdf")

    with pytest.raises(ValueError, match="Caption options require --caption"):
        _resolve_plan(
            [str(pdf)],
            caption=IngestCaptionOptions(caption_invoke_url="http://vlm:8000/v1/chat/completions"),
        )


def test_ingest_plan_auto_builds_audio_params(monkeypatch, tmp_path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        "nemo_retriever.operators.extract.audio.asr_actor.asr_params_from_env", lambda: ASRParams(segment_audio=False)
    )

    plan = _resolve_plan([str(audio)], media=IngestMediaOptions(segment_audio=True))

    assert [branch.family for branch in plan.branches] == ["audio"]
    assert plan.audio_chunk_params is not None
    assert plan.audio_chunk_params.split_type == "size"
    assert plan.audio_chunk_params.split_interval == 500000
    assert plan.asr_params is not None
    assert plan.asr_params.segment_audio is True
    assert plan.video_frame_params is None


def test_ingest_plan_preserves_env_asr_segment_audio_when_cli_unset(monkeypatch, tmp_path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        "nemo_retriever.operators.extract.audio.asr_actor.asr_params_from_env", lambda: ASRParams(segment_audio=True)
    )

    plan = _resolve_plan([str(audio)])

    assert plan.asr_params is not None
    assert plan.asr_params.segment_audio is True


def test_ingest_plan_auto_builds_video_params(monkeypatch, tmp_path) -> None:
    video = tmp_path / "scene.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "nemo_retriever.operators.extract.audio.asr_actor.asr_params_from_env", lambda: ASRParams(segment_audio=False)
    )

    plan = _resolve_plan([str(video)])

    assert [branch.family for branch in plan.branches] == ["video"]
    assert plan.extract_params.method == "pdfium"
    assert plan.audio_chunk_params is not None
    assert plan.audio_chunk_params.enabled is True
    assert plan.video_frame_params is not None
    assert plan.video_frame_params.fps == 0.5
    assert plan.video_frame_params.dedup is True
    assert plan.video_text_dedup_params is not None
    assert plan.video_text_dedup_params.max_dropped_frames == 2
    assert plan.av_fuse_params is not None
    assert plan.av_fuse_params.enabled is True


def test_ingest_plan_auto_allows_mixed_supported_branches(monkeypatch, tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    audio = tmp_path / "clip.wav"
    video = tmp_path / "scene.mp4"
    pdf.write_bytes(b"pdf")
    audio.write_bytes(b"audio")
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "nemo_retriever.operators.extract.audio.asr_actor.asr_params_from_env", lambda: ASRParams(segment_audio=False)
    )

    plan = _resolve_plan([str(pdf), str(audio), str(video)])

    assert [branch.family for branch in plan.branches] == ["pdf", "audio", "video"]
    assert plan.extract_params.method == "pdfium"
    assert plan.audio_chunk_params is not None
    assert plan.video_frame_params is not None


def test_ingest_plan_fast_text_validates_input_family(tmp_path) -> None:
    path = tmp_path / "scan.png"
    path.write_bytes(b"data")

    with pytest.raises(ValueError, match="--profile fast-text only supports PDF/document inputs"):
        _resolve_plan([str(path)], profile="fast-text")


@pytest.mark.parametrize("profile", ["ocr", "audio", "video", "multimodal"])
def test_ingest_plan_rejects_removed_profiles(profile: str, tmp_path) -> None:
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"data")

    with pytest.raises(ValueError, match="profile must be one of auto, fast-text"):
        _resolve_plan([str(path)], profile=profile)


def test_explicit_extraction_mode_bypasses_manifest_planning(tmp_path) -> None:
    image = tmp_path / "scan.png"
    image.write_bytes(b"png")
    ingestor = GraphIngestor(run_mode="inprocess").files([str(image)]).extract(extraction_mode="auto")

    assert ingestor._plan_default_extraction_branches() is None
    assert ingestor._resolve_effective_extraction_inputs().extraction_mode == "auto"


def test_inprocess_branch_execution_unions_schemas_and_runs_post_once(monkeypatch, tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    image = tmp_path / "scan.png"
    text = tmp_path / "notes.txt"
    pdf.write_bytes(b"pdf")
    image.write_bytes(b"png")
    text.write_text("notes", encoding="utf-8")
    extraction_calls: list[dict[str, Any]] = []
    post_calls: list[dict[str, Any]] = []

    def fake_build_graph(**kwargs: Any) -> Graph:
        extraction_calls.append(kwargs)
        return _graph_with(_TagOperator(tag=kwargs["extraction_mode"]))

    def fake_post_graph(**kwargs: Any) -> Graph:
        post_calls.append(kwargs)
        return _graph_with(_PostOperator())

    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.build_graph", fake_build_graph)
    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.build_post_extract_graph", fake_post_graph)

    result = (
        GraphIngestor(run_mode="inprocess", show_progress=False)
        .files([str(text), str(image), str(pdf)])
        .extract()
        .embed()
        .ingest()
    )

    assert [call["extraction_mode"] for call in extraction_calls] == ["pdf", "image", "text"]
    assert all(call.get("embed_params") is None for call in extraction_calls)
    assert len(post_calls) == 1
    assert post_calls[0]["embed_params"] is not None
    assert post_calls[0]["reshape_content_before_embed"] is True
    assert set(result.columns) == {"path", "pdf_value", "image_value", "text_value", "post_extract"}
    assert result["post_extract"].tolist() == [True, True, True]


def test_text_html_branch_execution_skips_content_reshape_before_embed(monkeypatch, tmp_path) -> None:
    text = tmp_path / "notes.txt"
    html = tmp_path / "index.html"
    text.write_text("notes", encoding="utf-8")
    html.write_text("<html></html>", encoding="utf-8")
    post_calls: list[dict[str, Any]] = []

    def fake_build_graph(**kwargs: Any) -> Graph:
        return _graph_with(_TagOperator(tag=kwargs["extraction_mode"]))

    def fake_post_graph(**kwargs: Any) -> Graph:
        post_calls.append(kwargs)
        return _graph_with(_PostOperator())

    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.build_graph", fake_build_graph)
    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.build_post_extract_graph", fake_post_graph)

    GraphIngestor(run_mode="inprocess", show_progress=False).files([str(text), str(html)]).extract().embed().ingest()

    assert post_calls[0]["reshape_content_before_embed"] is False


class _FakeDataset:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.unioned: list[_FakeDataset] = []
        self.normalized_columns: tuple[str, ...] | None = None

    def schema(self) -> Any:
        return SimpleNamespace(names=self.columns)

    def map_batches(self, *_args: Any, **kwargs: Any) -> "_FakeDataset":
        self.normalized_columns = kwargs["fn_kwargs"]["columns"]
        return self

    def union(self, other: "_FakeDataset") -> "_FakeDataset":
        self.unioned.append(other)
        return self


class _LazySchemaDataset:
    def __init__(self) -> None:
        self.map_batches_called = False

    def schema(self, *, fetch_if_missing: bool = True) -> None:
        assert fetch_if_missing is False
        return None

    def map_batches(self, *_args: Any, **_kwargs: Any) -> "_LazySchemaDataset":
        self.map_batches_called = True
        return self


def test_ray_schema_normalization_does_not_trigger_lazy_schema_fetch() -> None:
    datasets = [_LazySchemaDataset(), _LazySchemaDataset()]

    normalized = normalize_ray_branch_datasets(datasets)

    assert normalized == datasets
    assert all(not dataset.map_batches_called for dataset in datasets)


def test_batch_branch_execution_uses_dataset_union(monkeypatch, tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    image = tmp_path / "scan.png"
    pdf.write_bytes(b"pdf")
    image.write_bytes(b"png")
    datasets = [_FakeDataset(["path", "pdf_value"]), _FakeDataset(["path", "image_value"])]
    executor_calls: list[dict[str, Any]] = []

    class FakeCluster:
        def available_gpu_count(self) -> int:
            return 0

        def total_cpu_count(self) -> int:
            return 64

    class FakeExecutor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def build_dataset(self, data: Any, **kwargs: Any) -> Any:
            executor_calls.append({"method": "build_dataset", "data": data})
            return datasets.pop(0)

        def ingest(self, data: Any, **kwargs: Any) -> Any:
            executor_calls.append({"method": "ingest", "data": data})
            return pd.DataFrame({"done": [True]})

    monkeypatch.setattr(GraphIngestor, "_ensure_batch_runtime", lambda self: (None, FakeCluster()))
    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.RayDataExecutor", FakeExecutor)
    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.build_graph", lambda **_kwargs: Graph())
    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.build_post_extract_graph", lambda **_kwargs: Graph())

    result = GraphIngestor(run_mode="batch").files([str(pdf), str(image)]).extract().ingest()

    assert [call["method"] for call in executor_calls] == ["build_dataset", "build_dataset", "ingest"]
    combined = executor_calls[2]["data"]
    assert isinstance(combined, _FakeDataset)
    assert len(combined.unioned) == 1
    assert combined.normalized_columns == ("path", "pdf_value", "image_value")
    assert result["done"].tolist() == [True]
