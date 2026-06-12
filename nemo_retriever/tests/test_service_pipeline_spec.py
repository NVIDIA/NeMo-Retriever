# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 unit tests for the per-request PipelineSpec wire format.

Covers three layers:

* the client-side ``ServiceIngestor`` fluent builders translate to the
  right ``_pipeline_spec`` shape;
* the server-side ``validate_pipeline_spec`` policy accepts well-formed
  specs and rejects trust-sensitive overrides; and
* the worker-side merge preserves server-owned keys regardless of the
  client spec.
"""

from __future__ import annotations

import pytest

from nemo_retriever.common.params import DedupParams, EmbedParams, ExtractParams
from nemo_retriever.service.config import PipelineOverridesConfig
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.common.policy import PolicyError, validate_pipeline_spec
from nemo_retriever.service.services.pipeline_executor import (
    _build_graph_ingestor_from_spec,
    _merge_server_owned,
    _request_needs_asr_params,
    _resolve_service_extraction_mode,
    _run_pipeline_in_process,
    _TRUST_OWNED_EMBED_KEYS,
    _TRUST_OWNED_EXTRACT_KEYS,
)
from nemo_retriever.service.utils.file_type import infer_extraction_mode_from_filename
from nemo_retriever.service.service_ingestor import ServiceIngestor


class _TinyTokenizer:
    def __init__(self) -> None:
        self._text = ""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        self._text = text
        return list(range(len(text)))

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return "".join(self._text[i] for i in ids)


@pytest.fixture(autouse=True)
def _no_remote_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # _ParamsModel auto-resolves unset *api_key fields from these env vars,
    # which would then trip ServiceIngestor's server-owned-key guard.
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)


# ----------------------------------------------------------------------
# Client side: fluent → spec dict
# ----------------------------------------------------------------------


def test_serviceingestor_empty_spec_is_none() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    assert ing._pipeline_payload() is None


def test_extract_mode_only_omits_extract_params() -> None:
    """``.extract(extraction_mode='pdf')`` must not send client model defaults."""
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract(extraction_mode="pdf").all_tasks()
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["extraction_mode"] == "pdf"
    assert payload["stage_order"] == ["extract", "dedup", "embed"]
    assert "extract_params" not in payload


def test_extract_records_stage_and_params() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract(ExtractParams(extract_text=False, dpi=300))
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["extraction_mode"] == "auto"
    assert payload["stage_order"] == ["extract"]
    assert payload["extract_params"]["extract_text"] is False
    assert payload["extract_params"]["dpi"] == 300
    assert "page_elements_invoke_url" not in payload["extract_params"]
    assert "api_key" not in payload["extract_params"]
    assert "use_page_elements" not in payload["extract_params"]
    assert "batch_tuning" not in payload["extract_params"]


def test_extract_params_passes_default_policy_allowlist() -> None:
    """Regression: public ExtractParams must not send model defaults to nrl-service."""
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract(
        params=ExtractParams(
            extract_text=True,
            extract_images=False,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
        )
    )
    spec = PipelineSpec.model_validate(ing._pipeline_spec)
    validate_pipeline_spec(spec, PipelineOverridesConfig().to_policy())
    assert set(spec.extract_params) <= {
        "extract_text",
        "extract_images",
        "extract_tables",
        "extract_charts",
        "extract_infographics",
        "table_output_format",
    }


def test_extract_image_files_sets_image_mode() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract_image_files()
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["extraction_mode"] == "image"


def test_dedup_and_embed_add_stage_order() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.extract().dedup(DedupParams(iou_threshold=0.7)).embed(EmbedParams(inference_batch_size=64))
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["stage_order"] == ["extract", "dedup", "embed"]
    assert payload["dedup_params"]["iou_threshold"] == 0.7
    assert payload["embed_params"]["inference_batch_size"] == 64


def test_pdf_split_config_round_trips_via_spec() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.pdf_split_config(pages_per_chunk=16)
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["pdf_split"]["pages_per_chunk"] == 16


def test_split_method_records_split_config() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.split({"pdf": {"max_tokens": 512, "overlap_tokens": 32}})
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["split_config"] == {"pdf": {"max_tokens": 512, "overlap_tokens": 32}}


def test_all_tasks_seeds_canonical_stage_order() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    ing.all_tasks()
    payload = ing._pipeline_payload()
    assert payload is not None
    assert payload["stage_order"] == ["extract", "dedup", "embed"]


def test_client_rejects_server_owned_keys() -> None:
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(ValueError, match="server-owned"):
        ing.extract(ExtractParams(page_elements_invoke_url="http://attacker/"))


def test_future_phase_methods_raise_informative_error() -> None:
    """Methods deferred to follow-up phases still produce a clear error.

    ``store`` / ``webhook`` / ``vdb_upload`` (sinks) moved out in Phase 2,
    ``save_to_disk`` in Phase 3, ``caption`` in Phase 4, ``vdb_upload``
    sidecar metadata in Phase 6. ``udf`` is the only remaining stub.
    """
    ing = ServiceIngestor(base_url="http://example:7670")
    with pytest.raises(NotImplementedError, match="Phase 5"):
        ing.udf("noop")


# ----------------------------------------------------------------------
# Policy: accept / reject
# ----------------------------------------------------------------------


def test_validate_returns_none_for_empty_spec() -> None:
    policy = PipelineOverridesConfig().to_policy()
    assert validate_pipeline_spec(None, policy) is None
    assert validate_pipeline_spec(PipelineSpec(), policy) is None


def test_validate_accepts_default_allowlist() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(
        extract_params={"extract_text": False, "dpi": 300},
        embed_params={"inference_batch_size": 64},
        dedup_params={"iou_threshold": 0.5},
        stage_order=["extract", "dedup", "embed"],
    )
    out = validate_pipeline_spec(spec, policy)
    assert out is spec  # returned by reference when unchanged


def test_validate_rejects_endpoint_url() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(extract_params={"page_elements_invoke_url": "http://attacker/"})
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, policy)
    assert exc.value.status_code == 403
    assert "trust-sensitive" in exc.value.detail


def test_validate_rejects_api_key() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(embed_params={"api_key": "leaked-token"})
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, policy)
    assert exc.value.status_code == 403


def test_validate_rejects_unallowed_key_in_allow_list_mode() -> None:
    policy = PipelineOverridesConfig().to_policy()
    spec = PipelineSpec(extract_params={"not_a_real_field": True})
    with pytest.raises(PolicyError):
        validate_pipeline_spec(spec, policy)


def test_validate_allows_extra_key_when_operator_widens() -> None:
    cfg = PipelineOverridesConfig(extra_extract_keys=["weird_dev_flag"])
    spec = PipelineSpec(extract_params={"weird_dev_flag": True, "dpi": 300})
    out = validate_pipeline_spec(spec, cfg.to_policy())
    assert out is spec


def test_validate_reject_mode_blocks_any_override() -> None:
    cfg = PipelineOverridesConfig(mode="reject")
    spec = PipelineSpec(extract_params={"dpi": 300})
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, cfg.to_policy())
    assert exc.value.status_code == 403


def test_validate_allow_all_mode_still_blocks_endpoints() -> None:
    cfg = PipelineOverridesConfig(mode="allow_all")
    policy = cfg.to_policy()
    # "shape" keys pass freely:
    spec = PipelineSpec(extract_params={"any_dev_only_flag": True})
    assert validate_pipeline_spec(spec, policy) is spec
    # but the denylist still bites:
    spec2 = PipelineSpec(extract_params={"ocr_invoke_url": "http://x/"})
    with pytest.raises(PolicyError):
        validate_pipeline_spec(spec2, policy)


def test_validate_rejects_caption_without_endpoint() -> None:
    """Without an operator-configured caption endpoint, the stage is forbidden."""
    cfg = PipelineOverridesConfig()
    policy = cfg.to_policy(caption_enabled=False)
    spec = PipelineSpec(caption_params={"prompt": "Describe"})
    with pytest.raises(PolicyError) as exc:
        validate_pipeline_spec(spec, policy)
    assert exc.value.status_code == 403


# ----------------------------------------------------------------------
# Worker merge: server-owned keys always win
# ----------------------------------------------------------------------


def test_merge_preserves_server_extract_endpoints() -> None:
    base = {
        "page_elements_invoke_url": "http://server/page_elements",
        "ocr_invoke_url": "http://server/ocr",
        "api_key": "server-token",
        "dpi": 150,
    }
    override = {"dpi": 600, "page_elements_invoke_url": "http://attacker/"}
    merged = _merge_server_owned(base, override, _TRUST_OWNED_EXTRACT_KEYS)
    assert merged["dpi"] == 600
    assert merged["page_elements_invoke_url"] == "http://server/page_elements"
    assert merged["ocr_invoke_url"] == "http://server/ocr"
    assert merged["api_key"] == "server-token"


def test_merge_preserves_server_embed_endpoints() -> None:
    base = {"embed_invoke_url": "http://server/embed", "api_key": "k"}
    override = {"embed_invoke_url": "http://attacker/", "inference_batch_size": 8}
    merged = _merge_server_owned(base, override, _TRUST_OWNED_EMBED_KEYS)
    assert merged["embed_invoke_url"] == "http://server/embed"
    assert merged["api_key"] == "k"
    assert merged["inference_batch_size"] == 8


def test_build_graph_ingestor_applies_spec_extraction_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``extraction_mode='image'`` calls extract_image_files on the GraphIngestor."""
    base_extract = {"page_elements_invoke_url": "http://server/page_elements"}
    spec = {"extraction_mode": "image", "extract_params": {"dpi": 300}, "stage_order": ["extract"]}

    ingestor, mode, has_vdb = _build_graph_ingestor_from_spec(
        "stub.png",
        b"\x89PNG\r\n",
        base_extract,
        None,
        spec,
    )
    assert mode == "image"
    assert has_vdb is False
    assert ingestor._extraction_mode == "image"
    assert ingestor._extract_params is not None
    assert ingestor._extract_params.dpi == 300
    assert ingestor._extract_params.page_elements_invoke_url == "http://server/page_elements"


# ----------------------------------------------------------------------
# ASR-params gating
# ----------------------------------------------------------------------
#
# Regression coverage for the bug where the worker's ``ASRParams`` (built
# from ``serviceConfig.nimEndpoints.audioGrpcEndpoint``) leaked into every
# per-request ingestor and forced PDF uploads through the audio-only
# graph, crashing inside ``MediaChunkActor`` with
# ``RuntimeError: MediaChunkActor requires media dependencies; missing:
# ffmpeg, ffprobe``.


@pytest.mark.parametrize(
    ("extraction_mode", "filename", "expected"),
    [
        # Explicit audio/video intent: always attach.
        ("audio", "lecture.mp3", True),
        ("audio", "recording.wav", True),
        ("video", "talk.mp4", True),
        ("AUDIO", "recording.WAV", True),
        # auto + media extension: attach so MultiTypeExtractOperator can
        # dispatch the audio rows.
        ("auto", "lecture.mp3", True),
        ("auto", "talk.mp4", True),
        ("auto", "podcast.m4a", True),
        ("auto", "clip.mov", True),
        # auto + non-media extension: DO NOT attach. This is the PDF bug.
        ("auto", "report.pdf", False),
        ("auto", "scan.docx", False),
        ("auto", "spec.pptx", False),
        ("auto", "diagram.png", False),
        ("auto", "page.html", False),
        ("auto", "notes.txt", False),
        # Explicit non-media modes: never attach regardless of filename.
        ("pdf", "report.pdf", False),
        ("pdf", "weird.mp3", False),
        ("image", "diagram.png", False),
        ("text", "notes.txt", False),
        ("html", "page.html", False),
        # Unknown extension under auto: be conservative, don't attach.
        ("auto", "unknown.xyz", False),
        ("auto", "no_extension", False),
        # Missing/empty mode: same as unknown — don't attach.
        ("", "report.pdf", False),
        (None, "report.pdf", False),
    ],
)
def test_request_needs_asr_params(extraction_mode: str | None, filename: str, expected: bool) -> None:
    assert _request_needs_asr_params(extraction_mode, filename) is expected


def test_build_graph_ingestor_does_not_attach_asr_params_for_pdf_upload() -> None:
    """Regression: a worker with ``base_asr`` configured must not pin the
    cluster-wide ASR params onto PDF ingest requests.

    Before the fix the worker unconditionally executed
    ``ingestor._asr_params = asr_params`` whenever ``base_asr`` was
    truthy, which forced :func:`build_graph` into the audio-only branch
    and crashed inside :class:`MediaChunkActor` when ffmpeg was absent.
    """
    base_extract: dict[str, object] = {}
    base_asr = {"audio_endpoints": ["audio:50051", None]}
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}
    ingestor, mode, _ = _build_graph_ingestor_from_spec(
        "report.pdf",
        b"%PDF-1.4 stub",
        base_extract,
        None,
        spec,
        base_asr=base_asr,
    )

    assert mode == "pdf"
    assert (
        ingestor._asr_params is None
    ), f"PDF ingestion must not carry worker-wide ASR params. Got: {ingestor._asr_params!r}"


def test_build_graph_ingestor_attaches_asr_params_for_audio_upload() -> None:
    """A genuine audio upload under ``extraction_mode='auto'`` must still
    carry the ASR params so MultiTypeExtractOperator can dispatch ASR.
    """
    base_extract: dict[str, object] = {}
    base_asr = {"audio_endpoints": ["audio:50051", None]}
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}

    ingestor, _, _ = _build_graph_ingestor_from_spec(
        "lecture.mp3",
        b"ID3\x03",
        base_extract,
        None,
        spec,
        base_asr=base_asr,
    )

    assert ingestor._asr_params is not None
    assert tuple(ingestor._asr_params.audio_endpoints) == ("audio:50051", None)


def test_build_graph_ingestor_attaches_asr_params_for_explicit_audio_mode() -> None:
    """``extraction_mode='audio'`` must always attach the worker ASR params."""
    base_extract: dict[str, object] = {}
    base_asr = {"audio_endpoints": ["audio:50051", None]}
    spec = {"extraction_mode": "audio", "stage_order": ["extract"]}

    ingestor, mode, _ = _build_graph_ingestor_from_spec(
        # Filename without a media extension — explicit mode wins.
        "stream.bin",
        b"binary",
        base_extract,
        None,
        spec,
        base_asr=base_asr,
    )

    assert mode == "audio"
    assert ingestor._asr_params is not None


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("notes.txt", "text"),
        ("page.html", "html"),
        ("report.pdf", "pdf"),
        ("diagram.png", "image"),
        ("clip.mp4", "video"),
        ("unknown.xyz", None),
    ],
)
def test_infer_extraction_mode_from_filename(filename: str, expected: str | None) -> None:
    assert infer_extraction_mode_from_filename(filename) == expected


@pytest.mark.parametrize(
    ("extraction_mode", "filename", "resolved"),
    [
        ("auto", "notes.txt", "text"),
        ("auto", "page.html", "html"),
        ("auto", "report.pdf", "pdf"),
        ("pdf", "notes.txt", "pdf"),
        ("text", "page.html", "text"),
    ],
)
def test_resolve_service_extraction_mode(extraction_mode: str, filename: str, resolved: str) -> None:
    assert _resolve_service_extraction_mode(extraction_mode, filename) == resolved


def test_build_graph_ingestor_uses_typed_txt_html_shortcuts() -> None:
    base_extract: dict[str, object] = {}
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}

    txt_ingestor, txt_mode, _ = _build_graph_ingestor_from_spec(
        "notes.txt",
        b"The quick brown fox",
        base_extract,
        None,
        spec,
    )
    assert txt_mode == "text"
    assert txt_ingestor._extraction_mode == "text"
    assert txt_ingestor._text_params is not None

    html_ingestor, html_mode, _ = _build_graph_ingestor_from_spec(
        "page.html",
        b"<html><body><h1>Hi</h1></body></html>",
        base_extract,
        None,
        spec,
    )
    assert html_mode == "html"
    assert html_ingestor._extraction_mode == "html"
    assert html_ingestor._html_params is not None


def test_run_pipeline_in_process_rejects_empty_text_like_output() -> None:
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}
    with pytest.raises(ValueError, match="Extraction produced no rows"):
        _run_pipeline_in_process("empty.txt", b"", {}, None, None, spec)


def test_run_pipeline_in_process_html_txt_produce_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}
    monkeypatch.setattr(
        "nemo_retriever.common.modality.html.convert._get_txt_tokenizer", lambda *_, **__: _TinyTokenizer()
    )
    monkeypatch.setattr("nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *_, **__: _TinyTokenizer())
    html_rows, _, _ = _run_pipeline_in_process(
        "page.html",
        b"<html><body><h1>Title</h1><p>body</p></body></html>",
        {},
        None,
        None,
        spec,
    )
    txt_rows, _, _ = _run_pipeline_in_process(
        "notes.txt",
        b"Line one\nLine two\n",
        {},
        None,
        None,
        spec,
    )
    assert html_rows >= 1
    assert txt_rows >= 1


def test_build_graph_ingestor_omits_asr_params_when_worker_unconfigured() -> None:
    """When the worker has no ASR endpoint, nothing should be attached
    regardless of filename or extraction mode.
    """
    base_extract: dict[str, object] = {}
    spec = {"extraction_mode": "auto", "stage_order": ["extract"]}

    ingestor, _, _ = _build_graph_ingestor_from_spec(
        "lecture.mp3",
        b"ID3\x03",
        base_extract,
        None,
        spec,
        base_asr=None,
    )

    assert ingestor._asr_params is None
