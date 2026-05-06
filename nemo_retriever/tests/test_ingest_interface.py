import pytest

from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.ingestor import IngestorCreateParams, _merge_params, create_ingestor
from nemo_retriever.params import (
    ASRParams,
    AudioChunkParams,
    CaptionParams,
    DedupParams,
    EmbedParams,
    ExtractParams,
    HtmlChunkParams,
    TextChunkParams,
)


def test_merge_params_none_returns_kwargs() -> None:
    merged = _merge_params(None, {"documents": ["a.pdf"]})
    assert merged == {"documents": ["a.pdf"]}


def test_merge_params_with_model_copy_updates_values() -> None:
    params = IngestorCreateParams(documents=["before.pdf"], ray_log_to_driver=True)
    merged = _merge_params(params, {"documents": ["after.pdf"], "ray_log_to_driver": False})
    assert isinstance(merged, IngestorCreateParams)
    assert merged.documents == ["after.pdf"]
    assert merged.ray_log_to_driver is False


def test_create_ingestor_parses_kwargs_and_returns_graph_ingestor() -> None:
    ingestor = create_ingestor(run_mode="inprocess", documents=["doc.pdf"], base_url="http://example:7670")
    assert isinstance(ingestor, GraphIngestor)
    assert ingestor._run_mode == "inprocess"
    assert ingestor._documents == ["doc.pdf"]


def test_create_ingestor_rejects_unknown_kwargs() -> None:
    with pytest.raises(Exception):
        create_ingestor(run_mode="inprocess", unknown_field=True)


def test_create_ingestor_rejects_legacy_non_graph_modes() -> None:
    with pytest.raises(ValueError, match="supports run modes"):
        create_ingestor(run_mode="fused")  # type: ignore[arg-type]


def test_graph_ingestor_action_methods_materialize_default_params() -> None:
    ingestor = GraphIngestor(run_mode="inprocess")

    ingestor.extract()
    assert isinstance(ingestor._extract_params, ExtractParams)

    ingestor.extract_image_files()
    assert isinstance(ingestor._extract_params, ExtractParams)

    ingestor.extract_txt()
    assert isinstance(ingestor._text_params, TextChunkParams)

    ingestor.extract_html()
    assert isinstance(ingestor._html_params, HtmlChunkParams)

    ingestor.extract_audio()
    assert isinstance(ingestor._audio_chunk_params, AudioChunkParams)
    assert isinstance(ingestor._asr_params, ASRParams)

    ingestor.dedup()
    assert isinstance(ingestor._dedup_params, DedupParams)

    ingestor.caption()
    assert isinstance(ingestor._caption_params, CaptionParams)

    ingestor.embed()
    assert isinstance(ingestor._embed_params, EmbedParams)


def test_extract_unified_defaults() -> None:
    """`.extract()` defaults: extraction_mode='pdf' and no chunking unless opted in."""
    ingestor = GraphIngestor(run_mode="inprocess").extract()
    assert ingestor._extraction_mode == "pdf"
    assert all(ingestor._split_config[k] is None for k in ("text", "html", "pdf", "audio", "image", "video"))


def test_typed_shortcuts_preserve_legacy_no_default_chunking() -> None:
    """Typed shortcuts (extract_audio, extract_txt, ...) must NOT enable default
    split_config chunking. Default-ON is reserved for the unified .extract()
    path. extract_txt(custom_params) must propagate custom_params via the
    text_params fallback.
    """
    # extract_audio without split_config: no audio chunking.
    audio_ingestor = GraphIngestor(run_mode="inprocess").extract_audio()
    assert audio_ingestor._split_config["audio"] is None

    # extract_txt(custom): _split_config["text"] stays None so the operator
    # falls back to self.text_params (= custom) in _effective_chunk_params.
    custom = TextChunkParams(max_tokens=512)
    txt_ingestor = GraphIngestor(run_mode="inprocess").extract_txt(custom)
    assert txt_ingestor._split_config["text"] is None
    assert txt_ingestor._text_params is custom
