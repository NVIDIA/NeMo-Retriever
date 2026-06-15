import pandas as pd
import pytest
from PIL import Image

import nemo_retriever
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.ingestor.graph_ingestor import GraphIngestionError, GraphIngestor
from nemo_retriever.ingestor import IngestorCreateParams, _merge_params, create_ingestor
from nemo_retriever.common.params import (
    ASRParams,
    AudioChunkParams,
    CaptionParams,
    DedupParams,
    EmbedParams,
    ExtractParams,
    HtmlChunkParams,
    RemoteRetryParams,
    TextChunkParams,
)


def _graph_node_names(graph) -> list[str]:
    names: list[str] = []

    def visit(node) -> None:
        names.append(node.name)
        for child in node.children:
            visit(child)

    for root in graph.roots:
        visit(root)
    return names


def _effective_graph_node_names(ingestor: GraphIngestor) -> list[str]:
    effective = ingestor._resolve_effective_extraction_inputs()
    graph = build_graph(
        extraction_mode=effective.extraction_mode,
        extract_params=effective.extract_params,
        text_params=effective.text_params,
        html_params=effective.html_params,
        audio_chunk_params=effective.audio_chunk_params,
        asr_params=effective.asr_params,
        video_frame_params=effective.video_frame_params,
        video_text_dedup_params=effective.video_text_dedup_params,
        av_fuse_params=effective.av_fuse_params,
        split_config=ingestor._split_config,
    )
    return _graph_node_names(graph)


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


def test_create_ingestor_passes_error_policy_to_graph_ingestor() -> None:
    ingestor = create_ingestor(run_mode="inprocess", error_policy="collect")
    assert isinstance(ingestor, GraphIngestor)
    assert ingestor._error_policy == "collect"


def test_graph_ingestion_error_is_exported_from_top_level_package() -> None:
    assert "GraphIngestionError" in nemo_retriever.__all__
    assert nemo_retriever.GraphIngestionError is GraphIngestionError


def test_create_ingestor_rejects_unknown_kwargs() -> None:
    with pytest.raises(Exception):
        create_ingestor(run_mode="inprocess", unknown_field=True)


def test_create_ingestor_rejects_unknown_run_modes() -> None:
    with pytest.raises(ValueError, match="supports run modes"):
        create_ingestor(run_mode="parallel")  # type: ignore[arg-type]


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
    """`.extract()` defaults: infer extraction_mode at graph-build time and no chunking unless opted in."""
    ingestor = GraphIngestor(run_mode="inprocess").extract()
    assert ingestor._extraction_mode is None
    assert all(ingestor._split_config[k] is None for k in ("text", "html", "pdf", "audio", "image", "video"))


def test_extract_rejects_unknown_kwargs() -> None:
    """`.extract()` must fail loudly on kwargs that are not fields of ExtractParams.

    The audio-video.md hosted-Parakeet snippet historically passed
    ``extract_method``/``extract_audio_params`` (and ``document_type``) into
    ``.extract()``; those silently flowed through ``_coerce()`` /
    ``params.model_copy(update=...)``, bypassing ``ExtractParams``'s
    ``extra="forbid"`` config. Audio credentials never reached the ASR actor,
    which then fell back to the local HF model with no signal to the user.
    Regression-guard the strict validation that fixes that silent drop.
    """
    ingestor = GraphIngestor(run_mode="inprocess")

    with pytest.raises(TypeError, match="garbage_kwarg"):
        ingestor.extract(garbage_kwarg="x")

    with pytest.raises(TypeError) as exc_info:
        ingestor.extract(
            document_type="mp3",
            extract_method="audio",
            extract_audio_params={
                "grpc_endpoint": "grpc.nvcf.nvidia.com:443",
                "auth_token": "fake-key",
                "function_id": "fake-function-id",
                "segment_audio": True,
            },
        )
    message = str(exc_info.value)
    # Pin the rejected-keys list as a single repr so this test fails loudly if
    # any of these keys ever become real ExtractParams fields.
    expected_rejected = repr(sorted(["document_type", "extract_method", "extract_audio_params"]))
    assert expected_rejected in message
    assert "asr_params" in message


def test_extract_default_pdf_only_builds_dedicated_pdf_graph(tmp_path) -> None:
    document = tmp_path / "manual.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    ingestor = GraphIngestor(run_mode="inprocess").files([str(document)]).extract()

    node_names = _effective_graph_node_names(ingestor)
    assert "MultiTypeExtractOperator" not in node_names
    assert node_names[:4] == [
        "DocToPdfConversionActor",
        "PDFSplitActor",
        "PDFExtractionActor",
        "PageElementDetectionActor",
    ]


@pytest.mark.parametrize(("suffix", "image_format"), [(".bmp", "BMP"), (".tiff", "TIFF"), (".tif", "TIFF")])
def test_extract_default_direct_images_materialize_page_image(monkeypatch, tmp_path, suffix, image_format) -> None:
    image_path = tmp_path / f"page{suffix}"
    Image.new("RGB", (12, 8), color=(40, 90, 130)).save(image_path, format=image_format)

    def passthrough_detection(self, batch_df):
        return batch_df

    def fail_pdf_split(self, batch_df):
        raise AssertionError("direct image extraction routed through PDFSplitActor")

    monkeypatch.setattr(
        "nemo_retriever.operators.graph_ops.multi_type_extract_operator._MultiTypeExtractBase._run_detection_pipeline",
        passthrough_detection,
    )
    monkeypatch.setattr("nemo_retriever.operators.extract.pdf.split.PDFSplitActor.run", fail_pdf_split)

    result = (
        create_ingestor(run_mode="inprocess")
        .files([str(image_path)])
        .extract(
            params=ExtractParams(
                extract_text=True,
                extract_images=True,
                extract_tables=False,
                extract_charts=False,
                extract_infographics=False,
            )
        )
        .ingest()
    )

    assert len(result) == 1
    page_image = result.iloc[0]["page_image"]
    assert isinstance(page_image, dict)
    assert page_image["image_b64"]
    assert result.iloc[0]["metadata"]["source_path"] == str(image_path.resolve())


def test_extract_default_mixed_pdf_and_image_plans_ordered_branches(tmp_path) -> None:
    pdf = tmp_path / "manual.pdf"
    image = tmp_path / "scan.bmp"
    pdf.write_bytes(b"%PDF-1.4\n")
    image.write_bytes(b"bmp")

    ingestor = GraphIngestor(run_mode="inprocess").files([str(image), str(pdf)]).extract()

    branches = ingestor._plan_default_extraction_branches()
    assert [(branch.family, branch.extraction_mode, branch.input_paths) for branch in branches] == [
        ("pdf", "pdf", (str(pdf),)),
        ("image", "image", (str(image),)),
    ]


def test_extract_explicit_pdf_rejects_image_input(tmp_path) -> None:
    image = tmp_path / "scan.bmp"
    image.write_bytes(b"bmp")
    ingestor = GraphIngestor(run_mode="inprocess").files([str(image)]).extract(extraction_mode="pdf")

    with pytest.raises(ValueError, match="extraction_mode='pdf'"):
        ingestor.ingest()


def test_extract_default_rejects_unknown_input_type(tmp_path) -> None:
    document = tmp_path / "payload.bin"
    document.write_bytes(b"unknown")
    ingestor = GraphIngestor(run_mode="inprocess").files([str(document)]).extract()

    with pytest.raises(ValueError, match="Unsupported input file type"):
        ingestor.ingest()


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


@pytest.mark.integration
def test_graph_ingestor_raises_for_explicit_remote_stage_errors() -> None:
    ingestor = GraphIngestor(
        run_mode="inprocess",
        documents=["data/test.pdf"],
        show_progress=False,
    ).extract(
        page_elements_invoke_url="http://127.0.0.1:1/v1/nonexistent",
        extract_text=False,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
        inference_batch_size=1,
        remote_retry=RemoteRetryParams(
            remote_max_pool_workers=1,
            remote_max_retries=1,
            remote_max_429_retries=1,
        ),
    )

    with pytest.raises(RuntimeError, match="page_elements_v3"):
        ingestor.ingest()


@pytest.mark.integration
def test_graph_ingestor_collect_policy_returns_explicit_remote_stage_errors() -> None:
    result = (
        GraphIngestor(
            run_mode="inprocess",
            documents=["data/test.pdf"],
            show_progress=False,
            error_policy="collect",
        )
        .extract(
            page_elements_invoke_url="http://127.0.0.1:1/v1/nonexistent",
            extract_text=False,
            extract_images=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
            inference_batch_size=1,
            remote_retry=RemoteRetryParams(
                remote_max_pool_workers=1,
                remote_max_retries=1,
                remote_max_429_retries=1,
            ),
        )
        .ingest()
    )

    assert "page_elements_v3" in result.columns
    payload = result.iloc[0]["page_elements_v3"]
    assert payload["error"]["type"] == "ConnectionError"


def test_strict_remote_error_policy_ignores_unrelated_error_columns() -> None:
    ingestor = GraphIngestor(run_mode="inprocess").extract(
        page_elements_invoke_url="http://remote.example/v1/page-elements",
        extract_text=False,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
    )
    result = pd.DataFrame(
        {
            "page_elements_v3": [{"detections": [], "error": None}],
            "metadata": [
                {
                    "error": {
                        "stage": "local_postprocess",
                        "type": "ValueError",
                        "message": "local stage failed",
                    }
                }
            ],
        }
    )

    ingestor._raise_for_stage_errors(result)


def test_strict_remote_error_policy_accepts_batch_dataset_rows() -> None:
    class RayLikeDataset:
        columns = ["page_elements_v3", "metadata"]

        def iter_batches(self, *, batch_format: str):
            assert batch_format == "pandas"
            yield pd.DataFrame(
                {
                    "page_elements_v3": [
                        {
                            "timing": None,
                            "error": {
                                "stage": "remote_inference",
                                "type": "ConnectionError",
                                "message": "connection refused",
                            },
                        }
                    ],
                    "metadata": [{"source": "test.pdf"}],
                }
            )

    ingestor = GraphIngestor(run_mode="batch").extract(
        page_elements_invoke_url="http://remote.example/v1/page-elements",
        extract_text=False,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
    )

    with pytest.raises(GraphIngestionError, match="page_elements_v3") as exc_info:
        ingestor._raise_for_stage_errors(RayLikeDataset())

    assert exc_info.value.records[0]["row_index"] == 0


def test_graph_ingestion_error_sanitizes_remote_message_fields() -> None:
    sensitive_tail = "TAIL_SHOULD_NOT_APPEAR"
    long_message = "π" + ("x" * 600) + sensitive_tail

    err = GraphIngestionError(
        [
            {
                "row_index": 0,
                "column": "page_elements_v3",
                "path": "error",
                "error": {
                    "stage": "remote_inference",
                    "type": "BadRequest",
                    "message": long_message,
                },
            }
        ]
    )

    rendered = str(err)
    assert "π" not in rendered
    assert sensitive_tail not in rendered

    raw_string_err = GraphIngestionError(
        [
            {
                "row_index": 0,
                "column": "text_embeddings_1b_v2",
                "path": "error",
                "error": long_message,
            }
        ]
    )
    raw_rendered = str(raw_string_err)
    assert "π" not in raw_rendered
    assert sensitive_tail not in raw_rendered


def test_graph_ingestion_error_accepts_plain_string_records() -> None:
    err = GraphIngestionError(["Embedding error occurred: timed out"])
    rendered = str(err)
    assert "timed out" in rendered
    assert "path error" in rendered


def test_graph_ingestion_error_preserves_readable_remote_message_text() -> None:
    err = GraphIngestionError(
        [
            {
                "row_index": 0,
                "column": "page_elements_v3",
                "path": "error",
                "error": {
                    "stage": "remote_inference",
                    "type": "ConnectionError",
                    "message": "connection\nrefused",
                },
            }
        ]
    )

    rendered = str(err)
    assert "connection refused" in rendered
    assert "c o n n e c t i o n" not in rendered


def test_get_error_rows_accepts_inprocess_dataframe_stage_error_columns() -> None:
    ingestor = GraphIngestor(run_mode="inprocess")
    df = pd.DataFrame(
        {
            "table_structure_ocr_v1": [
                {
                    "timing": None,
                    "error": {
                        "stage": "remote_inference",
                        "type": "ConnectionError",
                        "message": "connection refused",
                    },
                },
                {"timing": None, "error": None},
            ],
            "text": ["first page", "second page"],
        }
    )

    errors = ingestor.get_error_rows(df)

    assert len(errors) == 1
    assert errors.iloc[0]["text"] == "first page"


def test_get_error_rows_maps_batch_dataset_with_columns_property() -> None:
    class RayLikeDataset:
        columns = ["page_elements_v3", "text"]

        def __getitem__(self, key: str):
            raise AssertionError(f"expected map_batches path, got pandas access for {key}")

        def map_batches(self, fn, *, batch_format: str):
            assert batch_format == "pandas"
            batch = pd.DataFrame(
                {
                    "page_elements_v3": [
                        {
                            "timing": None,
                            "error": {
                                "stage": "remote_inference",
                                "type": "ConnectionError",
                                "message": "connection refused",
                            },
                        },
                        {"timing": None, "error": None},
                    ],
                    "text": ["first page", "second page"],
                }
            )
            return fn(batch)

    errors = GraphIngestor(run_mode="batch").get_error_rows(RayLikeDataset())

    assert len(errors) == 1
    assert errors.iloc[0]["text"] == "first page"
