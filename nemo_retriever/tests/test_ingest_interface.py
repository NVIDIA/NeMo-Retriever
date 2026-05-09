import pandas as pd
import pytest

import nemo_retriever
from nemo_retriever.graph_ingestor import GraphIngestionError, GraphIngestor
from nemo_retriever.ingestor import IngestorCreateParams, _merge_params, create_ingestor
from nemo_retriever.params import (
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
