# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import typer

import nemo_retriever.pipeline as pipeline_pkg
from nemo_retriever.params import EmbedParams
from nemo_retriever.pipeline.__main__ import (
    _build_embed_params,
    _collect_results,
    _count_input_units,
    _count_uploadable_vdb_records,
    _parse_vdb_kwargs_json,
    _resolve_file_patterns,
)


def test_pipeline_package_exports_cli_app_and_run() -> None:
    from nemo_retriever.pipeline.__main__ import app, run

    assert pipeline_pkg.app is app
    assert pipeline_pkg.run is run
    assert set(pipeline_pkg.__all__) == {"app", "run"}


@pytest.mark.parametrize(
    ("input_type", "files", "expected_globs"),
    [
        ("pdf", ["nested/doc.pdf"], ["*.pdf"]),
        ("doc", ["deck.pptx", "report.docx"], ["*.docx", "*.pptx"]),
        ("image", ["plot.png"], ["*.png"]),
    ],
)
def test_resolve_file_patterns_recurses_directory_inputs(
    tmp_path: Path,
    input_type: str,
    files: list[str],
    expected_globs: list[str],
) -> None:
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    assert _resolve_file_patterns(tmp_path, input_type) == [str(tmp_path / "**" / glob) for glob in expected_globs]


def test_resolve_file_patterns_returns_existing_file_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"x")

    assert _resolve_file_patterns(path, "audio") == [str(path)]


def test_resolve_file_patterns_rejects_missing_or_empty_inputs(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter, match="Path does not exist"):
        _resolve_file_patterns(tmp_path / "missing", "pdf")

    (tmp_path / "sidecar.json").write_text("{}", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="No files found"):
        _resolve_file_patterns(tmp_path, "pdf")


def test_parse_vdb_kwargs_json_keeps_backend_kwargs_opaque() -> None:
    assert _parse_vdb_kwargs_json(None) == {}
    assert _parse_vdb_kwargs_json('{"collection_name": "docs", "uri": "http://localhost:19530"}') == {
        "collection_name": "docs",
        "uri": "http://localhost:19530",
    }


def test_parse_vdb_kwargs_json_rejects_non_object_json() -> None:
    with pytest.raises(typer.BadParameter, match="JSON object"):
        _parse_vdb_kwargs_json('["not", "an", "object"]')


def test_build_embed_params_forwards_remote_and_modality_flags() -> None:
    params = _build_embed_params(
        embed_model_name="nvidia/test-embed",
        embed_invoke_url="http://embed.example/v1",
        embed_remote_api_key="nvapi-secret",
        embed_modality="text_image",
        text_elements_modality="text",
        structured_elements_modality="image",
        embed_granularity="element",
        embed_actors=2,
        embed_batch_size=16,
        embed_cpus_per_actor=1.5,
        embed_gpus_per_actor=0.5,
    )

    assert isinstance(params, EmbedParams)
    assert params.model_name == "nvidia/test-embed"
    assert params.embed_invoke_url == "http://embed.example/v1"
    assert params.api_key == "nvapi-secret"
    assert params.embed_modality == "text_image"
    assert params.text_elements_modality == "text"
    assert params.structured_elements_modality == "image"
    assert params.embed_granularity == "element"
    assert params.inference_batch_size == 16
    assert params.batch_tuning.embed_workers == 2
    assert params.batch_tuning.embed_batch_size == 16
    assert params.batch_tuning.embed_cpus_per_actor == 1.5
    assert params.batch_tuning.gpu_embed == 0.0


class TestCollectResults:
    """Ingest returns a DataFrame (``ingestor.ingest()`` → ``ds.to_pandas()``); _collect_results consumes it."""

    def test_batch_mode_accepts_ingest_dataframe(self):
        rows = [
            {"source_id": "a", "text": "hello"},
            {"source_id": "a", "text": "world"},
            {"source_id": "b", "text": "!"},
        ]
        # Same shape as the graph executor return after ``Dataset.to_pandas()``.
        result_df = pd.DataFrame(rows)

        records, df, download_time, num_units = _collect_results("batch", result_df)

        assert records == rows
        assert df is result_df
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["source_id", "text"]
        assert len(df) == 3
        # ``source_id`` has two distinct values → that is the unit count.
        assert num_units == 2
        assert download_time >= 0.0

    def test_batch_mode_handles_empty_result(self):
        result_df = pd.DataFrame()
        records, df, download_time, num_units = _collect_results("batch", result_df)
        assert records == []
        assert df.empty
        # Empty DataFrame has no columns → falls through to len(df.index) == 0.
        assert num_units == 0
        assert download_time >= 0.0

    def test_inprocess_mode_accepts_dataframe_directly(self):
        rows = [
            {"source_id": "a", "text": "x"},
            {"source_id": "b", "text": "y"},
        ]
        df_in = pd.DataFrame(rows)

        records, df_out, download_time, num_units = _collect_results("inprocess", df_in)

        # The DataFrame is passed through unchanged (same object).
        assert df_out is df_in
        assert records == rows
        # inprocess mode never incurs Ray download time.
        assert download_time == 0.0
        assert num_units == 2


def test_collect_results_accepts_inprocess_dataframe() -> None:
    df_in = pd.DataFrame([{"source_path": "/a.pdf"}, {"source_path": "/b.pdf"}])

    records, df_out, download_time, num_units = _collect_results("inprocess", df_in)

    assert df_out is df_in
    assert records == [{"source_path": "/a.pdf"}, {"source_path": "/b.pdf"}]
    assert download_time == 0.0
    assert num_units == 2


def test_count_uploadable_vdb_records_filters_rows_without_embedding_or_text() -> None:
    rows = [
        {
            "text": "keep",
            "text_embeddings_1b_v2": {"embedding": [0.1, 0.2]},
            "source_id": "/tmp/doc-a.pdf",
            "page_number": 1,
        },
        {
            "text": "drop missing embedding",
            "source_id": "/tmp/doc-a.pdf",
            "page_number": 2,
        },
        {
            "text_embeddings_1b_v2": {"embedding": [0.3, 0.4]},
            "source_id": "/tmp/doc-a.pdf",
            "page_number": 3,
        },
    ]

    assert _count_uploadable_vdb_records(rows) == 1
    assert _count_uploadable_vdb_records([]) == 0


def test_count_input_units_prefers_source_id_then_source_path() -> None:
    assert _count_input_units(pd.DataFrame({"source_id": ["a", "a", "b"]})) == 2
    assert _count_input_units(pd.DataFrame({"source_path": ["/a", "/b", "/b"]})) == 2
    assert _count_input_units(pd.DataFrame({"text": ["x", "y", "z"]})) == 3
