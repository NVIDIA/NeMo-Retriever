# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import itertools
import json
import logging
import os
import re
import sys
from typing import Any
from unittest.mock import create_autospec

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import nemo_retriever.adapters.cli.sdk_workflow as sdk_workflow
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.params import (
    ASRParams,
    AudioChunkParams,
    AudioVisualFuseParams,
    CaptionParams,
    DedupParams,
    EmbedParams,
    ExtractParams,
    HtmlChunkParams,
    StoreParams,
    TextChunkParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
)


RUNNER = CliRunner()
cli_main = importlib.import_module("nemo_retriever.adapters.cli.main")


@pytest.fixture(autouse=True)
def _successful_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    # Most tests fake GraphIngestor; default row counts should look like a successful write.
    counts = itertools.count(1)
    monkeypatch.setattr(sdk_workflow, "_count_lancedb_rows", lambda *_, **__: next(counts))


def _make_fake_ingestor() -> Any:
    fake_ingestor = create_autospec(GraphIngestor, instance=True, spec_set=True)
    fake_ingestor.files.return_value = fake_ingestor
    fake_ingestor.extract.return_value = fake_ingestor
    fake_ingestor.dedup.return_value = fake_ingestor
    fake_ingestor.caption.return_value = fake_ingestor
    fake_ingestor.embed.return_value = fake_ingestor
    fake_ingestor.store.return_value = fake_ingestor
    fake_ingestor.vdb_upload.return_value = fake_ingestor
    fake_ingestor.ingest.return_value = [{"status": "ok"}]
    return fake_ingestor


def test_root_ingest_runs_default_sdk_chain(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    create_calls: list[dict[str, Any]] = []
    document = tmp_path / "multimodal_test.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    def fake_create_ingestor(**kwargs: Any) -> Any:
        create_calls.append(kwargs)
        return fake_ingestor

    monkeypatch.setattr(sdk_workflow, "create_ingestor", fake_create_ingestor)
    monkeypatch.setattr(sdk_workflow, "_count_lancedb_rows", lambda *_, **__: 7)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document)])

    assert result.exit_code == 0
    assert create_calls == [{"run_mode": "inprocess"}]
    assert [method_call[0] for method_call in fake_ingestor.method_calls] == [
        "files",
        "extract",
        "embed",
        "vdb_upload",
        "ingest",
    ]
    assert fake_ingestor.files.call_args.args == ([str(document)],)
    assert isinstance(fake_ingestor.extract.call_args.args[0], ExtractParams)
    assert fake_ingestor.extract.call_args.kwargs == {}
    assert fake_ingestor.embed.call_args.args == ()
    vdb_upload_params = fake_ingestor.vdb_upload.call_args.args[0]
    assert vdb_upload_params.vdb_op == "lancedb"
    assert vdb_upload_params.vdb_kwargs == {"uri": "lancedb", "table_name": "nemo-retriever", "overwrite": True}
    assert "Ingested 1 file(s) → 7 row(s) in LanceDB lancedb/nemo-retriever." in result.output


def test_root_ingest_passes_vdb_options_and_run_mode(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    create_calls: list[dict[str, Any]] = []
    first_document = tmp_path / "a.pdf"
    globbed_document = tmp_path / "b" / "c.pdf"
    first_document.write_bytes(b"%PDF-1.4\n")
    globbed_document.parent.mkdir()
    globbed_document.write_bytes(b"%PDF-1.4\n")

    def fake_create_ingestor(**kwargs: Any) -> Any:
        create_calls.append(kwargs)
        return fake_ingestor

    monkeypatch.setattr(sdk_workflow, "create_ingestor", fake_create_ingestor)
    monkeypatch.setattr(sdk_workflow, "_count_lancedb_rows", lambda *_, **__: 12)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(first_document),
            str(globbed_document.parent),
            "--run-mode",
            "batch",
            "--lancedb-uri",
            "/tmp/lancedb",
            "--table-name",
            "docs",
        ],
    )

    assert result.exit_code == 0
    assert create_calls == [{"run_mode": "batch"}]
    assert fake_ingestor.files.call_args.args == ([str(first_document), str(globbed_document)],)
    assert isinstance(fake_ingestor.extract.call_args.args[0], ExtractParams)
    assert fake_ingestor.extract.call_args.kwargs == {}
    assert fake_ingestor.vdb_upload.call_args.args[0].vdb_kwargs == {
        "uri": "/tmp/lancedb",
        "table_name": "docs",
        "overwrite": True,
    }
    assert "Ingested 2 file(s) → 12 row(s) in LanceDB /tmp/lancedb/docs." in result.output


def test_root_ingest_append_forwards_overwrite_false(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "multimodal_test.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document), "--append"])

    assert result.exit_code == 0
    assert fake_ingestor.vdb_upload.call_args.args[0].vdb_kwargs == {
        "uri": "lancedb",
        "table_name": "nemo-retriever",
        "overwrite": False,
    }


def test_root_ingest_fails_when_no_rows_landed(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "silent-stage-failure.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(sdk_workflow, "_count_lancedb_rows", lambda *_, **__: 0)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document)])

    assert result.exit_code == 1
    assert "retriever ingest produced 0 rows" in result.output
    assert "NVIDIA_API_KEY/NGC_API_KEY" in result.output
    assert "Ingested 1 file(s)" not in result.output


def test_root_ingest_append_fails_when_row_count_does_not_increase(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "silent-append-failure.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    counts = iter([3, 3])

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(sdk_workflow, "_count_lancedb_rows", lambda *_, **__: next(counts))

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document), "--append"])

    assert result.exit_code == 1
    assert "did not add rows" in result.output
    assert "row count stayed at 3" in result.output


def test_root_ingest_passes_nim_url_options(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "nim-routed.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    def fake_create_ingestor(**_kwargs: Any) -> Any:
        return fake_ingestor

    monkeypatch.setattr(sdk_workflow, "create_ingestor", fake_create_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(document),
            "--page-elements-invoke-url",
            "http://page-elements:8000/v1/infer",
            "--ocr-invoke-url",
            "http://ocr:8000/v1/infer",
            "--ocr-version",
            "v1",
            "--graphic-elements-invoke-url",
            "http://graphic-elements:8000/v1/infer",
            "--table-structure-invoke-url",
            "http://table-structure:8000/v1/infer",
            "--embed-invoke-url",
            "http://embed:8000/v1/embeddings",
            "--embed-model-name",
            "nvidia/llama-nemotron-embed-1b-v2",
        ],
    )

    assert result.exit_code == 0
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.page_elements_invoke_url == "http://page-elements:8000/v1/infer"
    assert extract_params.ocr_invoke_url == "http://ocr:8000/v1/infer"
    assert extract_params.ocr_version == "v1"
    assert extract_params.graphic_elements_invoke_url == "http://graphic-elements:8000/v1/infer"
    assert extract_params.table_structure_invoke_url == "http://table-structure:8000/v1/infer"

    embed_params = fake_ingestor.embed.call_args.args[0]
    assert isinstance(embed_params, EmbedParams)
    assert embed_params.embed_invoke_url == "http://embed:8000/v1/embeddings"
    assert embed_params.embedding_endpoint == "http://embed:8000/v1/embeddings"
    assert embed_params.model_name == "nvidia/llama-nemotron-embed-1b-v2"
    assert embed_params.embed_model_name == "nvidia/llama-nemotron-embed-1b-v2"


def test_root_ingest_passes_migrated_extraction_and_embedding_flags(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "jp20-style.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(document),
            "--use-graphic-elements",
            "--use-table-structure",
            "--embed-modality",
            "text_image",
            "--embed-granularity",
            "element",
            "--text-elements-modality",
            "text",
            "--structured-elements-modality",
            "image",
        ],
    )

    assert result.exit_code == 0
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.use_graphic_elements is True
    assert extract_params.use_table_structure is True

    embed_params = fake_ingestor.embed.call_args.args[0]
    assert isinstance(embed_params, EmbedParams)
    assert embed_params.embed_modality == "text_image"
    assert embed_params.embed_granularity == "element"
    assert embed_params.text_elements_modality == "text"
    assert embed_params.structured_elements_modality == "image"


def test_root_ingest_text_chunk_builds_split_config(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "chunked.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(document),
            "--text-chunk",
            "--text-chunk-max-tokens",
            "512",
            "--text-chunk-overlap-tokens",
            "64",
        ],
    )

    assert result.exit_code == 0
    assert fake_ingestor.extract.call_args.kwargs["split_config"] == {
        "pdf": {
            "max_tokens": 512,
            "overlap_tokens": 64,
            "tokenizer_model_id": None,
            "encoding": "utf-8",
            "tokenizer_cache_dir": None,
        }
    }


@pytest.mark.parametrize(
    ("filename", "param_key", "param_type"),
    [
        ("notes.txt", "text_params", TextChunkParams),
        ("page.html", "html_params", HtmlChunkParams),
    ],
)
def test_root_ingest_text_chunk_uses_dedicated_text_params(
    monkeypatch,
    tmp_path,
    filename: str,
    param_key: str,
    param_type: type[TextChunkParams | HtmlChunkParams],
) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / filename
    document.write_text("chunk me", encoding="utf-8")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(document),
            "--text-chunk",
            "--text-chunk-max-tokens",
            "512",
            "--text-chunk-overlap-tokens",
            "64",
        ],
    )

    assert result.exit_code == 0
    extract_kwargs = fake_ingestor.extract.call_args.kwargs
    assert "split_config" not in extract_kwargs
    chunk_params = extract_kwargs[param_key]
    assert isinstance(chunk_params, param_type)
    assert chunk_params.max_tokens == 512
    assert chunk_params.overlap_tokens == 64


def test_root_ingest_passes_ocr_lang_option(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "english-ocr.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document), "--ocr-lang", "english"])

    assert result.exit_code == 0
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.ocr_version == "v2"
    assert extract_params.ocr_lang == "english"


def test_root_ingest_rejects_ocr_lang_with_legacy_ocr_version(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "legacy-ocr.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        ["ingest", str(document), "--ocr-version", "v1", "--ocr-lang", "english"],
    )

    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    assert "ocr_lang is only supported when ocr_version='v2'" in result.output
    assert "Traceback" not in result.output
    fake_ingestor.extract.assert_not_called()


def test_root_ingest_passes_batch_tuning_options(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    create_calls: list[dict[str, Any]] = []
    document = tmp_path / "batch-tuned.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    def fake_create_ingestor(**kwargs: Any) -> Any:
        create_calls.append(kwargs)
        return fake_ingestor

    monkeypatch.setattr(sdk_workflow, "create_ingestor", fake_create_ingestor)
    monkeypatch.setattr(sdk_workflow, "_count_lancedb_rows", lambda *_, **__: 42)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(document),
            "--run-mode",
            "batch",
            "--ray-address",
            "ray://cluster:10001",
            "--no-ray-log-to-driver",
            "--pdf-extract-workers",
            "4",
            "--pdf-extract-batch-size",
            "2",
            "--pdf-extract-cpus-per-task",
            "1.5",
            "--page-elements-workers",
            "3",
            "--page-elements-batch-size",
            "8",
            "--page-elements-cpus-per-actor",
            "0.5",
            "--ocr-workers",
            "5",
            "--ocr-batch-size",
            "6",
            "--ocr-cpus-per-actor",
            "0.75",
            "--embed-workers",
            "7",
            "--embed-batch-size",
            "16",
            "--embed-cpus-per-actor",
            "0.25",
        ],
    )

    assert result.exit_code == 0
    assert create_calls == [
        {
            "run_mode": "batch",
            "ray_address": "ray://cluster:10001",
            "ray_log_to_driver": False,
        }
    ]

    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.batch_tuning.pdf_extract_workers == 4
    assert extract_params.batch_tuning.pdf_extract_batch_size == 2
    assert extract_params.batch_tuning.pdf_extract_num_cpus == 1.5
    assert extract_params.batch_tuning.page_elements_workers == 3
    assert extract_params.batch_tuning.page_elements_batch_size == 8
    assert extract_params.batch_tuning.page_elements_cpus_per_actor == 0.5
    assert extract_params.batch_tuning.ocr_workers == 5
    assert extract_params.batch_tuning.ocr_inference_batch_size == 6
    assert extract_params.batch_tuning.ocr_cpus_per_actor == 0.75

    embed_params = fake_ingestor.embed.call_args.args[0]
    assert isinstance(embed_params, EmbedParams)
    assert embed_params.batch_tuning.embed_workers == 7
    assert embed_params.batch_tuning.embed_batch_size == 16
    assert embed_params.batch_tuning.embed_cpus_per_actor == 0.25
    assert "Ingested 1 file(s) → 42 row(s) in LanceDB lancedb/nemo-retriever." in result.output


def test_root_ingest_passes_public_parity_options(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "parity.pdf"
    image_store = tmp_path / "images"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(sdk_workflow, "_count_lancedb_rows", lambda *_, **__: 14)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(document),
            "--api-key",
            "nvapi-secret",
            "--dedup",
            "--dedup-iou-threshold",
            "0.6",
            "--caption",
            "--caption-invoke-url",
            "http://vlm:8000/v1/chat/completions",
            "--store-images-uri",
            str(image_store),
            "--method",
            "nemotron_parse",
            "--pdf-split-batch-size",
            "2",
            "--nemotron-parse-workers",
            "3",
            "--nemotron-parse-batch-size",
            "4",
            "--nemotron-parse-gpus-per-actor",
            "0.5",
        ],
    )

    assert result.exit_code == 0
    assert [method_call[0] for method_call in fake_ingestor.method_calls] == [
        "files",
        "extract",
        "dedup",
        "caption",
        "embed",
        "store",
        "vdb_upload",
        "ingest",
    ]

    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.method == "nemotron_parse"
    assert extract_params.api_key == "nvapi-secret"
    assert extract_params.batch_tuning.pdf_split_batch_size == 2
    assert extract_params.batch_tuning.nemotron_parse_workers == 3
    assert extract_params.batch_tuning.nemotron_parse_batch_size == 4
    assert extract_params.batch_tuning.gpu_nemotron_parse == 0.5

    dedup_params = fake_ingestor.dedup.call_args.args[0]
    assert isinstance(dedup_params, DedupParams)
    assert dedup_params.iou_threshold == 0.6

    caption_params = fake_ingestor.caption.call_args.args[0]
    assert isinstance(caption_params, CaptionParams)
    assert caption_params.api_key == "nvapi-secret"

    embed_params = fake_ingestor.embed.call_args.args[0]
    assert isinstance(embed_params, EmbedParams)
    assert embed_params.api_key == "nvapi-secret"

    store_params = fake_ingestor.store.call_args.args[0]
    assert isinstance(store_params, StoreParams)
    assert store_params.storage_uri == str(image_store.resolve())
    assert "Ingested 1 file(s) → 14 row(s) in LanceDB lancedb/nemo-retriever." in result.output


def test_root_ingest_rejects_dedup_threshold_without_dedup(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "dedup-threshold.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document), "--dedup-iou-threshold", "0.6"])

    assert result.exit_code == 1
    assert "Dedup options require --dedup" in result.output
    fake_ingestor.dedup.assert_not_called()
    fake_ingestor.embed.assert_not_called()


def test_ingest_documents_accepts_legacy_public_api_kwargs(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "legacy-public-api.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = sdk_workflow.ingest_documents(
        [str(document)],
        input_type="pdf",
        table_output_format="markdown",
        local_ingest_embed_backend="hf",
        page_elements_gpus_per_actor=0.2,
        ocr_gpus_per_actor=0.3,
        table_structure_workers=6,
        table_structure_batch_size=12,
        table_structure_cpus_per_actor=0.4,
        table_structure_gpus_per_actor=0.25,
        pdf_split_batch_size=9,
        nemotron_parse_workers=10,
        nemotron_parse_batch_size=11,
        nemotron_parse_gpus_per_actor=0.6,
        embed_gpus_per_actor=0.5,
    )

    assert result["documents"] == [str(document)]
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.use_table_structure is True
    assert extract_params.table_output_format == "markdown"
    assert extract_params.batch_tuning.gpu_page_elements == 0.2
    assert extract_params.batch_tuning.gpu_ocr == 0.3
    assert extract_params.batch_tuning.table_structure_workers == 6
    assert extract_params.batch_tuning.table_structure_batch_size == 12
    assert extract_params.batch_tuning.table_structure_cpus_per_actor == 0.4
    assert extract_params.batch_tuning.gpu_table_structure == 0.25
    assert extract_params.batch_tuning.pdf_split_batch_size == 9
    assert extract_params.batch_tuning.nemotron_parse_workers == 10
    assert extract_params.batch_tuning.nemotron_parse_batch_size == 11
    assert extract_params.batch_tuning.gpu_nemotron_parse == 0.6

    embed_params = fake_ingestor.embed.call_args.args[0]
    assert isinstance(embed_params, EmbedParams)
    assert embed_params.local_ingest_embed_backend == "hf"
    assert embed_params.batch_tuning.gpu_embed == 0.5


def test_execute_ingest_plan_returns_structured_execution_data(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "execution-result.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(sdk_workflow, "_count_lancedb_rows", lambda *_, **__: 9)

    plan = sdk_workflow.resolve_ingest_plan(
        [str(document)],
        run_mode="inprocess",
        lancedb_uri="/tmp/nemo-test-lancedb",
        table_name="execution_result",
    )
    execution = sdk_workflow.execute_ingest_plan(plan)

    assert execution.documents == [str(document)]
    assert execution.lancedb_uri == "/tmp/nemo-test-lancedb"
    assert execution.table_name == "execution_result"
    assert execution.lancedb_target == "/tmp/nemo-test-lancedb/execution_result"
    assert execution.n_rows == 9
    assert execution.result == [{"status": "ok"}]
    assert execution.metadata["branch_summary"]
    assert execution.to_summary_dict()["n_rows"] == 9


def test_root_ingest_reports_empty_directory_error(tmp_path) -> None:
    result = RUNNER.invoke(cli_main.app, ["ingest", str(tmp_path)])

    assert result.exit_code == 1
    assert "No supported ingest files found under directory" in result.output


def test_root_ingest_reports_unknown_default_input_type(tmp_path) -> None:
    document = tmp_path / "payload.bin"
    document.write_bytes(b"unknown")

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document)])

    assert result.exit_code == 1
    assert "Unsupported input file type(s) for retriever ingest" in result.output


def test_root_ingest_routes_text_inputs_by_default_to_auto_planner(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "notes.txt"
    document.write_text("not a pdf", encoding="utf-8")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document)])

    assert result.exit_code == 0
    assert fake_ingestor.files.call_args.args == ([str(document)],)
    assert isinstance(fake_ingestor.extract.call_args.args[0], ExtractParams)
    assert isinstance(fake_ingestor.extract.call_args.kwargs["text_params"], TextChunkParams)


def test_root_ingest_help_does_not_expose_input_type() -> None:
    result = RUNNER.invoke(cli_main.app, ["ingest", "--help"])

    assert result.exit_code == 0
    assert "--input-type" not in result.output
    assert "--profile" in result.output
    assert "[auto|fast-text]" in result.output
    assert "--extract-images" in result.output
    # Rich help truncates long option names in narrow test terminals.
    assert "--use-graphic-el" in result.output
    assert "--use-table-stru" in result.output
    assert "--embed-modality" in result.output
    assert "--embed-granular" in result.output
    assert "--text-elements-" in result.output
    assert "--structured-ele" in result.output
    assert "--text-chunk" in result.output
    assert "--store-images-" in result.output
    assert "--api-key" in result.output
    assert "--dedup" in result.output
    assert "--nemotron-par" in result.output
    assert "--caption" in result.output
    assert "--run-mode" in result.output
    assert "[inprocess|batch" in result.output
    assert re.search(r"--no-caption(?!-)", result.output) is None


def test_root_ingest_dry_run_prints_plan_without_creating_ingestor(monkeypatch, tmp_path) -> None:
    document = tmp_path / "fast.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    def fail_create_ingestor(**_kwargs: Any) -> Any:
        raise AssertionError("create_ingestor should not be called for --dry-run")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", fail_create_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document), "--profile", "fast-text", "--dry-run"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["profile"] == "fast-text"
    assert payload["create_ingestor"] == {"run_mode": "inprocess"}
    assert payload["extract"]["method"] == "pdfium"
    assert payload["extract"]["extract_images"] is False
    assert payload["extract"]["use_page_elements"] is False
    assert payload["extract"]["extract_tables"] is False


def test_root_ingest_passes_extract_overrides_without_ocr_profile(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "manual.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(document),
            "--method",
            "pdfium",
            "--dpi",
            "250",
            "--no-extract-tables",
            "--no-extract-images",
            "--no-extract-charts",
            "--no-extract-infographics",
            "--no-extract-page-as-image",
            "--no-use-page-elements",
        ],
    )

    assert result.exit_code == 0
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.method == "pdfium"
    assert extract_params.dpi == 250
    assert extract_params.extract_text is True
    assert extract_params.extract_images is False
    assert extract_params.extract_tables is False
    assert extract_params.extract_charts is False
    assert extract_params.extract_infographics is False
    assert extract_params.extract_page_as_image is False
    assert extract_params.use_page_elements is False


def test_root_ingest_caption_is_optional_and_passes_minimal_caption_params(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "captioned.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(document),
            "--caption",
            "--caption-invoke-url",
            "http://vlm:8000/v1/chat/completions",
            "--caption-model-name",
            "nvidia/test-vlm",
            "--caption-context-text-max-chars",
            "512",
            "--caption-infographics",
        ],
    )

    assert result.exit_code == 0
    assert [method_call[0] for method_call in fake_ingestor.method_calls] == [
        "files",
        "extract",
        "caption",
        "embed",
        "vdb_upload",
        "ingest",
    ]
    caption_params = fake_ingestor.caption.call_args.args[0]
    assert isinstance(caption_params, CaptionParams)
    assert caption_params.endpoint_url == "http://vlm:8000/v1/chat/completions"
    assert caption_params.model_name == "nvidia/test-vlm"
    assert caption_params.context_text_max_chars == 512
    assert caption_params.caption_infographics is True


def test_root_ingest_rejects_caption_options_without_caption(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "not-captioned.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(
        cli_main.app,
        ["ingest", str(document), "--caption-invoke-url", "http://vlm:8000/v1/chat/completions"],
    )

    assert result.exit_code == 1
    assert "Caption options require --caption" in result.output
    fake_ingestor.caption.assert_not_called()
    fake_ingestor.embed.assert_not_called()


def test_root_ingest_auto_passes_audio_params(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "meeting.wav"
    document.write_bytes(b"audio")
    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr("nemo_retriever.audio.asr_actor.asr_params_from_env", lambda: ASRParams(segment_audio=False))

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(document),
            "--segment-audio",
            "--audio-split-type",
            "time",
            "--audio-split-interval",
            "42",
        ],
    )

    assert result.exit_code == 0
    kwargs = fake_ingestor.extract.call_args.kwargs
    assert isinstance(kwargs["audio_chunk_params"], AudioChunkParams)
    assert kwargs["audio_chunk_params"].split_type == "time"
    assert kwargs["audio_chunk_params"].split_interval == 42
    assert isinstance(kwargs["asr_params"], ASRParams)
    assert kwargs["asr_params"].segment_audio is True


def test_root_ingest_auto_passes_video_params(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "demo.mp4"
    document.write_bytes(b"video")
    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr("nemo_retriever.audio.asr_actor.asr_params_from_env", lambda: ASRParams(segment_audio=False))

    result = RUNNER.invoke(
        cli_main.app,
        [
            "ingest",
            str(document),
            "--no-video-extract-audio",
            "--video-frame-fps",
            "0.25",
            "--no-video-frame-dedup",
            "--no-video-frame-text-dedup",
            "--video-frame-text-dedup-max-dropped-frames",
            "5",
            "--no-video-av-fuse",
        ],
    )

    assert result.exit_code == 0
    extract_params = fake_ingestor.extract.call_args.args[0]
    assert isinstance(extract_params, ExtractParams)
    assert extract_params.method == "pdfium"
    kwargs = fake_ingestor.extract.call_args.kwargs
    assert isinstance(kwargs["audio_chunk_params"], AudioChunkParams)
    assert kwargs["audio_chunk_params"].enabled is False
    assert isinstance(kwargs["video_frame_params"], VideoFrameParams)
    assert kwargs["video_frame_params"].fps == 0.25
    assert kwargs["video_frame_params"].dedup is False
    assert isinstance(kwargs["video_text_dedup_params"], VideoFrameTextDedupParams)
    assert kwargs["video_text_dedup_params"].enabled is False
    assert kwargs["video_text_dedup_params"].max_dropped_frames == 5
    assert isinstance(kwargs["av_fuse_params"], AudioVisualFuseParams)
    assert kwargs["av_fuse_params"].enabled is False


def test_root_ingest_rejects_removed_profiles(tmp_path) -> None:
    document = tmp_path / "manual.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document), "--profile", "ocr"])

    assert result.exit_code == 2
    assert "is not one of 'auto', 'fast-text'" in result.output


def test_root_ingest_routes_tiff_inputs_by_default_to_auto_planner(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "scan.tiff"
    document.write_bytes(b"tiff")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document)])

    assert result.exit_code == 0
    assert fake_ingestor.files.call_args.args == ([str(document)],)
    assert isinstance(fake_ingestor.extract.call_args.args[0], ExtractParams)
    assert fake_ingestor.extract.call_args.kwargs == {}


def test_root_ingest_auto_mixed_directory_uses_auto_extraction(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    dataset = tmp_path / "dataset"
    nested = dataset / "nested"
    nested.mkdir(parents=True)
    pdf = dataset / "manual.pdf"
    text = nested / "notes.txt"
    image = nested / "diagram.png"
    pdf.write_bytes(b"%PDF-1.4\n")
    text.write_text("notes", encoding="utf-8")
    image.write_bytes(b"png")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(dataset)])

    assert result.exit_code == 0
    assert set(fake_ingestor.files.call_args.args[0]) == {str(pdf.resolve()), str(text.resolve()), str(image.resolve())}
    assert isinstance(fake_ingestor.extract.call_args.args[0], ExtractParams)
    assert isinstance(fake_ingestor.extract.call_args.kwargs["text_params"], TextChunkParams)


def test_root_ingest_reports_os_errors(monkeypatch) -> None:
    def fail_ingest_documents(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise PermissionError("permission denied")

    monkeypatch.setattr(cli_main, "ingest_documents", fail_ingest_documents)

    result = RUNNER.invoke(cli_main.app, ["ingest", "blocked.pdf"])

    assert result.exit_code == 1
    assert "Error: permission denied" in result.output


def test_root_cli_error_handler_includes_pydantic_validation_error() -> None:
    assert ValidationError in cli_main._ROOT_CLI_ERRORS


def test_ingest_documents_validates_run_mode_before_creating_ingestor(monkeypatch) -> None:
    def fail_create_ingestor(**_kwargs: Any) -> Any:
        raise AssertionError("create_ingestor should not be called for an invalid run mode")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", fail_create_ingestor)

    with pytest.raises(ValueError, match="run_mode must be one of"):
        sdk_workflow.ingest_documents(["ignored.pdf"], run_mode="parallel")  # type: ignore[arg-type]


def test_silence_noisy_libraries_sets_env_vars(monkeypatch) -> None:
    for var in (
        "VLLM_LOGGING_LEVEL",
        "TRANSFORMERS_VERBOSITY",
        "HF_HUB_VERBOSITY",
        "TQDM_DISABLE",
        "HF_HUB_DISABLE_PROGRESS_BARS",
    ):
        monkeypatch.delenv(var, raising=False)

    cli_main._silence_noisy_libraries()

    assert os.environ["VLLM_LOGGING_LEVEL"] == "ERROR"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
    assert os.environ["HF_HUB_VERBOSITY"] == "error"
    assert os.environ["TQDM_DISABLE"] == "1"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert logging.getLogger("vllm").level == logging.ERROR
    assert logging.getLogger("transformers").level == logging.ERROR


def test_quiet_capture_swallows_output_on_success(capfd: pytest.CaptureFixture[str]) -> None:
    with cli_main._quiet_capture():
        sys.stdout.write("noisy stdout\n")
        sys.stdout.flush()
        sys.stderr.write("noisy stderr\n")
        sys.stderr.flush()

    captured = capfd.readouterr()
    assert "noisy stdout" not in captured.out
    assert "noisy stderr" not in captured.err


def test_quiet_capture_flushes_captured_output_to_stderr_on_error(
    capfd: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with cli_main._quiet_capture():
            sys.stdout.write("about to fail\n")
            sys.stdout.flush()
            sys.stderr.write("diagnostic detail\n")
            sys.stderr.flush()
            raise RuntimeError("boom")

    captured = capfd.readouterr()
    # Both stdout and stderr output from the failing block are surfaced on
    # stderr so an operator/agent can debug the failure.
    assert "about to fail" in captured.err
    assert "diagnostic detail" in captured.err
    assert captured.out == ""


def test_root_ingest_quiet_invokes_silencing_and_capture(monkeypatch, tmp_path) -> None:
    import contextlib

    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "quiet.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)
    monkeypatch.setattr(sdk_workflow, "_count_lancedb_rows", lambda *_, **__: 3)

    silenced: list[bool] = []
    monkeypatch.setattr(cli_main, "_silence_noisy_libraries", lambda: silenced.append(True))

    captured_use: list[bool] = []

    @contextlib.contextmanager
    def fake_quiet_capture() -> Any:
        captured_use.append(True)
        yield

    monkeypatch.setattr(cli_main, "_quiet_capture", fake_quiet_capture)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document), "--quiet"])

    assert result.exit_code == 0
    assert silenced == [True]
    assert captured_use == [True]
    assert "Ingested 1 file(s) → 3 row(s) in LanceDB lancedb/nemo-retriever." in result.output
