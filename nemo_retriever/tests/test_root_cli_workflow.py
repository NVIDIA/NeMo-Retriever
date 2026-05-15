# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
from typing import Any
from unittest.mock import create_autospec

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import nemo_retriever.adapters.cli.sdk_workflow as sdk_workflow
from nemo_retriever.ingestor import ingestor as IngestorInterface
from nemo_retriever.params import EmbedParams, ExtractParams


RUNNER = CliRunner()
cli_main = importlib.import_module("nemo_retriever.adapters.cli.main")


def _make_fake_ingestor() -> Any:
    fake_ingestor = create_autospec(IngestorInterface, instance=True, spec_set=True)
    fake_ingestor.files.return_value = fake_ingestor
    fake_ingestor.extract.return_value = fake_ingestor
    fake_ingestor.embed.return_value = fake_ingestor
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
    assert fake_ingestor.extract.call_args.args == ()
    assert fake_ingestor.embed.call_args.args == ()
    vdb_upload_params = fake_ingestor.vdb_upload.call_args.args[0]
    assert vdb_upload_params.vdb_op == "lancedb"
    assert vdb_upload_params.vdb_kwargs == {"uri": "lancedb", "table_name": "nv-ingest", "overwrite": True}
    assert "Ingested 1 document(s) into LanceDB lancedb/nv-ingest." in result.output


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
    assert fake_ingestor.vdb_upload.call_args.args[0].vdb_kwargs == {
        "uri": "/tmp/lancedb",
        "table_name": "docs",
        "overwrite": True,
    }
    assert "Ingested 2 document(s) into LanceDB /tmp/lancedb/docs." in result.output


def test_root_ingest_append_forwards_overwrite_false(monkeypatch, tmp_path) -> None:
    fake_ingestor = _make_fake_ingestor()
    document = tmp_path / "multimodal_test.pdf"
    document.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(sdk_workflow, "create_ingestor", lambda **_kwargs: fake_ingestor)

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document), "--append"])

    assert result.exit_code == 0
    assert fake_ingestor.vdb_upload.call_args.args[0].vdb_kwargs == {
        "uri": "lancedb",
        "table_name": "nv-ingest",
        "overwrite": False,
    }


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
    assert "Ingested 1 document(s) into LanceDB lancedb/nv-ingest." in result.output


def test_root_ingest_reports_empty_directory_error(tmp_path) -> None:
    result = RUNNER.invoke(cli_main.app, ["ingest", str(tmp_path)])

    assert result.exit_code == 1
    assert "No PDF files found under directory" in result.output


def test_root_ingest_rejects_non_pdf_inputs(tmp_path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("not a pdf", encoding="utf-8")

    result = RUNNER.invoke(cli_main.app, ["ingest", str(document)])

    assert result.exit_code == 1
    assert "Only PDF inputs are supported by retriever ingest" in result.output


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


def test_root_query_passes_query_options_and_prints_json(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []
    query_calls: list[str] = []
    hits = [
        {"text": "passage", "page_number": 1, "_distance": 0.2},
        {"text": "other", "page_number": 2, "_distance": 0.4},
    ]

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str) -> list[dict[str, Any]]:
            query_calls.append(query)
            return hits

    monkeypatch.setattr(sdk_workflow, "Retriever", FakeRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "Which animal is responsible for typos?",
            "--top-k",
            "3",
            "--lancedb-uri",
            "/tmp/lancedb",
            "--table-name",
            "docs",
        ],
    )

    assert result.exit_code == 0
    assert retriever_calls == [{"top_k": 3, "vdb_kwargs": {"uri": "/tmp/lancedb", "table_name": "docs"}}]
    assert query_calls == ["Which animal is responsible for typos?"]
    assert json.loads(result.output) == hits
    assert result.output == json.dumps(hits, indent=2, sort_keys=True, default=str) + "\n"


def test_root_query_passes_embed_options(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []
    query_calls: list[str] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str) -> list[dict[str, Any]]:
            query_calls.append(query)
            return []

    monkeypatch.setattr(sdk_workflow, "Retriever", FakeRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "Which passages mention deployment?",
            "--embed-invoke-url",
            "http://embed:8000/v1/embeddings",
            "--embed-model-name",
            "nvidia/llama-nemotron-embed-1b-v2",
        ],
    )

    assert result.exit_code == 0
    assert retriever_calls == [
        {
            "top_k": 10,
            "vdb_kwargs": {"uri": "lancedb", "table_name": "nv-ingest"},
            "embed_kwargs": {
                "embed_invoke_url": "http://embed:8000/v1/embeddings",
                "embedding_endpoint": "http://embed:8000/v1/embeddings",
                "model_name": "nvidia/llama-nemotron-embed-1b-v2",
                "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
            },
        }
    ]
    assert query_calls == ["Which passages mention deployment?"]
    assert json.loads(result.output) == []


def test_root_query_passes_reranker_url(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []
    query_calls: list[str] = []
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str) -> list[dict[str, Any]]:
            query_calls.append(query)
            return []

    monkeypatch.setattr(sdk_workflow, "Retriever", FakeRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "Which passages mention deployment?",
            "--reranker-invoke-url",
            "http://rerank:8000/v1/ranking",
        ],
    )

    assert result.exit_code == 0
    assert retriever_calls == [
        {
            "top_k": 10,
            "vdb_kwargs": {"uri": "lancedb", "table_name": "nv-ingest"},
            "rerank": True,
            "rerank_kwargs": {
                "rerank_invoke_url": "http://rerank:8000/v1/ranking",
                "api_key": "nvapi-test",
            },
        }
    ]
    assert query_calls == ["Which passages mention deployment?"]
    assert json.loads(result.output) == []


def test_root_query_reports_os_errors(monkeypatch) -> None:
    def fail_query_documents(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise OSError("database unavailable")

    monkeypatch.setattr(cli_main, "query_documents", fail_query_documents)

    result = RUNNER.invoke(cli_main.app, ["query", "hello"])

    assert result.exit_code == 1
    assert "Error: database unavailable" in result.output
