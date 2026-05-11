# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
from typing import Any
from unittest.mock import create_autospec

import pytest
from typer.testing import CliRunner

import nemo_retriever.adapters.cli.sdk_workflow as sdk_workflow
from nemo_retriever.ingestor import ingestor as IngestorInterface


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
    vdb_upload_params = fake_ingestor.vdb_upload.call_args.args[0]
    assert vdb_upload_params.vdb_op == "lancedb"
    assert vdb_upload_params.vdb_kwargs == {"uri": "lancedb", "table_name": "nv-ingest"}
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
    assert fake_ingestor.vdb_upload.call_args.args[0].vdb_kwargs == {"uri": "/tmp/lancedb", "table_name": "docs"}
    assert "Ingested 2 document(s) into LanceDB /tmp/lancedb/docs." in result.output


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
