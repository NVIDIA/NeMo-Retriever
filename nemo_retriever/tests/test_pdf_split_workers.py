"""Native document splitting concurrency and its CLI-to-runtime contract."""

import importlib

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from nemo_retriever.common.params import BatchTuningParams, ExtractParams
from nemo_retriever.graph.ingestor_runtime import batch_tuning_to_node_overrides
from nemo_retriever.ingest.plan import IngestPlanRequest, IngestSourceOptions, resolve_ingest_plan


def test_splitter_document_batch_and_workers_are_independent_of_page_extraction():
    params = ExtractParams(
        batch_tuning=BatchTuningParams(
            pdf_split_workers=8, pdf_split_batch_size=2, pdf_extract_workers=32, pdf_extract_batch_size=16
        )
    )
    overrides = batch_tuning_to_node_overrides(extract_params=params, embed_params=None)
    assert overrides["PDFSplitActor"] == {"batch_size": 2, "concurrency": 8}
    assert overrides["PDFExtractionActor"]["concurrency"] == 32
    assert overrides["PDFExtractionActor"]["batch_size"] == 16


def test_default_splitter_concurrency_is_unchanged():
    overrides = batch_tuning_to_node_overrides(extract_params=ExtractParams(), embed_params=None)
    assert "concurrency" not in overrides.get("PDFSplitActor", {})


@pytest.mark.parametrize("workers", [0, -1])
def test_splitter_requires_positive_worker_count(workers):
    with pytest.raises(ValidationError):
        BatchTuningParams(pdf_split_workers=workers)


def test_batch_cli_forwards_splitters_through_plan_to_native_node(tmp_path, monkeypatch):
    from nemo_retriever.cli.ingest import graph_commands

    main = importlib.import_module("nemo_retriever.cli.main")
    document = tmp_path / "one.pdf"
    document.write_bytes(b"pdf")
    captured = []

    def capture(values, *, run_mode):
        assert run_mode == "batch"
        options = graph_commands._build_extract_options(
            values, batch=graph_commands._build_extract_batch_options(values, enabled=True)
        )
        plan = resolve_ingest_plan(
            IngestPlanRequest(source=IngestSourceOptions(documents=[str(document)]), extract=options)
        )
        captured.append(batch_tuning_to_node_overrides(extract_params=plan.extract_params, embed_params=None))

    monkeypatch.setattr(graph_commands, "_run_graph_ingest_from_parsed_options", capture)
    result = CliRunner().invoke(
        main.app,
        ["ingest", "batch", str(document), "--pdf-split-workers", "8", "--pdf-split-batch-size", "2"],
    )
    assert result.exit_code == 0, result.output
    assert captured[0]["PDFSplitActor"] == {"batch_size": 2, "concurrency": 8}


def test_splitter_workers_option_is_rejected_for_inprocess_ingestion(tmp_path):
    main = importlib.import_module("nemo_retriever.cli.main")
    document = tmp_path / "one.pdf"
    document.write_bytes(b"pdf")
    result = CliRunner().invoke(main.app, ["ingest", "local", str(document), "--pdf-split-workers", "8"])
    assert result.exit_code != 0
    assert "--pdf-split-workers" in result.output and "batch" in result.output
