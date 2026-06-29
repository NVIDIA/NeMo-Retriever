# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_retriever.harness.contracts import HarnessRunError
from nemo_retriever.harness.portal_job import run_portal_job_entry
from nemo_retriever.harness.service_execution import (
    _harness_config_from_overrides,
    run_service_portal_job,
)
from nemo_retriever.service.service_ingestor import ServiceIngestResult


def _service_result(**kwargs) -> ServiceIngestResult:
    result = ServiceIngestResult([{"document_id": "doc-1"}])
    result.elapsed_s = kwargs.get("elapsed_s", 1.0)
    result.failures = list(kwargs.get("failures", []))
    result.document_ids = list(kwargs.get("document_ids", ["doc-1"]))
    result.document_filenames = dict(kwargs.get("document_filenames", {"doc-1": "doc.pdf"}))
    result.job_id = kwargs.get("job_id", "svc-job")
    result.job_status = kwargs.get("job_status", "completed")
    return result


def test_harness_config_from_overrides_requires_service_url() -> None:
    with pytest.raises(HarnessRunError, match="service_url is required"):
        _harness_config_from_overrides(
            dataset_dir="/tmp/data",
            dataset_label="jp20",
            preset="single_gpu",
            overrides={"run_mode": "service"},
        )


def test_harness_config_from_overrides_infers_beir_evaluation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    query_csv = tmp_path / "queries.csv"
    query_csv.write_text("query\n", encoding="utf-8")

    cfg = _harness_config_from_overrides(
        dataset_dir=str(data_dir),
        dataset_label="jp20",
        preset="single_gpu",
        overrides={
            "run_mode": "service",
            "service_url": "http://localhost:7670",
            "beir_loader": "jp20_csv",
            "query_csv": str(query_csv),
        },
    )
    assert cfg.evaluation_mode == "beir"
    assert cfg.beir_loader == "jp20_csv"


def test_run_portal_job_entry_rejects_missing_service_url(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    result = run_portal_job_entry(
        run_name="job-1",
        config_file=None,
        session_dir=None,
        dataset=str(tmp_path),
        preset=None,
        sweep_overrides={"run_mode": "service"},
    )
    assert result["success"] is False
    assert "service_url" in (result.get("failure_reason") or "")


@patch("nemo_retriever.harness.service_execution.evaluate_service_beir")
@patch("nemo_retriever.service.service_ingestor.ServiceIngestor")
def test_run_service_portal_job_ingest_only(
    mock_ingestor_cls: MagicMock,
    mock_evaluate_beir: MagicMock,
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    mock_ingestor_cls.return_value.ingest.return_value = _service_result(elapsed_s=2.5)

    outcome = run_service_portal_job(
        dataset=str(tmp_path),
        preset="single_gpu",
        overrides={
            "run_mode": "service",
            "service_url": "http://localhost:7670",
            "input_type": "pdf",
            "evaluation_mode": "none",
        },
        run_name="svc-test",
        session_dir=None,
    )

    assert outcome.exit_code == 0
    assert outcome.results["success"] is True
    assert outcome.results["test_config"]["service_url"] == "http://localhost:7670"
    assert outcome.results["summary_metrics"]["files"] == 1
    mock_evaluate_beir.assert_not_called()


@patch("nemo_retriever.harness.service_execution.evaluate_service_beir")
@patch("nemo_retriever.service.service_ingestor.ServiceIngestor")
def test_run_portal_job_entry_service_mode(
    mock_ingestor_cls: MagicMock,
    mock_evaluate_beir: MagicMock,
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    query_csv = tmp_path / "queries.csv"
    query_csv.write_text("query\n", encoding="utf-8")

    mock_ingestor_cls.return_value.ingest.return_value = _service_result(elapsed_s=1.0)
    mock_evaluate_beir.return_value = (
        type("BeirDataset", (), {"query_ids": ["q1"]})(),
        [],
        {},
        {"recall@5": 0.9, "ndcg@10": 0.8},
    )

    result = run_portal_job_entry(
        run_name="job-service",
        config_file=None,
        session_dir=None,
        dataset=str(tmp_path),
        preset="single_gpu",
        sweep_overrides={
            "run_mode": "service",
            "service_url": "http://localhost:7670",
            "input_type": "pdf",
            "evaluation_mode": "beir",
            "beir_loader": "jp20_csv",
            "beir_dataset_name": "jp20",
            "query_csv": str(query_csv),
        },
    )

    assert result["success"] is True
    assert result["test_config"]["run_mode"] == "service"
    assert result["summary_metrics"]["recall_5"] == 0.9
    mock_evaluate_beir.assert_called_once()
