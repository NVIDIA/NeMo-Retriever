# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import requests
from typer.testing import CliRunner

from nemo_retriever.harness.benchmark_registry import VIDORE_V3_PUBLIC_DATASETS
from nemo_retriever.harness.baselines import HarnessBaseline, load_baselines
from nemo_retriever.harness.cli import app
from nemo_retriever.harness.slack import (
    DEFAULT_SLACK_METRIC_KEYS,
    HarnessRunReport,
    HarnessSessionReport,
    MAX_SLACK_TABLE_ROWS,
    VIDORE_V3_REPORT_DATASETS,
    build_slack_payload,
    load_replay_report,
    load_session_report,
    post_slack_payload,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_session(tmp_path: Path, *, dry_run: bool = False) -> Path:
    artifact_dir = tmp_path / "001_jp20_beir"
    artifact_dir.mkdir()
    _write_json(
        artifact_dir / "results.json",
        {
            "benchmark": "jp20_beir",
            "dataset": "jp20",
            "success": True,
            "exit_code": 0,
            "dry_run": dry_run,
            "summary_metrics": {
                "files": 20,
                "pages": 1940,
                "query_count": 115,
                "recall_5": 0.887,
                "recall_10": 0.948,
            },
            "failure": None,
        },
    )
    _write_json(
        artifact_dir / "environment.json",
        {
            "git_sha": "abc1234",
            "host": "benchmark-host",
            "gpu_sku": "NVIDIA H100 NVL",
            "gpu_count": 8,
            "python": "3.12.12",
            "ray_version": "2.49.0",
        },
    )
    _write_json(
        tmp_path / "session_summary.json",
        {
            "session_name": "library_beir",
            "session_type": "runfiles",
            "run_commit": "abc1234",
            "success": True,
            "all_passed": True,
            "exit_code": 0,
            "dry_run": dry_run,
            "runs": [
                {
                    "run_name": "jp20_beir",
                    "benchmark": "jp20_beir",
                    "artifact_dir": artifact_dir.name,
                    "exit_code": 0,
                    "success": True,
                    "summary_metrics": {"pages": 1940, "recall_5": 0.887},
                }
            ],
        },
    )
    return tmp_path


def _vidore_run(dataset: str, index: int = 0, *, success: bool = True) -> HarnessRunReport:
    return HarnessRunReport(
        run_name=f"{dataset}_beir",
        dataset=dataset,
        preset=None,
        success=success,
        return_code=0 if success else 1,
        failure_reason=None if success else "benchmark failed",
        artifact_dir=None,
        metrics={
            "pages": 100,
            "ingest_secs": 5.0 * (index + 1),
            "recall_5": 0.1 * (index + 1),
            "ndcg_10": 0.8 - 0.1 * index,
        },
    )


def _vidore_report(tmp_path: Path, results: list[HarnessRunReport]) -> HarnessSessionReport:
    return HarnessSessionReport(
        session_name="vidore-v3",
        session_dir=tmp_path,
        session_type="runfiles",
        timestamp=None,
        latest_commit="abc1234",
        all_passed=all(run.success for run in results),
        dry_run=False,
        results=results,
    )


def _table_rows(table: dict) -> list[list[str]]:
    return [[cell["elements"][0]["elements"][0]["text"] for cell in row] for row in table["rows"]]


def test_slack_report_loads_runfile_session_and_omits_local_paths(tmp_path):
    session_dir = _write_session(tmp_path)

    report = load_session_report(session_dir)
    payload = build_slack_payload(
        report,
        {
            "title": "nemo-retriever library benchmarks",
            "metric_keys": DEFAULT_SLACK_METRIC_KEYS,
            "post_artifact_paths": False,
        },
    )
    payload_text = json.dumps(payload)

    assert report.session_name == "library_beir"
    assert report.all_passed is True
    assert report.latest_commit == "abc1234"
    assert report.results[0].dataset == "jp20"
    assert report.results[0].return_code == 0
    assert report.results[0].metrics == {"pages": 1940, "recall_5": 0.887}
    assert report.results[0].run_metadata["python_version"] == "3.12.12"
    assert "nemo-retriever library benchmarks" in payload_text
    assert "recall@5" in payload_text
    assert str(tmp_path) not in payload_text
    assert ["-    gpu_sku", "NVIDIA H100 NVL"] in _table_rows(payload["blocks"][3])


def test_slack_report_labels_dry_run_without_reporting_pass(tmp_path):
    report = load_session_report(_write_session(tmp_path, dry_run=True))

    payload = build_slack_payload(
        report,
        {
            "metric_keys": DEFAULT_SLACK_METRIC_KEYS,
            "post_artifact_paths": False,
        },
    )
    payload_text = json.dumps(payload)

    assert "DRY RUN" in payload_text
    assert "PASS" not in payload_text


def test_session_report_resolves_child_artifacts_after_session_is_moved(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    _write_session(original)
    moved = tmp_path / "moved"
    original.rename(moved)

    report = load_session_report(moved)

    assert report.results[0].artifact_dir == moved / "001_jp20_beir"
    assert report.results[0].dataset == "jp20"
    assert report.results[0].run_metadata["gpu_count"] == 8


def test_session_report_rejects_malformed_run_entries(tmp_path):
    summary = tmp_path / "session_summary.json"
    _write_json(summary, {"session_name": "corrupt", "all_passed": True, "runs": ["not-an-object"]})

    with pytest.raises(ValueError, match="run at index 0 must be an object"):
        load_session_report(summary)


def test_run_artifact_replay_has_deterministic_identity(tmp_path):
    session_dir = _write_session(tmp_path)
    results_path = session_dir / "001_jp20_beir" / "results.json"

    first = load_replay_report([results_path])
    second = load_replay_report([results_path])

    assert first.session_name == second.session_name == "artifact_replay"
    assert first.timestamp is second.timestamp is None


def test_slack_payload_truncates_tables_at_slack_row_limit(tmp_path):
    results = [
        HarnessRunReport(
            run_name=f"run-{index}",
            dataset=f"dataset-{index}",
            preset=None,
            success=True,
            return_code=0,
            failure_reason=None,
            artifact_dir=None,
            metrics={key: index for key in DEFAULT_SLACK_METRIC_KEYS},
        )
        for index in range(12)
    ]
    report = HarnessSessionReport(
        session_name="large-session",
        session_dir=tmp_path,
        session_type="runfiles",
        timestamp=None,
        latest_commit="abc1234",
        all_passed=True,
        dry_run=False,
        results=results,
    )

    payload = build_slack_payload(
        report,
        {"metric_keys": DEFAULT_SLACK_METRIC_KEYS, "post_artifact_paths": False},
    )
    table = next(block for block in payload["blocks"] if block["type"] == "table")

    assert len(table["rows"]) == MAX_SLACK_TABLE_ROWS
    assert "TRUNCATED" in json.dumps(table["rows"][-1])
    assert "rows omitted" in json.dumps(table["rows"][-1])


def test_slack_payload_adds_explicitly_labeled_baseline_comparison(tmp_path):
    report = load_session_report(_write_session(tmp_path))
    baseline = HarnessBaseline(
        name="26.05 directional reference",
        dataset="jp20",
        environment={"gpu_sku": "H100 80GB HBM3", "gpu_count": 8},
        metrics={"pages": 1940, "recall_5": 0.8},
        comparability="directional_only",
        notes="Different GPU SKU and harness profile.",
    )

    payload = build_slack_payload(
        report,
        {"metric_keys": DEFAULT_SLACK_METRIC_KEYS, "post_artifact_paths": False},
        baselines=[baseline],
    )

    assert payload["blocks"][-2]["text"]["text"].endswith("Different GPU SKU and harness profile.")
    assert _table_rows(payload["blocks"][-1]) == [
        ["METRIC", "CURRENT", "BASELINE", "DELTA"],
        ["gpu_sku", "NVIDIA H100 NVL", "H100 80GB HBM3", "N/A"],
        ["gpu_count", "8", "8", "N/A"],
        ["pages", "1940", "1940", "+0.0%"],
        ["recall@5", "0.887", "0.800", "+10.9%"],
    ]


def test_load_baselines_validates_private_reference_file(tmp_path):
    baseline_path = tmp_path / "baselines.json"
    _write_json(
        baseline_path,
        {
            "schema_version": 1,
            "baselines": [
                {
                    "name": "release reference",
                    "dataset": "bo767",
                    "environment": {"gpu_sku": "H100 80GB HBM3", "gpu_count": 8},
                    "metrics": {"ingest_secs": 4000.0},
                    "comparability": "directional_only",
                    "source": {"release": "26.05"},
                }
            ],
        },
    )

    baselines = load_baselines(baseline_path)

    assert baselines[0].dataset == "bo767"
    assert baselines[0].metrics == {"ingest_secs": 4000.0}
    assert baselines[0].source == {"release": "26.05"}


def test_slack_payload_summarizes_complete_vidore_v3_suite(tmp_path):
    assert set(VIDORE_V3_REPORT_DATASETS) == set(VIDORE_V3_PUBLIC_DATASETS)
    results = [_vidore_run(dataset, index) for index, dataset in enumerate(VIDORE_V3_REPORT_DATASETS)]

    payload = build_slack_payload(
        _vidore_report(tmp_path, results),
        {"metric_keys": DEFAULT_SLACK_METRIC_KEYS, "post_artifact_paths": False},
    )
    tables = [block for block in payload["blocks"] if block["type"] == "table"]

    assert len(tables) == 2
    main_rows = _table_rows(tables[0])
    assert ["-    ViDoRe v3", "PASS (8/8)"] in main_rows
    assert ["-    total ingest time", "180.00s (03m : 00.00s)"] in main_rows
    assert ["-    aggregate pages/s", "4.44"] in main_rows
    assert not any("vidore_v3_" in cell for row in main_rows for cell in row)

    accuracy_rows = _table_rows(tables[1])
    assert accuracy_rows[:3] == [
        ["DATASET", "RECALL@5", "NDCG@10"],
        ["Avg (English)", "0.400", "0.500"],
        ["Avg (all)", "0.450", "0.450"],
    ]
    assert accuracy_rows[3] == ["finance_en", "0.100", "0.800"]
    assert accuracy_rows[-1] == ["finance_fr", "0.800", "0.100"]


def test_slack_payload_uses_compact_table_for_partial_vidore_v3_session(tmp_path):
    run = _vidore_run("vidore_v3_finance_en")

    payload = build_slack_payload(
        _vidore_report(tmp_path, [run]),
        {"metric_keys": DEFAULT_SLACK_METRIC_KEYS, "post_artifact_paths": False},
    )
    tables = [block for block in payload["blocks"] if block["type"] == "table"]

    assert len(tables) == 2
    assert _table_rows(tables[1]) == [
        ["DATASET", "RECALL@5", "NDCG@10"],
        ["finance_en", "0.100", "0.800"],
    ]


def test_slack_payload_does_not_average_failed_vidore_v3_suite(tmp_path):
    results = [_vidore_run(dataset, index) for index, dataset in enumerate(VIDORE_V3_REPORT_DATASETS)]
    results[-1] = _vidore_run("vidore_v3_finance_fr", len(results) - 1, success=False)

    payload = build_slack_payload(
        _vidore_report(tmp_path, results),
        {"metric_keys": DEFAULT_SLACK_METRIC_KEYS, "post_artifact_paths": False},
    )
    tables = [block for block in payload["blocks"] if block["type"] == "table"]

    assert ["Avg (all)", "N/A", "N/A"] in _table_rows(tables[1])
    assert ["-    finance_fr failure", "benchmark failed"] in _table_rows(tables[0])


def test_slack_payload_keeps_duplicate_vidore_runs_without_claiming_suite_averages(tmp_path):
    results = [_vidore_run(dataset, index) for index, dataset in enumerate(VIDORE_V3_REPORT_DATASETS)]
    results.append(_vidore_run("vidore_v3_finance_en"))

    payload = build_slack_payload(
        _vidore_report(tmp_path, results),
        {"metric_keys": DEFAULT_SLACK_METRIC_KEYS, "post_artifact_paths": False},
    )
    tables = [block for block in payload["blocks"] if block["type"] == "table"]
    main_rows = _table_rows(tables[0])
    accuracy_rows = _table_rows(tables[1])

    assert ["-    ViDoRe v3", "PASS (9/9)"] in main_rows
    assert ["-    total ingest time", "185.00s (03m : 05.00s)"] in main_rows
    assert ["-    aggregate pages/s", "4.86"] in main_rows
    assert not any(row[0].startswith("Avg") for row in accuracy_rows)
    assert sum(row[0] == "finance_en" for row in accuracy_rows) == 2


def test_slack_transport_error_does_not_expose_webhook(monkeypatch):
    webhook = "https://hooks.slack.com/services/TSECRET/BSECRET/XSECRET"

    def fail_post(*args, **kwargs):
        raise requests.ConnectionError(f"connection failed for {webhook}")

    monkeypatch.setattr(requests, "post", fail_post)

    with pytest.raises(RuntimeError) as exc_info:
        post_slack_payload({"text": "test"}, webhook)

    assert "TSECRET" not in str(exc_info.value)
    assert "webhook request could not be completed" in str(exc_info.value)


def test_slack_preview_matches_posted_payload_without_requiring_webhook(monkeypatch, tmp_path):
    session_dir = _write_session(tmp_path)
    runner = CliRunner()
    posted = []

    def capture_post(payload, webhook_url):
        posted.append((payload, webhook_url))

    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.setattr("nemo_retriever.harness.cli.post_slack_payload", capture_post)
    common_args = [
        "post-slack",
        "--title",
        "nemo-retriever library benchmarks",
        str(session_dir),
    ]

    preview_result = runner.invoke(app, [*common_args, "--preview"])

    assert preview_result.exit_code == 0
    assert posted == []
    preview_payload = json.loads(preview_result.stdout)

    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/test")
    post_result = runner.invoke(app, [*common_args, "--json"])

    assert post_result.exit_code == 0
    assert len(posted) == 1
    assert posted[0][1] == "https://hooks.slack.com/services/test"
    assert preview_payload == posted[0][0]
    assert preview_payload == json.loads(post_result.stdout)
    assert str(session_dir) not in json.dumps(preview_payload)
