# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import requests
from typer.testing import CliRunner

from nemo_retriever.harness.cli import app
from nemo_retriever.harness.slack import (
    DEFAULT_SLACK_METRIC_KEYS,
    HarnessRunReport,
    HarnessSessionReport,
    MAX_SLACK_TABLE_ROWS,
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
