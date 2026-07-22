# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
import json
import subprocess

import pytest

from nemo_retriever.harness.artifact_writer import (
    append_text,
    artifact_paths,
    ArtifactWriter,
    capture_output_to_log,
    redact,
)
from nemo_retriever.harness.beir_runner import _write_trec_run
from nemo_retriever.harness.contracts import (
    EXIT_ARTIFACT_WRITE_FAILURE,
    EXIT_INGEST_FAILURE,
    EXIT_INTERNAL_ERROR,
    FailurePayload,
    HarnessRunError,
)
from nemo_retriever.harness.diff import diff_artifact_dirs
from nemo_retriever.harness.environment import collect_environment
from nemo_retriever.harness.execution import _concise_message, _run_result_payload, _write_failure_result, run_benchmark


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_last_commit_preserves_the_full_git_sha(monkeypatch):
    import nemo_retriever.harness.artifacts as artifacts

    full_sha = "abcdef0123456789abcdef0123456789abcdef01"
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout=f"{full_sha}\n", stderr=""),
    )

    assert artifacts.last_commit() == full_sha


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [(0, "", False), (0, " M local.py\n", True), (128, "", None)],
)
def test_working_tree_dirty_reports_git_state(monkeypatch, returncode, stdout, expected):
    import nemo_retriever.harness.artifacts as artifacts

    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode, stdout=stdout, stderr=""),
    )

    assert artifacts.working_tree_dirty() is expected


def test_artifact_manifest_uses_relative_paths_and_includes_lancedb(tmp_path):
    writer = ArtifactWriter(artifact_dir=tmp_path, run_id="run-1", benchmark="jp20_beir")
    writer.status(status="running", phase="ingest")
    writer.path("environment.json").write_text("{}", encoding="utf-8")
    writer.path("lancedb").mkdir()

    assert artifact_paths(writer) == {
        "status": "status.json",
        "events": "events.jsonl",
        "environment": "environment.json",
        "lancedb": "lancedb",
    }


def test_artifact_writer_removes_only_stale_harness_outputs(tmp_path):
    (tmp_path / "summary_metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "beir_metrics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "lancedb").mkdir()
    (tmp_path / "lancedb" / "stale").write_text("old", encoding="utf-8")
    (tmp_path / "keep-me.txt").write_text("user-owned", encoding="utf-8")

    ArtifactWriter(artifact_dir=tmp_path, run_id="run-1", benchmark="jp20_beir")

    assert not (tmp_path / "summary_metrics.json").exists()
    assert not (tmp_path / "beir_metrics.json").exists()
    assert not (tmp_path / "lancedb").exists()
    assert (tmp_path / "keep-me.txt").read_text(encoding="utf-8") == "user-owned"


def test_write_json_preserves_previous_file_when_publish_fails(monkeypatch, tmp_path):
    import nemo_retriever.harness.json_io as json_io

    target = tmp_path / "status.json"
    target.write_text('{"status": "old"}\n', encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr(json_io.os, "replace", fail_replace)

    with pytest.raises(HarnessRunError) as error:
        json_io.write_json(target, {"status": "new"})

    assert error.value.exit_code == EXIT_ARTIFACT_WRITE_FAILURE
    assert target.read_text(encoding="utf-8") == '{"status": "old"}\n'
    assert list(tmp_path.glob(".status.json.*.tmp")) == []


def test_write_json_classifies_serialization_failures(tmp_path):
    from nemo_retriever.harness.json_io import write_json

    class Unserializable:
        def __str__(self):
            raise ValueError("cannot serialize")

    with pytest.raises(HarnessRunError) as error:
        write_json(tmp_path / "status.json", {"value": Unserializable()})

    assert error.value.exit_code == EXIT_ARTIFACT_WRITE_FAILURE
    assert not (tmp_path / "status.json").exists()


def test_redact_recurses_into_structured_override_values():
    override = 'query={"reranker_api_key":"secret-value","top_k":10}'

    assert redact(override) == 'query={"reranker_api_key": "<redacted>", "top_k": 10}'


def test_redact_preserves_token_limits():
    payload = {
        "reranker_api_key": "secret-value",
        "webhookUrl": "https://hooks.example.invalid/secret",
        "webhooks": ["https://hooks.example.invalid/one"],
        "passwords": ["password-value"],
        "client_secrets": ["client-secret"],
        "access_tokens": ["access-token"],
        "caption_max_tokens": 77,
        "text_chunk_overlap_tokens": 12,
        "tokenizer_model": "model-name",
    }

    assert redact(payload) == {
        "reranker_api_key": "<redacted>",
        "webhookUrl": "<redacted>",
        "webhooks": "<redacted>",
        "passwords": "<redacted>",
        "client_secrets": "<redacted>",
        "access_tokens": "<redacted>",
        "caption_max_tokens": 77,
        "text_chunk_overlap_tokens": 12,
        "tokenizer_model": "model-name",
    }


def test_text_artifact_writes_are_classified(tmp_path):
    with pytest.raises(HarnessRunError) as log_error:
        append_text(tmp_path, "log line")
    with pytest.raises(HarnessRunError) as trec_error:
        _write_trec_run(tmp_path, {"query": {"document": 1.0}})

    assert log_error.value.exit_code == EXIT_ARTIFACT_WRITE_FAILURE
    assert trec_error.value.exit_code == EXIT_ARTIFACT_WRITE_FAILURE


def test_invalid_run_config_preserves_existing_artifacts(tmp_path):
    (tmp_path / "results.json").write_text('{"old": true}', encoding="utf-8")
    (tmp_path / "lancedb").mkdir()
    (tmp_path / "lancedb" / "old-index").write_text("existing", encoding="utf-8")

    with pytest.raises(HarnessRunError):
        run_benchmark(
            "jp20_beir",
            output_dir=str(tmp_path),
            requirements=("not-a-gate",),
            dry_run=True,
        )

    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8")) == {"old": True}
    assert (tmp_path / "lancedb" / "old-index").read_text(encoding="utf-8") == "existing"


def test_result_payload_is_a_small_terminal_manifest(tmp_path):
    writer = ArtifactWriter(artifact_dir=tmp_path, run_id="run-1", benchmark="jp20_beir")
    writer.status(status="running", phase="query")
    writer.path("resolved_benchmark.json").write_text("{}", encoding="utf-8")
    writer.path("beir_metrics.json").write_text("{}", encoding="utf-8")
    writer.path("runfile.json").write_text("{}", encoding="utf-8")

    payload = _run_result_payload(
        writer,
        status="complete",
        success=True,
        exit_code=0,
        dry_run=False,
        resolved={"dataset": {"name": "jp20"}, "ingest": {"run_mode": "batch"}},
        summary_metrics={"files": 20},
        failure=None,
    )

    assert payload == {
        "run_id": "run-1",
        "benchmark": "jp20_beir",
        "dataset": "jp20",
        "mode": "batch",
        "status": "complete",
        "success": True,
        "exit_code": 0,
        "dry_run": False,
        "summary_metrics": {"files": 20},
        "failure": None,
        "artifacts": {
            "status": "status.json",
            "events": "events.jsonl",
            "runfile": "runfile.json",
            "resolved_benchmark": "resolved_benchmark.json",
            "beir_metrics": "beir_metrics.json",
        },
    }


def test_failure_result_is_concise_and_points_to_full_log(tmp_path):
    writer = ArtifactWriter(artifact_dir=tmp_path, run_id="run-1", benchmark="jp20_beir")
    writer.status(status="running", phase="ingest")
    writer.path("run.log").write_text("full traceback evidence", encoding="utf-8")
    long_message = "actor failed\n" + "traceback detail\n" * 1000

    payload = _write_failure_result(
        writer,
        failure=FailurePayload(
            failed_phase="ingest",
            failure_reason="ingest_failed",
            retryable=False,
            message=long_message,
            debug_artifacts=("ingest_plan.json", "run.log"),
        ),
        exit_code=EXIT_INGEST_FAILURE,
        dry_run=False,
        resolved={"dataset": {"name": "jp20"}, "ingest": {"run_mode": "inprocess"}},
        summary_metrics=None,
    )

    status = json.loads(writer.path("status.json").read_text(encoding="utf-8"))
    assert payload["failure"]["message"] == "actor failed"
    assert "status_payload" not in payload
    assert "resolved_benchmark" not in payload
    assert status["failure"] == payload["failure"]
    assert status["results_path"] == "results.json"
    assert not writer.path("summary_metrics.json").exists()


def test_run_benchmark_classifies_artifact_write_failures(monkeypatch, tmp_path):
    import nemo_retriever.harness.execution as execution

    original_write_json = execution.write_json

    def fail_results_write(path, payload):
        if path.name == "results.json":
            raise OSError("disk full")
        original_write_json(path, payload)

    monkeypatch.setattr(execution, "write_json", fail_results_write)

    outcome = run_benchmark(
        "jp20_beir",
        output_dir=str(tmp_path / "run"),
        overrides=(f'dataset.path="{tmp_path}"',),
        dry_run=True,
    )

    assert outcome.exit_code == EXIT_ARTIFACT_WRITE_FAILURE
    assert outcome.results["failure"]["failure_reason"] == "artifact_write_failed"
    assert outcome.results["failure"]["message"] == "OSError: disk full"


def test_run_benchmark_classifies_artifact_writer_initialization_failures(monkeypatch, tmp_path):
    import nemo_retriever.harness.execution as execution

    def fail_writer(**kwargs):
        raise OSError("artifact directory unavailable")

    monkeypatch.setattr(execution, "ArtifactWriter", fail_writer)

    with pytest.raises(HarnessRunError) as error:
        run_benchmark(
            "jp20_beir",
            output_dir=str(tmp_path / "run"),
            overrides=(f'dataset.path="{tmp_path}"',),
            dry_run=True,
        )

    assert error.value.exit_code == EXIT_ARTIFACT_WRITE_FAILURE
    assert error.value.failure.failure_reason == "artifact_write_failed"
    assert error.value.failure.message == "OSError: artifact directory unavailable"


def test_run_benchmark_preserves_run_log_write_failure(monkeypatch, tmp_path):
    import nemo_retriever.harness.execution as execution
    from nemo_retriever.harness.json_io import artifact_write_error

    class FakeIngestPlan:
        documents = ()

    @contextmanager
    def fail_log_capture(*args, **kwargs):
        raise artifact_write_error(OSError("run log unavailable"))
        yield

    monkeypatch.setattr(execution, "resolve_ingest_plan", lambda request: FakeIngestPlan())
    monkeypatch.setattr(execution, "run_ingest_workflow", lambda plan, dry_run: {})
    monkeypatch.setattr(execution, "resolve_query_plan", lambda request: object())
    monkeypatch.setattr(execution, "query_plan_payload", lambda plan: {})
    monkeypatch.setattr(execution, "capture_output_to_log", fail_log_capture)

    outcome = run_benchmark(
        "jp20_smoke",
        output_dir=str(tmp_path / "run"),
        overrides=(f'dataset.path="{tmp_path}"',),
    )

    assert outcome.exit_code == EXIT_ARTIFACT_WRITE_FAILURE
    assert outcome.results["failure"]["failure_reason"] == "artifact_write_failed"
    assert outcome.results["failure"]["message"] == "OSError: run log unavailable"


def test_run_benchmark_keeps_non_write_failures_internal(monkeypatch, tmp_path):
    import nemo_retriever.harness.execution as execution

    class FakeIngestPlan:
        documents = ()

    def fail_summary_metrics(*args, **kwargs):
        raise RuntimeError("metrics exploded")

    monkeypatch.setattr(execution, "resolve_ingest_plan", lambda request: FakeIngestPlan())
    monkeypatch.setattr(execution, "run_ingest_workflow", lambda plan, dry_run: {})
    monkeypatch.setattr(execution, "resolve_query_plan", lambda request: object())
    monkeypatch.setattr(execution, "query_plan_payload", lambda plan: {})
    monkeypatch.setattr(execution, "build_summary_metrics", fail_summary_metrics)

    outcome = run_benchmark(
        "jp20_beir",
        output_dir=str(tmp_path / "run"),
        overrides=(f'dataset.path="{tmp_path}"',),
        dry_run=True,
    )

    assert outcome.exit_code == EXIT_INTERNAL_ERROR
    assert outcome.results["failure"]["failure_reason"] == "unexpected_internal_error"
    assert outcome.results["failure"]["message"] == "RuntimeError: metrics exploded"


def test_concise_message_prefers_nested_root_exception():
    message = """The actor died during startup
Traceback (most recent call last):
requests.exceptions.HTTPError: 429 Client Error
huggingface_hub.errors.LocalEntryNotFoundError: model weights are not cached
"""

    assert _concise_message(message) == ("huggingface_hub.errors.LocalEntryNotFoundError: model weights are not cached")


def test_capture_output_records_exception_traceback(tmp_path):
    log_path = tmp_path / "run.log"

    with pytest.raises(RuntimeError, match="model startup failed"):
        with capture_output_to_log(log_path, label="ingest"):
            raise RuntimeError("model startup failed")

    text = log_path.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in text
    assert "RuntimeError: model startup failed" in text
    assert "ingest failed" in text


def test_diff_prefers_results_but_reads_legacy_summary(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    _write_json(left / "results.json", {"summary_metrics": {"recall_5": 0.8}})
    _write_json(right / "summary_metrics.json", {"recall_5": 0.9})

    payload = diff_artifact_dirs(left, right)

    assert payload["left"].endswith("left/results.json")
    assert payload["right"].endswith("right/summary_metrics.json")
    assert payload["summary_metrics"]["recall_5"]["delta"] == pytest.approx(0.1)


def test_environment_records_relevant_runtime_flags_without_credentials(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("NEMO_RETRIEVER_HF_CACHE_DIR", "/models/huggingface")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.example.invalid/secret")

    payload = collect_environment()

    assert payload["runtime_environment"]["HF_HUB_OFFLINE"] == "1"
    assert payload["runtime_environment"]["NEMO_RETRIEVER_HF_CACHE_DIR"] == "/models/huggingface"
    assert "SLACK_WEBHOOK_URL" not in payload["runtime_environment"]
