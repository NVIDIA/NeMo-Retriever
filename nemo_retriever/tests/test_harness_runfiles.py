# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_retriever.harness.contracts import (
    EXIT_ARTIFACT_WRITE_FAILURE,
    EXIT_INTERNAL_ERROR,
    EXIT_MISSING_INPUT,
    FailurePayload,
    HarnessRunError,
    RunOutcome,
)
from nemo_retriever.harness.runsets import _run_prepared_benchmark_isolated, run_runfiles, run_runset


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _successful_outcome(benchmark: str, output_dir: str) -> RunOutcome:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "benchmark": benchmark,
        "dataset": "jp20",
        "success": True,
        "exit_code": 0,
        "summary_metrics": {"files": 20},
        "failure": None,
    }
    results_path = artifact_dir / "results.json"
    _write_json(results_path, results)
    return RunOutcome(exit_code=0, artifact_dir=artifact_dir, results=results, results_path=results_path)


def test_run_files_applies_machine_paths_then_cli_overrides(monkeypatch, tmp_path):
    runfile = tmp_path / "jp20_beir.json"
    _write_json(
        runfile,
        {
            "schema_version": 1,
            "name": "jp20_beir",
            "benchmark": "jp20_beir",
            "mode": "batch",
            "require": ["files==20"],
            "set": {"query.top_k": 10},
        },
    )
    documents = tmp_path / "datasets" / "jp20"
    query_file = tmp_path / "datasets" / "jp20_query_gt.csv"
    documents.mkdir(parents=True)
    query_file.write_text("query_id,query\n1,test\n", encoding="utf-8")
    dataset_paths = tmp_path / "dataset_paths.yaml"
    dataset_paths.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "datasets:",
                "  jp20:",
                f"    path: {documents}",
                f"    query_file: {query_file}",
            )
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run_benchmark(prepared, **kwargs):
        calls.append((prepared, kwargs))
        return _successful_outcome(prepared.benchmark, kwargs["output_dir"])

    monkeypatch.setattr("nemo_retriever.harness.runsets.run_prepared_benchmark", fake_run_benchmark)
    monkeypatch.setattr("nemo_retriever.harness.runsets.working_tree_dirty", lambda: True)

    outcome = run_runfiles(
        [runfile],
        output_dir=str(tmp_path / "session"),
        session_name="library_beir",
        dataset_paths_file=dataset_paths,
        overrides=("query.top_k=20",),
        requirements=("pages==1940",),
    )

    prepared, kwargs = calls[0]
    assert prepared.benchmark == "jp20_beir"
    assert prepared.mode == "batch"
    assert prepared.requirements == ("files==20", "pages==1940")
    assert prepared.overrides == (
        "query.top_k=10",
        f'dataset.path="{documents}"',
        f'dataset.query_file="{query_file}"',
        f'evaluation.dataset_name="{query_file}"',
        "query.top_k=20",
    )
    assert outcome.results["session_name"] == "library_beir"
    assert outcome.results["working_tree_dirty"] is True
    assert outcome.results["runs"][0]["dataset"] == "jp20"
    assert outcome.results["runs"][0]["artifact_dir"] == "001_jp20_beir"
    assert outcome.results["runs"][0]["results_path"] == "001_jp20_beir/results.json"
    assert "results" not in outcome.results

    expanded = json.loads((outcome.artifact_dir / "expanded_runs.json").read_text(encoding="utf-8"))
    assert expanded["working_tree_dirty"] is True
    assert expanded["runfiles"][0]["dataset_paths"] == {
        "path": str(documents),
        "query_file": str(query_file),
    }
    assert expanded["runfiles"][0]["artifact_dir"] == "001_jp20_beir"


def test_run_files_applies_service_endpoint_only_to_service_children(monkeypatch, tmp_path):
    from nemo_retriever.harness.execution import PreparedBenchmark

    runfiles = []
    for name, mode in (("jp20_beir", "service"), ("bo767_beir", "local")):
        path = tmp_path / f"{name}.json"
        _write_json(path, {"schema_version": 1, "name": name, "benchmark": name, "mode": mode})
        runfiles.append(path)

    calls = []

    def fake_preflight(benchmark, **kwargs):
        calls.append((benchmark, kwargs["service_endpoint"]))
        return PreparedBenchmark(
            benchmark=benchmark,
            mode=kwargs["mode"],
            overrides=tuple(kwargs["overrides"]),
            requirements=tuple(kwargs["requirements"]),
            dry_run=kwargs["dry_run"],
            resolved={"dataset": {"name": benchmark}, "ingest": {}},
            dataset_path=tmp_path,
        )

    monkeypatch.setattr("nemo_retriever.harness.runsets.preflight_benchmark", fake_preflight)
    monkeypatch.setattr(
        "nemo_retriever.harness.runsets.run_prepared_benchmark",
        lambda prepared, **kwargs: _successful_outcome(prepared.benchmark, kwargs["output_dir"]),
    )

    outcome = run_runfiles(
        runfiles,
        output_dir=str(tmp_path / "session"),
        service_endpoint="http://localhost:17670",
        dry_run=True,
    )

    assert outcome.exit_code == 0
    assert calls == [("jp20_beir", "http://localhost:17670"), ("bo767_beir", None)]


def test_run_files_completes_remaining_runs_and_preserves_first_failure(monkeypatch, tmp_path):
    runfiles = []
    for name in ("jp20_beir", "bo767_beir"):
        path = tmp_path / f"{name}.json"
        _write_json(path, {"schema_version": 1, "name": name, "benchmark": name})
        runfiles.append(path)

    calls = []

    def fake_run_benchmark(prepared, **kwargs):
        benchmark = prepared.benchmark
        calls.append(benchmark)
        if benchmark == "jp20_beir":
            artifact_dir = Path(kwargs["output_dir"])
            artifact_dir.mkdir(parents=True)
            results_path = artifact_dir / "results.json"
            _write_json(results_path, {"summary_metrics": {}, "failure": {"message": "ingest failed"}})
            return RunOutcome(
                exit_code=10,
                artifact_dir=artifact_dir,
                results={
                    "summary_metrics": {},
                    "failure": {"message": "ingest failed"},
                },
                results_path=results_path,
            )
        return _successful_outcome(benchmark, kwargs["output_dir"])

    monkeypatch.setattr("nemo_retriever.harness.runsets.run_prepared_benchmark", fake_run_benchmark)

    outcome = run_runfiles(
        runfiles,
        output_dir=str(tmp_path / "session"),
        overrides=(f'dataset.path="{tmp_path}"',),
        dry_run=True,
    )

    assert calls == ["jp20_beir", "bo767_beir"]
    assert outcome.exit_code == 10
    assert outcome.results["all_passed"] is False
    assert [run["exit_code"] for run in outcome.results["runs"]] == [10, 0]


def test_run_files_can_isolate_each_child_process(monkeypatch, tmp_path):
    runfiles = []
    for name in ("jp20_beir", "bo767_beir"):
        path = tmp_path / f"{name}.json"
        _write_json(path, {"schema_version": 1, "name": name, "benchmark": name})
        runfiles.append(path)

    isolated_calls = []

    def fake_isolated_run(run, *, output_dir, run_id, timeout_seconds):
        isolated_calls.append((run.prepared.benchmark, run_id, timeout_seconds))
        return _successful_outcome(run.prepared.benchmark, output_dir)

    monkeypatch.setattr(
        "nemo_retriever.harness.runsets._run_prepared_benchmark_isolated",
        fake_isolated_run,
    )
    monkeypatch.setattr(
        "nemo_retriever.harness.runsets.run_prepared_benchmark",
        lambda *args, **kwargs: pytest.fail("in-process runner should not be used"),
    )

    outcome = run_runfiles(
        runfiles,
        output_dir=str(tmp_path / "session"),
        session_name="isolated",
        overrides=(f'dataset.path="{tmp_path}"',),
        dry_run=True,
        isolate_runs=True,
    )

    assert outcome.exit_code == 0
    assert outcome.results["isolate_runs"] is True
    assert isolated_calls == [
        ("jp20_beir", "isolated_001_jp20_beir", 6 * 60 * 60),
        ("bo767_beir", "isolated_002_bo767_beir", 6 * 60 * 60),
    ]
    assert len(outcome.results["runs"]) == 2


def test_run_files_isolated_timeout_records_failure_and_continues(monkeypatch, tmp_path):
    runfiles = []
    for name in ("jp20_beir", "bo767_beir"):
        path = tmp_path / f"{name}.json"
        _write_json(path, {"schema_version": 1, "name": name, "benchmark": name})
        runfiles.append(path)

    isolated_calls = []

    def fake_isolated_run(run, *, output_dir, run_id, timeout_seconds):
        isolated_calls.append((run.prepared.benchmark, timeout_seconds))
        if run.prepared.benchmark == "jp20_beir":
            raise TimeoutError("isolated benchmark exceeded the child timeout")
        return _successful_outcome(run.prepared.benchmark, output_dir)

    monkeypatch.setattr(
        "nemo_retriever.harness.runsets._run_prepared_benchmark_isolated",
        fake_isolated_run,
    )

    outcome = run_runfiles(
        runfiles,
        output_dir=str(tmp_path / "session"),
        overrides=(f'dataset.path="{tmp_path}"',),
        dry_run=True,
        isolate_runs=True,
        isolated_child_timeout_seconds=17,
    )

    assert outcome.exit_code == EXIT_INTERNAL_ERROR
    assert isolated_calls == [("jp20_beir", 17), ("bo767_beir", 17)]
    assert [run["success"] for run in outcome.results["runs"]] == [False, True]
    assert "TimeoutError" in outcome.results["runs"][0]["failure_reason"]
    assert outcome.results["isolated_child_timeout_seconds"] == 17


def test_isolated_child_timeout_uses_bounded_terminate_and_kill(monkeypatch):
    class FakeConnection:
        def __init__(self, *, readable):
            self.readable = readable
            self.closed = False
            self.poll_timeout = None

        def poll(self, timeout):
            self.poll_timeout = timeout
            return self.readable

        def recv(self):
            raise AssertionError("recv should not be called when poll times out")

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self):
            self.alive = True
            self.exitcode = None
            self.started = False
            self.terminated = False
            self.killed = False
            self.join_timeouts = []

        def start(self):
            self.started = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.alive = False
            self.exitcode = -9

        def join(self, timeout):
            self.join_timeouts.append(timeout)

    receive_connection = FakeConnection(readable=False)
    send_connection = FakeConnection(readable=False)
    process = FakeProcess()

    class FakeContext:
        @staticmethod
        def Pipe(*, duplex):
            assert duplex is False
            return receive_connection, send_connection

        @staticmethod
        def Process(**_kwargs):
            return process

    monkeypatch.setattr(
        "nemo_retriever.harness.runsets.multiprocessing.get_context",
        lambda method: FakeContext(),
    )

    run = SimpleNamespace(name="hung", artifact_name="hung")
    with pytest.raises(TimeoutError, match="exceeded the 1-second child timeout"):
        _run_prepared_benchmark_isolated(
            run,
            output_dir="/tmp/unused",
            run_id="hung",
            timeout_seconds=1,
        )

    assert process.started is True
    assert process.terminated is True
    assert process.killed is True
    assert process.join_timeouts == [30, 30]
    assert receive_connection.poll_timeout == 1
    assert receive_connection.closed is True
    assert send_connection.closed is True


def test_run_files_spawned_child_writes_terminal_summary(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    query_file = tmp_path / "queries.csv"
    query_file.write_text("query_id,query\n1,test\n", encoding="utf-8")
    dataset_paths = tmp_path / "dataset_paths.yaml"
    dataset_paths.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "datasets:",
                "  jp20:",
                f"    path: {documents}",
                f"    query_file: {query_file}",
            )
        ),
        encoding="utf-8",
    )
    runfile = tmp_path / "jp20_beir.json"
    _write_json(runfile, {"schema_version": 1, "name": "jp20_beir", "benchmark": "jp20_beir"})

    outcome = run_runfiles(
        [runfile],
        output_dir=str(tmp_path / "session"),
        dataset_paths_file=dataset_paths,
        mode="batch",
        dry_run=True,
        isolate_runs=True,
    )

    assert outcome.exit_code == 0
    assert outcome.results["isolate_runs"] is True
    assert outcome.results["runs"][0]["success"] is True
    assert (outcome.artifact_dir / "001_jp20_beir" / "results.json").exists()
    assert outcome.results_path == outcome.artifact_dir / "session_summary.json"


def test_run_files_removes_stale_summary_before_child_execution(monkeypatch, tmp_path):
    runfile = tmp_path / "jp20_beir.json"
    _write_json(runfile, {"schema_version": 1, "name": "jp20_beir", "benchmark": "jp20_beir"})
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    _write_json(session_dir / "session_summary.json", {"success": True, "old": True})

    def fake_run_benchmark(prepared, **kwargs):
        benchmark = prepared.benchmark
        assert not (session_dir / "session_summary.json").exists()
        return _successful_outcome(benchmark, kwargs["output_dir"])

    monkeypatch.setattr("nemo_retriever.harness.runsets.run_prepared_benchmark", fake_run_benchmark)

    outcome = run_runfiles(
        [runfile],
        output_dir=str(session_dir),
        overrides=(f'dataset.path="{tmp_path}"',),
        dry_run=True,
    )

    assert outcome.exit_code == 0
    summary = json.loads((session_dir / "session_summary.json").read_text(encoding="utf-8"))
    assert "old" not in summary
    assert summary["success"] is True


def test_run_files_classifies_session_summary_write_failure(monkeypatch, tmp_path):
    import nemo_retriever.harness.json_io as json_io

    runfile = tmp_path / "jp20_beir.json"
    _write_json(runfile, {"schema_version": 1, "name": "jp20_beir", "benchmark": "jp20_beir"})

    def fake_run_benchmark(prepared, **kwargs):
        return _successful_outcome(prepared.benchmark, kwargs["output_dir"])

    original_replace = json_io.os.replace

    def fail_session_summary(source, destination):
        if Path(destination).name == "session_summary.json":
            raise OSError("disk full")
        original_replace(source, destination)

    monkeypatch.setattr("nemo_retriever.harness.runsets.run_prepared_benchmark", fake_run_benchmark)
    monkeypatch.setattr(json_io.os, "replace", fail_session_summary)
    session_dir = tmp_path / "session"

    with pytest.raises(HarnessRunError) as error:
        run_runfiles(
            [runfile],
            output_dir=str(session_dir),
            overrides=(f'dataset.path="{tmp_path}"',),
            dry_run=True,
        )

    assert error.value.exit_code == EXIT_ARTIFACT_WRITE_FAILURE
    assert error.value.failure.failure_reason == "artifact_write_failed"
    assert not (session_dir / "session_summary.json").exists()


def test_run_files_converts_raised_child_failure_and_continues(monkeypatch, tmp_path):
    runfiles = []
    for name in ("jp20_beir", "bo767_beir"):
        path = tmp_path / f"{name}.json"
        _write_json(path, {"schema_version": 1, "name": name, "benchmark": name})
        runfiles.append(path)
    calls = []

    def fake_run_benchmark(prepared, **kwargs):
        benchmark = prepared.benchmark
        calls.append(benchmark)
        if benchmark == "jp20_beir":
            raise HarnessRunError(
                EXIT_MISSING_INPUT,
                FailurePayload(
                    failed_phase="resolve",
                    failure_reason="dataset_missing",
                    retryable=False,
                    message="dataset disappeared",
                ),
            )
        return _successful_outcome(benchmark, kwargs["output_dir"])

    monkeypatch.setattr("nemo_retriever.harness.runsets.run_prepared_benchmark", fake_run_benchmark)
    session_dir = tmp_path / "session"

    outcome = run_runfiles(
        runfiles,
        output_dir=str(session_dir),
        overrides=(f'dataset.path="{tmp_path}"',),
        dry_run=True,
    )

    assert calls == ["jp20_beir", "bo767_beir"]
    assert outcome.exit_code == EXIT_MISSING_INPUT
    assert [run["exit_code"] for run in outcome.results["runs"]] == [EXIT_MISSING_INPUT, 0]
    assert (session_dir / "001_jp20_beir" / "results.json").exists()
    assert (session_dir / "session_summary.json").exists()


def test_run_files_continues_when_failed_child_result_cannot_be_written(monkeypatch, tmp_path):
    import nemo_retriever.harness.runsets as runsets

    runfiles = []
    for name in ("jp20_beir", "bo767_beir"):
        path = tmp_path / f"{name}.json"
        _write_json(path, {"schema_version": 1, "name": name, "benchmark": name})
        runfiles.append(path)
    calls = []

    def fake_run_benchmark(prepared, **kwargs):
        benchmark = prepared.benchmark
        calls.append(benchmark)
        if benchmark == "jp20_beir":
            raise HarnessRunError(
                EXIT_MISSING_INPUT,
                FailurePayload(
                    failed_phase="resolve",
                    failure_reason="dataset_missing",
                    retryable=False,
                    message="dataset disappeared",
                ),
            )
        return _successful_outcome(benchmark, kwargs["output_dir"])

    original_write_json = runsets.write_json

    def fail_first_child_result(path, payload):
        if path.name == "results.json" and path.parent.name == "001_jp20_beir":
            raise OSError("child result unavailable")
        original_write_json(path, payload)

    monkeypatch.setattr(runsets, "run_prepared_benchmark", fake_run_benchmark)
    monkeypatch.setattr(runsets, "write_json", fail_first_child_result)
    session_dir = tmp_path / "session"

    outcome = run_runfiles(
        runfiles,
        output_dir=str(session_dir),
        overrides=(f'dataset.path="{tmp_path}"',),
        dry_run=True,
    )

    assert calls == ["jp20_beir", "bo767_beir"]
    assert outcome.exit_code == EXIT_ARTIFACT_WRITE_FAILURE
    assert [run["exit_code"] for run in outcome.results["runs"]] == [EXIT_ARTIFACT_WRITE_FAILURE, 0]
    assert outcome.results["runs"][0]["failure_reason"] == "OSError: child result unavailable"
    assert "results_path" not in outcome.results["runs"][0]
    assert not (session_dir / "001_jp20_beir" / "results.json").exists()
    assert (session_dir / "session_summary.json").exists()


def test_runset_converts_raised_child_failure_and_continues(monkeypatch, tmp_path):
    class FakeRunset:
        name = "pair"
        runs = ("jp20_beir", "bo767_beir")

        def to_dict(self):
            return {"name": self.name, "runs": list(self.runs)}

    calls = []

    def fake_run_benchmark(prepared, **kwargs):
        benchmark = prepared.benchmark
        calls.append(benchmark)
        if benchmark == "jp20_beir":
            raise HarnessRunError(
                EXIT_MISSING_INPUT,
                FailurePayload(
                    failed_phase="resolve",
                    failure_reason="dataset_missing",
                    retryable=False,
                    message="dataset disappeared",
                ),
            )
        return _successful_outcome(benchmark, kwargs["output_dir"])

    monkeypatch.setattr("nemo_retriever.harness.runsets._runset_or_error", lambda name: FakeRunset())
    monkeypatch.setattr("nemo_retriever.harness.runsets.run_prepared_benchmark", fake_run_benchmark)
    session_dir = tmp_path / "session"

    outcome = run_runset(
        "pair",
        output_dir=str(session_dir),
        overrides=(f'dataset.path="{tmp_path}"',),
        dry_run=True,
    )

    assert calls == ["jp20_beir", "bo767_beir"]
    assert outcome.exit_code == EXIT_MISSING_INPUT
    assert [run["exit_code"] for run in outcome.results["runs"]] == [EXIT_MISSING_INPUT, 0]
    assert (session_dir / "001_jp20_beir" / "results.json").exists()
    assert (session_dir / "session_summary.json").exists()


def test_run_files_preflights_every_run_before_execution(monkeypatch, tmp_path):
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "invalid.json"
    _write_json(
        valid,
        {
            "schema_version": 1,
            "name": "jp20",
            "benchmark": "jp20_beir",
            "set": {"dataset.path": str(tmp_path)},
        },
    )
    _write_json(
        invalid,
        {
            "schema_version": 1,
            "name": "bo767",
            "benchmark": "bo767_beir",
            "set": {"dataset.path": str(tmp_path), "query.nope": 1},
        },
    )
    calls = []

    def fake_run_benchmark(prepared, **kwargs):
        benchmark = prepared.benchmark
        calls.append(benchmark)
        return _successful_outcome(benchmark, kwargs["output_dir"])

    monkeypatch.setattr("nemo_retriever.harness.runsets.run_prepared_benchmark", fake_run_benchmark)
    session_dir = tmp_path / "session"

    with pytest.raises(HarnessRunError):
        run_runfiles([valid, invalid], output_dir=str(session_dir), dry_run=True)

    assert calls == []
    assert not session_dir.exists()


def test_run_files_redacts_sensitive_overrides_from_session_plan(monkeypatch, tmp_path):
    runfile = tmp_path / "jp20_beir.json"
    _write_json(
        runfile,
        {
            "schema_version": 1,
            "name": "jp20_beir",
            "benchmark": "jp20_beir",
            "set": {
                "query.reranker_api_key": "runfile-secret",
                "query.content_types": {"webhook_url": "structured-secret"},
            },
        },
    )
    calls = []

    def fake_run_benchmark(prepared, **kwargs):
        calls.append(prepared)
        return _successful_outcome(prepared.benchmark, kwargs["output_dir"])

    monkeypatch.setattr("nemo_retriever.harness.runsets.run_prepared_benchmark", fake_run_benchmark)

    outcome = run_runfiles(
        [runfile],
        output_dir=str(tmp_path / "session"),
        overrides=(f'dataset.path="{tmp_path}"', "query.reranker_api_key=cli-secret"),
        dry_run=True,
    )

    assert calls[0].overrides == (
        "query.reranker_api_key=runfile-secret",
        'query.content_types={"webhook_url": "structured-secret"}',
        f'dataset.path="{tmp_path}"',
        "query.reranker_api_key=cli-secret",
    )
    expanded_text = (outcome.artifact_dir / "expanded_runs.json").read_text(encoding="utf-8")
    assert "runfile-secret" not in expanded_text
    assert "structured-secret" not in expanded_text
    assert "cli-secret" not in expanded_text
    assert expanded_text.count("<redacted>") == 3


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("name", "unsafe/../../../escape"),
        ("output_dir", "/tmp/run-owned-output"),
        ("run_id", "run-owned-id"),
        ("dry_run", True),
    ),
)
def test_run_files_rejects_unsafe_or_run_owned_layout(field, value, tmp_path):
    runfile = tmp_path / "jp20_beir.json"
    payload = {"schema_version": 1, "name": "jp20_beir", "benchmark": "jp20_beir"}
    payload[field] = value
    _write_json(runfile, payload)
    session_dir = tmp_path / "session"

    with pytest.raises(HarnessRunError) as exc_info:
        run_runfiles([runfile], output_dir=str(session_dir))

    assert exc_info.value.exit_code == 2
    assert not session_dir.exists()
