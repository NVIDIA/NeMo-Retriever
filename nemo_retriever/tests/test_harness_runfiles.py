import json
from pathlib import Path

import pytest

from nemo_retriever.harness.benchmark_registry import get_runset
from nemo_retriever.harness.contracts import HarnessRunError, RunOutcome
from nemo_retriever.harness.runsets import run_runfiles


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _successful_outcome(benchmark: str, output_dir: str) -> RunOutcome:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True)
    results = {
        "benchmark": benchmark,
        "success": True,
        "exit_code": 0,
        "resolved_benchmark": {"dataset": {"name": "jp20"}},
        "summary_metrics": {"files": 20},
        "failure": None,
    }
    return RunOutcome(exit_code=0, artifact_dir=artifact_dir, results=results)


def test_library_nightly_runset_names_canonical_beir_benchmarks():
    assert get_runset("library_nightly").runs == (
        "jp20_beir",
        "bo767_beir",
        "earnings_beir",
        "financebench_beir",
    )


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

    def fake_run_benchmark(benchmark, **kwargs):
        calls.append((benchmark, kwargs))
        return _successful_outcome(benchmark, kwargs["output_dir"])

    monkeypatch.setattr("nemo_retriever.harness.runsets.run_benchmark", fake_run_benchmark)

    outcome = run_runfiles(
        [runfile],
        output_dir=str(tmp_path / "session"),
        session_name="library_nightly",
        dataset_paths_file=dataset_paths,
        overrides=("query.top_k=20",),
        requirements=("pages==1940",),
    )

    benchmark, kwargs = calls[0]
    assert benchmark == "jp20_beir"
    assert kwargs["mode"] == "batch"
    assert kwargs["requirements"] == ("files==20", "pages==1940")
    assert kwargs["overrides"] == (
        "query.top_k=10",
        f'dataset.path="{documents}"',
        f'dataset.query_file="{query_file}"',
        f'evaluation.dataset_name="{query_file}"',
        "query.top_k=20",
    )
    assert outcome.results["session_name"] == "library_nightly"
    assert outcome.results["runs"][0]["dataset"] == "jp20"
    assert "results" not in outcome.results

    expanded = json.loads((outcome.artifact_dir / "expanded_runs.json").read_text(encoding="utf-8"))
    assert expanded["runfiles"][0]["dataset_paths"] == {
        "path": str(documents),
        "query_file": str(query_file),
    }


def test_run_files_completes_remaining_runs_and_preserves_first_failure(monkeypatch, tmp_path):
    runfiles = []
    for name in ("jp20_beir", "bo767_beir"):
        path = tmp_path / f"{name}.json"
        _write_json(path, {"schema_version": 1, "name": name, "benchmark": name})
        runfiles.append(path)

    calls = []

    def fake_run_benchmark(benchmark, **kwargs):
        calls.append(benchmark)
        if benchmark == "jp20_beir":
            artifact_dir = Path(kwargs["output_dir"])
            artifact_dir.mkdir(parents=True)
            return RunOutcome(
                exit_code=10,
                artifact_dir=artifact_dir,
                results={
                    "summary_metrics": {},
                    "failure": {"message": "ingest failed"},
                },
            )
        return _successful_outcome(benchmark, kwargs["output_dir"])

    monkeypatch.setattr("nemo_retriever.harness.runsets.run_benchmark", fake_run_benchmark)

    outcome = run_runfiles(runfiles, output_dir=str(tmp_path / "session"))

    assert calls == ["jp20_beir", "bo767_beir"]
    assert outcome.exit_code == 10
    assert outcome.results["all_passed"] is False
    assert [run["exit_code"] for run in outcome.results["runs"]] == [10, 0]


def test_run_files_redacts_sensitive_overrides_from_session_plan(monkeypatch, tmp_path):
    runfile = tmp_path / "jp20_beir.json"
    _write_json(
        runfile,
        {
            "schema_version": 1,
            "name": "jp20_beir",
            "benchmark": "jp20_beir",
            "set": ["query.reranker_api_key=runfile-secret"],
        },
    )
    calls = []

    def fake_run_benchmark(benchmark, **kwargs):
        calls.append(kwargs)
        return _successful_outcome(benchmark, kwargs["output_dir"])

    monkeypatch.setattr("nemo_retriever.harness.runsets.run_benchmark", fake_run_benchmark)

    outcome = run_runfiles(
        [runfile],
        output_dir=str(tmp_path / "session"),
        overrides=("query.reranker_api_key=cli-secret",),
    )

    assert calls[0]["overrides"] == (
        "query.reranker_api_key=runfile-secret",
        "query.reranker_api_key=cli-secret",
    )
    expanded_text = (outcome.artifact_dir / "expanded_runs.json").read_text(encoding="utf-8")
    assert "runfile-secret" not in expanded_text
    assert "cli-secret" not in expanded_text
    assert expanded_text.count("<redacted>") == 2


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("name", "unsafe/../../../escape"),
        ("output_dir", "/tmp/run-owned-output"),
        ("run_id", "run-owned-id"),
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
