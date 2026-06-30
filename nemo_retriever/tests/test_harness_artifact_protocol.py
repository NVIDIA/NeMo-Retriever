# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nemo_retriever.harness import benchmark_registry
from nemo_retriever.harness.artifact_writer import ArtifactWriter, ArtifactWriteError
from nemo_retriever.harness.benchmark_specs import BenchmarkSpec, DatasetSpec, RunSet
from nemo_retriever.harness.cli import app
from nemo_retriever.harness import json_io
from nemo_retriever.harness.json_io import redact, write_json


RUNNER = CliRunner()
SECRET = "artifact-protocol-secret"


@pytest.fixture
def dry_run_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    name = "artifact_protocol_test"
    dataset_name = f"{name}_dataset"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "document.pdf").write_bytes(b"%PDF-1.4\n")

    source = benchmark_registry.BENCHMARKS["jp20_smoke"]
    monkeypatch.setitem(
        benchmark_registry.DATASETS,
        dataset_name,
        DatasetSpec(name=dataset_name, path=str(dataset_dir), input_type="pdf"),
    )
    monkeypatch.setitem(
        benchmark_registry.BENCHMARKS,
        name,
        BenchmarkSpec(
            name=name,
            dataset=dataset_name,
            ingest=deepcopy(source.ingest),
            query=deepcopy(source.query),
            evaluation={"mode": "none"},
            summary_keys=source.summary_keys,
            tags=("test",),
        ),
    )
    monkeypatch.setitem(
        benchmark_registry.RUNSETS,
        f"{name}_runset",
        RunSet(name=f"{name}_runset", runs=(name,), tags=("test",)),
    )
    return name


def test_redact_covers_nested_fields_and_override_strings() -> None:
    payload = redact(
        {
            "query": {"reranker_api_key": SECRET},
            "overrides": [f"query.reranker_api_key={SECRET}"],
            "input_tokens": 42,
        }
    )

    assert payload == {
        "query": {"reranker_api_key": "<redacted>"},
        "overrides": ["query.reranker_api_key=<redacted>"],
        "input_tokens": 42,
    }


def test_manifest_only_includes_artifacts_written_by_this_writer(tmp_path: Path) -> None:
    writer = ArtifactWriter(artifact_dir=tmp_path / "artifacts", run_id="run", benchmark="benchmark")
    writer.path("beir_metrics.json").write_text('{"stale": true}\n', encoding="utf-8")
    writer.write_json("summary_metrics.json", {"files": 1})

    assert writer.artifact_paths() == {
        "summary_metrics.json": str(writer.path("summary_metrics.json")),
    }


def test_atomic_json_failure_preserves_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"version": "old"}\n', encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(json_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        write_json(path, {"version": "new"})

    assert path.read_text(encoding="utf-8") == '{"version": "old"}\n'
    assert not list(tmp_path.glob(".artifact.json.*.tmp"))


def test_dry_run_redacts_json_output_and_structured_artifacts(
    tmp_path: Path,
    dry_run_benchmark: str,
) -> None:
    output_dir = tmp_path / "artifacts"
    result = RUNNER.invoke(
        app,
        [
            "run",
            dry_run_benchmark,
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--set",
            f"query.reranker_api_key={SECRET}",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert SECRET not in result.output
    stdout_payload = json.loads(result.stdout)
    assert stdout_payload["resolved_benchmark"]["query"]["reranker_api_key"] == "<redacted>"
    assert set(stdout_payload["artifacts"]) == {
        "status.json",
        "events.jsonl",
        "resolved_benchmark.json",
        "ingest_plan.json",
        "query_plan.json",
        "summary_metrics.json",
        "environment.json",
        "results.json",
    }
    for path in output_dir.iterdir():
        if path.suffix in {".json", ".jsonl"}:
            assert SECRET not in path.read_text(encoding="utf-8"), path
    assert not list(output_dir.glob(".*.tmp"))


def test_non_empty_output_dir_exits_30_without_touching_stale_artifacts(
    tmp_path: Path,
    dry_run_benchmark: str,
) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    stale = output_dir / "beir_metrics.json"
    stale.write_text('{"stale": true}\n', encoding="utf-8")

    result = RUNNER.invoke(
        app,
        ["run", dry_run_benchmark, "--dry-run", "--output-dir", str(output_dir)],
    )

    assert result.exit_code == 30
    assert "Artifact directory is not empty" in result.output
    assert "Traceback" not in result.output
    assert stale.read_text(encoding="utf-8") == '{"stale": true}\n'
    assert list(output_dir.iterdir()) == [stale]


def test_runset_redacts_overrides_and_refuses_output_dir_reuse(
    tmp_path: Path,
    dry_run_benchmark: str,
) -> None:
    output_dir = tmp_path / "runset-artifacts"
    runset = f"{dry_run_benchmark}_runset"
    args = [
        "run-set",
        runset,
        "--dry-run",
        "--output-dir",
        str(output_dir),
        "--set",
        f"query.reranker_api_key={SECRET}",
        "--json",
    ]

    first = RUNNER.invoke(app, args)

    assert first.exit_code == 0, first.output
    assert SECRET not in first.output
    expanded = json.loads((output_dir / "expanded_runs.json").read_text(encoding="utf-8"))
    assert expanded["runs"][0]["overrides"] == ["query.reranker_api_key=<redacted>"]
    assert all(SECRET not in path.read_text(encoding="utf-8") for path in output_dir.rglob("*.json"))

    second = RUNNER.invoke(app, args)

    assert second.exit_code == 30
    assert "Artifact directory is not empty" in second.output
    assert "Traceback" not in second.output


def test_artifact_write_failure_exits_30_without_traceback(
    tmp_path: Path,
    dry_run_benchmark: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write(*_args: object, **_kwargs: object) -> Path:
        raise ArtifactWriteError("injected artifact write failure")

    monkeypatch.setattr(ArtifactWriter, "write_json", fail_write)
    result = RUNNER.invoke(
        app,
        [
            "run",
            dry_run_benchmark,
            "--dry-run",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--json",
        ],
    )

    assert result.exit_code == 30
    assert "Traceback" not in result.output
    payload = json.loads(result.stdout)
    assert payload["exit_code"] == 30
    assert payload["failure"]["failure_reason"] == "artifact_write_failed"
    assert payload["artifacts"] == {}
