# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from nemo_retriever.harness import cli, helm_runner
from nemo_retriever.harness.contracts import EXIT_INGEST_FAILURE, EXIT_INVALID
from nemo_retriever.harness.helm_runner import EXIT_HELM_FAILURE

RUNNER = CliRunner()


def test_top_level_help_lists_run_helm() -> None:
    result = RUNNER.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "run-helm" in result.stdout


def test_run_helm_help_exposes_only_supported_dataset_paths_option() -> None:
    result = RUNNER.invoke(cli.app, ["run-helm", "--help"])
    assert result.exit_code == 0
    assert "{runfiles}..." in result.stdout
    assert "--config" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--dataset-paths" in result.stdout
    assert "--session-name" in result.stdout
    assert "--dataset " not in result.stdout


def test_run_helm_forwards_parsed_arguments_exactly_once(monkeypatch, tmp_path: Path) -> None:
    calls = []
    config = tmp_path / "helm.yaml"
    output_dir = tmp_path / "output"
    dataset_paths = tmp_path / "datasets.yaml"
    runfiles = [tmp_path / "one.json", tmp_path / "two.yaml"]

    def fake_run_helm_session(*args, **kwargs):
        calls.append((args, kwargs))
        return 0

    monkeypatch.setattr(helm_runner, "run_helm_session", fake_run_helm_session)
    result = RUNNER.invoke(
        cli.app,
        [
            "run-helm",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
            "--dataset-paths",
            str(dataset_paths),
            "--session-name",
            "bo767",
            *(str(path) for path in runfiles),
        ],
    )
    assert result.exit_code == 0
    assert calls == [
        ((config, runfiles), {"output_dir": output_dir, "session_name": "bo767", "dataset_paths": dataset_paths})
    ]


@pytest.mark.parametrize("exit_code", (0, EXIT_INGEST_FAILURE, EXIT_HELM_FAILURE))
def test_run_helm_propagates_session_and_helm_exit_codes(monkeypatch, tmp_path: Path, exit_code: int) -> None:
    monkeypatch.setattr(helm_runner, "run_helm_session", lambda *_args, **_kwargs: exit_code)
    result = RUNNER.invoke(
        cli.app,
        ["run-helm", "--config", str(tmp_path / "helm.yaml"), "--output-dir", str(tmp_path), "run.json"],
    )
    assert result.exit_code == exit_code


def test_run_helm_renders_invalid_config_concisely(monkeypatch, tmp_path: Path) -> None:
    def fail(*_args, **_kwargs):
        raise ValueError("bad config")

    monkeypatch.setattr(helm_runner, "run_helm_session", fail)
    result = RUNNER.invoke(
        cli.app,
        ["run-helm", "--config", str(tmp_path / "helm.yaml"), "--output-dir", str(tmp_path), "run.json"],
    )
    assert result.exit_code == EXIT_INVALID
    assert result.output.strip() == "bad config"
    assert "Traceback" not in result.output


def test_legacy_module_cli_still_parses_and_forwards(monkeypatch, tmp_path: Path) -> None:
    calls = []
    config = tmp_path / "helm.yaml"
    output_dir = tmp_path / "output"
    dataset_paths = tmp_path / "datasets.yaml"
    runfile = tmp_path / "run.json"
    monkeypatch.setattr(
        helm_runner,
        "run_helm_session",
        lambda *args, **kwargs: calls.append((args, kwargs)) or EXIT_INGEST_FAILURE,
    )
    with pytest.raises(SystemExit, match=str(EXIT_INGEST_FAILURE)):
        helm_runner.main(
            [
                "--config",
                str(config),
                "--output-dir",
                str(output_dir),
                "--dataset-paths",
                str(dataset_paths),
                "--session-name",
                "legacy",
                str(runfile),
            ]
        )
    assert calls == [
        ((config, [runfile]), {"output_dir": output_dir, "session_name": "legacy", "dataset_paths": dataset_paths})
    ]
