# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LATEST_MAIN_LAUNCHER = REPO_ROOT / "ops" / "retriever-nightly" / "run-latest-main.sh"


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", message)
    return _git(repository, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def latest_main_fixture(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(source)], check=True)
    _git(source, "config", "user.email", "nightly-test@example.com")
    _git(source, "config", "user.name", "Nightly Test")

    launcher = source / "ops" / "retriever-nightly" / "run-nightly.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "\n".join(
            (
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                'if [[ -n "${EXPECT_DEEP_GEMM_WARMUP+x}" && '
                '"${VLLM_DEEP_GEMM_WARMUP:-}" != "$EXPECT_DEEP_GEMM_WARMUP" ]]; then',
                "    exit 98",
                "fi",
                'checkout="$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)"',
                'commit="$(git -C "$checkout" rev-parse HEAD)"',
                'printf "%s|%s|%s\\n" "$commit" "$*" "${UV_PROJECT_ENVIRONMENT:-}" >>"$FAKE_LATEST_CALLS"',
                'for arg in "$@"; do',
                '    if [[ "$arg" == "--check-vidore-access" ]]; then',
                '        exit "${FAKE_ACCESS_RC:-0}"',
                "    fi",
                "done",
                'exit "${FAKE_RUN_RC:-0}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    (source / "version.txt").write_text("initial\n", encoding="utf-8")
    initial_commit = _commit(source, "initial")

    controller = tmp_path / "controller"
    subprocess.run(["git", "clone", "-q", str(source), str(controller)], check=True)

    (source / "version.txt").write_text("latest\n", encoding="utf-8")
    latest_commit = _commit(source, "latest")

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "nightly.env"
    config_file.write_text("HF_TOKEN=test-token\n", encoding="utf-8")
    config_file.chmod(0o600)

    checkout_root = tmp_path / "latest-checkouts"
    calls_path = tmp_path / "latest-calls.txt"
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "RETRIEVER_CONFIG_FILE": str(config_file),
            "RETRIEVER_UPDATE_REPOSITORY": str(controller),
            "RETRIEVER_LATEST_SOURCE": str(source),
            "RETRIEVER_LATEST_REF": "main",
            "RETRIEVER_LATEST_CHECKOUT_ROOT": str(checkout_root),
            "RETRIEVER_LATEST_KEEP_CHECKOUTS": "2",
            "FAKE_LATEST_CALLS": str(calls_path),
        }
    )

    def run(*args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(LATEST_MAIN_LAUNCHER), *args],
            cwd=tmp_path,
            env=env | (extra_env or {}),
            check=False,
            capture_output=True,
            text=True,
        )

    def calls() -> list[tuple[str, str, str]]:
        if not calls_path.exists():
            return []
        return [tuple(line.split("|", 2)) for line in calls_path.read_text(encoding="utf-8").splitlines()]

    return run, calls, source, controller, checkout_root, initial_commit, latest_commit


def test_scheduled_launcher_fetches_latest_main_into_immutable_worktree(latest_main_fixture) -> None:
    run, calls, _source, controller, checkout_root, initial_commit, latest_commit = latest_main_fixture

    result = run()

    assert result.returncode == 0, result.stderr
    assert _git(controller, "rev-parse", "HEAD").stdout.strip() == initial_commit
    assert calls() == [
        (latest_commit, "--check-vidore-access", str(checkout_root / ".venv")),
        (latest_commit, "", str(checkout_root / ".venv")),
    ]
    selected_checkout = checkout_root / f"main-{latest_commit}"
    assert selected_checkout.is_dir()
    assert _git(selected_checkout, "rev-parse", "HEAD").stdout.strip() == latest_commit


def test_help_does_not_require_configuration_or_fetch(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update({"HOME": str(tmp_path / "missing-home")})
    env.pop("RETRIEVER_CONFIG_FILE", None)

    result = subprocess.run(
        [str(LATEST_MAIN_LAUNCHER), "--help"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Fetch upstream/main" in result.stdout


@pytest.mark.parametrize("configured, expected", [(None, "skip"), ("full", "full")])
def test_scheduled_run_has_safe_warmup_default_and_allows_override(
    latest_main_fixture, configured: str | None, expected: str
) -> None:
    run, _calls, *_rest = latest_main_fixture
    extra_env = {"EXPECT_DEEP_GEMM_WARMUP": expected}
    if configured is not None:
        extra_env["VLLM_DEEP_GEMM_WARMUP"] = configured

    result = run("--dry-run", extra_env=extra_env)

    assert result.returncode == 0, result.stderr


def test_scheduled_launcher_fails_closed_when_access_preflight_fails(latest_main_fixture) -> None:
    run, calls, _source, _controller, _checkout_root, _initial_commit, latest_commit = latest_main_fixture

    result = run(extra_env={"FAKE_ACCESS_RC": "3"})

    assert result.returncode == 3
    assert calls() == [(latest_commit, "--check-vidore-access", str(_checkout_root / ".venv"))]


def test_scheduled_launcher_does_not_run_stale_commit_when_fetch_fails(latest_main_fixture) -> None:
    run, calls, _source, _controller, _checkout_root, _initial_commit, _latest_commit = latest_main_fixture

    result = run(extra_env={"RETRIEVER_LATEST_SOURCE": str(_checkout_root / "missing-source")})

    assert result.returncode != 0
    assert calls() == []


def test_scheduled_launcher_rejects_modified_controller(latest_main_fixture) -> None:
    run, calls, _source, controller, _checkout_root, _initial_commit, _latest_commit = latest_main_fixture
    (controller / "version.txt").write_text("modified\n", encoding="utf-8")

    result = run()

    assert result.returncode == 64
    assert calls() == []


def test_scheduled_launcher_prunes_only_old_managed_worktrees(latest_main_fixture) -> None:
    run, _calls, source, _controller, checkout_root, _initial_commit, latest_commit = latest_main_fixture
    assert run(extra_env={"RETRIEVER_LATEST_KEEP_CHECKOUTS": "1"}).returncode == 0

    (source / "version.txt").write_text("newer\n", encoding="utf-8")
    newer_commit = _commit(source, "newer")
    result = run(extra_env={"RETRIEVER_LATEST_KEEP_CHECKOUTS": "1"})

    assert result.returncode == 0, result.stderr
    assert not (checkout_root / f"main-{latest_commit}").exists()
    assert (checkout_root / f"main-{newer_commit}").is_dir()


def test_systemd_service_uses_latest_main_controller() -> None:
    service = (REPO_ROOT / "ops" / "retriever-nightly" / "systemd" / "nrl-harness-batch-hf.service").read_text(
        encoding="utf-8"
    )

    assert "ExecStart=" in service
    assert "/run-latest-main.sh" in service
