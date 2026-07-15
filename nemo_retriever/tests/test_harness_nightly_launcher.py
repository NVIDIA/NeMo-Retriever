# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "ops" / "retriever-nightly" / "run-nightly.sh"
DEFAULT_RUNFILES = (
    "nemo_retriever/harness/runfiles/jp20_beir.json",
    "nemo_retriever/harness/runfiles/bo767_beir.json",
    "nemo_retriever/harness/runfiles/earnings_beir.json",
    "nemo_retriever/harness/runfiles/financebench_beir.json",
    "nemo_retriever/harness/runfiles/vidore_v3_computer_science_beir.json",
    "nemo_retriever/harness/runfiles/vidore_v3_energy_beir.json",
    "nemo_retriever/harness/runfiles/vidore_v3_finance_en_beir.json",
    "nemo_retriever/harness/runfiles/vidore_v3_finance_fr_beir.json",
    "nemo_retriever/harness/runfiles/vidore_v3_hr_beir.json",
    "nemo_retriever/harness/runfiles/vidore_v3_industrial_beir.json",
    "nemo_retriever/harness/runfiles/vidore_v3_pharmaceuticals_beir.json",
    "nemo_retriever/harness/runfiles/vidore_v3_physics_beir.json",
)
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/test/webhook/value"


@pytest.fixture
def nightly_launcher(tmp_path: Path):
    checkout = tmp_path / "checkout"
    runfiles_dir = checkout / "nemo_retriever" / "harness" / "runfiles"
    runfiles_dir.mkdir(parents=True)
    for relative_path in DEFAULT_RUNFILES:
        (checkout / relative_path).write_text("{}\n", encoding="utf-8")
    default_dataset_paths = checkout / "ops" / "retriever-nightly" / "dataset_paths.datasets.yaml"
    default_dataset_paths.parent.mkdir(parents=True)
    default_dataset_paths.write_text("schema_version: 1\ndatasets: {}\n", encoding="utf-8")

    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.email", "harness-test@example.com"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Harness Test"], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "test fixture"], check=True)

    calls_path = tmp_path / "uv-calls.jsonl"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "\n".join(
            (
                f"#!{sys.executable}",
                "import json",
                "import os",
                "from pathlib import Path",
                "import sys",
                "args = sys.argv[1:]",
                "expected_warmup = os.environ.get('EXPECT_DEEP_GEMM_WARMUP')",
                "if expected_warmup is not None and os.environ.get('VLLM_DEEP_GEMM_WARMUP') != expected_warmup:",
                "    raise SystemExit(98)",
                "expected_hf_token = os.environ.get('EXPECT_HF_TOKEN')",
                "if expected_hf_token is not None and os.environ.get('HF_TOKEN') != expected_hf_token:",
                "    raise SystemExit(95)",
                "with Path(os.environ['FAKE_UV_CALLS']).open('a', encoding='utf-8') as stream:",
                "    stream.write(json.dumps(args) + '\\n')",
                "if 'run-files' in args:",
                "    if os.environ.get('SLACK_WEBHOOK_URL'):",
                "        raise SystemExit(97)",
                "    output_dir = Path(args[args.index('--output-dir') + 1])",
                "    output_dir.mkdir(parents=True, exist_ok=True)",
                "    (output_dir / 'session_summary.json').write_text('{}\\n', encoding='utf-8')",
                "    raise SystemExit(int(os.environ.get('FAKE_RUN_RC', '0')))",
                "if 'check-vidore-access' in args:",
                "    raise SystemExit(int(os.environ.get('FAKE_ACCESS_RC', '0')))",
                "if 'post-slack' in args:",
                "    if not os.environ.get('SLACK_WEBHOOK_URL'):",
                "        raise SystemExit(96)",
                "    raise SystemExit(int(os.environ.get('FAKE_POST_RC', '0')))",
                "raise SystemExit(99)",
                "",
            )
        ),
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "RETRIEVER_CHECKOUT": str(checkout),
            "RETRIEVER_SELECTED_CHECKOUT": str(checkout),
            "RETRIEVER_NIGHTLY_ROOT": str(tmp_path / "nightly-root"),
            "RETRIEVER_UV_BIN": str(fake_uv),
            "FAKE_UV_CALLS": str(calls_path),
        }
    )
    env.pop("VLLM_DEEP_GEMM_WARMUP", None)
    env.pop("SLACK_WEBHOOK_URL", None)

    def run(*args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        command_env = env | (extra_env or {})
        return subprocess.run(
            [str(LAUNCHER), *args],
            cwd=tmp_path,
            env=command_env,
            check=False,
            capture_output=True,
            text=True,
        )

    def calls() -> list[list[str]]:
        if not calls_path.exists():
            return []
        return [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]

    return run, calls


def test_normal_run_uses_root_and_checked_in_dataset_defaults(nightly_launcher, tmp_path: Path) -> None:
    run, calls = nightly_launcher

    result = run("--dry-run")

    assert result.returncode == 0, result.stderr
    invocation = calls()[0]
    assert invocation[invocation.index("--dataset-paths") + 1].endswith(
        "/ops/retriever-nightly/dataset_paths.datasets.yaml"
    )
    assert Path(invocation[invocation.index("--output-dir") + 1]).parent == (
        tmp_path / "nightly-root" / "retriever-nightly-artifacts"
    )


def test_launcher_rejects_untracked_checkout_files(nightly_launcher, tmp_path: Path) -> None:
    run, calls = nightly_launcher
    (tmp_path / "checkout" / "untracked.py").write_text("print('changed')\n", encoding="utf-8")

    result = run("--dry-run")

    assert result.returncode == 64
    assert calls() == []
    assert "untracked changes" in result.stderr


@pytest.mark.parametrize("configured, expected", [(None, "skip"), ("full", "full")])
def test_deep_gemm_warmup_has_safe_default_and_allows_override(
    nightly_launcher, configured: str | None, expected: str
) -> None:
    run, _calls = nightly_launcher
    extra_env = {"EXPECT_DEEP_GEMM_WARMUP": expected}
    if configured is not None:
        extra_env["VLLM_DEEP_GEMM_WARMUP"] = configured

    result = run("--check-vidore-access", extra_env=extra_env)

    assert result.returncode == 0, result.stderr


def test_manual_dry_run_launches_full_batch_suite_without_slack(nightly_launcher, tmp_path: Path) -> None:
    run, calls = nightly_launcher
    cli_dataset_paths = tmp_path / "cli-dataset-paths.yaml"
    cli_dataset_paths.write_text("schema_version: 1\ndatasets: {}\n", encoding="utf-8")

    result = run(
        "--dataset-paths",
        cli_dataset_paths.name,
        "--artifact-root",
        "cli-artifacts",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert len(calls()) == 1
    invocation = calls()[0]
    assert invocation[invocation.index("--dataset-paths") + 1] == str(cli_dataset_paths)
    assert Path(invocation[invocation.index("--output-dir") + 1]).parent == tmp_path / "cli-artifacts"
    assert invocation[invocation.index("--mode") + 1] == "batch"
    assert "--dry-run" in invocation
    assert "--isolate-runs" not in invocation
    assert tuple(invocation[-len(DEFAULT_RUNFILES) :]) == DEFAULT_RUNFILES


def test_access_preflight_checks_vidore_without_starting_a_session(nightly_launcher) -> None:
    run, calls = nightly_launcher

    result = run("--check-vidore-access")

    assert result.returncode == 0, result.stderr
    assert len(calls()) == 1
    invocation = calls()[0]
    assert invocation[-3:] == ["retriever", "harness", "check-vidore-access"]
    assert "run-files" not in invocation


def test_access_preflight_propagates_failure(nightly_launcher) -> None:
    run, calls = nightly_launcher

    result = run("--check-vidore-access", extra_env={"FAKE_ACCESS_RC": "3"})

    assert result.returncode == 3
    assert len(calls()) == 1


def test_configured_webhook_posts_terminal_session_to_slack(nightly_launcher, tmp_path: Path) -> None:
    run, calls = nightly_launcher
    config_dir = tmp_path / "nightly-root" / ".config" / "nemo-retriever" / "nightly"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "nightly.env"
    config_file.write_text(f"SLACK_WEBHOOK_URL={SLACK_WEBHOOK_URL}\n", encoding="utf-8")
    config_file.chmod(0o600)

    result = run()

    assert result.returncode == 0, result.stderr
    assert len(calls()) == 2
    run_invocation, post_invocation = calls()
    session_dir = Path(run_invocation[run_invocation.index("--output-dir") + 1])
    assert "--isolate-runs" in run_invocation
    assert "post-slack" in post_invocation
    assert post_invocation[-1] == str(session_dir)
    assert (session_dir / ".slack_post_attempted").exists()


def test_exported_secrets_post_without_config_file(nightly_launcher) -> None:
    run, calls = nightly_launcher

    result = run(
        extra_env={
            "HF_TOKEN": "exported-read-token",
            "EXPECT_HF_TOKEN": "exported-read-token",
            "SLACK_WEBHOOK_URL": SLACK_WEBHOOK_URL,
        }
    )

    assert result.returncode == 0, result.stderr
    assert len(calls()) == 2
    assert "run-files" in calls()[0]
    assert "post-slack" in calls()[1]


def test_missing_webhook_completes_without_slack(nightly_launcher) -> None:
    run, calls = nightly_launcher

    result = run()

    assert result.returncode == 0, result.stderr
    assert len(calls()) == 1
    assert "run-files" in calls()[0]


def test_no_slack_suppresses_configured_webhook(nightly_launcher) -> None:
    run, calls = nightly_launcher

    result = run("--no-slack", extra_env={"SLACK_WEBHOOK_URL": SLACK_WEBHOOK_URL})

    assert result.returncode == 0, result.stderr
    assert len(calls()) == 1
    invocation = calls()[0]
    session_dir = Path(invocation[invocation.index("--output-dir") + 1])
    assert "run-files" in invocation
    assert not (session_dir / ".slack_post_attempted").exists()


def test_dry_run_never_posts_when_webhook_is_present(nightly_launcher) -> None:
    run, calls = nightly_launcher

    result = run("--dry-run", extra_env={"SLACK_WEBHOOK_URL": SLACK_WEBHOOK_URL})

    assert result.returncode == 0, result.stderr
    assert len(calls()) == 1
    assert "--dry-run" in calls()[0]


def test_invalid_webhook_fails_before_work_unless_slack_is_suppressed(nightly_launcher) -> None:
    run, calls = nightly_launcher

    invalid_result = run(extra_env={"SLACK_WEBHOOK_URL": "not-a-webhook"})

    assert invalid_result.returncode == 64
    assert calls() == []

    suppressed_result = run("--no-slack", extra_env={"SLACK_WEBHOOK_URL": "not-a-webhook"})

    assert suppressed_result.returncode == 0, suppressed_result.stderr
    assert len(calls()) == 1
