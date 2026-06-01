from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
REUSABLE_PRE_COMMIT = "./.github/workflows/reusable-pre-commit.yml"
REUSABLE_DOCKER_BUILD_AND_TEST = "./.github/workflows/reusable-docker-build-and-test.yml"
THIS_FILE = Path(__file__).resolve()

requires_workflows = pytest.mark.skipif(
    not WORKFLOWS.exists(),
    reason="Workflow files are not present in the Docker image test environment.",
)


def _load_workflow(name):
    with (WORKFLOWS / name).open() as f:
        return yaml.safe_load(f)


@requires_workflows
def test_main_and_pr_ci_share_reusable_pre_commit_job():
    for workflow_name in ("ci-main.yml", "ci-pull-request.yml"):
        workflow = _load_workflow(workflow_name)
        job = workflow["jobs"]["pre-commit"]

        assert job == {
            "name": "Pre-commit Checks",
            "uses": REUSABLE_PRE_COMMIT,
        }


@requires_workflows
def test_reusable_pre_commit_installs_uv_before_pre_commit():
    workflow = _load_workflow("reusable-pre-commit.yml")
    steps = workflow["jobs"]["pre-commit"]["steps"]
    uses_steps = [step["uses"] for step in steps if "uses" in step]

    assert "astral-sh/setup-uv@v6" in uses_steps
    assert "pre-commit/action@v3.0.1" in uses_steps
    assert uses_steps.index("astral-sh/setup-uv@v6") < uses_steps.index("pre-commit/action@v3.0.1")


@requires_workflows
def test_main_ci_uses_single_job_docker_build_and_test():
    workflow = _load_workflow("ci-main.yml")
    jobs = workflow["jobs"]

    assert "docker-build" not in jobs
    assert "docker-test" not in jobs

    job = jobs["docker-build-and-test"]
    assert job["name"] == "Build & Test Docker (amd64)"
    assert job["uses"] == REUSABLE_DOCKER_BUILD_AND_TEST
    assert job["with"] == {
        "platform": "linux/amd64",
        "target": "service",
        "tags": "nrl-service:main-${{ github.sha }}",
        "base-image": "ubuntu",
        "base-image-tag": "jammy-20250415.1",
        "test-selection": "full",
        "pytest-markers": "not integration",
        "coverage": True,
        "runner": "linux-large-disk",
    }
    assert job["secrets"] == {
        "HF_ACCESS_TOKEN": "${{ secrets.HF_ACCESS_TOKEN }}",
    }


@requires_workflows
def test_legacy_ghcr_push_publish_workflow_is_removed():
    assert not (WORKFLOWS / "docker-build-publish-retriever.yml").exists()


def test_legacy_nv_ingest_compose_stack_is_removed():
    legacy_paths = (
        "docker-compose.yaml",
        "docker-compose.a100-40gb.yaml",
        "docker-compose.a10g.yaml",
        "docker-compose.l40s.yaml",
        "docker-compose.rtx-pro-4500.yaml",
        "nemo_retriever/docker.md",
        "ci/scripts/validate_deployment_configs.py",
        "skaffold/README.md",
        "skaffold/nv-ingest.skaffold.yaml",
        "skaffold/sensitive/.gitignore",
    )

    for relative_path in legacy_paths:
        assert not (REPO_ROOT / relative_path).exists(), relative_path


def test_legacy_tools_harness_is_removed():
    assert not (REPO_ROOT / "tools" / "harness").exists()

    legacy_tokens = ("tools/harness", "nv_ingest_harness", "nv-ingest-harness")
    ignored_dirs = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".worktrees",
        "__pycache__",
    }
    offenders: list[str] = []

    for path in REPO_ROOT.rglob("*"):
        relative = path.relative_to(REPO_ROOT)
        if path == THIS_FILE or not path.is_file() or any(part in ignored_dirs for part in relative.parts):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for token in legacy_tokens:
            if token in text:
                offenders.append(f"{relative}: {token}")

    assert offenders == []
