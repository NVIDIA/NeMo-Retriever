from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
REUSABLE_PRE_COMMIT = "./.github/workflows/reusable-pre-commit.yml"
REUSABLE_DOCKER_BUILD_AND_TEST = "./.github/workflows/reusable-docker-build-and-test.yml"

pytestmark = pytest.mark.skipif(
    not WORKFLOWS.exists(),
    reason="Workflow files are not present in the Docker image test environment.",
)


def _load_workflow(name):
    with (WORKFLOWS / name).open() as f:
        return yaml.safe_load(f)


def test_main_and_pr_ci_share_reusable_pre_commit_job():
    for workflow_name in ("ci-main.yml", "ci-pull-request.yml"):
        workflow = _load_workflow(workflow_name)
        job = workflow["jobs"]["pre-commit"]

        assert job == {
            "name": "Pre-commit Checks",
            "uses": REUSABLE_PRE_COMMIT,
        }


def test_reusable_pre_commit_installs_uv_before_pre_commit():
    workflow = _load_workflow("reusable-pre-commit.yml")
    steps = workflow["jobs"]["pre-commit"]["steps"]
    uses_steps = [step["uses"] for step in steps if "uses" in step]

    assert "astral-sh/setup-uv@v6" in uses_steps
    assert "pre-commit/action@v3.0.1" in uses_steps
    assert uses_steps.index("astral-sh/setup-uv@v6") < uses_steps.index("pre-commit/action@v3.0.1")


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


def test_legacy_ghcr_push_publish_workflow_is_removed():
    assert not (WORKFLOWS / "docker-build-publish-retriever.yml").exists()
