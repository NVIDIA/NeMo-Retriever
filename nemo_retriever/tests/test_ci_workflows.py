import os
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

IGNORED_SCAN_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".worktrees",
    "__pycache__",
    "artifacts",
    "lancedb",
    "tmp",
}
IGNORED_SCAN_SUFFIXES = (".egg-info",)
IGNORED_SCAN_PREFIXES = ("lancedb_",)


def _is_ignored_scan_path(relative: Path) -> bool:
    directory_parts = relative.parts[:-1]
    return (
        any(part in IGNORED_SCAN_DIRS for part in directory_parts)
        or any(part.endswith(IGNORED_SCAN_SUFFIXES) for part in directory_parts)
        or any(part.startswith(IGNORED_SCAN_PREFIXES) for part in directory_parts)
    )


def _legacy_text_offenders(tokens: tuple[str, ...], ignored_files: set[Path]) -> list[str]:
    ignored_resolved = {path.resolve() for path in ignored_files}
    offenders: list[str] = []

    # Prune ignored directories during traversal (rather than filtering after a
    # full rglob) so the scan never descends into large generated/scratch trees
    # such as .venv/ or tmp/. This keeps the scan fast regardless of local state.
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in IGNORED_SCAN_DIRS
            and not d.endswith(IGNORED_SCAN_SUFFIXES)
            and not d.startswith(IGNORED_SCAN_PREFIXES)
        ]
        for name in filenames:
            path = Path(dirpath) / name
            relative = path.relative_to(REPO_ROOT)
            if path.resolve() in ignored_resolved or _is_ignored_scan_path(relative):
                continue

            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            for token in tokens:
                if token in text:
                    offenders.append(f"{relative}: {token}")

    return offenders


def test_legacy_text_scan_skips_generated_output_dirs():
    assert _is_ignored_scan_path(Path("nemo_retriever/artifacts/run/output.json"))
    assert _is_ignored_scan_path(Path("lancedb/session.arrow"))
    assert _is_ignored_scan_path(Path("lancedb_session/data.arrow"))
    assert not _is_ignored_scan_path(Path("nemo_retriever/src/nemo_retriever/vdb/lancedb_bulk.py"))


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


@requires_workflows
def test_public_nightly_python_publish_workflows_do_not_target_testpypi():
    workflow_names = ("pypi-nightly-publish.yml", "huggingface-nightly.yml")

    for workflow_name in workflow_names:
        workflow = (WORKFLOWS / workflow_name).read_text(encoding="utf-8")

        assert "testpypi" not in workflow.lower(), workflow_name
        assert "test.pypi.org" not in workflow.lower(), workflow_name
        assert "https://upload.pypi.org/legacy/" in workflow, workflow_name
        assert "PYPI_API_TOKEN" in workflow, workflow_name


@requires_workflows
def test_public_nightly_python_publish_workflows_use_read_only_token_permissions():
    workflow_names = ("pypi-nightly-publish.yml", "huggingface-nightly.yml")

    for workflow_name in workflow_names:
        workflow = _load_workflow(workflow_name)

        assert workflow["permissions"] == {"contents": "read"}, workflow_name


@requires_workflows
def test_pypi_nightly_publish_uses_twine_password_env():
    workflow = _load_workflow("pypi-nightly-publish.yml")
    steps = workflow["jobs"]["build"]["steps"]
    publish_step = next(step for step in steps if step.get("name") == "Publish wheels")

    assert publish_step["env"] == {"TWINE_PASSWORD": "${{ secrets.PYPI_API_TOKEN }}"}
    assert ' -p "${token}"' not in publish_step["run"]
    assert "PYPI_API_TOKEN" not in publish_step["run"]


def test_legacy_nv_ingest_root_compose_stack_is_removed():
    legacy_paths = (
        "docker-compose.yaml",
        "docker-compose.a100-40gb.yaml",
        "docker-compose.a10g.yaml",
        "docker-compose.l40s.yaml",
        "docker-compose.rtx-pro-4500.yaml",
        "ci/scripts/validate_deployment_configs.py",
        "skaffold/README.md",
        "skaffold/nv-ingest.skaffold.yaml",
        "skaffold/sensitive/.gitignore",
    )

    for relative_path in legacy_paths:
        assert not (REPO_ROOT / relative_path).exists(), relative_path

    legacy_tokens = (
        "nv-ingest-ms-runtime",
        "nvcr.io/nvidia/nemo-microservices/nv-ingest",
        "target: runtime",
        "validate_deployment_configs",
        "skaffold/nv-ingest.skaffold.yaml",
    )
    ignored_files = {
        THIS_FILE,
        REPO_ROOT / "nemo_retriever" / "tests" / "test_harness_helm_profiles.py",
    }
    assert _legacy_text_offenders(legacy_tokens, ignored_files) == []


def test_dev_compose_helpers_are_feature_scoped():
    compose_dir = REPO_ROOT / "nemo_retriever" / "dev" / "compose"
    expected_services = {
        "judge.compose.yaml": "judge",
        "neo4j.compose.yaml": "neo4j",
    }

    compose_data = {}
    for filename, service_name in expected_services.items():
        compose_path = compose_dir / filename
        assert compose_path.exists(), filename
        text = compose_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        compose_data[filename] = data
        assert set(data["services"]) == {service_name}
        assert "nv-ingest-ms-runtime" not in text
        assert "ngcapikey" not in text
        assert "neo4jpassword" not in text

    judge_environment = compose_data["judge.compose.yaml"]["services"]["judge"]["environment"]
    assert any("NGC_API_KEY or NIM_NGC_API_KEY must be set" in item for item in judge_environment)

    neo4j_environment = compose_data["neo4j.compose.yaml"]["services"]["neo4j"]["environment"]
    assert "NEO4J_PASSWORD must be set" in neo4j_environment["NEO4J_AUTH"]

    neo4j_healthcheck = compose_data["neo4j.compose.yaml"]["services"]["neo4j"]["healthcheck"]["test"]
    assert neo4j_healthcheck[0] == "CMD-SHELL"
    assert "-u" not in neo4j_healthcheck
    assert "-p" not in neo4j_healthcheck
    assert any("NEO4J_AUTH%%/*" in part for part in neo4j_healthcheck)
    assert any("NEO4J_AUTH#*/" in part for part in neo4j_healthcheck)

    helper_readme = compose_dir / "README.md"
    assert helper_readme.exists()
    assert "docker compose -f nemo_retriever/dev/compose/judge.compose.yaml up -d judge" in helper_readme.read_text(
        encoding="utf-8"
    )
    assert "docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml up -d neo4j" in helper_readme.read_text(
        encoding="utf-8"
    )

    docker_doc = REPO_ROOT / "nemo_retriever" / "docker.md"
    assert docker_doc.exists()
    docker_doc_text = docker_doc.read_text(encoding="utf-8")
    assert "--target service" in docker_doc_text
    assert "retriever service start" in docker_doc_text
    assert "docker compose" not in docker_doc_text.lower()
    assert "docker-compose" not in docker_doc_text.lower()

    skill_eval_config = (
        REPO_ROOT / "nemo_retriever" / "src" / "nemo_retriever" / "tools" / "skill_eval" / "configs" / "skill_eval.yaml"
    ).read_text(encoding="utf-8")
    assert "nemo_retriever/dev/compose/judge.compose.yaml" in skill_eval_config

    neo4j_setup = (
        REPO_ROOT / "nemo_retriever" / "src" / "nemo_retriever" / "tabular_data" / "neo4j" / "SETUP.md"
    ).read_text(encoding="utf-8")
    assert "nemo_retriever/dev/compose/neo4j.compose.yaml" in neo4j_setup


def test_legacy_tools_harness_is_removed():
    assert not (REPO_ROOT / "tools" / "harness").exists()

    legacy_tokens = ("tools/harness", "nv_ingest_harness", "nv-ingest-harness")
    assert _legacy_text_offenders(legacy_tokens, {THIS_FILE}) == []
