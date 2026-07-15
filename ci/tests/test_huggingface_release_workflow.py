# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_nightly_build_publish_module() -> ModuleType:
    script_path = REPO_ROOT / "ci" / "scripts" / "nightly_build_publish.py"
    spec = importlib.util.spec_from_file_location("nightly_build_publish", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nightly_builder_can_patch_exact_release_version_in_pyproject(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        """
[build-system]
requires = ["hatchling"]

[project]
name = "example"
version = "2.0.0.dev20260520010101"
""".lstrip(),
        encoding="utf-8",
    )
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._patch_pyproject_version(project_dir, release_version="2.0.0")

    assert 'version = "2.0.0"' in pyproject.read_text(encoding="utf-8")


def test_nightly_builder_can_patch_exact_release_version_in_setup_cfg(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    setup_cfg = project_dir / "setup.cfg"
    setup_cfg.write_text(
        """
[metadata]
name = example
version = 2.0.0.dev20260520010101
""".lstrip(),
        encoding="utf-8",
    )
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._patch_setup_cfg_version(project_dir, release_version="2.0.0")

    assert "version = 2.0.0" in setup_cfg.read_text(encoding="utf-8")


def test_nightly_builder_relaxes_existing_requires_python(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        """
[build-system]
requires = ["hatchling"]

[project]
name = "example"
version = "2.0.0"
requires-python = ">=3.12,<3.13"
""".lstrip(),
        encoding="utf-8",
    )
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._patch_pyproject_requires_python(project_dir, ">=3.11,<3.14")

    text = pyproject.read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.14"' in text
    assert ">=3.12,<3.13" not in text


def test_nightly_builder_adds_requires_python_when_missing(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        """
[build-system]
requires = ["hatchling"]

[project]
name = "example"
version = "2.0.0"
""".lstrip(),
        encoding="utf-8",
    )
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._patch_pyproject_requires_python(project_dir, ">=3.11,<3.14")

    text = pyproject.read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.14"' in text
    # A no-op re-run must not duplicate the field.
    assert not nightly_build_publish._patch_pyproject_requires_python(project_dir, ">=3.11,<3.14")
    assert text.count("requires-python") == 1


@pytest.mark.parametrize(
    "version",
    ["", "2.0.0a1", "2.0.0rc1", "2.0.0+local", "2.0.0.dev1"],
)
def test_nightly_builder_rejects_non_stable_release_versions(version: str) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()

    with pytest.raises(ValueError, match="--release-version must be a stable public version"):
        nightly_build_publish._pep440_stable_release(version)


def test_nightly_builder_rejects_empty_release_version_with_nightly_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nightly_build_publish.py",
            "--repo-id",
            "example",
            "--repo-url",
            "https://huggingface.co/nvidia/example",
            "--nightly-base-version",
            "2.0.0",
            "--release-version",
            "",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        nightly_build_publish.main()

    assert exc_info.value.code == 2


def test_huggingface_workflow_has_manual_stable_ocr_release_controls() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "huggingface-nightly.yml").read_text(encoding="utf-8")

    assert "package:" in workflow
    assert "release_type:" in workflow
    assert "release_version:" in workflow
    assert "Stable releases must select a single package" in workflow
    assert "--release-version" in workflow
    assert 'expected_version="${INPUT_RELEASE_VERSION}"' in workflow
    assert "Built wheel metadata does not declare expected version" in workflow


def test_huggingface_non_ocr_nightlies_are_versioned_after_current_stable() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "huggingface-nightly.yml").read_text(encoding="utf-8")

    assert '--nightly-base-version "${{ matrix.repo.nightly_base_version }}"' in workflow
    assert "id: nemotron-page-elements-v3" in workflow
    assert 'nightly_base_version: "3.0.2"' in workflow
    assert "id: nemotron-table-structure-v1" in workflow
    assert workflow.count('nightly_base_version: "1.0.1"') == 1


def test_huggingface_ocr_builds_and_publishes_wheels_for_all_supported_pythons() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "huggingface-nightly.yml").read_text(encoding="utf-8")

    ocr_job = workflow.split("build_ocr_cuda:", 1)[1]

    # Python matrix drives setup-python so each leg builds an ABI-specific wheel.
    for version, tag in (("3.11", "cp311"), ("3.12", "cp312"), ("3.13", "cp313")):
        assert f'version: "{version}"' in ocr_job
        assert f"tag: {tag}" in ocr_job
    assert 'python-version: "${{ matrix.python.version }}"' in ocr_job

    # Published wheels stay pip-installable on every targeted interpreter.
    assert '--set-requires-python ">=3.11,<3.14"' in ocr_job

    # The identical sdist is emitted (and uploaded) exactly once across the matrix.
    assert "sdist_arg=" in ocr_job
    assert '"${{ matrix.platform.arch }}" == "x86_64" && "${{ matrix.python.version }}" == "3.12"' in ocr_job
    assert "${sdist_arg}" in ocr_job

    # Artifact names stay unique per architecture and Python version.
    assert "dist-${{ matrix.ocr.id }}-${{ matrix.platform.arch }}-${{ matrix.python.tag }}" in ocr_job


def test_huggingface_nightly_builder_defaults_to_public_pypi() -> None:
    script = (REPO_ROOT / "ci" / "scripts" / "nightly_build_publish.py").read_text(encoding="utf-8")

    assert 'default="https://upload.pypi.org/legacy/"' in script
    assert 'default="PYPI_API_TOKEN"' in script
