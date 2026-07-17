# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import zipfile
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


def test_huggingface_nightly_builder_defaults_to_public_pypi() -> None:
    script = (REPO_ROOT / "ci" / "scripts" / "nightly_build_publish.py").read_text(encoding="utf-8")

    assert 'default="https://upload.pypi.org/legacy/"' in script
    assert 'default="PYPI_API_TOKEN"' in script


def _write_wheel(path: Path, members: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for member in members:
            zf.writestr(member, "")


def test_nightly_builder_validates_required_members_in_every_wheel(tmp_path: Path) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()
    required_members = [
        "nemotron_ocr/inference/pipeline.py",
        "nemotron_ocr/inference/pipeline_v2.py",
    ]
    _write_wheel(tmp_path / "x86.whl", required_members)
    _write_wheel(tmp_path / "arm.whl", required_members + ["nemotron_ocr_cpp/extension.so"])

    nightly_build_publish._validate_required_wheel_members(tmp_path, required_members)


def test_nightly_builder_reports_wheel_and_missing_required_member(tmp_path: Path) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()
    _write_wheel(tmp_path / "nemotron_ocr.whl", ["nemotron_ocr/inference/pipeline_v2.py"])

    with pytest.raises(RuntimeError, match=r"nemotron_ocr\.whl: missing nemotron_ocr/inference/pipeline\.py"):
        nightly_build_publish._validate_required_wheel_members(
            tmp_path,
            ["nemotron_ocr/inference/pipeline.py", "nemotron_ocr/inference/pipeline_v2.py"],
        )


def test_nightly_builder_skips_wheel_validation_without_requirements(tmp_path: Path) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()

    nightly_build_publish._validate_required_wheel_members(tmp_path, [])


def test_nightly_builder_validates_wheels_before_upload() -> None:
    script = (REPO_ROOT / "ci" / "scripts" / "nightly_build_publish.py").read_text(encoding="utf-8")
    main_source = script.split("def main() -> int:", 1)[1]

    assert main_source.index("_validate_required_wheel_members(") < main_source.index("if args.upload:")


def test_huggingface_ocr_wheels_require_base_and_v2_pipeline_modules() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "huggingface-nightly.yml").read_text(encoding="utf-8")

    assert '--require-wheel-member "nemotron_ocr/inference/pipeline.py"' in workflow
    assert '--require-wheel-member "nemotron_ocr/inference/pipeline_v2.py"' in workflow
