# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest

from ci.scripts import nightly_build_publish as nightly


def test_patch_pyproject_version_can_override_nightly_base_version(tmp_path, monkeypatch):
    monkeypatch.setenv("NIGHTLY_DATE_SUFFIX", "20260501010101")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            """
            [project]
            name = "nemotron-ocr"
            version = "1.0.0"
            """
        ).lstrip(),
        encoding="utf-8",
    )

    assert nightly._patch_pyproject_version(tmp_path, nightly_base_version="1.0.2")

    patched = pyproject.read_text(encoding="utf-8")
    assert 'version = "1.0.2.dev20260501010101"' in patched


def test_patch_setup_cfg_version_can_override_nightly_base_version(tmp_path, monkeypatch):
    monkeypatch.setenv("NIGHTLY_DATE_SUFFIX", "20260501010101")
    setup_cfg = tmp_path / "setup.cfg"
    setup_cfg.write_text(
        textwrap.dedent(
            """
            [metadata]
            name = nemotron-ocr
            version = 1.0.0
            """
        ).lstrip(),
        encoding="utf-8",
    )

    assert nightly._patch_setup_cfg_version(tmp_path, nightly_base_version="1.0.2")

    patched = setup_cfg.read_text(encoding="utf-8")
    assert "version = 1.0.2.dev20260501010101" in patched


def test_patch_pyproject_runtime_dependency_pins_only_project_dependencies(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            """
            [project]
            name = "nemotron-ocr"
            version = "1.0.0"
            dependencies = [
                "huggingface_hub>=0.20.0",
                "torch>=2.8.0",
                "torchvision>=0.23.0",
                "shapely>=2.1.2,<3",
            ]

            [tool.hatch.build.targets.wheel.hooks.custom]
            path = "hatch_build.py"
            dependencies = ["setuptools>=68", "torch>=2.0"]
            """
        ).lstrip(),
        encoding="utf-8",
    )

    assert nightly._patch_pyproject_runtime_dependency_pins(
        tmp_path,
        {"torch": "2.10.0", "torchvision": "0.25.0"},
    )

    patched = pyproject.read_text(encoding="utf-8")
    assert '"torch~=2.10.0",' in patched
    assert '"torchvision~=0.25.0",' in patched
    assert '"shapely>=2.1.2,<3",' in patched
    assert 'dependencies = ["setuptools>=68", "torch>=2.0"]' in patched


def test_patch_pyproject_runtime_dependency_pins_requires_matching_dependency(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "example"
            version = "1.0.0"
            dependencies = [
                "numpy>=2",
            ]
            """
        ).lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No matching \\[project\\].dependencies entries"):
        nightly._patch_pyproject_runtime_dependency_pins(tmp_path, {"torch": "2.10.0"})


def test_patch_pyproject_runtime_dependency_pins_normalizes_dotted_names(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            """
            [project]
            name = "example"
            version = "1.0.0"
            dependencies = [
                "torch.special>=2.8.0",
            ]
            """
        ).lstrip(),
        encoding="utf-8",
    )

    assert nightly._patch_pyproject_runtime_dependency_pins(tmp_path, {"torch-special": "2.10.0"})

    patched = pyproject.read_text(encoding="utf-8")
    assert '"torch.special~=2.10.0",' in patched


def test_patch_pyproject_runtime_dependency_pins_searches_across_project_subtables(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            """
            [project]
            name = "nemotron-ocr"
            version = "1.0.0"

            [project.optional-dependencies]
            dev = ["pytest>=8"]

            dependencies = [
                "torch>=2.8.0",
            ]

            [tool.hatch.build.targets.wheel.hooks.custom]
            dependencies = ["torch>=2.0"]
            """
        ).lstrip(),
        encoding="utf-8",
    )

    assert nightly._patch_pyproject_runtime_dependency_pins(tmp_path, {"torch": "2.10.0"})

    patched = pyproject.read_text(encoding="utf-8")
    assert '"torch~=2.10.0",' in patched
    assert 'dependencies = ["torch>=2.0"]' in patched


def test_runtime_dependency_specifier_omits_local_suffix_and_allows_patch_releases():
    assert nightly._runtime_dependency_specifier("2.10.0+cu130") == "~=2.10.0"


def test_installed_distribution_public_version_strips_local_suffix(monkeypatch):
    run = Mock(
        return_value=nightly.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="2.10.0+cu130\n",
            stderr="",
        )
    )
    monkeypatch.setattr(nightly.subprocess, "run", run)

    version = nightly._installed_distribution_public_version(
        Path("/venv/bin/python"),
        "torch",
        cwd=Path("/project"),
        env={"PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )

    assert version == "2.10.0"
    run.assert_called_once()
    args, kwargs = run.call_args
    assert args[0][0] == "/venv/bin/python"
    assert args[0][-1] == "torch"
    assert kwargs["cwd"] == "/project"
    assert kwargs["env"] == {"PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    assert kwargs["text"] is True
    assert kwargs["stdout"] is nightly.subprocess.PIPE
    assert kwargs["stderr"] is nightly.subprocess.PIPE


def test_installed_distribution_public_version_raises_on_subprocess_failure(monkeypatch, capsys):
    run = Mock(
        return_value=nightly.subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="partial stdout\n",
            stderr="metadata failed\n",
        )
    )
    monkeypatch.setattr(nightly.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="Unable to read installed distribution version for 'torch'"):
        nightly._installed_distribution_public_version(
            Path("/venv/bin/python"),
            "torch",
            cwd=Path("/project"),
            env={},
        )

    captured = capsys.readouterr()
    assert "partial stdout" in captured.out
    assert "metadata failed" in captured.err
