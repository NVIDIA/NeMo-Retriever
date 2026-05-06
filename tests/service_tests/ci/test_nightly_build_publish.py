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


def test_patch_pyproject_project_name(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            """
            [project]
            name = "nemotron-ocr"
            version = "1.0.0"

            [tool.hatch.build.targets.wheel]
            packages = ["src/nemotron_ocr"]
            """
        ).lstrip(),
        encoding="utf-8",
    )

    assert nightly._patch_pyproject_project_name(tmp_path, "nemotron-ocr-v2")

    patched = pyproject.read_text(encoding="utf-8")
    assert 'name = "nemotron-ocr-v2"' in patched
    assert 'packages = ["src/nemotron_ocr"]' in patched


def test_patch_setup_cfg_project_name(tmp_path):
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

    assert nightly._patch_setup_cfg_project_name(tmp_path, "nemotron-ocr-v2")

    patched = setup_cfg.read_text(encoding="utf-8")
    assert "name = nemotron-ocr-v2" in patched


def test_rename_python_package_patches_source_package_without_touching_cpp_extension(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            """
            [project]
            name = "nemotron-ocr-v2"
            version = "2.0.0.dev20260501010101"

            [tool.hatch.build]
            artifacts = ["src/nemotron_ocr_cpp/*.so"]

            [tool.hatch.build.targets.wheel]
            packages = [
                "src/nemotron_ocr",
                "src/nemotron_ocr_cpp",
            ]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    package_dir = tmp_path / "src" / "nemotron_ocr"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "pipeline_v2.py").write_text(
        "from nemotron_ocr.inference import models\nimport nemotron_ocr_cpp\n",
        encoding="utf-8",
    )
    cpp_dir = tmp_path / "src" / "nemotron_ocr_cpp"
    cpp_dir.mkdir()
    (cpp_dir / "__init__.py").write_text("", encoding="utf-8")

    assert nightly._rename_python_package(tmp_path, "nemotron_ocr", "nemotron_ocr_v2")

    assert not package_dir.exists()
    assert (tmp_path / "src" / "nemotron_ocr_v2" / "__init__.py").exists()
    patched_pyproject = pyproject.read_text(encoding="utf-8")
    assert '"src/nemotron_ocr_v2",' in patched_pyproject
    assert '"src/nemotron_ocr_cpp",' in patched_pyproject
    assert 'artifacts = ["src/nemotron_ocr_cpp/*.so"]' in patched_pyproject
    patched_pipeline = (tmp_path / "src" / "nemotron_ocr_v2" / "pipeline_v2.py").read_text(encoding="utf-8")
    assert "from nemotron_ocr_v2.inference import models" in patched_pipeline
    assert "import nemotron_ocr_cpp" in patched_pipeline


def test_parse_package_rename_requires_old_equals_new():
    with pytest.raises(ValueError, match="OLD=NEW"):
        nightly._parse_package_rename("nemotron_ocr_v2")


def test_build_no_isolation_installs_pyproject_build_system_requires(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [build-system]
            requires = ["hatchling", "editables"]
            build-backend = "hatchling.build"
            """
        ).lstrip(),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    pip_install_calls = []

    monkeypatch.setattr(
        nightly,
        "_ensure_venv",
        lambda venv_dir, *, system_site_packages: Path("/venv/bin/python"),
    )

    def fake_pip_install(py, packages, *, cwd, env):
        pip_install_calls.append(list(packages))

    def fake_run(cmd, *, cwd=None, env=None):
        assert cmd[:3] == ["/venv/bin/python", "-m", "build"]
        assert "--no-isolation" in cmd
        dist_dir = project_dir / "dist"
        dist_dir.mkdir()
        (dist_dir / "example-0.0.0.tar.gz").write_text("sdist", encoding="utf-8")

    monkeypatch.setattr(nightly, "_pip_install", fake_pip_install)
    monkeypatch.setattr(nightly, "_run", fake_run)

    nightly._build(
        project_dir,
        out_dir,
        build_env=[],
        no_isolation=True,
        venv_system_site_packages=False,
        venv_pip_install=["torch==2.10.0"],
        pin_runtime_dependencies=[],
    )

    assert pip_install_calls == [
        ["build"],
        ["hatchling", "editables"],
        ["torch==2.10.0"],
    ]
    assert (out_dir / "example-0.0.0.tar.gz").exists()


def test_build_no_isolation_skips_empty_pyproject_build_system_requires(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "example"
            version = "0.0.0"
            """
        ).lstrip(),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    pip_install_calls = []

    monkeypatch.setattr(
        nightly,
        "_ensure_venv",
        lambda venv_dir, *, system_site_packages: Path("/venv/bin/python"),
    )

    def fake_pip_install(py, packages, *, cwd, env):
        pip_install_calls.append(list(packages))

    def fake_run(cmd, *, cwd=None, env=None):
        dist_dir = project_dir / "dist"
        dist_dir.mkdir()
        (dist_dir / "example-0.0.0.tar.gz").write_text("sdist", encoding="utf-8")

    monkeypatch.setattr(nightly, "_pip_install", fake_pip_install)
    monkeypatch.setattr(nightly, "_run", fake_run)

    nightly._build(
        project_dir,
        out_dir,
        build_env=[],
        no_isolation=True,
        venv_system_site_packages=False,
        venv_pip_install=["hatchling"],
        pin_runtime_dependencies=[],
    )

    assert pip_install_calls == [
        ["build"],
        ["hatchling"],
    ]


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
