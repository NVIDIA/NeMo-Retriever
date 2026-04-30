from __future__ import annotations

import textwrap

import pytest

from ci.scripts import nightly_build_publish as nightly


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


def test_runtime_dependency_specifier_omits_local_suffix_and_allows_patch_releases():
    assert nightly._runtime_dependency_specifier("2.10.0+cu130") == "~=2.10.0"
