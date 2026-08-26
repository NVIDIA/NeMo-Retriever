# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _extra_requirements(extra: str) -> list[Requirement]:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["optional-dependencies"][extra]
    return [Requirement(dep) for dep in deps]


def test_service_extra_includes_litellm_for_answer_generation() -> None:
    requirements = _extra_requirements("service")

    litellm = next((req for req in requirements if req.name == "litellm"), None)

    assert litellm is not None
    assert any(str(spec).startswith(">=") and "1.95.0rc3" in str(spec) for spec in litellm.specifier)


def test_core_dependencies_include_tokenizer_stack() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [Requirement(dep) for dep in pyproject["project"]["dependencies"]]
    names = {req.name for req in requirements}

    assert "tokenizers" in names
    assert "huggingface-hub" in names


def test_os_classifiers_match_supported_platforms() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    classifiers = pyproject["project"]["classifiers"]

    assert "Operating System :: OS Independent" not in classifiers
    assert "Operating System :: POSIX :: Linux" in classifiers
    assert "Operating System :: Microsoft :: Windows" in classifiers
    assert "Operating System :: MacOS" in classifiers
