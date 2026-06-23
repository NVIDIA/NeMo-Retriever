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
    assert any(str(spec).startswith(">=") and "1.86.0" in str(spec) for spec in litellm.specifier)
