# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Packaging contracts for the agent implementation."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_private_agent_prompt_resources_are_declared_for_distribution():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))

    dependencies = project["project"]["dependencies"]
    package_data = project["tool"]["setuptools"]["package-data"]

    assert any(dependency.lower().startswith("jinja2") for dependency in dependencies)
    assert package_data["nemo_retriever._agentic.nemo_agent.prompts"] == ["templates/**/*.j2"]
