# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Source hygiene checks for the private agent implementation."""

from __future__ import annotations

from pathlib import Path


def _source_files() -> list[Path]:
    root = Path(__file__).parents[1] / "src" / "nemo_retriever"
    paths = list((root / "_agentic").rglob("*.py"))
    paths += [
        root / "operators" / "graph_ops" / "react_agent_operator.py",
        root / "operators" / "graph_ops" / "selection_agent_operator.py",
    ]
    return paths


def test_private_agent_sources_do_not_advertise_public_api_or_vendor_boundary():
    text = "\n".join(path.read_text(encoding="utf-8") for path in _source_files())

    assert "Public surface:" not in text
    assert "public API" not in text
    assert "vendored" not in text
