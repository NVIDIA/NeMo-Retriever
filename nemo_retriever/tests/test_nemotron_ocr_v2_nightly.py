# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_local_extra_depends_on_versioned_ocr_v2_nightly() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    local_deps = pyproject["project"]["optional-dependencies"]["local"]
    uv_tool = pyproject["tool"]["uv"]
    uv_sources = uv_tool["sources"]

    assert "nemotron-ocr>=1.0.2.dev0,<1.0.2a0; sys_platform == 'linux'" in local_deps
    assert "nemotron-ocr" in uv_tool["no-build-package"]
    assert uv_sources["nemotron-ocr"] == {"index": "test-pypi"}


def test_local_ocr_v2_wrapper_imports_versioned_module() -> None:
    source = (PROJECT_ROOT / "src" / "nemo_retriever" / "model" / "local" / "nemotron_ocr_v2.py").read_text(
        encoding="utf-8"
    )

    assert "from nemotron_ocr.inference import pipeline_v2" in source
    assert "Local Nemotron OCR v2 requires the `nemotron_ocr_v2` package." in source
