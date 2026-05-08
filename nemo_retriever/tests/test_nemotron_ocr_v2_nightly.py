# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def test_local_extra_depends_on_ocr_2_nightly_only() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    local_deps = pyproject["project"]["optional-dependencies"]["local"]
    uv_tool = pyproject["tool"]["uv"]
    uv_sources = uv_tool["sources"]

    assert "nemotron-ocr>=2.0.0.dev0,<2.0.0a0; sys_platform == 'linux'" in local_deps
    assert not any(dep.startswith("nemotron-ocr-v2") for dep in local_deps)
    assert "nemotron-ocr" in uv_tool["no-build-package"]
    assert "nemotron-ocr-v2" not in uv_tool["no-build-package"]
    assert uv_sources["nemotron-ocr"] == {"index": "test-pypi"}
    assert "nemotron-ocr-v2" not in uv_sources


def test_local_ocr_v2_wrapper_uses_original_namespace_and_lang_selector() -> None:
    source = (PROJECT_ROOT / "src" / "nemo_retriever" / "model" / "local" / "nemotron_ocr_v2.py").read_text(
        encoding="utf-8"
    )

    assert "from nemotron_ocr.inference import pipeline_v2" in source
    assert "lang: str = \"v2_multi\"" in source
    assert "_NemotronOCRV2(model_dir=model_dir, lang=lang)" in source
    assert "_NemotronOCRV2(lang=lang)" in source
    assert "nemotron_ocr_v2" not in source
    assert "nemotron-ocr-v2` from TestPyPI" not in source


def test_huggingface_v2_nightly_publishes_to_ocr_project_without_namespace_rename() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "huggingface-nightly.yml").read_text(encoding="utf-8")
    v2_stanza = workflow.split("- id: nemotron-ocr-v2", 1)[1].split("container:", 1)[0]

    assert "project_name: \"\"" in v2_stanza
    assert "expected_project_name: nemotron-ocr\n" in v2_stanza
    assert "package_rename: \"\"" in v2_stanza
    assert "expected_package: nemotron_ocr\n" in v2_stanza
    assert "nightly_base_version: \"2.0.0\"" in v2_stanza
    assert "project_name: nemotron-ocr-v2" not in v2_stanza
    assert "expected_project_name: nemotron-ocr-v2" not in v2_stanza
    assert "package_rename: nemotron_ocr=nemotron_ocr_v2" not in v2_stanza
    assert "expected_package: nemotron_ocr_v2" not in v2_stanza
