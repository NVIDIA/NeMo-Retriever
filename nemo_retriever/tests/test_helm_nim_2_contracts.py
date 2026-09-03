# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permanent Helm runtime and API contracts for extraction NIM 2.0."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import SkipTest

import yaml

CHART = Path(__file__).resolve().parents[1] / "helm"
EXPECTED_MODELS = {
    "nemotron-page-elements-v3": (
        "nvidia/nemotron-page-elements-v3",
        "/model-store/page-elements",
    ),
    "nemotron-table-structure-v1": (
        "nvidia/nemotron-table-structure-v1",
        "/model-store/table-structure",
    ),
    "nemotron-ocr-v2": ("nvidia/nemotron-ocr-v2", "/model-store/ocr"),
}


def _render() -> list[dict]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available")
    command = [
        helm,
        "template",
        "nrl-nim-2-contracts",
        str(CHART),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
        "--api-versions",
        "apps.nvidia.com/v1alpha1",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return [document for document in yaml.safe_load_all(completed.stdout) if isinstance(document, dict)]


def _find(documents: list[dict], kind: str, name: str) -> dict:
    return next(
        document
        for document in documents
        if document.get("kind") == kind and document.get("metadata", {}).get("name") == name
    )


def test_extraction_nims_select_distinct_native_models() -> None:
    documents = _render()

    for name, (model_name, model_path) in EXPECTED_MODELS.items():
        service = _find(documents, "NIMService", name)
        env = {item["name"]: item.get("value") for item in service["spec"]["env"]}
        assert env["NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER"] == "ngc"
        assert env["NIM_ENGINE_MODEL_NAME"] == model_name
        assert env["NIM_ENGINE_MODEL_PATH"] == model_path
        assert env["NIM_PERFORMANCE_MODE"] == "1"
        assert env["NIM_ENGINE_COUNT"] == "1"
        assert env["NIM_PIPELINE_MAX_BATCH_SIZE"] == "1"
        assert "NIM_TRITON_MAX_BATCH_SIZE" not in env

    ocr_env = {
        item["name"]: item.get("value") for item in _find(documents, "NIMService", "nemotron-ocr-v2")["spec"]["env"]
    }
    assert ocr_env["NIM_ENGINE_MODEL_VARIANT"] == "multilingual"


def test_default_embed_nim_uses_nemotron_3_auto_precision() -> None:
    documents = _render()
    service = _find(documents, "NIMService", "nemotron-3-embed-1b")
    assert service["spec"]["image"] == {
        "repository": "nvcr.io/nim/nvidia/nemotron-3-embed-1b",
        "tag": "2.2.2",
        "pullPolicy": "IfNotPresent",
        "pullSecrets": ["ngc-secret"],
    }
    env = {item["name"]: item.get("value") for item in service["spec"]["env"]}
    assert env["NIM_ENGINE_COUNT"] == "1"
    assert env["NIM_PERFORMANCE_MODE"] == "1"
    assert "NIM_ENGINE_PRECISION" not in env
    assert "NIM_TRITON_PERFORMANCE_MODE" not in env


def test_operator_managed_urls_use_nim_2_contracts() -> None:
    documents = _render()
    rendered_config = "\n".join(
        document.get("data", {}).get("retriever-service.yaml", "")
        for document in documents
        if document.get("kind") == "ConfigMap"
    )

    assert 'page_elements_invoke_url: "http://nemotron-page-elements-v3:8000/v1/page-elements"' in rendered_config
    assert 'table_structure_invoke_url: "http://nemotron-table-structure-v1:8000/v1/table-structure"' in rendered_config
    assert 'ocr_invoke_url: "http://nemotron-ocr-v2:8000/v1/ocr"' in rendered_config
