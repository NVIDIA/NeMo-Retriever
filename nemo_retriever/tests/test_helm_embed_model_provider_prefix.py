# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helm rendering for the remote embedding provider prefix."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import SkipTest

import yaml

CHART = Path(__file__).resolve().parents[1] / "helm"


def _render(*extra_args: str) -> dict:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")

    command = [
        helm,
        "template",
        "embed-provider-prefix-test",
        str(CHART),
        "--set",
        "nims.enabled=false",
        "--set",
        "serviceConfig.nimEndpoints.embedInvokeUrl=http://embed:8000/v1/embeddings",
        *extra_args,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    documents = [document for document in yaml.safe_load_all(completed.stdout) if document]
    configmap = next(
        document
        for document in documents
        if document.get("kind") == "ConfigMap" and "retriever-service.yaml" in document.get("data", {})
    )
    return yaml.safe_load(configmap["data"]["retriever-service.yaml"])


def test_default_provider_prefix_preserves_null_schema_keys() -> None:
    config = _render()

    assert "embed_model_provider_prefix" in config["nim_endpoints"]
    assert config["nim_endpoints"]["embed_model_provider_prefix"] is None
    assert "embed_model_provider_prefix" in config["vectordb"]
    assert config["vectordb"]["embed_model_provider_prefix"] is None


def test_provider_prefix_override_is_rendered_in_both_sections() -> None:
    config = _render("--set", "serviceConfig.vectordb.embedModelProviderPrefix=nvidia")

    assert config["nim_endpoints"]["embed_model_provider_prefix"] == "nvidia"
    assert config["vectordb"]["embed_model_provider_prefix"] == "nvidia"
