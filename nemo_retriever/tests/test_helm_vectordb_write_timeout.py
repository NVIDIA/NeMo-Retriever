# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helm wiring for the worker-to-VectorDB write acknowledgement timeout."""

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

    completed = subprocess.run(
        [
            helm,
            "template",
            "vectordb-write-timeout-test",
            str(CHART),
            "--set",
            "nims.enabled=false",
            *extra_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    completed.check_returncode()
    documents = [document for document in yaml.safe_load_all(completed.stdout) if document]
    configmap = next(
        document
        for document in documents
        if document.get("kind") == "ConfigMap" and "retriever-service.yaml" in document.get("data", {})
    )
    return yaml.safe_load(configmap["data"]["retriever-service.yaml"])


def test_default_write_timeout_is_rendered() -> None:
    config = _render()

    assert config["vectordb"]["write_timeout_s"] == 300


def test_write_timeout_override_is_rendered() -> None:
    config = _render("--set", "serviceConfig.vectordb.writeTimeoutSeconds=900")

    assert config["vectordb"]["write_timeout_s"] == 900
