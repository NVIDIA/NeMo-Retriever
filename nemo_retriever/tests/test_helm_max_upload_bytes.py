# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helm wiring for the service upload-size limit."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import SkipTest

import pytest
import yaml


CHART = Path(__file__).resolve().parents[1] / "helm"


def _helm_template(*extra_args: str) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")

    command = [
        helm,
        "template",
        "max-upload-bytes-test",
        str(CHART),
        "--set",
        "nims.enabled=false",
        "--set",
        "serviceConfig.vectordb.enabled=false",
        *extra_args,
    ]
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _render(*extra_args: str) -> dict:
    completed = _helm_template(*extra_args)
    completed.check_returncode()
    documents = [document for document in yaml.safe_load_all(completed.stdout) if document]
    configmap = next(
        document
        for document in documents
        if document.get("kind") == "ConfigMap" and "retriever-service.yaml" in document.get("data", {})
    )
    return yaml.safe_load(configmap["data"]["retriever-service.yaml"])


def test_default_max_upload_bytes_is_rendered() -> None:
    config = _render()

    max_upload_bytes = config["resources"]["max_upload_bytes"]
    assert isinstance(max_upload_bytes, int)
    assert max_upload_bytes == 500_000_000


def test_max_upload_bytes_override_is_rendered() -> None:
    config = _render("--set", "serviceConfig.resources.maxUploadBytes=2000000000")

    max_upload_bytes = config["resources"]["max_upload_bytes"]
    assert isinstance(max_upload_bytes, int)
    assert max_upload_bytes == 2_000_000_000


@pytest.mark.parametrize(
    "extra_args",
    [
        ("--set-string", "serviceConfig.resources.maxUploadBytes="),
        ("--set", "serviceConfig.resources.maxUploadBytes=null"),
        ("--set-string", "serviceConfig.resources.maxUploadBytes=not-a-number"),
        ("--set", "serviceConfig.resources.maxUploadBytes=true"),
        ("--set", "serviceConfig.resources.maxUploadBytes=1.5"),
        ("--set", "serviceConfig.resources.maxUploadBytes=0"),
        ("--set", "serviceConfig.resources.maxUploadBytes=-1"),
    ],
)
def test_invalid_max_upload_bytes_fails_render(extra_args: tuple[str, ...]) -> None:
    completed = _helm_template(*extra_args)

    assert completed.returncode != 0
    assert "serviceConfig.resources.maxUploadBytes must be a positive integer" in completed.stderr
