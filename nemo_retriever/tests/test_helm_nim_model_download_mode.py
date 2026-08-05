# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for direct-PVC model downloads in affected 2.x NIMs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import SkipTest

import yaml

_CHART = Path(__file__).resolve().parents[1] / "helm"
_AFFECTED = {
    "nemotron-page-elements-v3": "25Gi",
    "nemotron-table-structure-v1": "25Gi",
    "nemotron-ocr-v2": "25Gi",
    "llama-nemotron-embed-vl-1b-v2": "50Gi",
    "llama-nemotron-rerank-vl-1b-v2": "50Gi",
}


def _render(*sets: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available")
    cmd = [
        helm,
        "template",
        "nrl-download-mode",
        str(_CHART),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
        "--api-versions",
        "apps.nvidia.com/v1alpha1",
    ]
    for value in sets:
        cmd.extend(["--set", value])
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if check:
        assert proc.returncode == 0, proc.stderr
    return proc


def _docs(rendered: str) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]


def _find(docs: list[dict], kind: str, name: str) -> dict:
    return next(doc for doc in docs if doc.get("kind") == kind and doc.get("metadata", {}).get("name") == name)


def test_defaults_use_direct_pvc_for_affected_2x_services() -> None:
    docs = _docs(_render("nimOperator.rerankqa.enabled=true").stdout)
    cache_names = {doc["metadata"]["name"] for doc in docs if doc.get("kind") == "NIMCache"}
    assert cache_names.isdisjoint(_AFFECTED)

    for name, size in _AFFECTED.items():
        service = _find(docs, "NIMService", name)
        pvc = service["spec"]["storage"]["pvc"]
        assert pvc == {
            "create": True,
            "storageClass": "",
            "size": size,
            "volumeAccessMode": "ReadWriteOnce",
        }
        assert "nimCache" not in service["spec"]["storage"]

    assert _find(docs, "NIMService", "llama-nemotron-embed-vl-1b-v2")["spec"]["image"]["tag"] == "2.3.0"
    assert _find(docs, "NIMService", "llama-nemotron-rerank-vl-1b-v2")["spec"]["image"]["tag"] == "2.3.0"

    embed = _find(docs, "NIMService", "llama-nemotron-embed-vl-1b-v2")["spec"]
    assert "command" not in embed
    assert "args" not in embed


def test_nimcache_mode_restores_legacy_resources() -> None:
    docs = _docs(
        _render(
            "nimOperator.page_elements.modelDownloadMode=nimCache",
            "nimOperator.table_structure.modelDownloadMode=nimCache",
            "nimOperator.ocr.modelDownloadMode=nimCache",
            "nimOperator.vlm_embed.modelDownloadMode=nimCache",
            "nimOperator.rerankqa.enabled=true",
            "nimOperator.rerankqa.modelDownloadMode=nimCache",
        ).stdout
    )
    for name in _AFFECTED:
        _find(docs, "NIMCache", name)
        service = _find(docs, "NIMService", name)
        assert service["spec"]["storage"]["nimCache"]["name"] == name
        assert "pvc" not in service["spec"]["storage"]


def test_invalid_mode_fails_with_clear_message() -> None:
    proc = _render("nimOperator.ocr.modelDownloadMode=invalid", check=False)
    assert proc.returncode != 0
    assert "nimOperator.ocr.modelDownloadMode must be one of: nimService, nimCache" in proc.stderr
