# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_CHART = _ROOT / "nemo_retriever" / "helm"


def _helm_template(*args: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("helm") is None:
        pytest.skip("helm executable is not installed")
    return subprocess.run(
        [
            "helm",
            "template",
            "retriever",
            str(_CHART),
            "--set",
            "serviceConfig.nimEndpoints.embedInvokeUrl=http://embed.example/v1/embeddings",
            *args,
        ],
        capture_output=True,
        check=False,
        text=True,
    )


@pytest.mark.parametrize("mode", ["auto", "dense", "hybrid"])
def test_vectordb_index_mode_renders_into_deployment(mode: str) -> None:
    result = _helm_template("--set", f"serviceConfig.vectordb.indexMode={mode}")
    assert result.returncode == 0, result.stderr
    documents = [doc for doc in yaml.safe_load_all(result.stdout) if doc]
    deployment = next(
        doc for doc in documents if doc.get("kind") == "Deployment" and doc["metadata"]["name"].endswith("-vectordb")
    )
    args = deployment["spec"]["template"]["spec"]["containers"][0]["args"]
    assert args[args.index("--index-mode") + 1] == mode


def test_vectordb_index_mode_rejects_invalid_value() -> None:
    result = _helm_template("--set", "serviceConfig.vectordb.indexMode=sparse")
    assert result.returncode != 0
    assert "indexMode must be one of" in result.stderr
