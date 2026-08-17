# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for manual Helm service-reranker configuration."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import SkipTest, TestCase


_RERANK_URL = "http://llama-nemotron-rerank-vl-1b-v2:8000/v1/ranking"
_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _helm_template(*args: str) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    return subprocess.run(
        [
            helm,
            "template",
            "retriever",
            str(_repo_root() / "nemo_retriever/helm"),
            "--set",
            "ngcImagePullSecret.create=false",
            "--set",
            "ngcApiSecret.create=false",
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


class HelmRerankEndpointTests(TestCase):
    def test_values_expose_manual_rerank_configuration(self) -> None:
        values = (_repo_root() / "nemo_retriever/helm/values.yaml").read_text(encoding="utf-8")
        self.assertIn('rerankInvokeUrl: ""', values)
        self.assertIn('rerankModelName: ""', values)

    def test_explicit_rerank_values_render_into_service_config(self) -> None:
        rendered = _helm_template(
            "--api-versions",
            "apps.nvidia.com/v1alpha1",
            "--set",
            f"serviceConfig.nimEndpoints.rerankInvokeUrl={_RERANK_URL}",
            "--set",
            f"serviceConfig.nimEndpoints.rerankModelName={_RERANK_MODEL}",
        )
        self.assertEqual(rendered.returncode, 0, rendered.stderr)
        self.assertIn(f'rerank_invoke_url: "{_RERANK_URL}"', rendered.stdout)
        self.assertIn(f'rerank_model_name: "{_RERANK_MODEL}"', rendered.stdout)

    def test_enabling_rerank_nim_does_not_auto_wire_service_config(self) -> None:
        rendered = _helm_template(
            "--api-versions",
            "apps.nvidia.com/v1alpha1",
            "--set",
            "nimOperator.rerankqa.enabled=true",
        )
        self.assertEqual(rendered.returncode, 0, rendered.stderr)
        self.assertIn("rerank_invoke_url: null", rendered.stdout)
        self.assertIn("rerank_model_name: null", rendered.stdout)
