# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rerank-endpoint auto-wiring in the Helm chart.

The chart can deploy the VL reranker (``llama-nemotron-rerank-vl-1b-v2``)
as a NIMService, but until now the retriever-service ConfigMap rendered no
``rerank_invoke_url``, so ``POST /v1/answer`` never reranked retrieval hits
even when the reranker was Ready in the cluster.

These tests pin the chart-side wiring:

* ``serviceConfig.nimEndpoints`` exposes ``rerankInvokeUrl`` and
  ``rerankModelName`` overrides, defaulting empty; ``serviceConfig.rerank``
  exposes the behaviour knobs (``enabled`` / ``refineFactor`` / ``maxLength``).
* ``templates/configmap.yaml`` resolves the rerank URL via the standard
  ``nim.endpointURL`` helper (operator-managed
  ``llama-nemotron-rerank-vl-1b-v2`` at ``/v1/ranking``) and renders both
  the ``nim_endpoints`` fields and the ``rerank`` section.
* Explicit ``rerankInvokeUrl`` overrides win; the model name defaults to the
  canonical VL rerank model id whenever any rerank URL is resolved; and
  ``rerank.enabled`` flips true automatically when a URL is resolved.

The integration tests shell out to ``helm template`` when ``helm`` is on
``$PATH``; otherwise they skip cleanly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest, TestCase, main


# Must match nemo_retriever.models.__init__.VL_RERANK_MODEL.
_VL_RERANK_MODEL_ID = "nvidia/llama-nemotron-rerank-vl-1b-v2"
_RERANK_OPERATOR_SERVICE = "llama-nemotron-rerank-vl-1b-v2"
_RERANK_INVOKE_PATH = "/v1/ranking"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_required_file(path: Path) -> str:
    if not path.is_file():
        raise SkipTest(f"Required file not present in this test environment: {path}")
    return path.read_text(encoding="utf-8")


def _helm_template(
    extra_args: Sequence[str] = (),
    api_versions: Sequence[str] = (),
) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    chart_path = _repo_root() / "nemo_retriever/helm"
    if not chart_path.is_dir():
        raise SkipTest(f"Chart directory missing: {chart_path}")

    cmd: list[str] = [
        helm,
        "template",
        "retriever",
        str(chart_path),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
    ]
    for v in api_versions:
        cmd += ["--api-versions", v]
    cmd += list(extra_args)
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _assert_helm_ok(self: TestCase, proc: subprocess.CompletedProcess[str]) -> None:
    self.assertEqual(
        proc.returncode,
        0,
        f"`helm template` failed unexpectedly:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
    )


class HelmRerankEndpointTests(TestCase):
    """Source-level + integration coverage of the rerank auto-wiring."""

    # ------------------------------------------------------------------
    # Source / values
    # ------------------------------------------------------------------

    def test_values_expose_rerank_endpoint_overrides(self) -> None:
        values = _read_required_file(_repo_root() / "nemo_retriever/helm/values.yaml")
        self.assertIn("rerankInvokeUrl:", values)
        self.assertIn("rerankModelName:", values)
        self.assertIn('rerankInvokeUrl: ""', values)
        self.assertIn('rerankModelName: ""', values)
        # Behaviour knobs.
        self.assertIn("refineFactor:", values)
        self.assertIn("maxLength:", values)

    def test_configmap_resolves_rerank_url_via_standard_helper(self) -> None:
        body = _read_required_file(_repo_root() / "nemo_retriever/helm/templates/configmap.yaml")
        self.assertIn('"key" "rerankqa"', body)
        self.assertIn(f'"serviceName" "{_RERANK_OPERATOR_SERVICE}"', body)
        self.assertIn(f'"invokePath" "{_RERANK_INVOKE_PATH}"', body)
        self.assertIn('"configKey" "rerankInvokeUrl"', body)
        # nim_endpoints fields + the rerank behaviour section must render.
        self.assertIn("rerank_invoke_url:", body)
        self.assertIn("rerank_model_name:", body)
        self.assertIn("rerank:", body)
        self.assertIn("refine_factor:", body)

    # ------------------------------------------------------------------
    # Integration: actual `helm template` against the chart
    # ------------------------------------------------------------------

    def test_helm_template_autowires_rerank_when_operator_enabled(self) -> None:
        proc = _helm_template(
            extra_args=("--set", "nimOperator.rerankqa.enabled=true"),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        expected_url = f'rerank_invoke_url: "http://{_RERANK_OPERATOR_SERVICE}:8000{_RERANK_INVOKE_PATH}"'
        expected_model = f'rerank_model_name: "{_VL_RERANK_MODEL_ID}"'
        self.assertIn(expected_url, proc.stdout)
        self.assertIn(expected_model, proc.stdout)
        # Resolving a URL auto-enables reranking on /v1/answer.
        self.assertIn("rerank:\n  enabled: true", proc.stdout)

    def test_helm_template_rerank_null_when_operator_disabled(self) -> None:
        proc = _helm_template(
            extra_args=("--set", "nimOperator.rerankqa.enabled=false"),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        self.assertIn("rerank_invoke_url: null", proc.stdout)
        self.assertIn("rerank_model_name: null", proc.stdout)
        self.assertIn("rerank:\n  enabled: false", proc.stdout)

    def test_helm_template_explicit_rerank_url_wins(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.rerankqa.enabled=true",
                "--set",
                "serviceConfig.nimEndpoints.rerankInvokeUrl=https://integrate.api.nvidia.com/v1/ranking",
                "--set",
                "serviceConfig.nimEndpoints.rerankModelName=nvidia/some-other-rerank",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        self.assertIn('rerank_invoke_url: "https://integrate.api.nvidia.com/v1/ranking"', proc.stdout)
        self.assertIn('rerank_model_name: "nvidia/some-other-rerank"', proc.stdout)

    def test_helm_template_explicit_url_defaults_model_to_vl_rerank(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.rerankqa.enabled=false",
                "--set",
                "serviceConfig.nimEndpoints.rerankInvokeUrl=https://integrate.api.nvidia.com/v1/ranking",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        self.assertIn('rerank_invoke_url: "https://integrate.api.nvidia.com/v1/ranking"', proc.stdout)
        self.assertIn(f'rerank_model_name: "{_VL_RERANK_MODEL_ID}"', proc.stdout)

    def test_helm_template_rerank_url_renders_in_split_mode(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.rerankqa.enabled=true",
                "--set",
                "topology.mode=split",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        url_count = proc.stdout.count(f"http://{_RERANK_OPERATOR_SERVICE}:8000{_RERANK_INVOKE_PATH}")
        self.assertGreaterEqual(url_count, 3)


if __name__ == "__main__":
    main()
