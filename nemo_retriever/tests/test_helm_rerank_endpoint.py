# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for VL reranker endpoint auto-wiring.

In 26.08 the chart could deploy the VL reranker as a NIMService and
accept ``serviceConfig.nimEndpoints.rerankInvokeUrl`` /
``rerankModelName``, but ``templates/configmap.yaml`` never rendered
``nim_endpoints.rerank_invoke_url`` / ``rerank_model_name``. Split-mode
gateway ConfigMaps therefore started without a configured reranker, so
``POST /v1/query`` with ``rerank=true`` returned HTTP 400 even when the
Rerank NIM was Ready.

These tests pin the chart-side fix:

* ``serviceConfig.nimEndpoints`` exposes ``rerankInvokeUrl`` and
  ``rerankModelName`` overrides, defaulting empty.
* ``templates/configmap.yaml`` resolves the rerank URL via the standard
  ``nim.endpointURL`` helper (operator-managed
  ``llama-nemotron-rerank-vl-1b-v2`` at ``/v1/ranking``) and renders both
  fields.
* Explicit ``rerankInvokeUrl`` overrides win; the model name defaults to
  the canonical VL reranker model id whenever any rerank URL is resolved.

The integration tests shell out to ``helm template`` when ``helm`` is
on ``$PATH``; otherwise they skip cleanly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest, TestCase, main


_RERANK_REMOTE_MODEL_ID = "nvidia/llama-nemotron-rerank-vl-1b-v2"
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
    """Source-level + integration coverage of the rerank auto-wiring fix."""

    def test_values_expose_rerank_endpoint_overrides(self) -> None:
        values = _read_required_file(_repo_root() / "nemo_retriever/helm/values.yaml")
        self.assertIn("rerankInvokeUrl:", values)
        self.assertIn("rerankModelName:", values)
        self.assertIn('rerankInvokeUrl: ""', values)
        self.assertIn('rerankModelName: ""', values)

    def test_configmap_resolves_rerank_url_via_standard_helper(self) -> None:
        body = _read_required_file(_repo_root() / "nemo_retriever/helm/templates/configmap.yaml")
        self.assertIn(
            '"key" "rerankqa"',
            body,
            "configmap.yaml must resolve the rerank URL via " "nemo-retriever.nim.endpointURL keyed on rerankqa.",
        )
        self.assertIn(f'"serviceName" "{_RERANK_OPERATOR_SERVICE}"', body)
        self.assertIn(f'"invokePath" "{_RERANK_INVOKE_PATH}"', body)
        self.assertIn('"configKey" "rerankInvokeUrl"', body)
        self.assertIn("rerank_invoke_url:", body)
        self.assertIn("rerank_model_name:", body)

    def test_readme_documents_rerank_autowiring(self) -> None:
        readme = _read_required_file(_repo_root() / "nemo_retriever/helm/README.md")
        self.assertIn("query-time-reranking", readme)
        self.assertIn("rerank_invoke_url", readme)
        self.assertIn(_RERANK_REMOTE_MODEL_ID, readme)

    def test_helm_template_autowires_rerank_when_rerankqa_enabled(self) -> None:
        """The exact customer-reported repro must now wire rerank fields."""
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.rerankqa.enabled=true",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        expected_url = f'rerank_invoke_url: "http://{_RERANK_OPERATOR_SERVICE}:8000{_RERANK_INVOKE_PATH}"'
        expected_model = f'rerank_model_name: "{_RERANK_REMOTE_MODEL_ID}"'
        self.assertIn(
            expected_url,
            proc.stdout,
            "configmap must auto-wire the operator-managed VL rerank URL into " "nim_endpoints.rerank_invoke_url.",
        )
        self.assertIn(
            expected_model,
            proc.stdout,
            "configmap must auto-set the canonical VL reranker model id.",
        )

    def test_helm_template_rerank_null_when_rerankqa_disabled(self) -> None:
        """Without the rerank NIM and no explicit URL, the rerank fields must be null."""
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.rerankqa.enabled=false",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        self.assertIn("rerank_invoke_url: null", proc.stdout)
        self.assertIn("rerank_model_name: null", proc.stdout)

    def test_helm_template_explicit_rerank_url_wins(self) -> None:
        """`rerankInvokeUrl` override must beat operator wiring."""
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.rerankqa.enabled=true",
                "--set",
                "serviceConfig.nimEndpoints.rerankInvokeUrl=http://external-rerank:8000/v1/ranking",
                "--set",
                "serviceConfig.nimEndpoints.rerankModelName=nvidia/some-other-reranker",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        self.assertIn(
            'rerank_invoke_url: "http://external-rerank:8000/v1/ranking"',
            proc.stdout,
        )
        self.assertIn('rerank_model_name: "nvidia/some-other-reranker"', proc.stdout)

    def test_helm_template_explicit_url_defaults_model_to_vl(self) -> None:
        """Explicit URL with no model override falls back to the VL model id."""
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.rerankqa.enabled=false",
                "--set",
                "serviceConfig.nimEndpoints.rerankInvokeUrl=http://external-rerank:8000/v1/ranking",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        self.assertIn(
            'rerank_invoke_url: "http://external-rerank:8000/v1/ranking"',
            proc.stdout,
        )
        self.assertIn(f'rerank_model_name: "{_RERANK_REMOTE_MODEL_ID}"', proc.stdout)

    def test_helm_template_rerank_operator_url_renders_in_split_mode(self) -> None:
        """Split-topology renders the same nim_endpoints block per role."""
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
        self.assertGreaterEqual(
            url_count,
            3,
            "expected the VL rerank URL to render in every per-role ConfigMap "
            f"(split mode), saw {url_count} occurrence(s).",
        )

    def test_helm_template_explicit_rerank_values_render_in_split_mode(self) -> None:
        """Customer repro: explicit serviceConfig values must reach split ConfigMaps."""
        proc = _helm_template(
            extra_args=(
                "--set",
                "topology.mode=split",
                "--set",
                "nimOperator.rerankqa.enabled=true",
                "--set",
                "serviceConfig.nimEndpoints.rerankInvokeUrl=" "http://llama-nemotron-rerank-vl-1b-v2:8000/v1/ranking",
                "--set",
                f"serviceConfig.nimEndpoints.rerankModelName={_RERANK_REMOTE_MODEL_ID}",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        self.assertIn(
            'rerank_invoke_url: "http://llama-nemotron-rerank-vl-1b-v2:8000/v1/ranking"',
            proc.stdout,
        )
        self.assertIn(f'rerank_model_name: "{_RERANK_REMOTE_MODEL_ID}"', proc.stdout)


if __name__ == "__main__":
    main()
