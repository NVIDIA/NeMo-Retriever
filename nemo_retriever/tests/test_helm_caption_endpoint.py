# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Omni 30B caption-endpoint auto-wiring.

In 26.05 RC2 the chart could deploy the Nemotron 3 Nano Omni VLM as a
NIMService, but the retriever-service ConfigMap rendered no
``caption_invoke_url`` / ``caption_model_name``.  The downstream service
deriving ``caption_enabled`` from
``nim_endpoints.caption_invoke_url`` would stay ``false``, so the
ingestion pipeline silently behaved as text-only even though Omni was
Ready in the cluster.

These tests pin the chart-side fix:

* ``serviceConfig.nimEndpoints`` exposes ``captionInvokeUrl`` and
  ``captionModelName`` overrides, defaulting empty.
* ``templates/configmap.yaml`` resolves the caption URL via the standard
  ``nim.endpointURL`` helper (operator-managed
  ``nemotron-3-nano-omni-30b-a3b-reasoning`` at
  ``/v1/chat/completions``) and renders both fields.
* Explicit ``captionInvokeUrl`` overrides win; the model name defaults
  to the canonical Omni remote model id whenever any caption URL is
  resolved.

The integration tests shell out to ``helm template`` when ``helm`` is
on ``$PATH``; otherwise they skip cleanly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest, TestCase, main


# Must match nemo_retriever.common.modality.caption.model_profiles.OMNI_REMOTE_MODEL_ID.
_OMNI_REMOTE_MODEL_ID = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
_OMNI_OPERATOR_SERVICE = "nemotron-3-nano-omni-30b-a3b-reasoning"
_OMNI_INVOKE_PATH = "/v1/chat/completions"


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


class HelmCaptionEndpointTests(TestCase):
    """Source-level + integration coverage of the caption auto-wiring fix."""

    # ------------------------------------------------------------------
    # Source / values
    # ------------------------------------------------------------------

    def test_values_expose_caption_endpoint_overrides(self) -> None:
        values = _read_required_file(_repo_root() / "nemo_retriever/helm/values.yaml")
        # Both knobs must be present so users can point at hosted endpoints
        # or override the model id.
        self.assertIn("captionInvokeUrl:", values)
        self.assertIn("captionModelName:", values)
        # And they must default to empty (so the chart only emits a value
        # when the operator NIM is enabled or the user opts in).
        self.assertIn('captionInvokeUrl: ""', values)
        self.assertIn('captionModelName: ""', values)

    def test_configmap_resolves_caption_url_via_standard_helper(self) -> None:
        body = _read_required_file(_repo_root() / "nemo_retriever/helm/templates/configmap.yaml")
        # The lookup must go through the shared resolver so the
        # explicit > operator > empty precedence stays consistent.
        self.assertIn(
            '"key" "nemotron_3_nano_omni_30b_a3b_reasoning"',
            body,
            "configmap.yaml must resolve the caption URL via " "nemo-retriever.nim.endpointURL keyed on the Omni NIM.",
        )
        self.assertIn(f'"serviceName" "{_OMNI_OPERATOR_SERVICE}"', body)
        self.assertIn(f'"invokePath" "{_OMNI_INVOKE_PATH}"', body)
        self.assertIn('"configKey" "captionInvokeUrl"', body)
        # Both fields must be rendered in the configBody — caption_invoke_url
        # is the trust gate that flips caption_enabled true.
        self.assertIn("caption_invoke_url:", body)
        self.assertIn("caption_model_name:", body)

    def test_readme_documents_caption_autowiring(self) -> None:
        readme = _read_required_file(_repo_root() / "nemo_retriever/helm/README.md")
        self.assertIn("image-captioning-omni-30b", readme)
        self.assertIn("caption_invoke_url", readme)
        self.assertIn(_OMNI_REMOTE_MODEL_ID, readme)

    # ------------------------------------------------------------------
    # Integration: actual `helm template` against the chart
    # ------------------------------------------------------------------

    def test_helm_template_autowires_caption_when_omni_enabled(self) -> None:
        """The exact customer-reported repro must now wire caption fields."""
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        expected_url = f'caption_invoke_url: "http://{_OMNI_OPERATOR_SERVICE}:8000{_OMNI_INVOKE_PATH}"'
        expected_model = f'caption_model_name: "{_OMNI_REMOTE_MODEL_ID}"'
        self.assertIn(
            expected_url,
            proc.stdout,
            "configmap must auto-wire the operator-managed Omni URL into " "nim_endpoints.caption_invoke_url.",
        )
        self.assertIn(
            expected_model,
            proc.stdout,
            "configmap must auto-set the canonical Omni remote model id.",
        )

    def test_helm_template_caption_null_when_omni_disabled(self) -> None:
        """Without the Omni NIM and no explicit URL, the caption fields must be null."""
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=false",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        self.assertIn("caption_invoke_url: null", proc.stdout)
        self.assertIn("caption_model_name: null", proc.stdout)

    def test_helm_template_explicit_caption_url_wins(self) -> None:
        """`captionInvokeUrl` override must beat operator wiring."""
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true",
                "--set",
                "serviceConfig.nimEndpoints.captionInvokeUrl=https://integrate.api.nvidia.com/v1/chat/completions",
                "--set",
                "serviceConfig.nimEndpoints.captionModelName=nvidia/some-other-vlm",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        self.assertIn(
            'caption_invoke_url: "https://integrate.api.nvidia.com/v1/chat/completions"',
            proc.stdout,
        )
        self.assertIn('caption_model_name: "nvidia/some-other-vlm"', proc.stdout)

    def test_helm_template_explicit_url_defaults_model_to_omni(self) -> None:
        """Explicit URL with no model override falls back to the canonical Omni id."""
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=false",
                "--set",
                "serviceConfig.nimEndpoints.captionInvokeUrl=https://integrate.api.nvidia.com/v1/chat/completions",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        self.assertIn(
            'caption_invoke_url: "https://integrate.api.nvidia.com/v1/chat/completions"',
            proc.stdout,
        )
        self.assertIn(f'caption_model_name: "{_OMNI_REMOTE_MODEL_ID}"', proc.stdout)

    def test_helm_template_omni_operator_url_renders_in_split_mode(self) -> None:
        """Split-topology renders the same nim_endpoints block per role."""
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true",
                "--set",
                "topology.mode=split",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        _assert_helm_ok(self, proc)
        # Each of gateway / realtime / batch ConfigMaps gets the wiring.
        # Don't be brittle about ordering: just count that the URL shows
        # up three times (one per role config).
        url_count = proc.stdout.count(f"http://{_OMNI_OPERATOR_SERVICE}:8000{_OMNI_INVOKE_PATH}")
        self.assertGreaterEqual(
            url_count,
            3,
            "expected the Omni caption URL to render in every per-role ConfigMap "
            f"(split mode), saw {url_count} occurrence(s).",
        )


if __name__ == "__main__":
    main()
