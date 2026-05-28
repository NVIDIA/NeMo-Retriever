# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for NIMService GPU resource rendering.

The NIM Operator does not reliably populate ``spec.resources.limits.nvidia.com/gpu``
from the model profile on all tested versions (for example v3.1.1 on A100/H100).
The chart therefore defaults to rendering ``nvidia.com/gpu: 1`` via
``nimOperator.nimServiceGpuLimit``.

Helm and the operator may both server-side-apply that field; see README
§GPU limits and ``helm upgrade`` for ``--force-conflicts`` guidance.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from unittest import SkipTest, TestCase, main


_NIMSERVICE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("audio.yaml", "audio"),
    ("llama-nemotron-embed-vl-1b-v2.yaml", "vlm_embed"),
    ("llama-nemotron-rerank-vl-1b-v2.yaml", "rerankqa"),
    ("nemotron-3-nano-omni-30b-a3b-reasoning.yaml", "nemotron_3_nano_omni_30b_a3b_reasoning"),
    ("nemotron-ocr-v1.yaml", "ocr"),
    ("nemotron-page-elements-v3.yaml", "page_elements"),
    ("nemotron-parse.yaml", "nemotron_parse"),
    ("nemotron-table-structure-v1.yaml", "table_structure"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_required_file(path: Path) -> str:
    if not path.is_file():
        raise SkipTest(f"Required file not present in this test environment: {path}")
    return path.read_text(encoding="utf-8")


def _helm_template(extra_sets: list[str] | None = None) -> str:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    chart_path = _repo_root() / "nemo_retriever/helm"
    if not chart_path.is_dir():
        raise SkipTest(f"Chart directory missing: {chart_path}")

    cmd = [
        helm,
        "template",
        "nrl-regression",
        str(chart_path),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
        "--set",
        "nimOperator.rerankqa.enabled=true",
        "--set",
        "nimOperator.audio.enabled=true",
        "--set",
        "nimOperator.nemotron_parse.enabled=true",
        "--set",
        "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true",
        "--api-versions",
        "apps.nvidia.com/v1alpha1",
    ]
    if extra_sets:
        for flag in extra_sets:
            cmd.extend(["--set", flag])

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"`helm template` failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout


def _nimservice_resources_blocks(rendered: str) -> list[str]:
    """Return the ``resources:`` subtree for each NIMService document."""
    blocks: list[str] = []
    for chunk in rendered.split("\n---\n"):
        if "\nkind: NIMService\n" not in chunk:
            continue
        match = re.search(r"\n  resources:\n(.*?)(?=\n  [a-zA-Z])", chunk, re.DOTALL)
        if match:
            blocks.append(match.group(1))
    return blocks


class HelmNimServiceResourcesTests(TestCase):
    def test_values_default_nim_service_gpu_limit(self) -> None:
        values = _read_required_file(_repo_root() / "nemo_retriever/helm/values.yaml")
        self.assertIn("nimServiceGpuLimit: 1", values)

    def test_each_nimservice_template_uses_resources_helper(self) -> None:
        templates_dir = _repo_root() / "nemo_retriever/helm/templates/nims"
        for filename, values_key in _NIMSERVICE_TEMPLATES:
            with self.subTest(template=filename):
                body = _read_required_file(templates_dir / filename)
                self.assertIn(
                    'include "nemo-retriever.nimServiceResources"',
                    body,
                    f"{filename} must render NIMService resources via the shared helper.",
                )
                self.assertNotIn(
                    f"{{{{- with .Values.nimOperator.{values_key}.resources }}}}",
                    body,
                    f"{filename} must not use the old `with resources` guard alone.",
                )

    def test_helpers_document_gpu_limit_behavior(self) -> None:
        helpers = _read_required_file(_repo_root() / "nemo_retriever/helm/templates/_helpers.tpl")
        self.assertIn("nimServiceGpuLimit", helpers)
        self.assertIn('define "nemo-retriever.nimServiceResources"', helpers)

    def test_readme_documents_gpu_limit_upgrade_caveat(self) -> None:
        readme = _read_required_file(_repo_root() / "nemo_retriever/helm/README.md")
        self.assertIn("gpu-limits-and-helm-upgrade", readme)
        self.assertIn("nimServiceGpuLimit", readme)
        self.assertIn("force-conflicts", readme)

    def test_helm_template_default_render_sets_gpu_limit_on_every_nimservice(self) -> None:
        rendered = _helm_template()
        blocks = _nimservice_resources_blocks(rendered)
        self.assertEqual(
            len(blocks),
            len(_NIMSERVICE_TEMPLATES),
            f"Expected {len(_NIMSERVICE_TEMPLATES)} NIMService resources blocks.",
        )
        for block in blocks:
            self.assertIn("nvidia.com/gpu: 1", block)

    def test_helm_template_operator_only_mode_omits_gpu_limit(self) -> None:
        rendered = _helm_template(["nimOperator.nimServiceGpuLimit=null"])
        self.assertNotIn("nvidia.com/gpu", rendered)

    def test_per_nim_resources_override_replaces_default(self) -> None:
        rendered = _helm_template(["nimOperator.page_elements.resources.limits.nvidia\\.com/gpu=2"])
        match = re.search(
            r"name: nemotron-page-elements-v3\nspec:.*?resources:\n(.*?)(?=\n  [a-z])",
            rendered,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        self.assertIn("nvidia.com/gpu: 2", match.group(1))


if __name__ == "__main__":
    main()
