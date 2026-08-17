# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for global NGC Secret name propagation into NIM CRs.

``ngcImagePullSecret.name`` and ``ngcApiSecret.name`` are documented as the
chart-wide Secret names for every Pod and every NIMCache / NIMService.
Per-NIM ``image.pullSecrets`` / ``authSecret`` default to empty so those
global names inherit; non-empty per-NIM values replace the chart-wide names
for that NIM only.

These tests pin:

* Default render uses ``ngc-secret`` / ``ngc-api`` on every NIMCache and
  NIMService.
* Overriding only the global names rewrites every rendered NIM CR.
* A non-empty per-NIM override wins for that NIM alone.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest, TestCase, main

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHART_DIR = _REPO_ROOT / "nemo_retriever/helm"
_VALUES_YAML = _REPO_ROOT / "nemo_retriever/helm/values.yaml"
_README_MD = _REPO_ROOT / "nemo_retriever/helm/README.md"

_NIM_KEYS: tuple[str, ...] = (
    "page_elements",
    "table_structure",
    "ocr",
    "vlm_embed",
    "rerankqa",
    "nemotron_parse",
    "answer_llm",
    "nemotron_3_nano_omni_30b_a3b_reasoning",
    "audio",
)


def _read_required_file(path: Path) -> str:
    if not path.is_file():
        raise SkipTest(f"Required file not present in this test environment: {path}")
    return path.read_text(encoding="utf-8")


def _helm_template(extra_args: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    if not _CHART_DIR.is_dir():
        raise SkipTest(f"Chart directory missing: {_CHART_DIR}")
    cmd = [
        helm,
        "template",
        "nrl-nim-secrets",
        str(_CHART_DIR),
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
        "nimOperator.answer_llm.enabled=true",
        "--set",
        "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true",
        "--api-versions",
        "apps.nvidia.com/v1alpha1",
    ]
    cmd += list(extra_args)
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _assert_helm_ok(self: TestCase, proc: subprocess.CompletedProcess[str]) -> None:
    self.assertEqual(
        proc.returncode,
        0,
        f"`helm template` failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
    )


def _iter_nim_docs(rendered: str, kind: str) -> list[dict]:
    docs: list[dict] = []
    for raw in yaml.safe_load_all(rendered):
        if isinstance(raw, dict) and raw.get("kind") == kind:
            docs.append(raw)
    return docs


def _assert_nim_secret_names(
    self: TestCase,
    rendered: str,
    *,
    pull_secret: str,
    auth_secret: str,
    per_nim_pull: dict[str, str] | None = None,
    per_nim_auth: dict[str, str] | None = None,
) -> None:
    per_nim_pull = per_nim_pull or {}
    per_nim_auth = per_nim_auth or {}

    caches = _iter_nim_docs(rendered, "NIMCache")
    services = _iter_nim_docs(rendered, "NIMService")
    self.assertGreaterEqual(len(caches), 1, "expected at least one NIMCache")
    self.assertGreaterEqual(len(services), 1, "expected at least one NIMService")
    self.assertEqual(len(caches), len(services))

    for cache in caches:
        name = cache["metadata"]["name"]
        ngc = cache["spec"]["source"]["ngc"]
        expected_pull = per_nim_pull.get(name, pull_secret)
        expected_auth = per_nim_auth.get(name, auth_secret)
        self.assertEqual(
            ngc.get("pullSecret"),
            expected_pull,
            f"NIMCache `{name}` pullSecret mismatch",
        )
        self.assertEqual(
            ngc.get("authSecret"),
            expected_auth,
            f"NIMCache `{name}` authSecret mismatch",
        )

    for service in services:
        name = service["metadata"]["name"]
        expected_pull = per_nim_pull.get(name, pull_secret)
        expected_auth = per_nim_auth.get(name, auth_secret)
        self.assertEqual(
            service["spec"]["image"].get("pullSecrets"),
            [expected_pull],
            f"NIMService `{name}` pullSecrets mismatch",
        )
        self.assertEqual(
            service["spec"].get("authSecret"),
            expected_auth,
            f"NIMService `{name}` authSecret mismatch",
        )


class TestHelmNimGlobalSecretNames(TestCase):
    def test_values_default_per_nim_secrets_are_empty(self) -> None:
        values = yaml.safe_load(_read_required_file(_VALUES_YAML))
        nim_operator = values["nimOperator"]
        for key in _NIM_KEYS:
            cfg = nim_operator[key]
            self.assertEqual(
                cfg["image"].get("pullSecrets") or [],
                [],
                f"nimOperator.{key}.image.pullSecrets must default to []",
            )
            self.assertEqual(
                cfg.get("authSecret") or "",
                "",
                f"nimOperator.{key}.authSecret must default to empty",
            )

    def test_readme_documents_global_inheritance(self) -> None:
        readme = _read_required_file(_README_MD)
        self.assertIn("Empty inherits `ngcImagePullSecret.name`", readme)
        self.assertIn("Empty inherits `ngcApiSecret.name`", readme)

    def test_default_render_uses_chart_wide_secret_names(self) -> None:
        proc = _helm_template()
        _assert_helm_ok(self, proc)
        _assert_nim_secret_names(
            self,
            proc.stdout,
            pull_secret="ngc-secret",
            auth_secret="ngc-api",
        )

    def test_global_secret_name_overrides_propagate_to_all_nim_crs(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "ngcImagePullSecret.name=rc4-global-pull",
                "--set",
                "ngcApiSecret.name=rc4-global-api",
            )
        )
        _assert_helm_ok(self, proc)
        _assert_nim_secret_names(
            self,
            proc.stdout,
            pull_secret="rc4-global-pull",
            auth_secret="rc4-global-api",
        )
        self.assertNotIn("ngc-secret", proc.stdout)
        # Default auth secret name must not remain on NIM CRs.
        for doc in _iter_nim_docs(proc.stdout, "NIMCache") + _iter_nim_docs(proc.stdout, "NIMService"):
            text = yaml.safe_dump(doc)
            self.assertNotIn("ngc-api", text)
            self.assertNotIn("ngc-secret", text)

    def test_per_nim_override_replaces_global_for_that_nim_only(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "ngcImagePullSecret.name=rc4-global-pull",
                "--set",
                "ngcApiSecret.name=rc4-global-api",
                "--set-json",
                'nimOperator.ocr.image.pullSecrets=["ocr-only-pull"]',
                "--set",
                "nimOperator.ocr.authSecret=ocr-only-api",
            )
        )
        _assert_helm_ok(self, proc)
        ocr_name = "nemotron-ocr-v2"
        _assert_nim_secret_names(
            self,
            proc.stdout,
            pull_secret="rc4-global-pull",
            auth_secret="rc4-global-api",
            per_nim_pull={ocr_name: "ocr-only-pull"},
            per_nim_auth={ocr_name: "ocr-only-api"},
        )


if __name__ == "__main__":
    main()
