# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the vectordb / embed-endpoint fail-fast guard.

When ``serviceConfig.vectordb.enabled=true`` and no NIM embedding
endpoint can be resolved (neither an explicit
``serviceConfig.nimEndpoints.embedInvokeUrl`` nor an operator-managed
``vlm_embed`` URL), the chart used to render a "healthy" vectordb
Deployment with ``--embed-endpoint ""``.  Its ``/v1/health`` probe
passed; the first ``/v1/query`` request then died with
``HTTP 501 No embedding endpoint configured.`` — an install-time
configuration error surfaced only after ingestion.

``templates/deployment-vectordb.yaml`` now uses ``{{ fail ... }}`` to
halt rendering in that exact state.  These tests pin the guard so it
cannot be silently removed:

* the template source still contains the ``fail`` guard and the
  resolution lookup it depends on;
* (integration) ``helm template`` actually exits non-zero on the
  customer-reported reproduction values and rejects each of the three
  documented escape valves correctly.

The integration test is skipped automatically when ``helm`` is not on
the ``$PATH``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest, TestCase, main


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


class HelmVectorDBEmbedRequiredTests(TestCase):
    """Source-level + integration coverage of the vectordb fail-fast guard."""

    # ------------------------------------------------------------------
    # Source guard
    # ------------------------------------------------------------------

    def test_template_contains_fail_guard_for_unresolved_embed(self) -> None:
        body = _read_required_file(_repo_root() / "nemo_retriever/helm/templates/deployment-vectordb.yaml")

        # The guard must look at the resolved embed URL and call `fail`
        # with a message that tells the user every supported override.
        self.assertIn(
            "{{- if and (not $embedURL) (not $localEmbed) }}",
            body,
            "deployment-vectordb.yaml must guard on resolved embed URL and local embed.",
        )
        self.assertIn("{{- fail ", body, "deployment-vectordb.yaml must call `fail`.")
        for needle in (
            "serviceConfig.nimEndpoints.embedInvokeUrl",
            "nimOperator.vlm_embed.enabled=true",
            "serviceConfig.localModels.enabled=true",
            "serviceConfig.vectordb.enabled=false",
        ):
            self.assertIn(
                needle,
                body,
                f"fail-fast message must reference the `{needle}` escape valve.",
            )

    def test_readme_documents_vectordb_embed_requirement(self) -> None:
        readme = _read_required_file(_repo_root() / "nemo_retriever/helm/README.md")
        self.assertIn("vectordb-and-the-embed-endpoint", readme)
        self.assertIn("HTTP 501", readme)
        self.assertIn("--set serviceConfig.vectordb.enabled=false", readme)

    # ------------------------------------------------------------------
    # Integration: actual `helm template` against the chart
    # ------------------------------------------------------------------

    def test_helm_template_fails_when_vectordb_enabled_without_embed(self) -> None:
        """The exact customer-reported reproduction must now fail at template time."""
        proc = _helm_template(
            extra_args=(
                "--set",
                "serviceConfig.vectordb.enabled=true",
                "--set",
                "nimOperator.vlm_embed.enabled=false",
            ),
        )
        self.assertNotEqual(
            proc.returncode,
            0,
            "`helm template` must refuse to render vectordb with no embed "
            f"endpoint resolved. STDOUT:\n{proc.stdout}",
        )
        # The error surface must reach the user (Helm sends `fail` to stderr).
        combined = proc.stdout + proc.stderr
        self.assertIn("no query embedding backend could be resolved", combined)
        self.assertIn("serviceConfig.vectordb.enabled=false", combined)
        self.assertIn(
            "serviceConfig.nimEndpoints.embedInvokeUrl",
            combined,
            "the error must point users at the explicit-URL escape valve.",
        )

    def test_helm_template_passes_with_explicit_embed_url(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "serviceConfig.vectordb.enabled=true",
                "--set",
                "nimOperator.vlm_embed.enabled=false",
                "--set",
                "serviceConfig.nimEndpoints.embedInvokeUrl=http://embed.svc:8000/v1/embeddings",
            ),
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"`helm template` should succeed with an explicit embed URL.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        # vectordb container args must carry the explicit URL.
        self.assertIn(
            '--embed-endpoint\n            - "http://embed.svc:8000/v1/embeddings"',
            proc.stdout,
        )

    def test_helm_template_passes_with_in_cluster_embed_nim(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "serviceConfig.vectordb.enabled=true",
                "--set",
                "nimOperator.vlm_embed.enabled=true",
            ),
            api_versions=("apps.nvidia.com/v1alpha1",),
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"`helm template` should succeed when vlm_embed is operator-managed.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        # vectordb container args must carry the operator-resolved URL.
        self.assertIn("--embed-endpoint", proc.stdout)
        self.assertIn("/v1/embeddings", proc.stdout)

    def test_helm_template_passes_with_local_embed_enabled(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "serviceConfig.vectordb.enabled=true",
                "--set",
                "nimOperator.vlm_embed.enabled=false",
                "--set",
                "serviceConfig.localModels.enabled=true",
                "--set",
                "serviceConfig.localModels.embed.enabled=true",
            ),
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"`helm template` should succeed with in-pod local query embedding.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        self.assertIn("--local-embed", proc.stdout)
        self.assertIn("--local-embed-backend", proc.stdout)
        self.assertNotIn('--embed-endpoint\n            - ""', proc.stdout)

    def test_helm_template_passes_with_vectordb_disabled(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "serviceConfig.vectordb.enabled=false",
                "--set",
                "nimOperator.vlm_embed.enabled=false",
            ),
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"`helm template` should succeed when vectordb is disabled.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        self.assertNotIn(
            "kind: Deployment\nmetadata:\n  name: retriever-nemo-retriever-vectordb",
            proc.stdout,
            "vectordb Deployment must NOT render when its switch is off.",
        )


if __name__ == "__main__":
    main()
