# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the NIMService ``resources`` field-ownership fix.

The NIM Operator reconciles ``NIMService.spec.resources.limits.nvidia.com/gpu``
from the model profile.  If the Helm chart also writes that field, both
Helm and the operator become server-side-apply owners of it, and a
subsequent ``helm upgrade --install`` (even a no-op one) fails with:

    Error: UPGRADE FAILED: conflict occurred while applying object
      <ns>/<nim> apps.nvidia.com/v1alpha1, Kind=NIMService:
      Apply failed with 1 conflict:
      conflict with "manager" using apps.nvidia.com/v1alpha1:
        .spec.resources.limits.nvidia.com/gpu

To stay idempotent the chart must:

* default ``nimOperator.<key>.resources`` to ``{}`` in ``values.yaml``,
  and
* wrap the NIMService ``resources:`` block in ``{{- with ... }}`` on
  every ``templates/nims/*.yaml`` so the field is **not rendered** when
  the user has not overridden it.

These two invariants are pinned below.  An optional end-to-end check
shells out to ``helm template`` when the binary is available and asserts
that no ``nvidia.com/gpu`` key appears anywhere in the default render.
"""

from __future__ import annotations

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


class HelmNimServiceResourcesTests(TestCase):
    """Field-ownership invariants for ``NIMService.spec.resources``."""

    def test_values_default_resources_to_empty_for_every_nim(self) -> None:
        """Defaults must be ``{}`` — anything else means Helm claims SSA ownership."""
        values = _read_required_file(_repo_root() / "nemo_retriever/helm/values.yaml")

        self.assertNotIn(
            "nvidia.com/gpu: 1",
            values,
            "values.yaml must not default any nimOperator.<key>.resources.limits "
            "to a GPU count — the NIM Operator reconciles that field. See "
            "templates/_helpers.tpl §NIM Operator field ownership notes.",
        )
        # Every per-NIM block should end the resources entry with `{}`.
        self.assertEqual(
            values.count("    resources: {}"),
            len(_NIMSERVICE_TEMPLATES),
            "Every nimOperator.<key>.resources block must default to `{}`.",
        )

    def test_each_nimservice_template_renders_resources_conditionally(self) -> None:
        """The NIMService ``resources:`` block must be wrapped in ``{{ with }}``."""
        templates_dir = _repo_root() / "nemo_retriever/helm/templates/nims"

        for filename, values_key in _NIMSERVICE_TEMPLATES:
            with self.subTest(template=filename):
                body = _read_required_file(templates_dir / filename)

                expected_guard = f"{{{{- with .Values.nimOperator.{values_key}.resources }}}}"
                self.assertIn(
                    expected_guard,
                    body,
                    f"{filename} must guard the NIMService resources block with "
                    f"`{{{{- with .Values.nimOperator.{values_key}.resources }}}}` "
                    "so an empty default does not render `resources: {}` (which "
                    "still grants Helm SSA ownership of "
                    "`spec.resources.limits.nvidia.com/gpu` and conflicts with the "
                    "NIM Operator on every `helm upgrade --install`).",
                )

                # The unconditional `toYaml ... .resources | indent 4` form is
                # exactly what the bug used; make sure it does not creep back.
                self.assertNotIn(
                    f"  resources:\n{{{{ toYaml .Values.nimOperator.{values_key}.resources | indent 4 }}}}",
                    body,
                    f"{filename} still renders the NIMService resources block "
                    "unconditionally — that was the field-ownership bug.",
                )

    def test_helpers_document_the_field_ownership_rationale(self) -> None:
        helpers = _read_required_file(_repo_root() / "nemo_retriever/helm/templates/_helpers.tpl")
        self.assertIn("NIM Operator field ownership notes", helpers)
        self.assertIn(".spec.resources.limits.nvidia.com/gpu", helpers)

    def test_readme_documents_gpu_limit_upgrade_caveat(self) -> None:
        readme = _read_required_file(_repo_root() / "nemo_retriever/helm/README.md")
        self.assertIn("gpu-limits-and-helm-upgrade", readme)
        self.assertIn("force-conflicts", readme)

    # ------------------------------------------------------------------
    # Optional integration check — only runs when `helm` is available.
    # ------------------------------------------------------------------

    def test_helm_template_default_render_has_no_nvidia_gpu_limit(self) -> None:
        """No `nvidia.com/gpu` field on any rendered NIMService, even when all 8 are enabled.

        The SSA-conflict bug is field-level, not NIM-level — every
        ``templates/nims/*.yaml`` that renders must keep the operator as
        the single owner of ``spec.resources.limits.nvidia.com/gpu``.
        We therefore opt in to the NIMs that are now disabled by
        default (``rerankqa``, ``audio``, ``nemotron_parse``, and
        ``nemotron_3_nano_omni_30b_a3b_reasoning``; see
        :mod:`test_helm_optional_nims_disabled_by_default` for the
        regression that pins the new defaults) so the check still
        exercises **every** NIMService template.
        """
        helm = shutil.which("helm")
        if helm is None:
            raise SkipTest("`helm` binary not available in this environment.")
        chart_path = _repo_root() / "nemo_retriever/helm"
        if not chart_path.is_dir():
            raise SkipTest(f"Chart directory missing: {chart_path}")

        proc = subprocess.run(
            [
                helm,
                "template",
                "nrl-regression",
                str(chart_path),
                "--set",
                "ngcImagePullSecret.create=false",
                "--set",
                "ngcApiSecret.create=false",
                # Opt every optional NIM in so this test still asserts
                # the SSA-conflict invariant across all 8 NIMService
                # templates. The actual defaults (rerankqa + audio +
                # Parse + Omni off) are covered separately to keep
                # concerns separated.
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
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"`helm template` failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )

        rendered = proc.stdout
        self.assertNotIn(
            "nvidia.com/gpu",
            rendered,
            "Default `helm template` render must not contain `nvidia.com/gpu` — "
            "the NIM Operator owns that field. Found it in the rendered "
            "manifest, which reintroduces the no-op `helm upgrade --install` "
            "SSA conflict.",
        )

        nimservice_count = rendered.count("\nkind: NIMService\n")
        self.assertEqual(
            nimservice_count,
            len(_NIMSERVICE_TEMPLATES),
            f"Expected {len(_NIMSERVICE_TEMPLATES)} NIMService objects in the "
            f"default + opt-in render, got {nimservice_count}.",
        )


if __name__ == "__main__":
    main()
