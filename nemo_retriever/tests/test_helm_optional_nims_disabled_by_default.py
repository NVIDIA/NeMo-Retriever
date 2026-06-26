# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the 26.05 "optional and disabled by default" contract.

The 26.05 docs at ``docs/extraction/deployment-options.md`` mark the
**VL reranker** (``llama-nemotron-rerank-vl-1b-v2``), **Nemotron Parse**,
and the **Nemotron 3 Nano Omni 30B** caption NIM as optional and not
auto-wired into the retriever-service.  Through 26.05 RC2 the Helm
chart did the opposite — all three NIMs were ``enabled: true`` in
``values.yaml`` — so a plain ``helm install`` (matching the documented
quick-start) silently pulled tens of GiB of model weights and claimed a
dedicated GPU per NIM with no opt-in.  The rerank block additionally
pointed at the **text-only** ``llama-nemotron-rerank-1b-v2`` SKU, which
silently degrades multimodal reranking — the docs cite the VL build.

These tests pin the chart-side fix:

* ``nimOperator.rerankqa.enabled`` defaults to ``false`` and the
  pinned image is the VL SKU (``llama-nemotron-rerank-vl-1b-v2``), not
  the text-only one.
* ``nimOperator.nemotron_parse.enabled`` defaults to ``false``.
* ``nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled``
  defaults to ``false``.
* A ``helm template`` with **no overrides** renders no ``NIMCache`` /
  ``NIMService`` for any of the three NIMs (and no caption
  auto-wiring).
* Explicit opt-in still reconciles them, so the documented
  ``--set nimOperator.<key>.enabled=true`` workflow keeps working.
* The README and ``values.yaml`` document the ``1.7.0-variant`` tag
  used by Parse + Omni so air-gapped mirror pipelines and
  reproducibility audits can map it to the 26.05 release.

The integration tests shell out to ``helm template`` when ``helm`` is
on ``$PATH``; otherwise they skip cleanly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest, TestCase, main


# Repo-relative paths exercised by every test in this module.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALUES_YAML = _REPO_ROOT / "nemo_retriever/helm/values.yaml"
_README_MD = _REPO_ROOT / "nemo_retriever/helm/README.md"
_CHART_DIR = _REPO_ROOT / "nemo_retriever/helm"

# Per-NIM block headers in values.yaml — each is followed (within a
# handful of lines) by exactly one ``enabled:`` field. We anchor on the
# block header rather than scanning the whole file so an unrelated
# ``enabled:`` (e.g. ``service.gpu.enabled``) cannot accidentally
# satisfy the assertion.
_RERANKQA_BLOCK = "  rerankqa:"
_PARSE_BLOCK = "  nemotron_parse:"
_OMNI_BLOCK = "  nemotron_3_nano_omni_30b_a3b_reasoning:"

# NIMService manifest names produced by ``templates/nims/*.yaml``. The
# ``\nname: <name>\n`` form pins the metadata.name slot specifically;
# the bare names would also appear in env vars, helper comments, etc.
_RERANK_VL_SERVICE_NAME = "name: llama-nemotron-rerank-vl-1b-v2"
_RERANK_TEXT_SERVICE_NAME = "name: llama-nemotron-rerank-1b-v2"
_PARSE_SERVICE_NAME = "name: nemotron-parse"
_OMNI_SERVICE_NAME = "name: nemotron-3-nano-omni-30b-a3b-reasoning"

# Image tag the chart pins for both NIMs in 26.05. Documenting it on
# both ends (values.yaml comments + README) keeps air-gapped mirror
# pipelines pointed at the right NGC tag.
_VARIANT_TAG = "1.7.0-variant"

# Repositories the rerank NIM may be pinned to. The chart MUST point at
# the VL SKU — the text-only SKU silently degrades multimodal
# reranking, which is the bug surfaced in the 26.05 report.
_RERANK_VL_REPOSITORY = "nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2"
_RERANK_TEXT_REPOSITORY = "nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2"


def _read_required_file(path: Path) -> str:
    if not path.is_file():
        raise SkipTest(f"Required file not present in this test environment: {path}")
    return path.read_text(encoding="utf-8")


def _enabled_value_for_block(values_text: str, block_header: str) -> str:
    """Return the literal ``enabled:`` value beneath ``block_header``.

    Looks at the first ``enabled:`` line that follows ``block_header``
    within a small window so we don't read ahead into the next NIM
    block. Returns the value verbatim (``"true"``, ``"false"``, …).
    """
    lines = values_text.splitlines()
    try:
        start = lines.index(block_header)
    except ValueError as exc:
        raise AssertionError(
            f"Could not find block header {block_header!r} in values.yaml; " "did the block name change?"
        ) from exc
    window = 12  # The block's first 12 lines are more than enough.
    for line in lines[start + 1 : start + 1 + window]:
        stripped = line.lstrip()
        if stripped.startswith("enabled:"):
            return stripped.split(":", 1)[1].strip()
    raise AssertionError(f"No `enabled:` field found within {window} lines after " f"{block_header!r} in values.yaml.")


def _helm_template(extra_args: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
    """Run ``helm template`` against the chart with NIM Operator CRDs available."""
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    if not _CHART_DIR.is_dir():
        raise SkipTest(f"Chart directory missing: {_CHART_DIR}")
    cmd = [
        helm,
        "template",
        "nrl-regression",
        str(_CHART_DIR),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
        # Pretend the NIM Operator CRDs are installed so the templates
        # would otherwise render the NIMService manifests — this is the
        # only way ``helm template`` produces operator resources and is
        # required to make the "no-overrides → no Parse/Omni" assertion
        # meaningful.
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


class OptionalNimsDefaultDisabledTests(TestCase):
    """26.05 contract: Parse and Omni are off until the user opts in."""

    # ------------------------------------------------------------------
    # values.yaml — source-level invariants
    # ------------------------------------------------------------------

    def test_values_parse_enabled_defaults_to_false(self) -> None:
        """``nimOperator.nemotron_parse.enabled`` must default to ``false``.

        Setting this to ``true`` reintroduces the customer-facing
        regression: the Parse pod auto-deploys on every default install,
        consuming an additional dedicated GPU and ~3.5 GiB of GPU memory
        for a NIM the docs explicitly mark optional and not auto-wired.
        """
        values = _read_required_file(_VALUES_YAML)
        value = _enabled_value_for_block(values, _PARSE_BLOCK)
        self.assertEqual(
            value,
            "false",
            "nimOperator.nemotron_parse.enabled must default to `false` "
            "per docs/extraction/deployment-options.md (Nemotron Parse "
            "is optional and not auto-wired). Set to `true` only when the "
            'pipeline runs `extract_method="nemotron_parse"`.',
        )

    def test_values_omni_enabled_defaults_to_false(self) -> None:
        """``nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled`` must default to ``false``.

        Omni 30B is the heaviest NIM in the chart (~62 GiB BF16 weights,
        ~80 GB on-disk NIM cache, requires its own ≥ 80 GiB GPU). It must
        not deploy on a "default" install — that contradicts the docs and
        the README's [Recommended minimal install (26.05)] guidance.
        """
        values = _read_required_file(_VALUES_YAML)
        value = _enabled_value_for_block(values, _OMNI_BLOCK)
        self.assertEqual(
            value,
            "false",
            "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled "
            "must default to `false` per docs/extraction/deployment-options.md "
            "(Omni 30B is optional and not auto-wired). Opt-in workflow "
            "is documented in helm/README.md `Image captioning (Omni 30B)`.",
        )

    def test_values_rerankqa_enabled_defaults_to_false(self) -> None:
        """``nimOperator.rerankqa.enabled`` must default to ``false``.

        Through 26.05 RC2 this defaulted to ``true``, so a plain
        ``helm install`` provisioned an extra ≈ 3.1 GiB GPU NIM with no
        opt-in. The docs explicitly mark the VL reranker as optional
        and disabled by default (``docs/extraction/deployment-options.md``
        L21).
        """
        values = _read_required_file(_VALUES_YAML)
        value = _enabled_value_for_block(values, _RERANKQA_BLOCK)
        self.assertEqual(
            value,
            "false",
            "nimOperator.rerankqa.enabled must default to `false` per "
            "docs/extraction/deployment-options.md (the VL reranker is "
            "optional and not auto-wired). Opt in with "
            "`--set nimOperator.rerankqa.enabled=true`.",
        )

    def test_values_rerankqa_image_is_vl_sku(self) -> None:
        """The pinned image must be the VL reranker, not the text-only SKU.

        ``docs/extraction/prerequisites-support-matrix.md`` L92 / L128
        documents ``llama-nemotron-rerank-vl-1b-v2`` as the supported
        reranker NIM for 26.05. Through RC2 the chart shipped the
        text-only ``llama-nemotron-rerank-1b-v2`` — that SKU silently
        degrades multimodal reranking and is not the documented POR.
        """
        values = _read_required_file(_VALUES_YAML)
        self.assertIn(
            f"repository: {_RERANK_VL_REPOSITORY}",
            values,
            "nimOperator.rerankqa.image.repository must pin the VL SKU "
            f"`{_RERANK_VL_REPOSITORY}` per "
            "docs/extraction/prerequisites-support-matrix.md.",
        )
        # And the text-only repository must not appear anywhere in
        # values.yaml — the bug surfaces when the chart silently
        # substitutes the text-only build.
        self.assertNotIn(
            f"repository: {_RERANK_TEXT_REPOSITORY}",
            values,
            "values.yaml must not pin the text-only rerank SKU "
            f"`{_RERANK_TEXT_REPOSITORY}` — that silently degrades "
            "multimodal reranking and contradicts the 26.05 docs. Use "
            "the VL build instead.",
        )

    def test_values_document_the_variant_tag(self) -> None:
        """The ``1.7.0-variant`` tag must be explained in ``values.yaml``.

        Customer-facing pain (3) from the bug report: ``1.7.0-variant``
        is unsearchable on NGC and has no docs entry, so air-gapped
        mirror pipelines and reproducibility audits cannot map the tag
        to a known release. An inline comment in ``values.yaml`` is the
        minimum bar — the README's "Image tag conventions" subsection
        covers it in more depth.
        """
        values = _read_required_file(_VALUES_YAML)
        self.assertEqual(
            values.count(f'tag: "{_VARIANT_TAG}"'),
            2,
            "Expected exactly two NIM image tags to be pinned to "
            f"{_VARIANT_TAG!r} (Parse + Omni). If you bumped one tag, "
            "bump the other together or split this test.",
        )
        # The comment must explain what `-variant` means, not just
        # mention the literal string.
        self.assertIn(
            "-variant",
            values,
            "values.yaml should reference the `-variant` tag family in "
            "comments so operators reading the file understand what the "
            "tag means without leaving the chart source.",
        )
        self.assertIn(
            "TensorRT engine",
            values,
            "values.yaml comments should explain the `-variant` suffix "
            "(per-GPU TensorRT engine variants selected by the NIM "
            "Operator). Air-gapped mirror pipelines depend on this.",
        )

    # ------------------------------------------------------------------
    # README — operator-facing documentation
    # ------------------------------------------------------------------

    def test_readme_per_nim_table_reflects_new_defaults(self) -> None:
        """The README's per-NIM defaults table must show ``false`` for rerankqa + Parse + Omni."""
        readme = _read_required_file(_README_MD)
        # The defaults table uses ``| <path> | `false` | ...`` formatting;
        # we look for the path *and* the same-line `false` cell to avoid
        # matching any stray reference elsewhere.
        for path in (
            "nimOperator.rerankqa.enabled",
            "nimOperator.nemotron_parse.enabled",
            "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled",
        ):
            self.assertRegex(
                readme,
                rf"`{path}`.*\|\s*`false`",
                f"README per-NIM defaults table must show `{path}` defaulting " "to `false` after the 26.05 fix.",
            )

    def test_readme_image_table_pins_vl_rerank_sku(self) -> None:
        """The README mirror-image table must list the VL reranker, not the text-only SKU.

        The image table doubles as the air-gapped mirror checklist —
        listing the text-only SKU there would point operators at the
        wrong NGC repository.
        """
        readme = _read_required_file(_README_MD)
        self.assertIn(
            f"{_RERANK_VL_REPOSITORY}:1.10.0",
            readme,
            "README mirror-image table must list the VL reranker " f"`{_RERANK_VL_REPOSITORY}:1.10.0`.",
        )
        self.assertNotIn(
            f"{_RERANK_TEXT_REPOSITORY}:1.10.0",
            readme,
            "README mirror-image table must not list the text-only "
            f"rerank SKU `{_RERANK_TEXT_REPOSITORY}:1.10.0` — that "
            "would silently degrade multimodal reranking for air-gapped "
            "mirror setups.",
        )

    def test_readme_documents_image_tag_conventions(self) -> None:
        """A dedicated subsection must explain the ``1.7.0-variant`` tag.

        Without this, the customer-facing complaint that ``1.7.0-variant``
        is undocumented stays valid even after the defaults flip.
        """
        readme = _read_required_file(_README_MD)
        self.assertIn(
            "image-tag-conventions",
            readme,
            "README must expose an `Image tag conventions` anchor so the "
            "values.yaml entries and per-NIM table can link to it.",
        )
        self.assertIn(
            _VARIANT_TAG,
            readme,
            f"README must mention the {_VARIANT_TAG!r} tag verbatim so a "
            "`grep` for the tag inside the chart docs returns the "
            "explanation.",
        )
        # The README explicitly tells operators NOT to substitute :latest
        # — this is the actionable guidance the bug report asks for.
        self.assertIn(
            ":latest",
            readme,
            "README should explicitly warn against substituting `:latest` "
            "for the pinned tag — air-gapped mirror pipelines need an "
            "exact reference, not a moving NGC alias.",
        )

    def test_readme_minimal_install_no_longer_disables_parse_or_omni(self) -> None:
        """The minimal-install recipe should not include redundant Parse/Omni flags.

        With the new defaults the flags are no-ops — keeping them in
        the example would mislead operators into thinking the chart
        still enables Parse and Omni by default.
        """
        readme = _read_required_file(_README_MD)
        # Find the heredoc-style minimal install command. The recipe
        # ends with `audio.enabled=false`; the block above that is what
        # we inspect.
        marker = "Recommended minimal install (26.05)"
        idx = readme.find(marker)
        self.assertNotEqual(
            idx,
            -1,
            "README must keep a `Recommended minimal install (26.05)` "
            "section even after the defaults flip — it documents the "
            "two flags that are still needed (`rerankqa` + `audio`).",
        )
        # Inspect a window after the marker so we only check the recipe,
        # not unrelated mentions elsewhere in the README.
        window = readme[idx : idx + 1500]
        self.assertNotIn(
            "nimOperator.nemotron_parse.enabled=false",
            window,
            "Minimal-install recipe must not set "
            "`nimOperator.nemotron_parse.enabled=false` — that's the "
            "default now and listing it implies the chart still enables "
            "Parse on a plain install.",
        )
        self.assertNotIn(
            "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=false",
            window,
            "Minimal-install recipe must not set the Omni `enabled=false` " "flag — that's the default now.",
        )
        self.assertNotIn(
            "nimOperator.rerankqa.enabled=false",
            window,
            "Minimal-install recipe must not set "
            "`nimOperator.rerankqa.enabled=false` — that's the default "
            "in 26.05 now and listing it implies the chart still "
            "provisions the VL reranker on a plain install.",
        )

    # ------------------------------------------------------------------
    # `helm template` — actually render the chart
    # ------------------------------------------------------------------

    def test_helm_template_default_render_omits_parse_and_omni(self) -> None:
        """Plain ``helm install`` (no overrides) must produce no Parse / Omni resources.

        This is the exact customer repro from the bug report:

            helm install nrl ./nemo_retriever/helm \\
              --set imagePullSecret.password=$NGC_API_KEY \\
              --set nims.ngcApiKey=$NGC_API_KEY

        After the fix the rendered manifest must contain no
        ``NIMCache`` / ``NIMService`` for Parse or Omni — anything else
        re-introduces the regression.
        """
        proc = _helm_template()
        _assert_helm_ok(self, proc)

        self.assertNotIn(
            _PARSE_SERVICE_NAME,
            proc.stdout,
            "Default helm template render must not contain a "
            "`name: nemotron-parse` resource — Parse is optional and "
            "disabled by default in 26.05.",
        )
        self.assertNotIn(
            _OMNI_SERVICE_NAME,
            proc.stdout,
            "Default helm template render must not contain a "
            "`name: nemotron-3-nano-omni-30b-a3b-reasoning` resource — "
            "Omni 30B is optional and disabled by default in 26.05.",
        )
        # Caption auto-wiring must stay off too, otherwise the service
        # would call a non-existent NIM Service.
        self.assertIn(
            "caption_invoke_url: null",
            proc.stdout,
            "With Omni disabled by default the configmap must render "
            "`caption_invoke_url: null` — anything else means the "
            "caption auto-wiring is silently active without an Omni "
            "Pod to back it.",
        )

    def test_helm_template_default_render_omits_rerankqa(self) -> None:
        """Plain ``helm install`` must produce no reranker resources.

        Replays the customer repro from the bug report — a default
        ``helm install`` must not provision either the VL rerank pod
        nor the text-only one.
        """
        proc = _helm_template()
        _assert_helm_ok(self, proc)
        for name in (_RERANK_VL_SERVICE_NAME, _RERANK_TEXT_SERVICE_NAME):
            self.assertNotIn(
                name,
                proc.stdout,
                "Default helm template render must not contain a "
                f"`{name}` resource — the VL reranker is optional and "
                "disabled by default in 26.05 (the text-only SKU must "
                "never appear at all).",
            )

    def test_helm_template_parse_opt_in_renders_nimservice(self) -> None:
        """Explicit ``--set nimOperator.nemotron_parse.enabled=true`` reconciles Parse."""
        proc = _helm_template(
            extra_args=("--set", "nimOperator.nemotron_parse.enabled=true"),
        )
        _assert_helm_ok(self, proc)
        self.assertIn(
            _PARSE_SERVICE_NAME,
            proc.stdout,
            "Opt-in `nimOperator.nemotron_parse.enabled=true` must render "
            "a `NIMService name: nemotron-parse` resource. If this fails "
            "the chart has broken the opt-in path while flipping the "
            "default.",
        )
        # The pinned tag must travel with the opt-in.
        self.assertIn(
            f"tag: {_VARIANT_TAG}",
            proc.stdout,
            f"Parse opt-in must render with the pinned {_VARIANT_TAG!r} tag.",
        )

    def test_helm_template_omni_opt_in_renders_nimservice_and_caption(self) -> None:
        """Explicit Omni opt-in reconciles the NIM **and** auto-wires the caption URL."""
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true",
            ),
        )
        _assert_helm_ok(self, proc)
        self.assertIn(
            _OMNI_SERVICE_NAME,
            proc.stdout,
            "Opt-in `nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning."
            "enabled=true` must render the matching NIMService.",
        )
        # The caption auto-wiring must come back on for the opt-in
        # path; this is the regression covered separately by
        # test_helm_caption_endpoint.py but worth re-asserting here so
        # a careless defaults flip doesn't also disable captioning.
        self.assertIn(
            'caption_invoke_url: "http://nemotron-3-nano-omni-30b-a3b-reasoning:8000/v1/chat/completions"',
            proc.stdout,
            "Omni opt-in must restore the caption URL auto-wiring. If "
            "this fails the defaults flip also broke the captioning "
            "feature wiring.",
        )

    def test_helm_template_rerankqa_opt_in_renders_vl_nimservice(self) -> None:
        """Explicit opt-in must render the VL NIMService (not the text-only one)."""
        proc = _helm_template(
            extra_args=("--set", "nimOperator.rerankqa.enabled=true"),
        )
        _assert_helm_ok(self, proc)
        self.assertIn(
            _RERANK_VL_SERVICE_NAME,
            proc.stdout,
            "Opt-in `nimOperator.rerankqa.enabled=true` must render a "
            "`NIMService name: llama-nemotron-rerank-vl-1b-v2` resource. "
            "If this fails the chart has either broken the opt-in path "
            "or silently substituted the text-only SKU.",
        )
        self.assertNotIn(
            _RERANK_TEXT_SERVICE_NAME,
            proc.stdout,
            "Opt-in must never render the text-only "
            "`name: llama-nemotron-rerank-1b-v2` resource — that SKU "
            "silently degrades multimodal reranking.",
        )
        self.assertIn(
            _RERANK_VL_REPOSITORY,
            proc.stdout,
            f"Rendered manifest must reference the VL repository " f"`{_RERANK_VL_REPOSITORY}`.",
        )
        self.assertNotIn(
            f"{_RERANK_TEXT_REPOSITORY}:",
            proc.stdout,
            "Rendered manifest must not reference the text-only rerank "
            "repository — that is the bug the 26.05 fix exists to "
            "prevent.",
        )

    def test_helm_template_omni_image_tag_pins_to_variant(self) -> None:
        """Opt-in Omni must render with the pinned ``1.7.0-variant`` tag, not ``:latest``.

        The bug report's reproducibility concern: substituting
        ``:latest`` would silently move to a different NIM build.
        """
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true",
            ),
        )
        _assert_helm_ok(self, proc)
        self.assertIn(
            f"tag: {_VARIANT_TAG}",
            proc.stdout,
            f"Omni opt-in must render with the pinned {_VARIANT_TAG!r} tag.",
        )
        # And there is no stray `:latest` reference in the rendered
        # NIMCache/NIMService manifests for either heavy-weight NIM.
        self.assertNotIn(
            "nemotron-3-nano-omni-30b-a3b-reasoning:latest",
            proc.stdout,
            "Omni image must never resolve to `:latest` — that's a "
            "moving NGC alias and breaks air-gapped mirror pipelines.",
        )


if __name__ == "__main__":
    main()
