# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests pinning the ``image_caption`` stage's concurrency.

Customer-reported head-of-line block (26.05):

    Client A ingests a small text-only PDF (``--task=extract:{...}``,
    no caption work).  On its own A finishes in ~0.45 s.

    Client B concurrently ingests a larger PDF with the full caption
    pipeline enabled (``--task=extract:{...} --task=caption:{}``).
    B takes ~110 s, dominated by caption work.

    When B is already in flight, A's wall-clock stretches from
    ~0.45 s to ~59 s, and the CLI trace shows the entirety of the
    extra time accruing in ``image_caption_channel_in`` — i.e. A is
    sitting in the ``image_caption`` stage's input queue waiting for
    the *single* stage replica to finish B's images, even though A
    has zero caption work to do.

Root cause:

    Both ``config/default_pipeline.yaml`` and
    ``config/custom_summarization_pipeline.yaml`` declared

        replicas:
          static_replicas: { value: 1 }
          max_replicas:    { value: 1 }

    on the ``image_caption`` stage.  Every item flows through every
    stage in the Ray-actor pipeline (stages no-op items whose task
    list does not include their task, but they still have to dequeue
    them serially), so a single replica serializes A behind B.

Fix:

    Both YAMLs now declare multiple replicas — mirroring the
    ``text_embedder`` stage, which is architecturally similar (remote
    HTTP-only call to a NIM endpoint).  This test pins minimum
    invariants on the per-stage replica configuration so a future
    edit can not silently revert to the single-replica configuration
    that caused the customer-visible regression.

The actual end-to-end "A finishes in << B" benchmark requires a
running nv-ingest + VLM service and is out of scope for unit tests;
the configuration invariants checked here are the static surface
that the bug fix relies on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from nemo_retriever.common.modality.caption.model_profiles import DEFAULT_REMOTE_CAPTION_MODEL_ID


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


# Stage name we are pinning.  Lives in two YAMLs today; both must keep
# the multi-replica configuration.
_STAGE_NAME = "image_caption"

# Pipeline YAMLs that ship caption configuration. Each name maps to its
# absolute file path; new pipelines that add caption support should be
# added here so the same invariants apply automatically.
_PIPELINE_YAMLS = {
    "default_pipeline.yaml": CONFIG_DIR / "default_pipeline.yaml",
    "custom_summarization_pipeline.yaml": CONFIG_DIR / "custom_summarization_pipeline.yaml",
}

# Minimum static_replicas the test enforces.  Picked to be strictly
# greater than 1 (which was the regressed value) but small enough to
# allow operators to tune downward for tight resource budgets without
# bringing the count all the way back to 1.  The "shipped" defaults
# today are higher (4); the test sets the floor, not the ceiling.
_MIN_STATIC_REPLICAS = 2

# Minimum max_replicas. The bug shipped with max_replicas=1 (no
# scale-out headroom even under sustained burst). 4 is a conservative
# floor that gives the autoscaler real room while leaving operators
# free to cap at less than the shipped 8 if their VLM is tightly
# provisioned.
_MIN_MAX_REPLICAS = 4

_CANONICAL_REMOTE_CAPTION_MODEL = DEFAULT_REMOTE_CAPTION_MODEL_ID
_CAPTION_MODEL_ENV_EXPRESSION = f'$VLM_CAPTION_MODEL_NAME|"{_CANONICAL_REMOTE_CAPTION_MODEL}"'


# ---------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------


def _load_pipeline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        pytest.skip(f"Pipeline YAML not present in this test environment: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_stage(pipeline: dict[str, Any], stage_name: str) -> dict[str, Any]:
    """Return the stage dict with ``name == stage_name``.

    Raises a clear assertion failure if missing — these YAMLs are
    expected to contain the ``image_caption`` stage; if it's been
    renamed or removed the test has a real problem to surface, not
    a silent skip.
    """
    stages = pipeline.get("stages", [])
    for stage in stages:
        if stage.get("name") == stage_name:
            return stage
    raise AssertionError(
        f"Stage {stage_name!r} not found in pipeline YAML "
        f"(found stages: {[s.get('name') for s in stages]!r}). "
        "If the stage was renamed update _STAGE_NAME and the "
        "comment block at the top of this file."
    )


def _replica_value(stage: dict[str, Any], slot: str) -> int:
    """Extract the integer ``value`` from a ``static`` replica strategy.

    The pipeline YAML schema nests counts under
    ``replicas.<slot>.value`` with ``strategy: "static"``.  Non-static
    strategies (e.g. ``memory_thresholding`` on the PDF extractor)
    are valid but they do not apply to ``image_caption`` — caption is
    a remote HTTP-only stage with no local memory pressure, so a
    static replica policy is the right knob.  The test deliberately
    fails loudly on a non-static strategy here so a future schema
    change has to update the assertion explicitly.
    """
    replicas = stage.get("replicas") or {}
    block = replicas.get(slot)
    assert block is not None, (
        f"image_caption.replicas.{slot} is missing; the stage must "
        "declare both `static_replicas` and `max_replicas` so the "
        "scheduler knows the initial pool size and the burst cap."
    )
    strategy = block.get("strategy")
    assert strategy == "static", (
        f"image_caption.replicas.{slot}.strategy = {strategy!r}; "
        "this regression test was written against the `static` "
        "strategy. If you switch to dynamic scaling for this stage, "
        "update the assertions here to validate the new strategy "
        "preserves the HOL-blocking fix (multiple concurrent items)."
    )
    value = block.get("value")
    assert isinstance(value, int), f"image_caption.replicas.{slot}.value must be an int, " f"got {value!r}."
    return value


# ---------------------------------------------------------------------
# Per-pipeline assertions
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_name, yaml_path",
    sorted(_PIPELINE_YAMLS.items()),
    ids=sorted(_PIPELINE_YAMLS.keys()),
)
def test_image_caption_uses_canonical_remote_model_with_env_override(yaml_name: str, yaml_path: Path) -> None:
    pipeline = _load_pipeline(yaml_path)
    stage = _find_stage(pipeline, _STAGE_NAME)

    assert stage["config"]["model_name"] == _CAPTION_MODEL_ENV_EXPRESSION, (
        f"{yaml_name}: keep VLM_CAPTION_MODEL_NAME as the highest-precedence override "
        f"and use {_CANONICAL_REMOTE_CAPTION_MODEL!r} as the shipped fallback."
    )


@pytest.mark.parametrize(
    "yaml_name, yaml_path",
    sorted(_PIPELINE_YAMLS.items()),
    ids=sorted(_PIPELINE_YAMLS.keys()),
)
def test_image_caption_static_replicas_above_one(yaml_name: str, yaml_path: Path) -> None:
    """``static_replicas.value`` must be > 1 — the regressed value was 1.

    With a single replica the ``image_caption`` stage drains items
    strictly serially.  A no-caption client's item sits in
    ``image_caption_channel_in`` until the lone replica finishes
    whatever caption-heavy item is in front of it, multiplying the
    no-caption client's wall-clock by ``queue_depth × per_image_VLM_latency``.
    """
    pipeline = _load_pipeline(yaml_path)
    stage = _find_stage(pipeline, _STAGE_NAME)
    value = _replica_value(stage, "static_replicas")
    assert value >= _MIN_STATIC_REPLICAS, (
        f"{yaml_name}: image_caption.static_replicas.value must be "
        f">= {_MIN_STATIC_REPLICAS}, got {value}. A single replica "
        "head-of-line-blocks unrelated no-caption clients behind the "
        "busy client's queued images (customer regression: 0.45s "
        "→ 59s for a text-only doc while a caption-heavy doc is in "
        "flight)."
    )


@pytest.mark.parametrize(
    "yaml_name, yaml_path",
    sorted(_PIPELINE_YAMLS.items()),
    ids=sorted(_PIPELINE_YAMLS.keys()),
)
def test_image_caption_max_replicas_allows_scale_out(yaml_name: str, yaml_path: Path) -> None:
    """``max_replicas.value`` must allow scale-out beyond the static floor.

    Even if a future tuning lowers ``static_replicas`` toward 2,
    ``max_replicas`` must still allow the autoscaler to grow the
    stage during sustained caption bursts — otherwise the same HOL
    block reappears once the static pool saturates.
    """
    pipeline = _load_pipeline(yaml_path)
    stage = _find_stage(pipeline, _STAGE_NAME)
    value = _replica_value(stage, "max_replicas")
    assert value >= _MIN_MAX_REPLICAS, (
        f"{yaml_name}: image_caption.max_replicas.value must be "
        f">= {_MIN_MAX_REPLICAS}, got {value}. The stage needs real "
        "burst headroom — capping at 1 reintroduces the regression "
        "the moment the static pool saturates."
    )


@pytest.mark.parametrize(
    "yaml_name, yaml_path",
    sorted(_PIPELINE_YAMLS.items()),
    ids=sorted(_PIPELINE_YAMLS.keys()),
)
def test_image_caption_max_replicas_not_below_static(yaml_name: str, yaml_path: Path) -> None:
    """``max_replicas`` must be >= ``static_replicas``.

    The scheduler is undefined if the maximum is smaller than the
    static floor — both regressed-to-1 and a future typo (e.g. a
    pair like ``static_replicas=4, max_replicas=2``) should be
    flagged here so the misconfiguration is caught at lint time
    instead of at runtime when the stage refuses to start.
    """
    pipeline = _load_pipeline(yaml_path)
    stage = _find_stage(pipeline, _STAGE_NAME)
    static_v = _replica_value(stage, "static_replicas")
    max_v = _replica_value(stage, "max_replicas")
    assert max_v >= static_v, (
        f"{yaml_name}: image_caption.max_replicas ({max_v}) must be "
        f">= static_replicas ({static_v}); the static pool can not "
        "exceed the autoscaler cap."
    )


def test_image_caption_replica_counts_match_across_pipelines() -> None:
    """The default and custom_summarization pipelines must agree on the HOL fix.

    Both YAMLs share the same Ray-actor architecture and the same
    upstream/downstream stage shape; if one ships with a multi-replica
    caption stage and the other reverts to a single replica, customers
    using the second one would hit the exact bug we are fixing.  This
    test pins parity — if you intentionally diverge the configurations
    (e.g. because the custom_summarization pipeline uses a different
    VLM with different latency characteristics), update this test
    with the explicit divergence justification.
    """
    counts: dict[str, tuple[int, int]] = {}
    for yaml_name, yaml_path in _PIPELINE_YAMLS.items():
        pipeline = _load_pipeline(yaml_path)
        stage = _find_stage(pipeline, _STAGE_NAME)
        counts[yaml_name] = (
            _replica_value(stage, "static_replicas"),
            _replica_value(stage, "max_replicas"),
        )
    unique_pairs = set(counts.values())
    assert len(unique_pairs) == 1, (
        "image_caption replica counts diverged across pipeline YAMLs:\n"
        + "\n".join(f"  {name}: static={s}, max={m}" for name, (s, m) in sorted(counts.items()))
        + "\nKeep both pipelines aligned, or update this test with "
        "the rationale for the intentional divergence."
    )
