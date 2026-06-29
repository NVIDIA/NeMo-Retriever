# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and persist replayable harness CLI commands for portal display."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataclasses import fields

from nemo_retriever.harness.config import HarnessConfig


def build_harness_run_command(
    benchmark: str,
    *,
    output_dir: str | Path,
    run_id: str | None = None,
    mode: str = "local",
    overrides: Sequence[str] = (),
    requirements: Sequence[str] = (),
    dry_run: bool = False,
) -> str:
    """Return an equivalent ``retriever harness run`` shell command."""
    parts = ["retriever harness run", shlex.quote(benchmark)]
    parts.append(f"--output-dir {shlex.quote(str(output_dir))}")
    if run_id:
        parts.append(f"--run-id {shlex.quote(run_id)}")
    if mode and mode != "local":
        parts.append(f"--mode {shlex.quote(mode)}")
    for override in overrides:
        parts.append(f"--set {shlex.quote(str(override))}")
    for requirement in requirements:
        parts.append(f"--require {shlex.quote(str(requirement))}")
    if dry_run:
        parts.append("--dry-run")
    return " ".join(parts)


def build_service_replay_command(cfg: HarnessConfig) -> str:
    """Return a human-readable replay command for service-mode portal jobs."""
    parts = [
        "# portal service-mode job",
        f"--service-url {shlex.quote(str(cfg.service_url or ''))}",
    ]
    if cfg.dataset_dir:
        parts.append(f"--dataset {shlex.quote(cfg.dataset_dir)}")
    if cfg.preset:
        parts.append(f"--preset {shlex.quote(cfg.preset)}")
    parts.append(f"--service-max-concurrency {int(cfg.service_max_concurrency)}")
    if cfg.input_type:
        parts.append(f"--input-type {shlex.quote(cfg.input_type)}")
    if cfg.evaluation_mode and cfg.evaluation_mode != "none":
        parts.append(f"--evaluation-mode {shlex.quote(cfg.evaluation_mode)}")
    if cfg.artifacts_dir:
        parts.append(f"# artifacts_dir={shlex.quote(cfg.artifacts_dir)}")
    return " ".join(part for part in parts if part)


def persist_replay_command(artifact_dir: Path, command: str) -> dict[str, str]:
    """Write ``command.txt`` and return artifact metadata for run results."""
    path = (artifact_dir / "command.txt").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(command.rstrip() + "\n", encoding="utf-8")
    return {
        "command_file": str(path),
        "replay_command": command.rstrip(),
    }


def reconstruct_command_from_record(
    raw: Mapping[str, Any],
    job: Mapping[str, Any] | None = None,
) -> str | None:
    """Best-effort CLI reconstruction when ``command.txt`` is unavailable."""
    replay = raw.get("replay_command")
    if isinstance(replay, str) and replay.strip():
        return replay.strip()

    test_config = raw.get("test_config") or {}
    if not isinstance(test_config, Mapping):
        test_config = {}

    benchmark = raw.get("benchmark") or test_config.get("benchmark")
    artifact_dir = raw.get("artifact_dir")
    if job and not artifact_dir:
        artifact_dir = job.get("artifact_dir")

    run_mode = test_config.get("run_mode")
    mode = "local"
    overrides: list[str] = []
    if run_mode == "batch":
        mode = "batch"
    elif run_mode == "service":
        cfg_fields = {
            "service_url": test_config.get("service_url"),
            "dataset_dir": test_config.get("dataset_dir"),
            "preset": test_config.get("preset"),
            "service_max_concurrency": test_config.get("service_max_concurrency") or 8,
            "input_type": test_config.get("input_type") or "pdf",
            "evaluation_mode": test_config.get("evaluation_mode") or "none",
            "artifacts_dir": str(artifact_dir) if artifact_dir else None,
            "dataset_label": test_config.get("dataset_label") or "unknown",
        }
        allowed = {field.name for field in fields(HarnessConfig)}
        cfg = HarnessConfig(**{k: v for k, v in cfg_fields.items() if k in allowed and v is not None})
        return build_service_replay_command(cfg)

    if test_config.get("dataset_dir"):
        overrides.append(f"dataset.path={test_config['dataset_dir']}")
    if test_config.get("embed_model_name"):
        overrides.append(f"query.embed_model_name={test_config['embed_model_name']}")
    if run_mode:
        overrides.append(f"ingest.run_mode={run_mode}")

    if benchmark and artifact_dir:
        return build_harness_run_command(
            str(benchmark),
            output_dir=artifact_dir,
            mode=mode,
            overrides=overrides,
        )

    if job:
        dataset = job.get("dataset")
        preset = job.get("preset")
        if dataset:
            parts = ["# portal job replay (approximate)", f"dataset={shlex.quote(str(dataset))}"]
            if preset:
                parts.append(f"preset={shlex.quote(str(preset))}")
            if run_mode:
                parts.append(f"run_mode={run_mode}")
            return " ".join(parts)

    return None
