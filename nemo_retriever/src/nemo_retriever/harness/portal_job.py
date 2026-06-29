# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bridge portal/runner job payloads to the revamped harness benchmark runner."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from nemo_retriever.harness.artifacts import last_commit, now_timestr
from nemo_retriever.harness.benchmark_registry import BENCHMARKS, benchmark_names
from nemo_retriever.harness.execution import run_benchmark
from nemo_retriever.harness.json_io import read_json_object
from nemo_retriever.harness.resolution import make_run_id, resolve_artifact_dir

_LEGACY_OVERRIDE_MAP: dict[str, str] = {
    "dataset_dir": "dataset.path",
    "query_csv": "dataset.query_file",
    "input_type": "dataset.input_type",
    "run_mode": "ingest.run_mode",
    "ray_address": "ingest.ray_address",
    "evaluation_mode": "evaluation.mode",
    "beir_loader": "evaluation.loader",
    "beir_dataset_name": "evaluation.dataset_name",
    "beir_split": "evaluation.split",
    "beir_query_language": "evaluation.query_language",
    "beir_doc_id_field": "evaluation.doc_id_field",
    "beir_ks": "evaluation.ks",
    "embed_model_name": "ingest.embed.embed_model_name",
    "embed_invoke_url": "query.embed_invoke_url",
    "lancedb_table_name": "ingest.storage.table_name",
    "embed_modality": "ingest.embed.embed_modality",
    "embed_granularity": "ingest.embed.embed_granularity",
    "extract_page_as_image": "ingest.extract.extract_page_as_image",
    "extract_infographics": "ingest.extract.extract_infographics",
    "ocr_version": "ingest.extract.ocr_version",
    "ocr_lang": "ingest.extract.ocr_lang",
    "pdf_extract_workers": "ingest.extract.batch.pdf_extract_workers",
    "pdf_extract_num_cpus": "ingest.extract.batch.pdf_extract_cpus_per_task",
    "pdf_extract_batch_size": "ingest.extract.batch.pdf_extract_batch_size",
    "pdf_split_batch_size": "ingest.extract.batch.pdf_split_batch_size",
    "page_elements_batch_size": "ingest.extract.batch.page_elements_batch_size",
    "page_elements_workers": "ingest.extract.batch.page_elements_workers",
    "page_elements_cpus_per_actor": "ingest.extract.batch.page_elements_cpus_per_actor",
    "ocr_workers": "ingest.extract.batch.ocr_workers",
    "ocr_batch_size": "ingest.extract.batch.ocr_batch_size",
    "ocr_cpus_per_actor": "ingest.extract.batch.ocr_cpus_per_actor",
    "embed_workers": "ingest.embed.batch.embed_workers",
    "embed_batch_size": "ingest.embed.batch.embed_batch_size",
    "embed_cpus_per_actor": "ingest.embed.batch.embed_cpus_per_actor",
    "gpu_page_elements": "ingest.extract.batch.page_elements_gpus_per_actor",
    "gpu_ocr": "ingest.extract.batch.ocr_gpus_per_actor",
    "gpu_embed": "ingest.embed.batch.embed_gpus_per_actor",
}


def _dataset_label(dataset: str | None, overrides: dict[str, Any]) -> str:
    if overrides.get("dataset_label"):
        return str(overrides["dataset_label"])
    if dataset and "/" not in dataset and "\\" not in dataset:
        return str(dataset)
    raw = overrides.get("dataset_dir") or dataset or "unknown"
    return Path(str(raw)).name or "unknown"


def _wants_beir_evaluation(overrides: dict[str, Any]) -> bool:
    if overrides.get("recall_required"):
        return True
    mode = overrides.get("evaluation_mode")
    if mode and str(mode).lower() not in {"none", ""}:
        return True
    return bool(overrides.get("query_csv"))


def _resolve_portal_benchmark(dataset: str | None, overrides: dict[str, Any]) -> str:
    if overrides.get("benchmark"):
        return str(overrides["benchmark"])

    label = _dataset_label(dataset, overrides)
    if label in benchmark_names():
        return label

    candidates = [name for name, spec in BENCHMARKS.items() if spec.dataset == label]
    if candidates:
        if _wants_beir_evaluation(overrides):
            for name in candidates:
                if BENCHMARKS[name].evaluation.get("mode") == "beir":
                    return name
        for name in candidates:
            if BENCHMARKS[name].evaluation.get("mode") == "none":
                return name
        return candidates[0]

    for guess in (f"{label}_beir", f"{label}_smoke", f"{label}_beir_fast_text"):
        if guess in BENCHMARKS:
            return guess

    raise ValueError(
        f"Could not resolve a harness benchmark for dataset {label!r}. "
        "Register a matching benchmark or set benchmark in job overrides."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def _portal_overrides_to_set_args(overrides: dict[str, Any]) -> list[str]:
    set_args: list[str] = []
    tuning = overrides.get("tuning")
    if isinstance(tuning, dict):
        merged = dict(overrides)
        merged.update(tuning)
        overrides = merged

    for key, value in overrides.items():
        if key in {"dataset", "preset", "benchmark", "dataset_label", "tuning", "tags"}:
            continue
        if value is None:
            continue
        mapped = _LEGACY_OVERRIDE_MAP.get(key)
        if mapped is None:
            continue
        set_args.append(f"{mapped}={_format_override_value(value)}")

    if overrides.get("hybrid") is True:
        set_args.append("ingest.storage.index_mode=hybrid")
    if overrides.get("embed_model_name"):
        model = _format_override_value(overrides["embed_model_name"])
        set_args.append(f"query.embed_model_name={model}")

    return set_args


def _legacy_result(
    outcome,
    *,
    dataset_label: str,
    preset: str | None,
    overrides: dict[str, Any],
    tags: list[str] | None,
) -> dict[str, Any]:
    results = outcome.results or {}
    summary_metrics = dict(results.get("summary_metrics") or {})
    failure = results.get("failure") or {}
    resolved = results.get("resolved_benchmark") or {}
    dataset_spec = resolved.get("dataset") or {}
    ingest = resolved.get("ingest") or {}
    success = bool(results.get("success"))

    test_config = {
        "benchmark": results.get("benchmark"),
        "dataset_label": dataset_label,
        "dataset_dir": overrides.get("dataset_dir") or dataset_spec.get("path"),
        "preset": preset,
        "run_mode": overrides.get("run_mode") or ingest.get("run_mode"),
        "service_url": overrides.get("service_url"),
        "service_max_concurrency": overrides.get("service_max_concurrency"),
        "query_csv": overrides.get("query_csv") or dataset_spec.get("query_file"),
        "input_type": overrides.get("input_type") or dataset_spec.get("input_type"),
        "recall_required": overrides.get("recall_required"),
        "evaluation_mode": (resolved.get("evaluation") or {}).get("mode"),
        "embed_model_name": overrides.get("embed_model_name"),
        "embed_modality": overrides.get("embed_modality"),
        "embed_granularity": overrides.get("embed_granularity"),
        "ray_address": overrides.get("ray_address") or ingest.get("ray_address"),
    }

    try:
        host = socket.gethostname().strip() or "unknown"
    except Exception:
        host = "unknown"

    env_path = outcome.artifact_dir / "environment.json"
    env_payload: dict[str, Any] = {}
    if env_path.is_file():
        try:
            env_payload = read_json_object(env_path)
        except (OSError, ValueError):
            env_payload = {}

    artifacts = dict(results.get("artifacts") or {})
    if results.get("replay_command"):
        artifacts.setdefault("replay_command", results["replay_command"])

    return {
        "timestamp": now_timestr(),
        "latest_commit": last_commit(),
        "success": success,
        "return_code": outcome.exit_code,
        "failure_reason": failure.get("message") if not success else None,
        "test_config": test_config,
        "metrics": summary_metrics,
        "summary_metrics": summary_metrics,
        "run_metadata": {
            "host": host,
            "gpu_type": env_payload.get("gpu_type"),
        },
        "artifacts": artifacts,
        "artifact_dir": str(outcome.artifact_dir),
        "replay_command": results.get("replay_command") or artifacts.get("replay_command"),
        "tags": tags,
    }


def run_portal_job_entry(
    *,
    run_name: str | None,
    config_file: str | None,
    session_dir: Path | None,
    dataset: str | None,
    preset: str | None,
    sweep_overrides: dict[str, Any] | None = None,
    cli_overrides: list[str] | None = None,
    cli_helm_set: list[str] | None = None,
    recall_required: bool | None = None,
    tags: list[str] | None = None,
    skip_local_history: bool = False,
    graph_code: str | None = None,
) -> dict[str, Any]:
    """Execute a portal-style harness job and return the legacy result payload."""
    del config_file, cli_overrides, cli_helm_set, recall_required, skip_local_history

    if graph_code:
        return {
            "success": False,
            "return_code": 2,
            "failure_reason": "Graph pipeline portal jobs are not supported by the revamped harness yet.",
            "timestamp": now_timestr(),
        }

    overrides = dict(sweep_overrides or {})
    if dataset and "dataset_dir" not in overrides:
        overrides["dataset_dir"] = dataset
    if preset and "preset" not in overrides:
        overrides["preset"] = preset

    run_mode = str(overrides.get("run_mode") or "batch")

    if run_mode == "service":
        from nemo_retriever.harness.service_execution import run_service_portal_job

        outcome = run_service_portal_job(
            dataset=dataset,
            preset=preset,
            overrides=overrides,
            run_name=run_name,
            session_dir=session_dir,
        )
        return _legacy_result(
            outcome,
            dataset_label=_dataset_label(dataset, overrides),
            preset=preset,
            overrides=overrides,
            tags=tags,
        )

    try:
        benchmark = _resolve_portal_benchmark(dataset, overrides)
    except ValueError as exc:
        return {
            "success": False,
            "return_code": 2,
            "failure_reason": str(exc),
            "timestamp": now_timestr(),
        }

    set_args = _portal_overrides_to_set_args(overrides)
    effective_run_id = run_name or make_run_id(benchmark)
    if session_dir is not None:
        output_dir = str((session_dir / effective_run_id).resolve())
    else:
        output_dir = str(resolve_artifact_dir(benchmark, effective_run_id, overrides.get("artifacts_dir")))

    outcome = run_benchmark(
        benchmark,
        output_dir=output_dir,
        run_id=effective_run_id,
        mode="local",
        overrides=tuple(set_args),
    )
    return _legacy_result(
        outcome,
        dataset_label=_dataset_label(dataset, overrides),
        preset=preset,
        overrides=overrides,
        tags=tags,
    )


def run_portal_job_from_dict(job: dict[str, Any]) -> dict[str, Any]:
    """Execute a portal history job record."""
    overrides = job.get("dataset_overrides") or {}
    if isinstance(overrides, str):
        overrides = json.loads(overrides)
    dataset_value = job.get("dataset_path") or job.get("dataset")
    return run_portal_job_entry(
        run_name=job.get("id"),
        config_file=job.get("config"),
        session_dir=None,
        dataset=dataset_value,
        preset=job.get("preset"),
        sweep_overrides=overrides,
        tags=job.get("tags"),
        skip_local_history=True,
        graph_code=job.get("graph_code"),
    )


if __name__ == "__main__":
    import os
    import sys

    if os.environ.get("HARNESS_JOB_JSON"):
        payload = json.loads(os.environ["HARNESS_JOB_JSON"])
    elif len(sys.argv) == 2:
        payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    else:
        raise SystemExit(
            "Usage: HARNESS_JOB_JSON='{...}' python -m nemo_retriever.harness.portal_job "
            "or python -m nemo_retriever.harness.portal_job <job.json>"
        )
    outcome = run_portal_job_from_dict(payload)
    sys.stdout.write(json.dumps(outcome))
    raise SystemExit(0 if outcome.get("success") else int(outcome.get("return_code") or 1))
