# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-mode harness execution for portal/runner jobs."""

from __future__ import annotations

import time
import traceback
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from nemo_retriever.common.input_files import resolve_input_files
from nemo_retriever.harness.artifact_writer import ArtifactWriter, capture_output_to_log
from nemo_retriever.harness.artifacts import last_commit, now_timestr
from nemo_retriever.harness.benchmark_registry import DEFAULT_EMBED_MODEL, DEFAULT_SUMMARY_KEYS, DEFAULT_TABLE_NAME
from nemo_retriever.harness.config import HarnessConfig
from nemo_retriever.harness.contracts import (
    EXIT_INGEST_FAILURE,
    EXIT_INTERNAL_ERROR,
    EXIT_INVALID,
    EXIT_SUCCESS,
    FailurePayload,
    HarnessRunError,
    RunOutcome,
)
from nemo_retriever.harness.environment import collect_environment
from nemo_retriever.harness.json_io import write_json
from nemo_retriever.harness.metrics import _normalize_metric_key, _safe_pdf_page_count
from nemo_retriever.harness.replay_command import build_service_replay_command, persist_replay_command
from nemo_retriever.harness.resolution import make_run_id, resolve_artifact_dir
from nemo_retriever.tools.recall.beir import BeirConfig, evaluate_service_beir, resolve_beir_dataset_options

_CSV_BEIR_LOADERS = {"bo767_csv", "bo10k_csv", "earnings_csv", "financebench_json", "jp20_csv"}


def _service_failure_key(failure: Any) -> str:
    if isinstance(failure, (list, tuple)) and failure:
        return str(failure[0])
    return str(failure)


def _filename_page_counts(paths: list[Path], page_counts: list[int | None]) -> dict[str, int]:
    filename_to_pages: dict[str, int] = {}
    seen_filenames: set[str] = set()
    ambiguous_filenames: set[str] = set()
    for path, page_count in zip(paths, page_counts):
        filename = path.name
        if filename in seen_filenames:
            ambiguous_filenames.add(filename)
            filename_to_pages.pop(filename, None)
            continue
        seen_filenames.add(filename)
        if page_count is not None:
            filename_to_pages[filename] = page_count
    for filename in ambiguous_filenames:
        filename_to_pages.pop(filename, None)
    return filename_to_pages


def _count_failed_pdf_pages(
    failures: list[Any],
    document_filenames: dict[str, str],
    filename_to_pages: dict[str, int],
) -> int:
    pages_failed = 0
    for failure in failures:
        failure_key = _service_failure_key(failure)
        filename = document_filenames.get(failure_key, failure_key)
        page_count = filename_to_pages.get(Path(filename).name)
        if page_count is None:
            page_count = 1
        pages_failed += page_count
    return pages_failed


def _infer_evaluation_mode(overrides: dict[str, Any]) -> str:
    mode = overrides.get("evaluation_mode")
    if mode:
        return str(mode)
    if overrides.get("beir_loader"):
        return "beir"
    if overrides.get("query_csv") and overrides.get("recall_required"):
        return "beir"
    return "none"


def _harness_config_from_overrides(
    *,
    dataset_dir: str,
    dataset_label: str,
    preset: str | None,
    overrides: dict[str, Any],
) -> HarnessConfig:
    merged = dict(overrides)
    merged["dataset_dir"] = dataset_dir
    merged["dataset_label"] = dataset_label
    merged["preset"] = preset or merged.get("preset") or "single_gpu"
    merged["run_mode"] = "service"
    merged.setdefault("service_max_concurrency", 8)
    merged.setdefault("input_type", "pdf")
    merged.setdefault("recall_required", bool(merged.get("query_csv")))
    merged.setdefault("evaluation_mode", _infer_evaluation_mode(merged))
    if merged["evaluation_mode"] == "beir":
        merged.setdefault("beir_dataset_name", dataset_label)
        if merged.get("query_csv") and not merged.get("beir_loader"):
            merged.setdefault("beir_loader", f"{dataset_label}_csv")
    allowed = {field.name for field in fields(HarnessConfig)}
    cfg = HarnessConfig(**{key: value for key, value in merged.items() if key in allowed})
    errors = cfg.validate()
    if errors:
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="invalid_service_config",
                retryable=False,
                message="Service configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors),
            ),
        )
    return cfg


def _service_beir_dataset_name(cfg: HarnessConfig) -> str:
    dataset_name = cfg.beir_dataset_name or cfg.dataset_label
    if cfg.beir_loader in _CSV_BEIR_LOADERS and cfg.query_csv:
        return str(Path(cfg.query_csv).resolve())
    return str(dataset_name)


def _run_service_beir_evaluation(cfg: HarnessConfig) -> tuple[float, dict[str, float], int]:
    beir_options = resolve_beir_dataset_options(
        dataset_name=_service_beir_dataset_name(cfg),
        loader=cfg.beir_loader,
        doc_id_field=cfg.beir_doc_id_field,
        ks=cfg.beir_ks,
    )
    if not beir_options.loader:
        raise ValueError("beir_loader is required for service-mode BEIR evaluation")
    if not beir_options.dataset_name:
        raise ValueError("beir_dataset_name is required for service-mode BEIR evaluation")

    beir_cfg = BeirConfig(
        lancedb_uri=str(cfg.lancedb_uri or "lancedb"),
        lancedb_table=str(cfg.lancedb_table_name or DEFAULT_TABLE_NAME),
        embedding_model=cfg.embed_model_name,
        loader=str(beir_options.loader),
        dataset_name=str(beir_options.dataset_name),
        split=str(cfg.beir_split),
        query_language=cfg.beir_query_language,
        doc_id_field=str(beir_options.doc_id_field),
        ks=beir_options.ks,
        hybrid=bool(cfg.hybrid),
        service_url=cfg.service_url,
        service_api_token=cfg.api_key,
        service_max_concurrent=int(cfg.service_max_concurrency),
    )
    eval_start = time.perf_counter()
    beir_dataset, _raw_hits, _run, metrics = evaluate_service_beir(beir_cfg)
    return time.perf_counter() - eval_start, metrics, len(beir_dataset.query_ids)


def _build_summary_metrics(
    metrics_payload: dict[str, Any],
    *,
    evaluation_metrics: dict[str, float] | None = None,
    evaluation_count: int = 0,
) -> dict[str, Any]:
    summary = {
        "files": metrics_payload.get("files"),
        "pages": metrics_payload.get("pages"),
        "rows_processed": None,
        "ingest_secs": metrics_payload.get("ingest_secs"),
        "pages_per_sec_ingest": metrics_payload.get("pages_per_sec_ingest"),
        "query_count": evaluation_count,
        "query_latency_p50_ms": None,
        "query_latency_p95_ms": None,
    }
    for key, value in (evaluation_metrics or {}).items():
        summary[_normalize_metric_key(key)] = value
    return {key: summary.get(key) for key in DEFAULT_SUMMARY_KEYS}


def _test_config_payload(cfg: HarnessConfig) -> dict[str, Any]:
    return {
        "dataset_label": cfg.dataset_label,
        "dataset_dir": cfg.dataset_dir,
        "preset": cfg.preset,
        "run_mode": "service",
        "service_url": cfg.service_url,
        "service_max_concurrency": cfg.service_max_concurrency,
        "manage_service": cfg.manage_service,
        "input_type": cfg.input_type,
        "api_key": "(set)" if cfg.api_key else None,
        "query_csv": cfg.query_csv,
        "effective_query_csv": cfg.query_csv if cfg.evaluation_mode == "beir" else None,
        "recall_required": cfg.recall_required,
        "evaluation_mode": cfg.evaluation_mode,
        "beir_loader": cfg.beir_loader,
        "beir_dataset_name": cfg.beir_dataset_name,
        "beir_split": cfg.beir_split,
        "beir_query_language": cfg.beir_query_language,
        "beir_doc_id_field": cfg.beir_doc_id_field,
        "beir_ks": list(cfg.beir_ks),
    }


def _service_result_payload(
    writer: ArtifactWriter,
    *,
    cfg: HarnessConfig,
    success: bool,
    exit_code: int,
    failure_reason: str | None,
    metrics_payload: dict[str, Any],
    summary_metrics: dict[str, Any],
    runtime_summary: dict[str, Any] | None = None,
    service_fields: dict[str, Any] | None = None,
    failures: list[Any] | None = None,
) -> dict[str, Any]:
    runtime_dir = writer.artifact_dir / "runtime_metrics"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    replay_meta = persist_replay_command(writer.artifact_dir, build_service_replay_command(cfg))
    env_payload = collect_environment()
    write_json(writer.path("environment.json"), env_payload)
    result: dict[str, Any] = {
        "run_id": writer.run_id,
        "benchmark": writer.benchmark,
        "timestamp": now_timestr(),
        "latest_commit": last_commit(),
        "status": "complete" if success else "failed",
        "success": success,
        "exit_code": exit_code,
        "return_code": exit_code,
        "failure_reason": failure_reason,
        "failure": (
            {
                "failed_phase": "ingest",
                "failure_reason": "service_ingest_failed",
                "retryable": False,
                "message": failure_reason or "service run failed",
            }
            if not success
            else None
        ),
        "test_config": _test_config_payload(cfg),
        "metrics": metrics_payload,
        "summary_metrics": summary_metrics,
        "run_metadata": {
            "host": env_payload.get("host"),
            "gpu_type": env_payload.get("gpu_type"),
            "gpu_count": env_payload.get("gpu_count"),
            "ray_cluster_mode": "none (service mode)",
        },
        "runtime_summary": runtime_summary,
        "detection_summary": None,
        "artifacts": {
            "runtime_metrics_dir": str(runtime_dir.resolve()),
            "artifact_dir": str(writer.artifact_dir),
            **replay_meta,
        },
        "artifact_dir": str(writer.artifact_dir),
        "replay_command": replay_meta["replay_command"],
    }
    if service_fields:
        result.update(service_fields)
    if failures:
        result["failures"] = failures
    write_json(writer.path("results.json"), result)
    write_json(writer.path("summary_metrics.json"), summary_metrics)
    writer.status(status="complete" if success else "failed", phase="write_artifacts")
    return result


def _execute_service_mode(
    cfg: HarnessConfig,
    *,
    writer: ArtifactWriter,
) -> RunOutcome:
    from nemo_retriever.service.service_ingestor import ServiceIngestor

    dataset_path = Path(cfg.dataset_dir)
    input_files = resolve_input_files(dataset_path, cfg.input_type)
    if not input_files:
        failure = f"No {cfg.input_type} files found in {cfg.dataset_dir}"
        result = _service_result_payload(
            writer,
            cfg=cfg,
            success=False,
            exit_code=EXIT_INVALID,
            failure_reason=failure,
            metrics_payload={"files": 0, "pages": 0, "ingest_secs": None},
            summary_metrics=_build_summary_metrics({"files": 0, "pages": 0}),
        )
        return RunOutcome(exit_code=EXIT_INVALID, artifact_dir=writer.artifact_dir, results=result)

    ingestor = ServiceIngestor(
        base_url=str(cfg.service_url),
        documents=[str(path) for path in input_files],
        max_concurrency=int(cfg.service_max_concurrency),
        api_token=cfg.api_key,
    )

    wall_start = time.perf_counter()
    try:
        with capture_output_to_log(writer.path("run.log"), label="service_ingest"):
            result_obj = ingestor.ingest()
    except Exception as exc:
        elapsed = time.perf_counter() - wall_start
        result = _service_result_payload(
            writer,
            cfg=cfg,
            success=False,
            exit_code=EXIT_INGEST_FAILURE,
            failure_reason=f"{type(exc).__name__}: {exc}",
            metrics_payload={"files": len(input_files), "ingest_secs": round(elapsed, 2)},
            summary_metrics=_build_summary_metrics(
                {"files": len(input_files), "ingest_secs": round(elapsed, 2)},
            ),
        )
        result["error_detail"] = traceback.format_exc()
        write_json(writer.path("results.json"), result)
        return RunOutcome(exit_code=EXIT_INGEST_FAILURE, artifact_dir=writer.artifact_dir, results=result)

    elapsed = float(getattr(result_obj, "elapsed_s", 0.0))
    failures = list(getattr(result_obj, "failures", []))
    document_ids = list(getattr(result_obj, "document_ids", []))
    raw_document_filenames = getattr(result_obj, "document_filenames", {})
    if isinstance(raw_document_filenames, dict):
        document_filenames = {str(doc_id): str(filename) for doc_id, filename in raw_document_filenames.items()}
    else:
        document_filenames = {}

    input_page_counts: list[int | None] = []
    counted_input_pages: int | None = None
    if cfg.input_type == "pdf":
        total_counted_pages = 0
        counted_any_pdf = False
        for path in input_files:
            page_count = _safe_pdf_page_count(path)
            input_page_counts.append(page_count)
            if page_count is None:
                continue
            counted_any_pdf = True
            total_counted_pages += page_count
        if counted_any_pdf:
            counted_input_pages = total_counted_pages

    pages_failed = len(failures)
    if counted_input_pages is not None:
        total_pages = counted_input_pages
        filename_to_pages = _filename_page_counts(input_files, input_page_counts)
        pages_failed = _count_failed_pdf_pages(failures, document_filenames, filename_to_pages)
        pages_processed = max(total_pages - pages_failed, 0)
    else:
        if isinstance(result_obj, list):
            pages_processed = len(result_obj)
        else:
            pages_processed = len(document_ids) if document_ids else max(len(input_files) - pages_failed, 0)
        total_pages = pages_processed + pages_failed

    pps = round(total_pages / elapsed, 2) if elapsed > 0 else None
    metrics_payload: dict[str, Any] = {
        "files": len(input_files),
        "pages": total_pages,
        "pages_processed": pages_processed,
        "pages_failed": pages_failed,
        "ingest_secs": round(elapsed, 2),
        "pages_per_sec_ingest": pps,
    }

    runtime_summary: dict[str, Any] | None = None
    evaluation_metrics: dict[str, float] = {}
    evaluation_count = 0
    evaluation_failure: str | None = None
    if cfg.evaluation_mode == "beir":
        writer.status(status="running", phase="evaluate")
        try:
            with capture_output_to_log(writer.path("run.log"), label="service_beir"):
                evaluation_secs, evaluation_metrics, evaluation_count = _run_service_beir_evaluation(cfg)
            runtime_summary = {
                "evaluation_label": "BEIR",
                "evaluation_time_secs": round(evaluation_secs, 2),
                "evaluation_metrics": evaluation_metrics,
                "evaluation_count": evaluation_count,
            }
            metrics_payload.update({_normalize_metric_key(name): value for name, value in evaluation_metrics.items()})
        except Exception as exc:
            evaluation_failure = f"{type(exc).__name__}: {exc}"
    elif cfg.evaluation_mode not in {"none", ""}:
        evaluation_failure = f"evaluation_mode={cfg.evaluation_mode!r} is not supported in service mode"

    success = True
    exit_code = EXIT_SUCCESS
    failure_reason: str | None = None
    if pages_failed:
        success = False
        exit_code = EXIT_INGEST_FAILURE
        failure_reason = f"{pages_failed} page(s) failed during service ingestion"
    if evaluation_failure:
        success = False
        exit_code = EXIT_INGEST_FAILURE
        failure_reason = evaluation_failure
    if cfg.evaluation_mode == "beir" and cfg.recall_required and not evaluation_metrics:
        success = False
        exit_code = EXIT_INGEST_FAILURE
        failure_reason = failure_reason or "missing_beir_metrics"

    summary_metrics = _build_summary_metrics(
        metrics_payload,
        evaluation_metrics=evaluation_metrics,
        evaluation_count=evaluation_count,
    )
    result = _service_result_payload(
        writer,
        cfg=cfg,
        success=success,
        exit_code=exit_code,
        failure_reason=failure_reason,
        metrics_payload=metrics_payload,
        summary_metrics=summary_metrics,
        runtime_summary=runtime_summary,
        service_fields={
            "service_job_id": getattr(result_obj, "job_id", None),
            "service_document_ids": document_ids,
            "service_job_status": getattr(result_obj, "job_status", None),
        },
        failures=failures or None,
    )
    return RunOutcome(exit_code=exit_code, artifact_dir=writer.artifact_dir, results=result)


def _execute_managed_service_mode(cfg: HarnessConfig, *, writer: ArtifactWriter) -> RunOutcome:
    from nemo_retriever.harness.helm_manager import HelmServiceManager

    manager = HelmServiceManager(cfg)
    start_error: str | None = None
    start_rc = 1
    try:
        try:
            start_rc = manager.start()
        except Exception as exc:
            start_error = f"{type(exc).__name__}: {exc}"
            start_rc = 1

        if start_rc != 0:
            result = _service_result_payload(
                writer,
                cfg=cfg,
                success=False,
                exit_code=start_rc,
                failure_reason=start_error or f"managed Helm service failed to become ready (exit {start_rc})",
                metrics_payload={"files": None, "pages": None, "ingest_secs": None},
                summary_metrics=_build_summary_metrics({}),
            )
            try:
                manager.dump_logs(writer.artifact_dir)
            except Exception as exc:
                result["service_log_collection_error"] = f"{type(exc).__name__}: {exc}"
            write_json(writer.path("results.json"), result)
            return RunOutcome(exit_code=start_rc, artifact_dir=writer.artifact_dir, results=result)

        service_cfg = replace(cfg, service_url=manager.get_service_url())
        outcome = _execute_service_mode(service_cfg, writer=writer)
        outcome.results["managed_service"] = {
            "helm_release": cfg.helm_release,
            "helm_namespace": cfg.helm_namespace or cfg.helm_release,
            "service_url": service_cfg.service_url,
            "kept_up": bool(cfg.keep_up),
        }
        if not outcome.results.get("success"):
            try:
                manager.dump_logs(writer.artifact_dir)
            except Exception as exc:
                outcome.results["service_log_collection_error"] = f"{type(exc).__name__}: {exc}"
        write_json(writer.path("results.json"), outcome.results)
        return outcome
    finally:
        if not cfg.keep_up:
            manager.stop()


def run_service_portal_job(
    *,
    dataset: str | None,
    preset: str | None,
    overrides: dict[str, Any],
    run_name: str | None,
    session_dir: Path | None,
) -> RunOutcome:
    """Execute a portal service-mode job and return a harness-style outcome."""
    dataset_dir = str(overrides.get("dataset_dir") or dataset or "")
    dataset_label = Path(dataset_dir).name if dataset_dir else "unknown"
    if dataset and "/" not in dataset and "\\" not in dataset and not overrides.get("dataset_label"):
        dataset_label = str(dataset)

    effective_run_id = run_name or make_run_id(f"service_{dataset_label}")
    if session_dir is not None:
        artifact_dir = (session_dir / effective_run_id).resolve()
    else:
        artifact_dir = resolve_artifact_dir(f"service_{dataset_label}", effective_run_id, overrides.get("artifacts_dir"))

    writer = ArtifactWriter(
        artifact_dir=artifact_dir,
        run_id=effective_run_id,
        benchmark=f"service_{dataset_label}",
    )
    writer.status(status="planned", phase="resolve")

    try:
        cfg = _harness_config_from_overrides(
            dataset_dir=dataset_dir,
            dataset_label=dataset_label,
            preset=preset,
            overrides=overrides,
        )
        if cfg.manage_service:
            return _execute_managed_service_mode(cfg, writer=writer)
        return _execute_service_mode(cfg, writer=writer)
    except HarnessRunError as exc:
        result = {
            "run_id": effective_run_id,
            "benchmark": writer.benchmark,
            "timestamp": now_timestr(),
            "success": False,
            "exit_code": exc.exit_code,
            "return_code": exc.exit_code,
            "failure_reason": exc.failure.message,
            "failure": exc.failure.to_dict(),
            "test_config": {
                "dataset_label": dataset_label,
                "dataset_dir": dataset_dir,
                "preset": preset,
                "run_mode": "service",
                "service_url": overrides.get("service_url"),
            },
            "metrics": {},
            "summary_metrics": {},
            "artifacts": {"artifact_dir": str(writer.artifact_dir)},
            "artifact_dir": str(writer.artifact_dir),
        }
        write_json(writer.path("results.json"), result)
        return RunOutcome(exit_code=exc.exit_code, artifact_dir=writer.artifact_dir, results=result)
    except Exception as exc:
        result = {
            "run_id": effective_run_id,
            "benchmark": writer.benchmark,
            "timestamp": now_timestr(),
            "success": False,
            "exit_code": EXIT_INTERNAL_ERROR,
            "return_code": EXIT_INTERNAL_ERROR,
            "failure_reason": str(exc),
            "error_detail": traceback.format_exc(),
            "test_config": {
                "dataset_label": dataset_label,
                "dataset_dir": dataset_dir,
                "preset": preset,
                "run_mode": "service",
                "service_url": overrides.get("service_url"),
            },
            "metrics": {},
            "summary_metrics": {},
            "artifacts": {"artifact_dir": str(writer.artifact_dir)},
            "artifact_dir": str(writer.artifact_dir),
        }
        write_json(writer.path("results.json"), result)
        return RunOutcome(exit_code=EXIT_INTERNAL_ERROR, artifact_dir=writer.artifact_dir, results=result)
