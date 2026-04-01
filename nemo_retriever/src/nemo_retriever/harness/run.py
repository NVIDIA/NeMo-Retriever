# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import metadata
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer

from nemo_retriever.application.modes.reports import (
    RunArtifactConfig,
    RunEvaluationConfig,
    flatten_report_metrics,
    normalize_metric_key,
)
from nemo_retriever.application.modes.run_batch import BatchPipelineConfig, run_batch_pipeline
from nemo_retriever.application.modes.run_inprocess import InProcessPipelineConfig, run_inprocess_pipeline
from nemo_retriever.application.modes.shared import DEFAULT_LANCEDB_TABLE
from nemo_retriever.harness.artifacts import (
    create_run_artifact_dir,
    create_session_dir,
    last_commit,
    now_timestr,
    write_json,
    write_session_summary,
)
from nemo_retriever.harness.config import (
    DEFAULT_NIGHTLY_CONFIG_PATH,
    HarnessConfig,
    load_harness_config,
    load_nightly_config,
    tuning_fields_for_run_mode,
)
from nemo_retriever.harness.recall_adapters import prepare_recall_query_file
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.params import VdbUploadParams
from nemo_retriever.utils.input_files import resolve_input_files


def _collect_gpu_metadata() -> tuple[int | None, str | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None

    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    combined_output = f"{result.stdout}\n{result.stderr}"
    if "No devices were found" in combined_output:
        return 0, None
    if result.returncode != 0:
        return None, None
    if not output_lines:
        return 0, None
    return len(output_lines), output_lines[0]


def _collect_run_metadata() -> dict[str, Any]:
    try:
        host = socket.gethostname().strip() or "unknown"
    except OSError:
        host = "unknown"

    version_info = getattr(sys, "version_info", None)
    if version_info is None:
        python_version = "unknown"
    else:
        python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    try:
        ray_version = metadata.version("ray")
    except metadata.PackageNotFoundError:
        ray_version = "unknown"

    gpu_count, cuda_driver = _collect_gpu_metadata()
    return {
        "host": host,
        "gpu_count": gpu_count,
        "cuda_driver": cuda_driver,
        "ray_version": ray_version,
        "python_version": python_version,
    }


def _normalize_tags(tags: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in tags or []:
        tag = str(raw).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


def _normalize_recall_metric_key(key: str) -> str:
    return normalize_metric_key(key)


def _empty_summary_metrics() -> dict[str, Any]:
    return {
        "pages": None,
        "ingest_secs": None,
        "pages_per_sec_ingest": None,
        "recall_5": None,
        "ndcg_10": None,
    }


def _safe_pdf_page_count(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(str(path))
        try:
            try:
                count = int(len(doc))
            except Exception:
                count = int(doc.get_page_count())  # type: ignore[attr-defined]
        finally:
            try:
                doc.close()
            except Exception:
                pass
        return max(count, 0)
    except Exception:
        return None


def _resolve_summary_metrics(
    cfg: HarnessConfig,
    metrics_payload: dict[str, Any],
    runtime_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    summary_metrics = {
        "pages": metrics_payload.get("pages"),
        "ingest_secs": metrics_payload.get("ingest_secs"),
        "pages_per_sec_ingest": metrics_payload.get("pages_per_sec_ingest"),
        "recall_5": metrics_payload.get("recall_5"),
        "ndcg_10": metrics_payload.get("ndcg_10"),
    }

    if summary_metrics["pages"] is None and isinstance(runtime_summary, dict):
        runtime_pages = runtime_summary.get("processed_pages")
        if runtime_pages is None:
            runtime_pages = runtime_summary.get("input_pages")
        if runtime_pages is not None:
            try:
                summary_metrics["pages"] = int(runtime_pages)
            except (TypeError, ValueError):
                summary_metrics["pages"] = None

    if summary_metrics["pages"] is None and cfg.input_type == "pdf":
        total_pages = 0
        counted_any = False
        for path in resolve_input_files(Path(cfg.dataset_dir), cfg.input_type):
            page_count = _safe_pdf_page_count(path)
            if page_count is None:
                continue
            counted_any = True
            total_pages += page_count
        if counted_any:
            summary_metrics["pages"] = total_pages

    if summary_metrics["pages_per_sec_ingest"] is None:
        pages = summary_metrics.get("pages")
        ingest_secs = summary_metrics.get("ingest_secs")
        if pages is not None and ingest_secs not in {None, 0, 0.0}:
            try:
                summary_metrics["pages_per_sec_ingest"] = round(float(pages) / float(ingest_secs), 2)
            except (TypeError, ValueError, ZeroDivisionError):
                summary_metrics["pages_per_sec_ingest"] = None

    return summary_metrics


def _resolve_lancedb_uri(cfg: HarnessConfig, artifact_dir: Path) -> str:
    raw = str(cfg.lancedb_uri or "lancedb")
    if raw == "lancedb":
        return str((artifact_dir / "lancedb").resolve())
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return str(p)


def _parse_gpu_devices(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _prepare_run_paths(
    cfg: HarnessConfig,
    artifact_dir: Path,
) -> tuple[Path, Path | None, Path | None]:
    runtime_dir = artifact_dir / "runtime_metrics"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    detection_summary_file = (
        artifact_dir / "detection_summary.json" if cfg.write_detection_file else runtime_dir / ".detection_summary.json"
    )
    effective_query_csv: Path | None = None
    if cfg.evaluation_mode != "beir":
        effective_query_csv = prepare_recall_query_file(
            query_csv=Path(cfg.query_csv) if cfg.query_csv else None,
            recall_adapter=cfg.recall_adapter,
            output_dir=runtime_dir,
        )
    return runtime_dir, detection_summary_file, effective_query_csv


def _build_command(
    cfg: HarnessConfig,
    artifact_dir: Path,
    run_id: str,
) -> tuple[list[str], Path, Path | None, Path | None]:
    runtime_dir, detection_summary_file, effective_query_csv = _prepare_run_paths(cfg, artifact_dir)
    lancedb_uri = _resolve_lancedb_uri(cfg, artifact_dir)
    module_name = {
        "batch": "nemo_retriever.examples.batch_pipeline",
        "inprocess": "nemo_retriever.examples.inprocess_pipeline",
    }[cfg.run_mode]
    cmd = [sys.executable, "-m", module_name, str(Path(cfg.dataset_dir).resolve())]

    if cfg.input_type != "pdf":
        cmd += ["--input-type", cfg.input_type]

    if cfg.run_mode == "batch":
        cmd += [
            "--evaluation-mode",
            cfg.evaluation_mode,
            "--pdf-extract-tasks",
            str(cfg.pdf_extract_workers),
            "--pdf-extract-cpus-per-task",
            str(cfg.pdf_extract_num_cpus),
            "--pdf-extract-batch-size",
            str(cfg.pdf_extract_batch_size),
            "--pdf-split-batch-size",
            str(cfg.pdf_split_batch_size),
            "--page-elements-actors",
            str(cfg.page_elements_workers),
            "--page-elements-batch-size",
            str(cfg.page_elements_batch_size),
            "--ocr-actors",
            str(cfg.ocr_workers),
            "--ocr-batch-size",
            str(cfg.ocr_batch_size),
            "--embed-actors",
            str(cfg.embed_workers),
            "--embed-batch-size",
            str(cfg.embed_batch_size),
            "--page-elements-cpus-per-actor",
            str(cfg.page_elements_cpus_per_actor),
            "--ocr-cpus-per-actor",
            str(cfg.ocr_cpus_per_actor),
            "--embed-cpus-per-actor",
            str(cfg.embed_cpus_per_actor),
            "--page-elements-gpus-per-actor",
            str(cfg.gpu_page_elements),
            "--ocr-gpus-per-actor",
            str(cfg.gpu_ocr),
            "--embed-gpus-per-actor",
            str(cfg.gpu_embed),
            "--embed-model-name",
            cfg.embed_model_name,
            "--embed-modality",
            cfg.embed_modality,
            "--embed-granularity",
            cfg.embed_granularity,
            "--runtime-metrics-dir",
            str(runtime_dir),
            "--runtime-metrics-prefix",
            run_id,
            "--lancedb-uri",
            lancedb_uri,
        ]
        cmd += ["--detection-summary-file", str(detection_summary_file)]
        if cfg.evaluation_mode == "beir":
            cmd += [
                "--beir-loader",
                str(cfg.beir_loader),
                "--beir-dataset-name",
                str(cfg.beir_dataset_name or cfg.dataset_label),
                "--beir-split",
                cfg.beir_split,
                "--beir-doc-id-field",
                cfg.beir_doc_id_field,
            ]
            if cfg.beir_query_language:
                cmd += ["--beir-query-language", cfg.beir_query_language]
            for k in cfg.beir_ks:
                cmd += ["--beir-k", str(int(k))]
        elif effective_query_csv is not None:
            cmd += ["--query-csv", str(effective_query_csv), "--recall-match-mode", cfg.recall_match_mode]
        cmd += ["--extract-page-as-image" if cfg.extract_page_as_image else "--no-extract-page-as-image"]
        if cfg.extract_infographics:
            cmd += ["--extract-infographics"]
        if cfg.embed_modality:
            cmd += ["--structured-elements-modality", cfg.embed_modality]
        if cfg.hybrid:
            cmd += ["--hybrid"]
        if cfg.ray_address:
            cmd += ["--ray-address", cfg.ray_address]
    elif cfg.run_mode == "inprocess":
        cmd += ["--max-workers", str(cfg.max_workers)]
        if cfg.gpu_devices:
            cmd += ["--gpu-devices", cfg.gpu_devices]
        if effective_query_csv is not None:
            cmd += ["--query-csv", str(effective_query_csv)]
        if cfg.hybrid:
            cmd += ["--hybrid"]
        cmd += ["--embed-modality", cfg.embed_modality, "--embed-granularity", cfg.embed_granularity]
    else:
        raise ValueError(f"Unsupported run_mode: {cfg.run_mode}")

    return cmd, runtime_dir, detection_summary_file, effective_query_csv


def _common_evaluation_config(cfg: HarnessConfig, effective_query_csv: Path | None) -> RunEvaluationConfig:
    return RunEvaluationConfig(
        evaluation_mode=cfg.evaluation_mode,
        query_csv=str(effective_query_csv) if effective_query_csv is not None else None,
        recall_match_mode=cfg.recall_match_mode,
        beir_loader=cfg.beir_loader,
        beir_dataset_name=cfg.beir_dataset_name or cfg.dataset_label,
        beir_split=cfg.beir_split,
        beir_query_language=cfg.beir_query_language,
        beir_doc_id_field=cfg.beir_doc_id_field,
        beir_ks=tuple(cfg.beir_ks),
    )


def _common_extract_params(cfg: HarnessConfig, *, batch_tuning: dict[str, Any] | None = None) -> ExtractParams:
    kwargs: dict[str, Any] = {
        "extract_text": True,
        "extract_tables": True,
        "extract_charts": True,
        "extract_infographics": cfg.extract_infographics,
        "extract_page_as_image": cfg.extract_page_as_image,
    }
    if batch_tuning is not None:
        kwargs["batch_tuning"] = batch_tuning
    return ExtractParams(**kwargs)


def _common_embed_params(
    cfg: HarnessConfig,
    *,
    batch_tuning: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> EmbedParams:
    kwargs: dict[str, Any] = {
        "model_name": model_name or cfg.embed_model_name,
        "embed_modality": cfg.embed_modality,
        "embed_granularity": cfg.embed_granularity,
    }
    if batch_tuning is not None:
        kwargs["batch_tuning"] = batch_tuning
    return EmbedParams(**kwargs)


def _build_runner_config(
    cfg: HarnessConfig,
    artifact_dir: Path,
    run_id: str,
    runtime_dir: Path,
    detection_summary_file: Path | None,
    effective_query_csv: Path | None,
):
    lancedb_uri = _resolve_lancedb_uri(cfg, artifact_dir)
    artifacts = RunArtifactConfig(
        lancedb_uri=lancedb_uri,
        lancedb_table=DEFAULT_LANCEDB_TABLE,
        detection_summary_file=str(detection_summary_file) if detection_summary_file is not None else None,
    )
    evaluation = _common_evaluation_config(cfg, effective_query_csv)
    runtime_metrics_dir = str(runtime_dir)
    text_chunk_params = TextChunkParams()

    if cfg.run_mode == "batch":
        batch_tuning = {
            "pdf_extract_workers": cfg.pdf_extract_workers,
            "pdf_extract_num_cpus": cfg.pdf_extract_num_cpus,
            "pdf_extract_batch_size": cfg.pdf_extract_batch_size,
            "pdf_split_batch_size": cfg.pdf_split_batch_size,
            "page_elements_batch_size": cfg.page_elements_batch_size,
            "page_elements_workers": cfg.page_elements_workers,
            "detect_workers": cfg.ocr_workers,
            "ocr_inference_batch_size": cfg.ocr_batch_size,
            "detect_batch_size": cfg.ocr_batch_size,
            "embed_workers": cfg.embed_workers,
            "embed_batch_size": cfg.embed_batch_size,
            "page_elements_cpus_per_actor": cfg.page_elements_cpus_per_actor,
            "ocr_cpus_per_actor": cfg.ocr_cpus_per_actor,
            "embed_cpus_per_actor": cfg.embed_cpus_per_actor,
            "gpu_page_elements": cfg.gpu_page_elements,
            "gpu_ocr": cfg.gpu_ocr,
            "gpu_embed": cfg.gpu_embed,
        }
        extract_params = (
            None if cfg.input_type in {"txt", "html"} else _common_extract_params(cfg, batch_tuning=batch_tuning)
        )
        return BatchPipelineConfig(
            input_path=cfg.dataset_dir,
            input_type=cfg.input_type,
            create_params=IngestorCreateParams(ray_address=cfg.ray_address, ray_log_to_driver=True, debug=False),
            execute_params=IngestExecuteParams(
                runtime_metrics_dir=runtime_metrics_dir,
                runtime_metrics_prefix=run_id,
            ),
            extract_params=extract_params,
            embed_params=_common_embed_params(cfg, batch_tuning=batch_tuning),
            text_chunk_params=text_chunk_params,
            enable_text_chunk=False,
            evaluation=evaluation,
            artifacts=artifacts,
            hybrid=cfg.hybrid,
        )

    if cfg.run_mode == "inprocess":
        return InProcessPipelineConfig(
            input_path=cfg.dataset_dir,
            input_type=cfg.input_type,
            execute_params=IngestExecuteParams(
                parallel=True,
                max_workers=cfg.max_workers,
                gpu_devices=_parse_gpu_devices(cfg.gpu_devices),
                show_progress=False,
                runtime_metrics_dir=runtime_metrics_dir,
                runtime_metrics_prefix=run_id,
            ),
            extract_params=_common_extract_params(cfg),
            embed_params=_common_embed_params(cfg),
            text_chunk_params=text_chunk_params,
            enable_text_chunk=False,
            vdb_upload_params=VdbUploadParams(
                lancedb={
                    "lancedb_uri": lancedb_uri,
                    "table_name": DEFAULT_LANCEDB_TABLE,
                    "overwrite": True,
                    "create_index": True,
                    "hybrid": cfg.hybrid,
                }
            ),
            evaluation=evaluation,
            artifacts=artifacts,
        )

    raise ValueError(f"Unsupported run_mode: {cfg.run_mode}")


def _execute_runner(cfg: HarnessConfig, runner_cfg):
    if cfg.run_mode == "batch":
        return run_batch_pipeline(runner_cfg)
    if cfg.run_mode == "inprocess":
        return run_inprocess_pipeline(runner_cfg)
    raise ValueError(f"Unsupported run_mode: {cfg.run_mode}")


def _evaluate_run_outcome(
    process_rc: int | None = None,
    evaluation_mode: str | None = None,
    recall_required: bool | None = None,
    recall_metrics: dict[str, float] | None = None,
    evaluation_metrics: dict[str, float] | None = None,
    *,
    runner_error: Exception | None = None,
) -> tuple[int, str, bool]:
    if runner_error is not None:
        return 1, f"runner_exception_{type(runner_error).__name__}", False
    if process_rc not in {None, 0}:
        return int(process_rc), f"subprocess_exit_{int(process_rc)}", False

    metrics = evaluation_metrics or recall_metrics or {}
    if evaluation_mode == "beir" and not metrics:
        return 97, "missing_beir_metrics", False
    if evaluation_mode == "recall" and recall_required and not metrics:
        return 98, "missing_recall_metrics", False
    return 0, "", True


def _run_single(cfg: HarnessConfig, artifact_dir: Path, run_id: str, tags: list[str] | None = None) -> dict[str, Any]:
    cmd, runtime_dir, detection_summary_file, effective_query_csv = _build_command(cfg, artifact_dir, run_id)
    command_text = " ".join(shlex.quote(token) for token in cmd)
    # Persist a reproducible CLI equivalent for debugging and handoff notes.
    # The harness executes below via the structured runner config.
    (artifact_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

    typer.echo(f"\n=== Running {run_id} ===")
    typer.echo(command_text)

    run_metadata = _collect_run_metadata()
    runner_error: Exception | None = None
    report = None
    try:
        runner_cfg = _build_runner_config(
            cfg, artifact_dir, run_id, runtime_dir, detection_summary_file, effective_query_csv
        )
        report = _execute_runner(cfg, runner_cfg)
    except Exception as exc:
        runner_error = exc
        typer.echo(f"Run failed: {type(exc).__name__}: {exc}", err=True)

    metrics_payload = flatten_report_metrics(report) if report is not None else {}
    runtime_summary = report.runtime_summary if report is not None else None
    summary_metrics = (
        _resolve_summary_metrics(cfg, metrics_payload, runtime_summary)
        if report is not None
        else _empty_summary_metrics()
    )
    detection_summary = report.detection_summary if report is not None else None
    evaluation_metrics = dict(report.evaluation.metrics) if report is not None else {}
    effective_rc, failure_reason, success = _evaluate_run_outcome(
        process_rc=0,
        evaluation_mode=cfg.evaluation_mode,
        recall_required=bool(cfg.recall_required),
        recall_metrics=evaluation_metrics,
        evaluation_metrics=evaluation_metrics,
        runner_error=runner_error,
    )

    result_payload: dict[str, Any] = {
        "timestamp": now_timestr(),
        "latest_commit": last_commit(),
        "success": success,
        "return_code": effective_rc,
        "failure_reason": failure_reason or None,
        "test_config": {
            "dataset_label": cfg.dataset_label,
            "dataset_dir": cfg.dataset_dir,
            "preset": cfg.preset,
            "run_mode": cfg.run_mode,
            "query_csv": cfg.query_csv,
            "effective_query_csv": str(effective_query_csv) if effective_query_csv is not None else None,
            "input_type": cfg.input_type,
            "recall_required": cfg.recall_required,
            "recall_match_mode": cfg.recall_match_mode,
            "recall_adapter": cfg.recall_adapter,
            "evaluation_mode": cfg.evaluation_mode,
            "beir_loader": cfg.beir_loader,
            "beir_dataset_name": cfg.beir_dataset_name,
            "beir_split": cfg.beir_split,
            "beir_query_language": cfg.beir_query_language,
            "beir_doc_id_field": cfg.beir_doc_id_field,
            "beir_ks": list(cfg.beir_ks),
            "ray_address": cfg.ray_address,
            "hybrid": cfg.hybrid,
            "embed_model_name": cfg.embed_model_name,
            "embed_modality": cfg.embed_modality,
            "embed_granularity": cfg.embed_granularity,
            "extract_page_as_image": cfg.extract_page_as_image,
            "extract_infographics": cfg.extract_infographics,
            "write_detection_file": cfg.write_detection_file,
            "lancedb_uri": _resolve_lancedb_uri(cfg, artifact_dir),
            "tuning": {field: getattr(cfg, field) for field in sorted(tuning_fields_for_run_mode(cfg.run_mode))},
        },
        "metrics": metrics_payload,
        "summary_metrics": summary_metrics,
        "run_metadata": run_metadata,
        "runtime_summary": runtime_summary,
        "detection_summary": detection_summary,
        "artifacts": {
            "command_file": str((artifact_dir / "command.txt").resolve()),
            "runtime_metrics_dir": str(runtime_dir.resolve()),
        },
    }
    if report is not None:
        result_payload["run_report"] = report.model_dump(mode="python")
        if report.artifacts.report_file:
            result_payload["artifacts"]["mode_run_report_file"] = report.artifacts.report_file
        if report.artifacts.runtime_summary_file:
            result_payload["artifacts"]["runtime_summary_file"] = report.artifacts.runtime_summary_file
        if cfg.write_detection_file and report.artifacts.detection_summary_file:
            result_payload["artifacts"]["detection_summary_file"] = report.artifacts.detection_summary_file
    if runner_error is not None:
        result_payload["error"] = {"type": type(runner_error).__name__, "message": str(runner_error)}
    if tags:
        result_payload["tags"] = list(tags)

    write_json(artifact_dir / "results.json", result_payload)
    return result_payload


def _run_entry(
    *,
    run_name: str | None,
    config_file: str | None,
    session_dir: Path | None,
    dataset: str | None,
    preset: str | None,
    sweep_overrides: dict[str, Any] | None = None,
    cli_overrides: list[str] | None = None,
    recall_required: bool | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    cfg = load_harness_config(
        config_file=config_file,
        dataset=dataset,
        preset=preset,
        sweep_overrides=sweep_overrides,
        cli_overrides=cli_overrides,
        cli_recall_required=recall_required,
    )

    if session_dir is None:
        artifact_dir = create_run_artifact_dir(cfg.dataset_label, run_name=run_name, base_dir=cfg.artifacts_dir)
    else:
        resolved_run_name = run_name or cfg.dataset_label
        artifact_dir = session_dir / resolved_run_name
        artifact_dir.mkdir(parents=True, exist_ok=True)

    resolved_run_name = run_name or cfg.dataset_label
    normalized_tags = _normalize_tags(tags)
    result = _run_single(cfg, artifact_dir, run_id=resolved_run_name, tags=normalized_tags)
    run_result = {
        "run_name": resolved_run_name,
        "dataset": cfg.dataset_label,
        "preset": cfg.preset,
        "artifact_dir": str(artifact_dir.resolve()),
        "success": bool(result["success"]),
        "return_code": int(result["return_code"]),
        "failure_reason": result.get("failure_reason"),
        "metrics": dict(result.get("summary_metrics", result.get("metrics", {}))),
    }
    if normalized_tags:
        run_result["tags"] = normalized_tags
    return run_result


def execute_runs(
    *,
    runs: list[dict[str, Any]],
    config_file: str | None,
    session_prefix: str,
    preset_override: str | None,
    base_artifacts_dir: str | None = None,
    tags: list[str] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    session_dir = create_session_dir(session_prefix, base_dir=base_artifacts_dir)
    run_results: list[dict[str, Any]] = []

    for idx, run in enumerate(runs):
        run_name = str(run.get("name") or f"run_{idx + 1:03d}")
        run_result = _run_entry(
            run_name=run_name,
            config_file=config_file,
            session_dir=session_dir,
            dataset=run.get("dataset"),
            preset=run.get("preset") if preset_override is None else preset_override,
            sweep_overrides=run.get("overrides") if isinstance(run.get("overrides"), dict) else run,
            recall_required=run.get("recall_required"),
            tags=tags,
        )
        run_results.append(run_result)

    return session_dir, run_results


def run_command(
    dataset: str = typer.Option(..., "--dataset", help="Dataset name from config or direct path."),
    preset: str | None = typer.Option(None, "--preset", help="Preset override."),
    config: str | None = typer.Option(None, "--config", help="Path to harness test config YAML."),
    run_name: str | None = typer.Option(None, "--run-name", help="Optional run name label."),
    override: list[str] = typer.Option([], "--override", help="Override values with KEY=VALUE."),
    tag: list[str] = typer.Option([], "--tag", help="Run tag to persist in harness artifacts. Repeatable."),
    recall_required: bool | None = typer.Option(
        None, "--recall-required/--no-recall-required", help="Override recall-required gate for this run."
    ),
) -> None:
    result = _run_entry(
        run_name=run_name,
        config_file=config,
        session_dir=None,
        dataset=dataset,
        preset=preset,
        cli_overrides=override,
        recall_required=recall_required,
        tags=tag,
    )
    typer.echo(
        f"\nResult: {'PASS' if result['success'] else 'FAIL'} | "
        f"return_code={result['return_code']} | artifact_dir={result['artifact_dir']}"
    )
    raise typer.Exit(code=0 if result["success"] else 1)


def sweep_command(
    config: str | None = typer.Option(None, "--config", help="Path to harness test config YAML."),
    runs_config: str = typer.Option(str(DEFAULT_NIGHTLY_CONFIG_PATH), "--runs-config", help="Path to sweep runs YAML."),
    preset: str | None = typer.Option(None, "--preset", help="Force preset for all sweep runs."),
    session_prefix: str = typer.Option("sweep", "--session-prefix", help="Session directory prefix."),
    tag: list[str] = typer.Option([], "--tag", help="Session tag to persist on each run. Repeatable."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print run plan without executing."),
) -> None:
    normalized_tags = _normalize_tags(tag)
    sweep_cfg = load_nightly_config(runs_config)
    runs = sweep_cfg["runs"]
    resolved_preset = preset or sweep_cfg.get("preset")
    if dry_run:
        typer.echo("Sweep dry run:")
        for idx, run in enumerate(runs):
            tag_text = f" tags={normalized_tags}" if normalized_tags else ""
            run_preset = run.get("preset") if run.get("preset") is not None else resolved_preset
            plan_line = (
                f"  {idx + 1:03d}: name={run.get('name')} "
                f"dataset={run.get('dataset')} preset={run_preset}{tag_text}"
            )
            typer.echo(plan_line)
        raise typer.Exit(code=0)

    session_dir, run_results = execute_runs(
        runs=runs,
        config_file=config,
        session_prefix=session_prefix,
        preset_override=resolved_preset,
        tags=normalized_tags,
    )
    summary_path = write_session_summary(
        session_dir,
        run_results,
        session_type="sweep",
        config_path=str(Path(runs_config).expanduser().resolve()),
    )

    typer.echo(f"\nSweep session: {session_dir}")
    typer.echo(f"Session summary: {summary_path}")
    failed = [r for r in run_results if not r["success"]]
    raise typer.Exit(code=0 if not failed else 1)
