# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import Any, Sequence

from nemo_retriever.cli.ingest_workflow import run_ingest_workflow
from nemo_retriever.cli.shared import silence_noisy_libraries
from nemo_retriever.harness.artifact_writer import (
    artifact_paths,
    ArtifactWriter,
    capture_output_to_log,
    redact,
)
from nemo_retriever.harness.beir_runner import run_beir_queries
from nemo_retriever.harness.contracts import (
    EXIT_INGEST_FAILURE,
    EXIT_INTERNAL_ERROR,
    EXIT_INVALID,
    EXIT_MISSING_INPUT,
    EXIT_SUCCESS,
    FailurePayload,
    HarnessRunError,
    PHASE_VALUES,
    RunOutcome,
)
from nemo_retriever.harness.environment import collect_environment
from nemo_retriever.harness.json_io import write_json
from nemo_retriever.harness.metrics import build_summary_metrics
from nemo_retriever.harness.metric_gates import enforce_metric_gates, parse_metric_gates
from nemo_retriever.harness.resolution import (
    build_ingest_request,
    build_query_request,
    make_run_id,
    query_plan_payload,
    resolve_artifact_dir,
    resolve_benchmark,
    validate_dataset_inputs,
)
from nemo_retriever.harness.replay_command import build_harness_run_command, persist_replay_command
from nemo_retriever.ingest.plan import resolve_ingest_plan
from nemo_retriever.query.workflow import resolve_query_plan


def _run_result_payload(
    writer: ArtifactWriter,
    *,
    status: str,
    success: bool,
    exit_code: int,
    dry_run: bool,
    resolved: dict[str, Any] | None,
    summary_metrics: dict[str, Any],
    failure: FailurePayload | None,
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "run_id": writer.run_id,
        "benchmark": writer.benchmark,
        "status": status,
        "success": success,
        "exit_code": exit_code,
        "dry_run": bool(dry_run),
        "resolved_benchmark": resolved,
        "summary_metrics": summary_metrics,
        "failure": failure.to_dict() if failure is not None else None,
        "artifacts": artifact_paths(writer),
    }
    result.update(extra)
    return result


def _write_failure_result(
    writer: ArtifactWriter,
    *,
    failure: FailurePayload,
    exit_code: int,
    dry_run: bool,
    resolved: dict[str, Any] | None,
    summary_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    if summary_metrics is None:
        summary_metrics = {}
    try:
        write_json(writer.path("summary_metrics.json"), summary_metrics)
    except Exception:
        pass
    status_payload = writer.status(
        status="failed",
        phase=failure.failed_phase if failure.failed_phase in PHASE_VALUES else "write_artifacts",
        failure=failure,
        summary_metrics_path=writer.path("summary_metrics.json"),
    )
    result = _run_result_payload(
        writer,
        status="failed",
        success=False,
        exit_code=exit_code,
        dry_run=dry_run,
        resolved=resolved,
        summary_metrics=summary_metrics,
        failure=failure,
        status_payload=status_payload,
    )
    write_json(writer.path("results.json"), result)
    return result


def _mark_dry_run_metrics_unavailable(summary_metrics: dict[str, Any]) -> None:
    static_keys = {"files", "pages"}
    for key in list(summary_metrics):
        if key not in static_keys:
            summary_metrics[key] = None


def _merge_replay_metadata(result: dict[str, Any], replay_meta: dict[str, str]) -> dict[str, Any]:
    result["replay_command"] = replay_meta["replay_command"]
    artifacts = dict(result.get("artifacts") or {})
    artifacts.update(replay_meta)
    result["artifacts"] = artifacts
    return result


def run_benchmark(
    benchmark: str,
    *,
    output_dir: str | None = None,
    run_id: str | None = None,
    mode: str = "local",
    overrides: Sequence[str] = (),
    requirements: Sequence[str] = (),
    dry_run: bool = False,
    runfile_payload: dict[str, Any] | None = None,
    runfile_path: str | None = None,
) -> RunOutcome:
    effective_run_id = run_id or make_run_id(benchmark)
    writer = ArtifactWriter(
        artifact_dir=resolve_artifact_dir(benchmark, effective_run_id, output_dir),
        run_id=effective_run_id,
        benchmark=benchmark,
    )
    replay_meta = persist_replay_command(
        writer.artifact_dir,
        build_harness_run_command(
            benchmark,
            output_dir=writer.artifact_dir,
            run_id=effective_run_id,
            mode=mode,
            overrides=overrides,
            requirements=requirements,
            dry_run=dry_run,
        ),
    )
    silence_noisy_libraries()
    resolved: dict[str, Any] | None = None
    summary_metrics: dict[str, Any] | None = None
    try:
        writer.status(status="planned", phase="resolve")
        if runfile_payload is not None:
            write_json(
                writer.path("runfile.json"),
                {
                    "source_path": runfile_path,
                    "payload": runfile_payload,
                },
            )
        parse_metric_gates(requirements)
        resolved = resolve_benchmark(benchmark, mode=mode, overrides=overrides)
        dataset_path, _query_path = validate_dataset_inputs(resolved, dry_run=dry_run)
        write_json(writer.path("environment.json"), collect_environment())

        writer.status(status="running", phase="ingest_plan")
        ingest_request = build_ingest_request(resolved, dataset_path, writer.artifact_dir)
        try:
            ingest_plan = resolve_ingest_plan(ingest_request)
            ingest_plan_payload = run_ingest_workflow(ingest_plan, dry_run=True)
        except FileNotFoundError as exc:
            raise HarnessRunError(
                EXIT_MISSING_INPUT,
                FailurePayload(
                    failed_phase="ingest_plan",
                    failure_reason="dataset_missing",
                    retryable=False,
                    message=str(exc),
                    debug_artifacts=("resolved_benchmark.json",),
                ),
            ) from exc
        except ValueError as exc:
            raise HarnessRunError(
                EXIT_INVALID,
                FailurePayload(
                    failed_phase="ingest_plan",
                    failure_reason="ingest_plan_failed",
                    retryable=False,
                    message=str(exc),
                    debug_artifacts=("resolved_benchmark.json",),
                ),
            ) from exc
        write_json(writer.path("ingest_plan.json"), redact(ingest_plan_payload))

        writer.status(status="running", phase="query_plan")
        query_request = build_query_request(resolved, "")
        query_plan = resolve_query_plan(query_request)
        write_json(writer.path("query_plan.json"), query_plan_payload(query_plan))
        write_json(writer.path("resolved_benchmark.json"), resolved)

        ingest_summary: dict[str, Any] | None = None
        ingest_secs: float | None = None
        query_latencies_ms: list[float] = []
        beir_metrics: dict[str, float] = {}
        query_count = 0

        if dry_run:
            writer.event("write_artifacts", "dry_run", "Dry-run completed without executing ingest or query")
        else:
            writer.status(status="running", phase="ingest")
            writer.event("ingest", "ingest_start", f"Ingesting {len(ingest_plan.documents)} document(s)")
            ingest_start = time.perf_counter()
            try:
                with capture_output_to_log(writer.path("run.log"), label="ingest"):
                    ingest_summary = run_ingest_workflow(ingest_plan, dry_run=False)
            except Exception as exc:
                raise HarnessRunError(
                    EXIT_INGEST_FAILURE,
                    FailurePayload(
                        failed_phase="ingest",
                        failure_reason="ingest_failed",
                        retryable=False,
                        message=str(exc),
                        debug_artifacts=("ingest_plan.json", "run.log"),
                    ),
                ) from exc
            ingest_secs = round(time.perf_counter() - ingest_start, 3)

            if (resolved.get("evaluation") or {}).get("mode") == "beir":
                with capture_output_to_log(writer.path("run.log"), label="query_evaluate"):
                    query_latencies_ms, beir_metrics, query_count = run_beir_queries(writer, resolved, query_plan)

        summary_metrics = build_summary_metrics(
            resolved,
            documents=ingest_plan.documents,
            ingest_summary=ingest_summary,
            ingest_secs=ingest_secs,
            query_latencies_ms=query_latencies_ms,
            beir_metrics=beir_metrics,
        )
        if query_count and summary_metrics.get("query_count") == 0:
            summary_metrics["query_count"] = query_count
        if dry_run:
            _mark_dry_run_metrics_unavailable(summary_metrics)

        writer.status(status="running", phase="write_artifacts")
        write_json(writer.path("summary_metrics.json"), summary_metrics)
        skipped_metric_gates = enforce_metric_gates(summary_metrics, requirements, skip_missing=dry_run)
        result = _run_result_payload(
            writer,
            status="complete",
            success=True,
            exit_code=EXIT_SUCCESS,
            dry_run=dry_run,
            resolved=resolved,
            summary_metrics=summary_metrics,
            failure=None,
            ingest_summary=ingest_summary,
            beir_metrics=beir_metrics,
            metric_gates=list(requirements),
            skipped_metric_gates=list(skipped_metric_gates),
            runfile={"source_path": runfile_path, "payload": runfile_payload} if runfile_payload is not None else None,
        )
        _merge_replay_metadata(result, replay_meta)
        write_json(writer.path("results.json"), result)
        writer.status(
            status="complete",
            phase="write_artifacts",
            summary_metrics_path=writer.path("summary_metrics.json"),
        )
        return RunOutcome(exit_code=EXIT_SUCCESS, artifact_dir=writer.artifact_dir, results=result)
    except HarnessRunError as exc:
        result = _write_failure_result(
            writer,
            failure=exc.failure,
            exit_code=exc.exit_code,
            dry_run=dry_run,
            resolved=resolved,
            summary_metrics=summary_metrics,
        )
        _merge_replay_metadata(result, replay_meta)
        write_json(writer.path("results.json"), result)
        return RunOutcome(exit_code=exc.exit_code, artifact_dir=writer.artifact_dir, results=result)
    except Exception as exc:
        failure = FailurePayload(
            failed_phase="write_artifacts",
            failure_reason="unexpected_internal_error",
            retryable=False,
            message=str(exc),
            debug_artifacts=("status.json", "events.jsonl"),
        )
        result = _write_failure_result(
            writer,
            failure=failure,
            exit_code=EXIT_INTERNAL_ERROR,
            dry_run=dry_run,
            resolved=resolved,
            summary_metrics=summary_metrics,
        )
        _merge_replay_metadata(result, replay_meta)
        write_json(writer.path("results.json"), result)
        return RunOutcome(exit_code=EXIT_INTERNAL_ERROR, artifact_dir=writer.artifact_dir, results=result)
