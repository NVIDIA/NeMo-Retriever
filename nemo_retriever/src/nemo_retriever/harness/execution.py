# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from nemo_retriever.cli.ingest_workflow import run_ingest_workflow
from nemo_retriever.cli.shared import silence_noisy_libraries
from nemo_retriever.harness.artifact_writer import (
    append_text,
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
from nemo_retriever.ingest.plan import resolve_ingest_plan
from nemo_retriever.query.workflow import resolve_query_plan

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_EXCEPTION_LINE_RE = re.compile(r"(?:Error|Exception):")
_MAX_FAILURE_MESSAGE_CHARS = 500


def _concise_message(message: str) -> str:
    plain = _ANSI_ESCAPE_RE.sub("", str(message))
    lines = [line.strip() for line in plain.splitlines() if line.strip()]
    exception_lines = [line for line in lines if _EXCEPTION_LINE_RE.search(line)]
    selected = exception_lines[-1] if exception_lines else lines[0] if lines else "Unknown harness failure"
    if len(selected) <= _MAX_FAILURE_MESSAGE_CHARS:
        return selected
    return selected[: _MAX_FAILURE_MESSAGE_CHARS - 3].rstrip() + "..."


def _concise_exception_message(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {_concise_message(str(exc))}"


def _concise_failure(failure: FailurePayload) -> FailurePayload:
    return FailurePayload(
        failed_phase=failure.failed_phase,
        failure_reason=failure.failure_reason,
        retryable=failure.retryable,
        message=_concise_message(failure.message),
        debug_artifacts=failure.debug_artifacts,
    )


def _result_identity(resolved: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(resolved, dict):
        return {}
    identity: dict[str, str] = {}
    dataset = resolved.get("dataset")
    if isinstance(dataset, dict) and dataset.get("name"):
        identity["dataset"] = str(dataset["name"])
    ingest = resolved.get("ingest")
    if isinstance(ingest, dict) and ingest.get("run_mode"):
        run_mode = str(ingest["run_mode"])
        identity["mode"] = "local" if run_mode == "inprocess" else run_mode
    return identity


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
        **_result_identity(resolved),
        "status": status,
        "success": success,
        "exit_code": exit_code,
        "dry_run": bool(dry_run),
        "summary_metrics": summary_metrics,
        "failure": failure.to_dict() if failure is not None else None,
        "artifacts": artifact_paths(writer),
    }
    result.update(extra)
    return redact(result)


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
    failure = _concise_failure(failure)
    result = _run_result_payload(
        writer,
        status="failed",
        success=False,
        exit_code=exit_code,
        dry_run=dry_run,
        resolved=resolved,
        summary_metrics=summary_metrics,
        failure=failure,
    )
    write_json(writer.path("results.json"), result)
    writer.status(
        status="failed",
        phase=failure.failed_phase if failure.failed_phase in PHASE_VALUES else "write_artifacts",
        failure=failure,
        results_path=writer.path("results.json"),
    )
    return result


def _mark_dry_run_metrics_unavailable(summary_metrics: dict[str, Any]) -> None:
    static_keys = {"files", "pages"}
    for key in list(summary_metrics):
        if key not in static_keys:
            summary_metrics[key] = None


def preflight_benchmark(
    benchmark: str,
    *,
    mode: str,
    overrides: Sequence[str],
    requirements: Sequence[str],
    dry_run: bool,
) -> tuple[dict[str, Any], Path]:
    """Resolve and validate one run without creating or replacing artifacts."""
    parse_metric_gates(requirements)
    resolved = resolve_benchmark(benchmark, mode=mode, overrides=overrides)
    dataset_path, _query_path = validate_dataset_inputs(resolved, dry_run=dry_run)
    return resolved, dataset_path


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
    resolved, dataset_path = preflight_benchmark(
        benchmark,
        mode=mode,
        overrides=overrides,
        requirements=requirements,
        dry_run=dry_run,
    )
    effective_run_id = run_id or make_run_id(benchmark)
    writer = ArtifactWriter(
        artifact_dir=resolve_artifact_dir(benchmark, effective_run_id, output_dir),
        run_id=effective_run_id,
        benchmark=benchmark,
    )
    silence_noisy_libraries()
    summary_metrics: dict[str, Any] | None = None
    try:
        writer.status(status="planned", phase="resolve")
        if runfile_payload is not None:
            write_json(
                writer.path("runfile.json"),
                redact(
                    {
                        "source_path": runfile_path,
                        "payload": runfile_payload,
                    }
                ),
            )
        write_json(writer.path("resolved_benchmark.json"), redact(resolved))
        write_json(writer.path("environment.json"), collect_environment())

        writer.status(status="running", phase="ingest_plan")
        ingest_request = build_ingest_request(resolved, dataset_path, writer.artifact_dir)
        write_json(writer.path("resolved_benchmark.json"), redact(resolved))
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
                        message=_concise_exception_message(exc),
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
            metric_gates=list(requirements),
            skipped_metric_gates=list(skipped_metric_gates),
        )
        write_json(writer.path("results.json"), result)
        writer.status(
            status="complete",
            phase="write_artifacts",
            results_path=writer.path("results.json"),
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
        return RunOutcome(exit_code=exc.exit_code, artifact_dir=writer.artifact_dir, results=result)
    except Exception as exc:
        append_text(writer.path("run.log"), f"\n## {traceback.format_exc()}")
        failure = FailurePayload(
            failed_phase="write_artifacts",
            failure_reason="unexpected_internal_error",
            retryable=False,
            message=_concise_exception_message(exc),
            debug_artifacts=("status.json", "events.jsonl", "run.log"),
        )
        result = _write_failure_result(
            writer,
            failure=failure,
            exit_code=EXIT_INTERNAL_ERROR,
            dry_run=dry_run,
            resolved=resolved,
            summary_metrics=summary_metrics,
        )
        return RunOutcome(exit_code=EXIT_INTERNAL_ERROR, artifact_dir=writer.artifact_dir, results=result)
