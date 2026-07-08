# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
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
from nemo_retriever.harness.beir_runner import run_beir_queries, run_service_beir_queries
from nemo_retriever.harness.contracts import (
    EXIT_ARTIFACT_WRITE_FAILURE,
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
from nemo_retriever.harness.json_io import artifact_write_error, write_json
from nemo_retriever.harness.metrics import build_summary_metrics
from nemo_retriever.harness.metric_gates import enforce_metric_gates, parse_metric_gates
from nemo_retriever.harness.resolution import (
    build_ingest_request,
    build_query_request,
    build_service_ingest_plan_request,
    build_service_query_request,
    make_run_id,
    query_plan_payload,
    resolve_artifact_dir,
    resolve_benchmark,
    service_plan_payload,
    validate_dataset_inputs,
)
from nemo_retriever.ingest.plan import resolve_ingest_plan
from nemo_retriever.ingest.service import execute_service_ingest_request, resolve_service_ingest_request
from nemo_retriever.query.workflow import resolve_query_plan

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_EXCEPTION_LINE_RE = re.compile(r"(?:Error|Exception):")
_MAX_FAILURE_MESSAGE_CHARS = 500


@dataclass(frozen=True)
class PreparedBenchmark:
    """Validated benchmark inputs that can be executed without another preflight."""

    benchmark: str
    mode: str
    overrides: tuple[str, ...]
    requirements: tuple[str, ...]
    dry_run: bool
    resolved: dict[str, Any]
    dataset_path: Path
    service_endpoint: str | None = None
    service_ingest_request: Any | None = None
    service_query_request: Any | None = None


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


def _artifact_write_failure(exc: BaseException) -> FailurePayload:
    return FailurePayload(
        failed_phase="write_artifacts",
        failure_reason="artifact_write_failed",
        retryable=False,
        message=_concise_exception_message(exc),
        debug_artifacts=("status.json", "events.jsonl", "run.log"),
    )


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


def _failure_outcome(
    writer: ArtifactWriter,
    *,
    failure: FailurePayload,
    exit_code: int,
    dry_run: bool,
    resolved: dict[str, Any] | None,
    summary_metrics: dict[str, Any] | None,
) -> RunOutcome:
    try:
        result = _write_failure_result(
            writer,
            failure=failure,
            exit_code=exit_code,
            dry_run=dry_run,
            resolved=resolved,
            summary_metrics=summary_metrics,
        )
        results_path: Path | None = writer.path("results.json")
    except Exception as write_exc:
        exit_code = EXIT_ARTIFACT_WRITE_FAILURE
        if isinstance(write_exc, HarnessRunError) and write_exc.exit_code == EXIT_ARTIFACT_WRITE_FAILURE:
            failure = write_exc.failure
        else:
            failure = _artifact_write_failure(write_exc)
        result = _run_result_payload(
            writer,
            status="failed",
            success=False,
            exit_code=exit_code,
            dry_run=dry_run,
            resolved=resolved,
            summary_metrics=summary_metrics or {},
            failure=failure,
        )
        results_path = None
    return RunOutcome(
        exit_code=exit_code,
        artifact_dir=writer.artifact_dir,
        results=result,
        results_path=results_path,
    )


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
    service_endpoint: str | None = None,
) -> PreparedBenchmark:
    """Resolve and validate one run without creating or replacing artifacts."""
    parse_metric_gates(requirements)
    resolved = resolve_benchmark(benchmark, mode=mode, overrides=overrides)
    dataset_path, _query_path = validate_dataset_inputs(resolved, dry_run=dry_run)
    resolved_service_endpoint: str | None = None
    service_ingest_request: Any | None = None
    service_query_request: Any | None = None
    if mode == "service":
        resolved_service_endpoint = service_endpoint or "http://localhost:7670"
        api_token = os.environ.get("HARNESS_SERVICE_API_TOKEN") or None
        try:
            service_ingest_request = resolve_service_ingest_request(
                build_service_ingest_plan_request(
                    resolved,
                    dataset_path,
                    service_url=resolved_service_endpoint,
                    service_concurrency=8,
                    service_api_token=api_token,
                )
            )
            service_query_request = build_service_query_request(
                resolved,
                "",
                service_url=resolved_service_endpoint,
                service_api_token=api_token,
            )
        except FileNotFoundError as exc:
            raise HarnessRunError(
                EXIT_MISSING_INPUT,
                FailurePayload(
                    failed_phase="ingest_plan",
                    failure_reason="dataset_missing",
                    retryable=False,
                    message=str(exc),
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
                ),
            ) from exc
    return PreparedBenchmark(
        benchmark=benchmark,
        mode=mode,
        overrides=tuple(overrides),
        requirements=tuple(requirements),
        dry_run=bool(dry_run),
        resolved=resolved,
        dataset_path=dataset_path,
        service_endpoint=resolved_service_endpoint,
        service_ingest_request=service_ingest_request,
        service_query_request=service_query_request,
    )


def run_benchmark(
    benchmark: str,
    *,
    output_dir: str | None = None,
    run_id: str | None = None,
    mode: str = "local",
    overrides: Sequence[str] = (),
    requirements: Sequence[str] = (),
    dry_run: bool = False,
    service_endpoint: str | None = None,
    runfile_payload: dict[str, Any] | None = None,
    runfile_path: str | None = None,
) -> RunOutcome:
    prepared = preflight_benchmark(
        benchmark,
        mode=mode,
        overrides=overrides,
        requirements=requirements,
        dry_run=dry_run,
        service_endpoint=service_endpoint,
    )
    return run_prepared_benchmark(
        prepared,
        output_dir=output_dir,
        run_id=run_id,
        runfile_payload=runfile_payload,
        runfile_path=runfile_path,
    )


def run_prepared_benchmark(
    prepared: PreparedBenchmark,
    *,
    output_dir: str | None = None,
    run_id: str | None = None,
    runfile_payload: dict[str, Any] | None = None,
    runfile_path: str | None = None,
) -> RunOutcome:
    benchmark = prepared.benchmark
    resolved = deepcopy(prepared.resolved)
    dataset_path = prepared.dataset_path
    requirements = prepared.requirements
    dry_run = prepared.dry_run
    service_mode = prepared.mode == "service"
    effective_run_id = run_id or make_run_id(benchmark)
    artifact_dir = resolve_artifact_dir(benchmark, effective_run_id, output_dir)
    try:
        writer = ArtifactWriter(
            artifact_dir=artifact_dir,
            run_id=effective_run_id,
            benchmark=benchmark,
        )
    except OSError as exc:
        raise artifact_write_error(exc) from exc
    silence_noisy_libraries()
    summary_metrics: dict[str, Any] | None = None
    try:
        environment = collect_environment()
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
        # Keep the pre-plan config when request validation fails. build_ingest_request
        # then adds artifact-local storage paths, so the final config is written again.
        write_json(writer.path("resolved_benchmark.json"), redact(resolved))
        write_json(writer.path("environment.json"), environment)

        writer.status(status="running", phase="ingest_plan")
        if service_mode:
            ingest_request = prepared.service_ingest_request
            query_request = prepared.service_query_request
            if ingest_request is None or query_request is None:
                raise RuntimeError("Service-mode benchmark was not prepared with service requests")
            plan_payload = service_plan_payload(ingest_request, query_request)
            ingest_documents = ingest_request.documents
            write_json(writer.path("ingest_plan.json"), plan_payload)
            writer.status(status="running", phase="query_plan")
            write_json(writer.path("query_plan.json"), plan_payload["query"])
            query_plan = None
        else:
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
            ingest_documents = ingest_plan.documents
            write_json(writer.path("ingest_plan.json"), redact(ingest_plan_payload))
            writer.status(status="running", phase="query_plan")
            query_request = build_query_request(resolved, "")
            query_plan = resolve_query_plan(query_request)
            query_plan_data = query_plan_payload(query_plan)
            write_json(writer.path("query_plan.json"), query_plan_data)

        ingest_summary: dict[str, Any] | None = None
        ingest_secs: float | None = None
        query_latencies_ms: list[float] = []
        beir_metrics: dict[str, float] = {}
        query_count = 0

        if dry_run:
            writer.event("write_artifacts", "dry_run", "Dry-run completed without executing ingest or query")
        else:
            writer.status(status="running", phase="ingest")
            writer.event("ingest", "ingest_start", f"Ingesting {len(ingest_documents)} document(s)")
            ingest_start = time.perf_counter()
            try:
                with capture_output_to_log(
                    writer.path("run.log"), label="service_ingest" if service_mode else "ingest"
                ):
                    if service_mode:
                        ingest_summary = execute_service_ingest_request(ingest_request).to_summary_dict()
                    else:
                        ingest_summary = run_ingest_workflow(ingest_plan, dry_run=False)
            except Exception as exc:
                if isinstance(exc, HarnessRunError) and exc.exit_code == EXIT_ARTIFACT_WRITE_FAILURE:
                    raise
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
                    if service_mode:
                        query_latencies_ms, beir_metrics, query_count = run_service_beir_queries(
                            writer, resolved, query_request
                        )
                    else:
                        query_latencies_ms, beir_metrics, query_count = run_beir_queries(
                            writer, resolved, query_plan, query_request
                        )

        summary_metrics = build_summary_metrics(
            resolved,
            documents=ingest_documents,
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
        return RunOutcome(
            exit_code=EXIT_SUCCESS,
            artifact_dir=writer.artifact_dir,
            results=result,
            results_path=writer.path("results.json"),
        )
    except HarnessRunError as exc:
        return _failure_outcome(
            writer,
            failure=exc.failure,
            exit_code=exc.exit_code,
            dry_run=dry_run,
            resolved=resolved,
            summary_metrics=summary_metrics,
        )
    except Exception as exc:
        try:
            append_text(writer.path("run.log"), f"\n## {traceback.format_exc()}")
        except Exception:
            pass
        failure = FailurePayload(
            failed_phase="write_artifacts",
            failure_reason="unexpected_internal_error",
            retryable=False,
            message=_concise_exception_message(exc),
            debug_artifacts=("status.json", "events.jsonl", "run.log"),
        )
        return _failure_outcome(
            writer,
            failure=failure,
            exit_code=EXIT_INTERNAL_ERROR,
            dry_run=dry_run,
            resolved=resolved,
            summary_metrics=summary_metrics,
        )
