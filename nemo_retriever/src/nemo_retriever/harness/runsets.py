# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
import multiprocessing
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from nemo_retriever.harness.artifact_writer import redact, utc_now
from nemo_retriever.harness.artifacts import get_artifacts_root, last_commit, now_timestr, working_tree_dirty
from nemo_retriever.harness.benchmark_registry import get_benchmark, get_runset, runset_names
from nemo_retriever.harness.contracts import (
    EXIT_ARTIFACT_WRITE_FAILURE,
    EXIT_INTERNAL_ERROR,
    EXIT_INVALID,
    EXIT_SUCCESS,
    FailurePayload,
    HarnessRunError,
    PHASE_VALUES,
    RunOutcome,
)
from nemo_retriever.harness.dataset_paths import load_dataset_paths
from nemo_retriever.harness.execution import PreparedBenchmark, preflight_benchmark, run_prepared_benchmark
from nemo_retriever.harness.json_io import artifact_write_error, read_json_object, write_json
from nemo_retriever.harness.resolution import make_run_id, resolve_artifact_dir
from nemo_retriever.harness.runfile import load_runfile

_SESSION_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ISOLATED_CHILD_TIMEOUT_SECONDS = 6 * 60 * 60
_ISOLATED_PROCESS_STOP_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class PreparedRun:
    """One preflighted child in a runset or runfile session."""

    name: str
    artifact_name: str
    prepared: PreparedBenchmark
    runfile_payload: dict[str, Any] | None = None
    runfile_path: str | None = None
    summary_mode: str | None = None
    timeout_seconds: float | None = None


def _invalid_runfile(message: str) -> HarnessRunError:
    return HarnessRunError(
        EXIT_INVALID,
        FailurePayload(
            failed_phase="resolve",
            failure_reason="invalid_runfile",
            retryable=False,
            message=message,
        ),
    )


def _validate_session_label(value: str, *, field: str) -> str:
    if not _SESSION_LABEL_RE.fullmatch(value):
        raise _invalid_runfile(f"{field} must contain only letters, numbers, '.', '_', and '-': {value!r}")
    return value


def _session_id(runset: str) -> str:
    return f"{runset}_{now_timestr()}"


def _session_dir(runset: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (get_artifacts_root() / _session_id(runset)).resolve()


def _remove_stale_session_summary(session_dir: Path) -> None:
    summary_path = session_dir / "session_summary.json"
    try:
        if summary_path.exists() or summary_path.is_symlink():
            summary_path.unlink()
    except OSError as exc:
        raise artifact_write_error(exc) from exc


def _runset_or_error(name: str):
    try:
        return get_runset(name)
    except KeyError as exc:
        suggestion = get_close_matches(name, runset_names(), n=1)
        suffix = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="invalid_benchmark",
                retryable=False,
                message=f"Unknown runset {name!r}.{suffix}",
            ),
        ) from exc


def _run_outcome_summary(
    benchmark: str,
    outcome: RunOutcome,
    *,
    session_dir: Path,
    run_name: str | None = None,
    runfile_path: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    failure = outcome.results.get("failure")
    artifact_dir = outcome.artifact_dir.resolve()

    def session_path(path: Path) -> str:
        resolved_path = path.resolve()
        try:
            return resolved_path.relative_to(session_dir.resolve()).as_posix()
        except ValueError:
            return str(resolved_path)

    artifact_dir_value = session_path(artifact_dir)
    payload = {
        "run_name": run_name or benchmark,
        "benchmark": benchmark,
        "artifact_dir": artifact_dir_value,
        "exit_code": outcome.exit_code,
        "success": outcome.exit_code == EXIT_SUCCESS,
        "summary_metrics": outcome.results.get("summary_metrics", {}),
    }
    if outcome.results_path is not None:
        payload["results_path"] = session_path(outcome.results_path)
    dataset = outcome.results.get("dataset")
    if not dataset:
        try:
            dataset = get_benchmark(benchmark).dataset
        except KeyError:
            dataset = None
    if dataset:
        payload["dataset"] = str(dataset)
    if runfile_path is not None:
        payload["runfile_path"] = runfile_path
    if mode is not None:
        payload["mode"] = mode
    if isinstance(failure, dict):
        payload["failure_reason"] = failure.get("message") or failure.get("failure_reason")
    return payload


def _failed_result_payload(
    *,
    benchmark: str,
    dry_run: bool,
    exit_code: int,
    failure: FailurePayload,
) -> dict[str, Any]:
    return redact(
        {
            "benchmark": benchmark,
            "status": "failed",
            "success": False,
            "exit_code": exit_code,
            "dry_run": bool(dry_run),
            "summary_metrics": {},
            "failure": failure.to_dict(),
        }
    )


def _failed_child_outcome(
    *,
    benchmark: str,
    run_id: str,
    artifact_dir: Path,
    dry_run: bool,
    exc: Exception,
) -> RunOutcome:
    if isinstance(exc, HarnessRunError):
        exit_code = exc.exit_code
        failure = exc.failure
    else:
        exit_code = EXIT_INTERNAL_ERROR
        failure = FailurePayload(
            failed_phase="resolve",
            failure_reason="unexpected_internal_error",
            retryable=False,
            message=f"{type(exc).__name__}: {exc}",
        )

    result = _failed_result_payload(
        benchmark=benchmark,
        dry_run=dry_run,
        exit_code=exit_code,
        failure=failure,
    )
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        results_path: Path | None = artifact_dir / "results.json"
        write_json(results_path, result)
        status_path = artifact_dir / "status.json"
        try:
            previous_status = read_json_object(status_path)
        except (FileNotFoundError, OSError, ValueError):
            previous_status = {}
        if previous_status.get("run_id") != run_id:
            previous_status = {}
        phase = previous_status.get("phase")
        if not isinstance(phase, str) or phase not in PHASE_VALUES:
            phase = failure.failed_phase if failure.failed_phase in PHASE_VALUES else "resolve"
        write_json(
            status_path,
            {
                "run_id": run_id,
                "benchmark": benchmark,
                "status": "failed",
                "phase": phase,
                "started_at": previous_status.get("started_at") or utc_now(),
                "updated_at": utc_now(),
                "artifact_dir": str(artifact_dir.resolve()),
                "results_path": "results.json",
                "failure": failure.to_dict(),
            },
        )
    except Exception as write_exc:
        exit_code = EXIT_ARTIFACT_WRITE_FAILURE
        if isinstance(write_exc, HarnessRunError) and write_exc.exit_code == EXIT_ARTIFACT_WRITE_FAILURE:
            failure = write_exc.failure
        else:
            failure = FailurePayload(
                failed_phase="write_artifacts",
                failure_reason="artifact_write_failed",
                retryable=False,
                message=f"{type(write_exc).__name__}: {write_exc}",
            )
        result = _failed_result_payload(
            benchmark=benchmark,
            dry_run=dry_run,
            exit_code=exit_code,
            failure=failure,
        )
        results_path = None
    return RunOutcome(
        exit_code=exit_code,
        artifact_dir=artifact_dir,
        results=result,
        results_path=results_path,
    )


def _isolated_run_worker(connection: Any, run: PreparedRun, *, output_dir: str, run_id: str) -> None:
    """Execute one prepared benchmark in a fresh spawned process."""

    try:
        outcome = run_prepared_benchmark(
            run.prepared,
            output_dir=output_dir,
            run_id=run_id,
            runfile_payload=run.runfile_payload,
            runfile_path=run.runfile_path,
        )
        connection.send(("outcome", outcome))
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _stop_isolated_process(process: multiprocessing.Process) -> None:
    """Best-effort bounded cleanup for an isolated benchmark process."""

    if process.is_alive():
        process.terminate()
    process.join(timeout=_ISOLATED_PROCESS_STOP_TIMEOUT_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(timeout=_ISOLATED_PROCESS_STOP_TIMEOUT_SECONDS)


def _run_prepared_benchmark_isolated(
    run: PreparedRun,
    *,
    output_dir: str,
    run_id: str,
) -> RunOutcome:
    """Run one child with a process boundary that releases Ray and dataframe memory."""

    # The child creates a fresh status file after startup. Clear any status from
    # a previous run before launching it so a startup failure cannot reuse it.
    try:
        (Path(output_dir) / "status.json").unlink(missing_ok=True)
    except OSError as exc:
        raise artifact_write_error(exc) from exc

    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_run_worker,
        kwargs={
            "connection": send_connection,
            "run": run,
            "output_dir": output_dir,
            "run_id": run_id,
        },
        name=f"retriever-harness-{run.artifact_name}",
    )
    try:
        process.start()
    except BaseException:
        receive_connection.close()
        send_connection.close()
        raise
    send_connection.close()

    message: tuple[str, Any] | None = None
    try:
        timeout_seconds = getattr(run, "timeout_seconds", None) or _ISOLATED_CHILD_TIMEOUT_SECONDS
        if not receive_connection.poll(timeout_seconds):
            raise TimeoutError(
                f"isolated benchmark {run.name!r} exceeded the " f"{timeout_seconds:g}-second child timeout"
            )
        try:
            message = receive_connection.recv()
        except EOFError:
            pass
    except BaseException:
        _stop_isolated_process(process)
        raise
    finally:
        receive_connection.close()

    process.join(timeout=_ISOLATED_PROCESS_STOP_TIMEOUT_SECONDS)
    if process.is_alive():
        _stop_isolated_process(process)
        raise RuntimeError(
            f"isolated benchmark {run.name!r} returned an outcome but did not exit within "
            f"{_ISOLATED_PROCESS_STOP_TIMEOUT_SECONDS} seconds"
        )
    if message is None:
        if process.exitcode != 0:
            raise RuntimeError(f"isolated benchmark process exited with code {process.exitcode}")
        raise RuntimeError("isolated benchmark process returned no outcome")

    message_type, payload = message
    if message_type == "error":
        raise RuntimeError(str(payload))
    if message_type != "outcome" or not isinstance(payload, RunOutcome):
        raise RuntimeError("isolated benchmark process returned an invalid outcome")
    return payload


def run_benchmark_with_timeout(
    benchmark: str,
    *,
    timeout_seconds: float,
    output_dir: str | None = None,
    run_id: str | None = None,
    mode: str = "local",
    overrides: Sequence[str] = (),
    requirements: Sequence[str] = (),
    service_endpoint: str | None = None,
    runfile_payload: dict[str, Any] | None = None,
    runfile_path: str | None = None,
) -> RunOutcome:
    """Run one benchmark in an isolated process with a deadline.

    A child timeout or process failure produces a failed outcome and writes
    ``results.json`` and ``status.json`` in the artifact directory. Preflight
    failures occur before the child starts and raise ``HarnessRunError``.

    Args:
        benchmark: Name of a registered benchmark.
        timeout_seconds: Finite, positive deadline in seconds for child execution.
        output_dir: Artifact directory. Uses the default artifacts root if omitted.
        run_id: Artifact run identifier. Generates one if omitted.
        mode: Execution mode (``local``, ``batch``, or ``service``).
        overrides: Benchmark configuration overrides as ``KEY=VALUE`` strings.
        requirements: Summary metric gates to enforce after execution.
        service_endpoint: Retriever service URL for service mode.
        runfile_payload: Runfile content to include in the run artifacts.
        runfile_path: Source runfile path to include in the run artifacts.

    Returns:
        The benchmark outcome, including the exit code and artifact paths.

    Raises:
        HarnessRunError: Preflight finds an invalid benchmark, configuration,
            metric gate, or missing dataset input. Correct the reported input
            and retry the run.
    """

    prepared = preflight_benchmark(
        benchmark,
        mode=mode,
        overrides=overrides,
        requirements=requirements,
        dry_run=False,
        service_endpoint=service_endpoint,
    )
    effective_run_id = run_id or make_run_id(benchmark)
    artifact_dir = resolve_artifact_dir(benchmark, effective_run_id, output_dir)
    run = PreparedRun(
        name=benchmark,
        artifact_name=effective_run_id,
        prepared=prepared,
        runfile_payload=runfile_payload,
        runfile_path=runfile_path,
        timeout_seconds=timeout_seconds,
    )
    try:
        return _run_prepared_benchmark_isolated(run, output_dir=str(artifact_dir), run_id=effective_run_id)
    except Exception as exc:
        return _failed_child_outcome(
            benchmark=benchmark,
            run_id=effective_run_id,
            artifact_dir=artifact_dir,
            dry_run=False,
            exc=exc,
        )


def _session_summary(
    *,
    session_type: str,
    exit_code: int,
    dry_run: bool,
    run_results: list[dict[str, Any]],
    run_commit: str,
    **extra: Any,
) -> dict[str, Any]:
    success = exit_code == EXIT_SUCCESS
    return {
        "session_type": session_type,
        "timestamp": now_timestr(),
        "run_commit": run_commit,
        "latest_commit": run_commit,
        "success": success,
        "all_passed": success,
        "exit_code": exit_code,
        "dry_run": bool(dry_run),
        "runs": run_results,
        **extra,
    }


def _run_session(
    *,
    session_type: str,
    session_name: str,
    session_dir: Path,
    runs: Sequence[PreparedRun],
    expanded_payload: Mapping[str, Any],
    run_commit: str,
    dry_run: bool,
    summary_extra: Mapping[str, Any],
    isolate_runs: bool = False,
) -> RunOutcome:
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise artifact_write_error(exc) from exc
    _remove_stale_session_summary(session_dir)
    write_json(session_dir / "expanded_runs.json", redact(dict(expanded_payload)))

    run_results: list[dict[str, Any]] = []
    exit_code = EXIT_SUCCESS
    for run in runs:
        artifact_dir = session_dir / run.artifact_name
        run_id = f"{session_name}_{run.artifact_name}"
        try:
            if isolate_runs:
                outcome = _run_prepared_benchmark_isolated(
                    run,
                    output_dir=str(artifact_dir),
                    run_id=run_id,
                )
            else:
                outcome = run_prepared_benchmark(
                    run.prepared,
                    output_dir=str(artifact_dir),
                    run_id=run_id,
                    runfile_payload=run.runfile_payload,
                    runfile_path=run.runfile_path,
                )
        except Exception as exc:
            outcome = _failed_child_outcome(
                benchmark=run.prepared.benchmark,
                run_id=run_id,
                artifact_dir=artifact_dir,
                dry_run=run.prepared.dry_run,
                exc=exc,
            )
        run_results.append(
            _run_outcome_summary(
                run.prepared.benchmark,
                outcome,
                session_dir=session_dir,
                run_name=run.name,
                runfile_path=run.runfile_path,
                mode=run.summary_mode,
            )
        )
        if exit_code == EXIT_SUCCESS and outcome.exit_code != EXIT_SUCCESS:
            exit_code = outcome.exit_code

    session_summary = _session_summary(
        session_type=session_type,
        exit_code=exit_code,
        dry_run=dry_run,
        run_results=run_results,
        run_commit=run_commit,
        **dict(summary_extra),
    )
    summary_path = session_dir / "session_summary.json"
    write_json(summary_path, session_summary)
    return RunOutcome(
        exit_code=exit_code,
        artifact_dir=session_dir,
        results=session_summary,
        results_path=summary_path,
    )


def run_runset(
    runset: str,
    *,
    output_dir: str | None = None,
    mode: str = "local",
    overrides: Sequence[str] = (),
    requirements: Sequence[str] = (),
    dry_run: bool = False,
) -> RunOutcome:
    spec = _runset_or_error(runset)
    runs: list[PreparedRun] = []
    expanded_runs: list[dict[str, Any]] = []
    for index, benchmark in enumerate(spec.runs, start=1):
        prepared = preflight_benchmark(
            benchmark,
            mode=mode,
            overrides=overrides,
            requirements=requirements,
            dry_run=dry_run,
        )
        artifact_name = f"{index:03d}_{benchmark}"
        runs.append(
            PreparedRun(
                name=benchmark,
                artifact_name=artifact_name,
                prepared=prepared,
            )
        )
        expanded_runs.append(
            {
                "index": index,
                "benchmark": benchmark,
                "artifact_dir": artifact_name,
                "mode": mode,
                "overrides": list(overrides),
                "dry_run": bool(dry_run),
            }
        )

    return _run_session(
        session_type="runset",
        session_name=runset,
        session_dir=_session_dir(runset, output_dir),
        runs=runs,
        expanded_payload={"runset": spec.to_dict(), "runs": expanded_runs},
        run_commit=last_commit(),
        dry_run=dry_run,
        summary_extra={"runset": spec.name, "working_tree_dirty": working_tree_dirty()},
    )


def run_runfiles(
    runfiles: Sequence[Path],
    *,
    output_dir: str | None = None,
    session_name: str = "runfiles",
    dataset_paths_file: Path | None = None,
    mode: str | None = None,
    service_endpoint: str | None = None,
    overrides: Sequence[str] = (),
    requirements: Sequence[str] = (),
    dry_run: bool = False,
) -> RunOutcome:
    if not runfiles:
        raise _invalid_runfile("At least one runfile path is required.")

    session_name = _validate_session_label(session_name, field="--session-name")
    run_commit = last_commit()
    source_worktree_dirty = working_tree_dirty()
    local_dataset_paths = load_dataset_paths(dataset_paths_file)
    requests = [load_runfile(path) for path in runfiles]
    runs: list[PreparedRun] = []
    expanded_runs: list[dict[str, Any]] = []
    for index, request in enumerate(requests, start=1):
        run_name = _validate_session_label(request.name or request.benchmark, field="Runfile name")
        if request.output_dir is not None or request.run_id is not None or request.dry_run is not None:
            raise _invalid_runfile(
                f"Runfile {request.source_path} cannot set 'output_dir', 'run_id', or 'dry_run' when used with "
                "run-files; the session owns artifact paths, run IDs, and dry-run behavior."
            )
        dataset_name = get_benchmark(request.benchmark).dataset
        dataset_paths = local_dataset_paths.get(dataset_name)
        dataset_overrides = dataset_paths.overrides() if dataset_paths is not None else ()
        effective_mode = mode or request.mode or "local"
        effective_overrides = (*request.overrides, *dataset_overrides, *overrides)
        effective_requirements = (*request.requirements, *requirements)
        prepared = preflight_benchmark(
            request.benchmark,
            mode=effective_mode,
            overrides=effective_overrides,
            requirements=effective_requirements,
            dry_run=dry_run,
            service_endpoint=service_endpoint if effective_mode == "service" else None,
        )
        artifact_name = f"{index:03d}_{run_name}"
        runs.append(
            PreparedRun(
                name=run_name,
                artifact_name=artifact_name,
                prepared=prepared,
                runfile_payload=dict(request.payload),
                runfile_path=str(request.source_path),
                summary_mode=effective_mode,
                timeout_seconds=request.timeout_seconds,
            )
        )
        expanded_runs.append(
            {
                "index": index,
                "name": run_name,
                "benchmark": request.benchmark,
                "dataset": dataset_name,
                "runfile_path": str(request.source_path),
                "artifact_dir": artifact_name,
                "mode": effective_mode,
                "service_endpoint": service_endpoint if effective_mode == "service" else None,
                "dataset_paths": dataset_paths.to_dict() if dataset_paths is not None else None,
                "overrides": list(effective_overrides),
                "requirements": list(effective_requirements),
                "dry_run": bool(dry_run),
                "timeout_seconds": request.timeout_seconds or _ISOLATED_CHILD_TIMEOUT_SECONDS,
            }
        )

    dataset_paths_value = str(dataset_paths_file.expanduser().resolve()) if dataset_paths_file else None
    isolate_runs = not dry_run
    return _run_session(
        session_type="runfiles",
        session_name=session_name,
        session_dir=_session_dir(session_name, output_dir),
        runs=runs,
        expanded_payload={
            "session_name": session_name,
            "run_commit": run_commit,
            "working_tree_dirty": source_worktree_dirty,
            "dataset_paths_file": dataset_paths_value,
            "isolate_runs": bool(isolate_runs),
            "runfiles": expanded_runs,
        },
        run_commit=run_commit,
        dry_run=dry_run,
        summary_extra={
            "session_name": session_name,
            "dataset_paths_file": dataset_paths_value,
            "isolate_runs": bool(isolate_runs),
            "working_tree_dirty": source_worktree_dirty,
        },
        isolate_runs=isolate_runs,
    )
