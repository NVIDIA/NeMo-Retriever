# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
import re
from typing import Any, Sequence

from nemo_retriever.harness.artifact_writer import redact
from nemo_retriever.harness.artifacts import get_artifacts_root, last_commit, now_timestr
from nemo_retriever.harness.benchmark_registry import get_benchmark, get_runset, runset_names
from nemo_retriever.harness.contracts import (
    EXIT_ARTIFACT_WRITE_FAILURE,
    EXIT_INTERNAL_ERROR,
    EXIT_INVALID,
    EXIT_SUCCESS,
    FailurePayload,
    HarnessRunError,
    RunOutcome,
)
from nemo_retriever.harness.dataset_paths import load_dataset_paths
from nemo_retriever.harness.execution import preflight_benchmark, run_benchmark
from nemo_retriever.harness.json_io import write_json
from nemo_retriever.harness.runfile import load_runfile

_SESSION_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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
    if summary_path.exists() or summary_path.is_symlink():
        summary_path.unlink()


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
    try:
        artifact_dir_value = artifact_dir.relative_to(session_dir.resolve()).as_posix()
        results_path_value = f"{artifact_dir_value}/results.json"
    except ValueError:
        artifact_dir_value = str(artifact_dir)
        results_path_value = str(artifact_dir / "results.json")
    payload = {
        "run_name": run_name or benchmark,
        "benchmark": benchmark,
        "artifact_dir": artifact_dir_value,
        "exit_code": outcome.exit_code,
        "success": outcome.exit_code == EXIT_SUCCESS,
        "summary_metrics": outcome.results.get("summary_metrics", {}),
        "results_path": results_path_value,
    }
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


def _failed_child_outcome(
    *,
    benchmark: str,
    artifact_dir: Path,
    dry_run: bool,
    exc: BaseException,
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

    def failure_result() -> dict[str, Any]:
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

    result = failure_result()
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        write_json(artifact_dir / "results.json", result)
    except Exception as write_exc:
        exit_code = EXIT_ARTIFACT_WRITE_FAILURE
        failure = FailurePayload(
            failed_phase="write_artifacts",
            failure_reason="artifact_write_failed",
            retryable=False,
            message=f"{type(write_exc).__name__}: {write_exc}",
        )
        result = failure_result()
    return RunOutcome(exit_code=exit_code, artifact_dir=artifact_dir, results=result)


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
    for benchmark in spec.runs:
        preflight_benchmark(
            benchmark,
            mode=mode,
            overrides=overrides,
            requirements=requirements,
            dry_run=dry_run,
        )
    run_commit = last_commit()
    session_dir = _session_dir(runset, output_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_session_summary(session_dir)
    expanded_runs = [
        {
            "index": index,
            "benchmark": benchmark,
            "artifact_dir": f"{index:03d}_{benchmark}",
            "mode": mode,
            "overrides": list(overrides),
            "dry_run": bool(dry_run),
        }
        for index, benchmark in enumerate(spec.runs, start=1)
    ]
    write_json(
        session_dir / "expanded_runs.json",
        redact(
            {
                "runset": spec.to_dict(),
                "runs": expanded_runs,
            }
        ),
    )

    run_results: list[dict[str, Any]] = []
    exit_code = EXIT_SUCCESS
    for expanded in expanded_runs:
        artifact_dir = session_dir / str(expanded["artifact_dir"])
        try:
            outcome = run_benchmark(
                str(expanded["benchmark"]),
                output_dir=str(artifact_dir),
                run_id=f"{runset}_{expanded['index']:03d}_{expanded['benchmark']}",
                mode=mode,
                overrides=overrides,
                requirements=requirements,
                dry_run=dry_run,
            )
        except Exception as exc:
            outcome = _failed_child_outcome(
                benchmark=str(expanded["benchmark"]),
                artifact_dir=artifact_dir,
                dry_run=dry_run,
                exc=exc,
            )
        run_results.append(_run_outcome_summary(str(expanded["benchmark"]), outcome, session_dir=session_dir))
        if exit_code == EXIT_SUCCESS and outcome.exit_code != EXIT_SUCCESS:
            exit_code = outcome.exit_code

    session_summary = _session_summary(
        session_type="runset",
        exit_code=exit_code,
        dry_run=dry_run,
        run_results=run_results,
        run_commit=run_commit,
        runset=spec.name,
    )
    write_json(session_dir / "session_summary.json", session_summary)
    return RunOutcome(exit_code=exit_code, artifact_dir=session_dir, results=session_summary)


def run_runfiles(
    runfiles: Sequence[Path],
    *,
    output_dir: str | None = None,
    session_name: str = "runfiles",
    dataset_paths_file: Path | None = None,
    mode: str | None = None,
    overrides: Sequence[str] = (),
    requirements: Sequence[str] = (),
    dry_run: bool = False,
) -> RunOutcome:
    if not runfiles:
        raise _invalid_runfile("At least one runfile path is required.")

    session_name = _validate_session_label(session_name, field="--session-name")
    run_commit = last_commit()
    local_dataset_paths = load_dataset_paths(dataset_paths_file)
    requests = [load_runfile(path) for path in runfiles]
    session_dir = _session_dir(session_name, output_dir)
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
        preflight_benchmark(
            request.benchmark,
            mode=effective_mode,
            overrides=effective_overrides,
            requirements=effective_requirements,
            dry_run=dry_run,
        )
        expanded_runs.append(
            {
                "index": index,
                "name": run_name,
                "benchmark": request.benchmark,
                "dataset": dataset_name,
                "runfile_path": str(request.source_path),
                "artifact_dir": f"{index:03d}_{run_name}",
                "mode": effective_mode,
                "dataset_paths": dataset_paths.to_dict() if dataset_paths is not None else None,
                "overrides": list(effective_overrides),
                "requirements": list(effective_requirements),
                "dry_run": bool(dry_run),
            }
        )

    session_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_session_summary(session_dir)
    write_json(
        session_dir / "expanded_runs.json",
        redact(
            {
                "session_name": session_name,
                "run_commit": run_commit,
                "dataset_paths_file": str(dataset_paths_file.expanduser().resolve()) if dataset_paths_file else None,
                "runfiles": expanded_runs,
            }
        ),
    )

    run_results: list[dict[str, Any]] = []
    exit_code = EXIT_SUCCESS
    for expanded, request in zip(expanded_runs, requests, strict=True):
        artifact_dir = session_dir / str(expanded["artifact_dir"])
        try:
            outcome = run_benchmark(
                request.benchmark,
                output_dir=str(artifact_dir),
                run_id=f"{session_name}_{expanded['index']:03d}_{expanded['name']}",
                mode=str(expanded["mode"]),
                overrides=tuple(expanded["overrides"]),
                requirements=tuple(expanded["requirements"]),
                dry_run=bool(expanded["dry_run"]),
                runfile_payload=dict(request.payload),
                runfile_path=str(request.source_path),
            )
        except Exception as exc:
            outcome = _failed_child_outcome(
                benchmark=request.benchmark,
                artifact_dir=artifact_dir,
                dry_run=bool(expanded["dry_run"]),
                exc=exc,
            )
        run_results.append(
            _run_outcome_summary(
                request.benchmark,
                outcome,
                session_dir=session_dir,
                run_name=str(expanded["name"]),
                runfile_path=str(request.source_path),
                mode=str(expanded["mode"]),
            )
        )
        if exit_code == EXIT_SUCCESS and outcome.exit_code != EXIT_SUCCESS:
            exit_code = outcome.exit_code

    session_summary = _session_summary(
        session_type="runfiles",
        exit_code=exit_code,
        dry_run=dry_run,
        run_results=run_results,
        run_commit=run_commit,
        session_name=session_name,
        dataset_paths_file=str(dataset_paths_file.expanduser().resolve()) if dataset_paths_file else None,
    )
    write_json(session_dir / "session_summary.json", session_summary)
    return RunOutcome(exit_code=exit_code, artifact_dir=session_dir, results=session_summary)
