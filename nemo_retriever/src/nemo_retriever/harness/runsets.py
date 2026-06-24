# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Sequence

from nemo_retriever.harness.artifact_writer import write_json
from nemo_retriever.harness.artifacts import get_artifacts_root
from nemo_retriever.harness.benchmark_registry import get_runset, runset_names
from nemo_retriever.harness.contracts import EXIT_INVALID, EXIT_SUCCESS, FailurePayload, HarnessRunError, RunOutcome
from nemo_retriever.harness.execution import run_benchmark


def _session_id(runset: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    return f"{runset}_{stamp}"


def _session_dir(runset: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (get_artifacts_root() / _session_id(runset)).resolve()


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
    session_dir = _session_dir(runset, output_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    expanded_runs = [
        {
            "index": index,
            "benchmark": benchmark,
            "artifact_dir": str((session_dir / f"{index:03d}_{benchmark}").resolve()),
            "mode": mode,
            "overrides": list(overrides),
            "dry_run": bool(dry_run),
        }
        for index, benchmark in enumerate(spec.runs, start=1)
    ]
    write_json(
        session_dir / "expanded_runs.json",
        {
            "runset": spec.to_dict(),
            "runs": expanded_runs,
        },
    )

    run_results: list[dict[str, Any]] = []
    exit_code = EXIT_SUCCESS
    for expanded in expanded_runs:
        outcome = run_benchmark(
            str(expanded["benchmark"]),
            output_dir=str(expanded["artifact_dir"]),
            run_id=f"{runset}_{expanded['index']:03d}_{expanded['benchmark']}",
            mode=mode,
            overrides=overrides,
            requirements=requirements,
            dry_run=dry_run,
        )
        run_result = {
            "benchmark": expanded["benchmark"],
            "artifact_dir": str(outcome.artifact_dir),
            "exit_code": outcome.exit_code,
            "success": outcome.exit_code == EXIT_SUCCESS,
            "summary_metrics": outcome.results.get("summary_metrics", {}),
            "results_path": str(outcome.artifact_dir / "results.json"),
        }
        run_results.append(run_result)
        if exit_code == EXIT_SUCCESS and outcome.exit_code != EXIT_SUCCESS:
            exit_code = outcome.exit_code

    session_summary = {
        "runset": spec.name,
        "success": exit_code == EXIT_SUCCESS,
        "exit_code": exit_code,
        "dry_run": bool(dry_run),
        "runs": run_results,
    }
    write_json(session_dir / "session_summary.json", session_summary)
    return RunOutcome(exit_code=exit_code, artifact_dir=session_dir, results=session_summary)
