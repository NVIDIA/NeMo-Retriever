# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nemo_retriever.harness.benchmark_registry import (
    benchmark_payload,
    list_benchmarks,
    list_runsets,
)
from nemo_retriever.harness.contracts import EXIT_INVALID, FailurePayload
from nemo_retriever.harness.revamp_runner import (
    HarnessRunError,
    run_benchmark,
    show_benchmark_payload,
)
from nemo_retriever.harness.diff import diff_artifact_dirs
from nemo_retriever.harness.resolution import make_run_id
from nemo_retriever.harness.runfile import load_runfile
from nemo_retriever.harness.runsets import run_runset

app = typer.Typer(help="Artifact-first Retriever benchmark harness.")


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=False))


@app.command("list")
def list_command(
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
    runsets: Annotated[bool, typer.Option("--runsets", help="Include runsets in the listing.")] = False,
) -> None:
    """List code-owned benchmark registry entries."""
    payload = {
        "benchmarks": [benchmark_payload(spec) for spec in list_benchmarks()],
    }
    if runsets:
        payload["runsets"] = [runset.to_dict() for runset in list_runsets()]
    if json_output:
        _echo_json(payload)
        return

    for item in payload["benchmarks"]:
        tags = ", ".join(item.get("tags") or [])
        suffix = f" [{tags}]" if tags else ""
        typer.echo(f"{item['name']}{suffix}")
    if runsets:
        typer.echo("\nRunsets:")
        for runset in payload.get("runsets", []):
            typer.echo(f"{runset['name']}: {', '.join(runset.get('runs') or [])}")


@app.command("show")
def show_command(
    benchmark: Annotated[str, typer.Argument(help="Benchmark name.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Show one benchmark spec from the registry."""
    try:
        payload = show_benchmark_payload(benchmark)
    except HarnessRunError as exc:
        typer.echo(exc.failure.message, err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    if json_output:
        _echo_json(payload)
        return

    typer.echo(f"{payload['name']}")
    if payload.get("description"):
        typer.echo(str(payload["description"]))
    dataset = payload.get("dataset_spec") or {}
    typer.echo(f"dataset: {dataset.get('name')} ({dataset.get('path')})")
    typer.echo(f"evaluation: {(payload.get('evaluation') or {}).get('mode', 'none')}")


@app.command("run")
def run_command(
    benchmark: Annotated[str | None, typer.Argument(help="Benchmark name. Omit when using --runfile.")] = None,
    runfile: Annotated[Path | None, typer.Option("--runfile", help="JSON/YAML file for one concrete run.")] = None,
    output_dir: Annotated[str | None, typer.Option("--output-dir", help="Directory for run artifacts.")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id", help="Stable run identifier.")] = None,
    mode: Annotated[str | None, typer.Option("--mode", help="Ingest mode: local or batch.")] = None,
    set_values: Annotated[
        list[str] | None,
        typer.Option("--set", help="Apply a small KEY=VALUE override. Repeatable."),
    ] = None,
    requirements: Annotated[
        list[str] | None,
        typer.Option("--require", help="Require a summary metric gate, e.g. recall_5>=0.80. Repeatable."),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Resolve plans and artifacts without execution.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit results JSON to stdout.")] = False,
) -> None:
    """Run one benchmark and write stable artifacts."""
    try:
        runfile_payload = None
        runfile_path = None
        if runfile is not None:
            if benchmark is not None:
                raise HarnessRunError(
                    EXIT_INVALID,
                    failure=FailurePayload(
                        failed_phase="resolve",
                        failure_reason="invalid_runfile",
                        retryable=False,
                        message="Pass either a benchmark argument or --runfile, not both.",
                    ),
                )
            request = load_runfile(runfile)
            benchmark = request.benchmark
            output_dir = output_dir or request.output_dir
            run_id = run_id or request.run_id or make_run_id(request.name or request.benchmark)
            mode = mode or request.mode
            set_values = list(request.overrides) + list(set_values or ())
            requirements = list(request.requirements) + list(requirements or ())
            dry_run = dry_run or bool(request.dry_run)
            runfile_payload = dict(request.payload)
            runfile_path = str(request.source_path)
        if benchmark is None:
            raise HarnessRunError(
                EXIT_INVALID,
                failure=FailurePayload(
                    failed_phase="resolve",
                    failure_reason="invalid_benchmark",
                    retryable=False,
                    message="Pass a benchmark argument or --runfile.",
                ),
            )
        outcome = run_benchmark(
            benchmark,
            output_dir=output_dir,
            run_id=run_id,
            mode=mode or "local",
            overrides=set_values or (),
            requirements=requirements or (),
            dry_run=dry_run,
            runfile_payload=runfile_payload,
            runfile_path=runfile_path,
        )
    except HarnessRunError as exc:
        typer.echo(exc.failure.message, err=True)
        raise typer.Exit(code=exc.exit_code) from exc
    if json_output:
        _echo_json(outcome.results)
    elif outcome.exit_code == 0:
        typer.echo(f"Artifacts: {outcome.artifact_dir}")
        typer.echo(f"Results: {outcome.artifact_dir / 'results.json'}")
    else:
        failure = outcome.results.get("failure") or {}
        message = failure.get("message") or f"Benchmark failed with exit code {outcome.exit_code}"
        typer.echo(message, err=True)
        typer.echo(f"Artifacts: {outcome.artifact_dir}", err=True)
    raise typer.Exit(code=outcome.exit_code)


@app.command("run-set")
def run_set_command(
    runset: Annotated[str, typer.Argument(help="Runset name.")],
    output_dir: Annotated[str | None, typer.Option("--output-dir", help="Directory for session artifacts.")] = None,
    mode: Annotated[str, typer.Option("--mode", help="Ingest mode: local or batch.")] = "local",
    set_values: Annotated[
        list[str] | None,
        typer.Option("--set", help="Apply a small KEY=VALUE override to every run. Repeatable."),
    ] = None,
    requirements: Annotated[
        list[str] | None,
        typer.Option("--require", help="Require a summary metric gate for every run. Repeatable."),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Resolve plans and artifacts without execution.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit session summary JSON to stdout.")] = False,
) -> None:
    """Expand and run a code-owned benchmark runset."""
    try:
        outcome = run_runset(
            runset,
            output_dir=output_dir,
            mode=mode,
            overrides=set_values or (),
            requirements=requirements or (),
            dry_run=dry_run,
        )
    except HarnessRunError as exc:
        typer.echo(exc.failure.message, err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    if json_output:
        _echo_json(outcome.results)
    elif outcome.exit_code == 0:
        typer.echo(f"Session artifacts: {outcome.artifact_dir}")
        typer.echo(f"Session summary: {outcome.artifact_dir / 'session_summary.json'}")
    else:
        typer.echo(f"Runset failed with exit code {outcome.exit_code}", err=True)
        typer.echo(f"Session artifacts: {outcome.artifact_dir}", err=True)
    raise typer.Exit(code=outcome.exit_code)


@app.command("diff")
def diff_command(
    left: Annotated[Path, typer.Argument(help="Left run artifact directory or summary_metrics.json file.")],
    right: Annotated[Path, typer.Argument(help="Right run artifact directory or summary_metrics.json file.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Diff two run artifact directories by summary metrics."""
    try:
        payload = diff_artifact_dirs(left, right)
    except HarnessRunError as exc:
        typer.echo(exc.failure.message, err=True)
        raise typer.Exit(code=exc.exit_code) from exc

    if json_output:
        _echo_json(payload)
        return

    for key, delta in payload["summary_metrics"].items():
        if not delta["changed"]:
            continue
        typer.echo(f"{key}: {delta['left']} -> {delta['right']}")


runner_app = typer.Typer(help="Runner agent commands.")
from nemo_retriever.harness.runner import runner_start_command  # noqa: E402

runner_app.command("start")(runner_start_command)
app.add_typer(runner_app, name="runner")


@app.command("portal")
def portal_command(
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = 8100,
) -> None:
    """Launch the harness web portal."""
    import uvicorn

    typer.echo(f"Starting harness portal at http://{host}:{port}")
    uvicorn.run("nemo_retriever.harness.portal.app:app", host=host, port=port, log_level="info")


def main() -> None:
    app()
