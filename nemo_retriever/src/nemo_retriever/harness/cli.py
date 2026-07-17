# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from nemo_retriever.harness.benchmark_registry import (
    benchmark_payload,
    list_benchmarks,
    list_runsets,
)
from nemo_retriever.harness.baselines import load_baselines
from nemo_retriever.harness.contracts import EXIT_INVALID, EXIT_MISSING_INPUT, FailurePayload
from nemo_retriever.harness.revamp_runner import (
    HarnessRunError,
    run_benchmark,
    show_benchmark_payload,
)
from nemo_retriever.harness.diff import diff_artifact_dirs
from nemo_retriever.harness.resolution import make_run_id
from nemo_retriever.harness.runfile import load_runfile
from nemo_retriever.harness.runsets import run_runfiles, run_runset
from nemo_retriever.harness.slack import (
    DEFAULT_SLACK_METRIC_KEYS,
    build_slack_payload,
    load_replay_report,
    post_slack_payload,
    resolve_slack_webhook_url,
)
from nemo_retriever.harness.vidore_access import VidoreAccessError, check_vidore_access

app = typer.Typer(
    help=(
        "Developer benchmark and evaluation harness. Use 'retriever ingest' and "
        "'retriever query' for direct product workflows."
    )
)


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=False))


@app.command("check-vidore-access")
def check_vidore_access_command(
    dataset_names: Annotated[
        list[str] | None,
        typer.Option("--dataset", help="Check one ViDoRe dataset name. Repeat to check multiple datasets."),
    ] = None,
    timeout_seconds: Annotated[
        float,
        typer.Option("--timeout-seconds", min=1.0, help="HTTP timeout for each streamed range request."),
    ] = 20.0,
    require_token: Annotated[
        bool,
        typer.Option("--require-token/--allow-anonymous", help="Require HF_TOKEN before checking public data."),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Verify ViDoRe queries, qrels, and corpus objects without downloading them."""

    try:
        results = check_vidore_access(
            dataset_names=dataset_names,
            timeout_seconds=timeout_seconds,
            require_token=require_token,
        )
    except VidoreAccessError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_MISSING_INPUT) from exc

    payload = {"success": True, "datasets": [result.to_dict() for result in results]}
    if json_output:
        _echo_json(payload)
        return
    for result in results:
        typer.echo(f"{result.dataset}: access ok ({', '.join(result.checked_partitions)})")


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
    mode: Annotated[
        str | None, typer.Option("--mode", help="System-under-test mode: local, batch, or service.")
    ] = None,
    service_endpoint: Annotated[
        str | None,
        typer.Option("--service-endpoint", help="Machine-local Retriever service URL for service mode."),
    ] = None,
    set_values: Annotated[
        list[str] | None,
        typer.Option("--set", help="Apply a small KEY=VALUE override. Repeatable."),
    ] = None,
    requirements: Annotated[
        list[str] | None,
        typer.Option("--require", help="Require a summary metric gate, e.g. recall_5>=0.80. Repeatable."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Resolve configuration and write plans without executing ingest or query."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit results JSON to stdout.")] = False,
) -> None:
    """Run one registered benchmark and write stable artifacts."""
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
            service_endpoint=service_endpoint,
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
    mode: Annotated[str, typer.Option("--mode", help="System-under-test mode: local, batch, or service.")] = "local",
    set_values: Annotated[
        list[str] | None,
        typer.Option("--set", help="Apply a small KEY=VALUE override to every run. Repeatable."),
    ] = None,
    requirements: Annotated[
        list[str] | None,
        typer.Option("--require", help="Require a summary metric gate for every run. Repeatable."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Resolve configuration and write plans without executing ingest or query."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit session summary JSON to stdout.")] = False,
) -> None:
    """Run a code-owned benchmark group using registry dataset paths."""
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


@app.command("run-files")
def run_files_command(
    runfiles: Annotated[list[Path], typer.Argument(help="Runfile paths to execute as one session.")],
    output_dir: Annotated[str | None, typer.Option("--output-dir", help="Directory for session artifacts.")] = None,
    session_name: Annotated[str, typer.Option("--session-name", help="Stable session label.")] = "runfiles",
    dataset_paths: Annotated[
        Path | None,
        typer.Option(
            "--dataset-paths",
            help="Machine-local YAML file that maps registered datasets to document and query paths.",
        ),
    ] = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="Override system-under-test mode for every runfile."),
    ] = None,
    service_endpoint: Annotated[
        str | None,
        typer.Option(
            "--service-endpoint",
            help="Machine-local Retriever service URL, applied only to service-mode runfiles.",
        ),
    ] = None,
    set_values: Annotated[
        list[str] | None,
        typer.Option("--set", help="Apply a small KEY=VALUE override to every run. Repeatable."),
    ] = None,
    requirements: Annotated[
        list[str] | None,
        typer.Option("--require", help="Require a summary metric gate for every run. Repeatable."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Resolve configuration and write plans without executing ingest or query."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit session summary JSON to stdout.")] = False,
) -> None:
    """Run one or more runfiles, optionally with machine-local dataset paths."""
    try:
        outcome = run_runfiles(
            runfiles,
            output_dir=output_dir,
            session_name=session_name,
            dataset_paths_file=dataset_paths,
            mode=mode,
            service_endpoint=service_endpoint,
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
        typer.echo(f"Runfile session failed with exit code {outcome.exit_code}", err=True)
        typer.echo(f"Session artifacts: {outcome.artifact_dir}", err=True)
    raise typer.Exit(code=outcome.exit_code)


@app.command("post-slack")
def post_slack_command(
    paths: Annotated[
        list[Path],
        typer.Argument(help="One session directory, or one or more run artifact directories/results.json files."),
    ],
    title: Annotated[str, typer.Option("--title", help="Slack message title.")] = "nemo_retriever Harness Report",
    metric_keys: Annotated[
        list[str] | None,
        typer.Option("--metric-key", help="Summary metric key to include. Repeatable."),
    ] = None,
    post_artifact_paths: Annotated[
        bool,
        typer.Option("--artifact-paths/--no-artifact-paths", help="Include local artifact paths in the Slack post."),
    ] = False,
    baseline_file: Annotated[
        Path | None,
        typer.Option(
            "--baseline-file",
            help="Optional local JSON baseline file used only to render comparison tables.",
        ),
    ] = None,
    preview: Annotated[
        bool,
        typer.Option("--preview", help="Render the Slack payload as JSON without reading a webhook or posting."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit posted Slack payload JSON to stdout.")] = False,
) -> None:
    """Render or post existing harness artifacts without running benchmarks."""
    try:
        report = load_replay_report(paths)
        slack_config = {
            "title": title,
            "metric_keys": metric_keys or DEFAULT_SLACK_METRIC_KEYS,
            "post_artifact_paths": post_artifact_paths,
        }
        reference_file = baseline_file
        if reference_file is None and (configured_reference := os.environ.get("RETRIEVER_HARNESS_REFERENCE_FILE")):
            reference_file = Path(configured_reference)
        baselines = load_baselines(reference_file) if reference_file is not None else None
        payload = build_slack_payload(report, slack_config, baselines=baselines)
        if not preview:
            post_slack_payload(payload, resolve_slack_webhook_url())
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if preview or json_output:
        _echo_json(payload)
    else:
        typer.echo(f"Posted Slack report for {report.session_name}")


@app.command("diff")
def diff_command(
    left: Annotated[Path, typer.Argument(help="Left run artifact directory or results.json file.")],
    right: Annotated[Path, typer.Argument(help="Right run artifact directory or results.json file.")],
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


def main() -> None:
    app()
