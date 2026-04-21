# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import typer

from nemo_retriever.harness.nightly import nightly_command
from nemo_retriever.harness.reporting import compare_command, summary_command
from nemo_retriever.harness.run import run_command, sweep_command
from nemo_retriever.harness.runner import runner_start_command

app = typer.Typer(help="Harness commands for benchmark orchestration.")
app.command("run")(run_command)
app.command("sweep")(sweep_command)
app.command("nightly")(nightly_command)
app.command("summary")(summary_command)
app.command("compare")(compare_command)

runner_app = typer.Typer(help="Runner agent commands.")
runner_app.command("start")(runner_start_command)
app.add_typer(runner_app, name="runner")


@app.command("portal")
def portal_command(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address."),
    port: int = typer.Option(8100, "--port", help="Bind port."),
) -> None:
    """Launch the harness web portal."""
    import uvicorn

    typer.echo(f"Starting harness portal at http://{host}:{port}")
    uvicorn.run("nemo_retriever.harness.portal.app:app", host=host, port=port, log_level="info")


@app.command("backfill")
def backfill_command(
    artifacts_dir: str | None = typer.Option(None, "--artifacts-dir", help="Root artifacts directory to scan."),
    db_path: str | None = typer.Option(None, "--db", help="Path to history database."),
) -> None:
    """Import existing artifact results.json files into the history database."""
    from nemo_retriever.harness.history import backfill_from_artifacts

    root = Path(artifacts_dir) if artifacts_dir else None
    count = backfill_from_artifacts(artifacts_root=root, db_path=db_path)
    typer.echo(f"Imported {count} run(s) into the history database.")


def main() -> None:
    app()
