# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from nemo_retriever.cli.ingest import app as ingest_app
from nemo_retriever.cli.query import app as query_app
from nemo_retriever.harness import app as harness_app
from nemo_retriever.version import get_version_info

app = typer.Typer(
    help=(
        "NeMo Retriever product workflows: ingest content, query an index, "
        "run benchmark harnesses, or operate the service."
    )
)

# Service sub-app is always available (lightweight, no GPU deps).
from nemo_retriever.service.cli import app as service_app  # noqa: E402

app.add_typer(service_app, name="service")
app.add_typer(ingest_app, name="ingest")
app.add_typer(query_app, name="query")
app.add_typer(harness_app, name="harness")


def _version_callback(value: bool) -> None:
    if not value:
        return
    info = get_version_info()
    typer.echo(info["full_version"])
    raise typer.Exit()


def main() -> None:
    app()


@app.callback()
def _callback(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show retriever version metadata and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    _ = version
