# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable
from typing import Any

import typer

from nemo_retriever.cli.shared import ROOT_CLI_ERRORS, quiet_capture, silence_noisy_libraries

logger = logging.getLogger(__name__)


def run_cli_workflow(make_summary: Callable[[], dict[str, Any]], *, quiet: bool) -> None:
    if quiet:
        silence_noisy_libraries()
    capture = quiet_capture() if quiet else contextlib.nullcontext()
    try:
        with capture:
            summary = make_summary()
    except ROOT_CLI_ERRORS as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    print_ingest_summary(summary)


def print_ingest_summary(summary: dict[str, Any]) -> None:
    if summary.get("dry_run") is True:
        typer.echo(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return

    if summary.get("run_mode") == "service":
        n_files = len(summary["documents"])
        service_target = summary["service_url"]
        n_rows = summary.get("n_rows")
        if n_rows is None:
            typer.echo(
                f"Ingested {n_files} file(s) through retriever service {service_target} " "(row count unavailable)."
            )
        else:
            typer.echo(f"Ingested {n_files} file(s) → {n_rows} row(s) through retriever service {service_target}.")
        return

    n_files = summary["n_documents"]
    table_path = f"{summary['lancedb_uri']}/{summary['table_name']}"
    n_rows = summary.get("n_rows")
    if n_rows is None:
        typer.echo(f"Ingested {n_files} file(s) into LanceDB {table_path} (row count unavailable).")
    else:
        typer.echo(f"Ingested {n_files} file(s) → {n_rows} row(s) in LanceDB {table_path}.")
