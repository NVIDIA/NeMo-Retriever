# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import logging

import typer

from nemo_retriever.cli.ingest import app as ingest_app
from nemo_retriever.cli.query import app as query_app
from nemo_retriever.version import get_version_info

logger = logging.getLogger(__name__)

app = typer.Typer(help="Retriever")

# Service sub-app is always available (lightweight, no GPU deps).
from nemo_retriever.service.cli import app as service_app  # noqa: E402

app.add_typer(service_app, name="service")
app.add_typer(ingest_app, name="ingest")
app.add_typer(query_app, name="query")

# All other sub-apps are registered lazily so that missing optional
# dependencies (tritonclient, torch, …) don't prevent the service
# from starting.
_LAZY_SUBAPPS: list[tuple[str, str, str]] = [
    ("audio", "nemo_retriever.cli.audio.cli", "app"),
    ("image", "nemo_retriever.cli.image", "app"),
    ("pdf", "nemo_retriever.cli.pdf.__main__", "app"),
    ("local", "nemo_retriever.cli.local", "app"),
    ("chart", "nemo_retriever.cli.chart.commands", "app"),
    ("compare", "nemo_retriever.cli.compare", "app"),
    ("eval", "nemo_retriever.tools.evaluation.cli", "app"),
    ("benchmark", "nemo_retriever.tools.benchmark", "app"),
    ("harness", "nemo_retriever.harness", "app"),
    ("recall", "nemo_retriever.tools.recall", "app"),
    ("skill-eval", "nemo_retriever.tools.skill_eval", "app"),
    ("txt", "nemo_retriever.cli.txt.__main__", "app"),
    ("html", "nemo_retriever.cli.html.__main__", "app"),
    ("pipeline", "nemo_retriever.cli.pipeline.__main__", "app"),
]

for _name, _module, _attr in _LAZY_SUBAPPS:
    try:
        _mod = importlib.import_module(_module)
        app.add_typer(getattr(_mod, _attr), name=_name)
    except Exception:
        logger.debug("Skipping '%s' sub-command (import failed)", _name)


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
