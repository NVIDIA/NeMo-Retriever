# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click
import typer
from typer.core import TyperGroup

from nemo_retriever.cli.ingest.graph_commands import _graph_ingest_command
from nemo_retriever.cli.ingest.service import _service_command

_DEFAULT_COMMAND = "local"
_GROUP_OPTIONS = {"--help", "-h"}


class DefaultLocalIngestGroup(TyperGroup):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in _GROUP_OPTIONS:
            args = [_DEFAULT_COMMAND, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=DefaultLocalIngestGroup,
    help=(
        "Ingest documents into Retriever indexes. Omitting a mode runs local ingest. "
        "Use local, batch, or service --help for mode-specific options."
    ),
    no_args_is_help=True,
)

app.command("local")(_graph_ingest_command)
app.command("batch")(_graph_ingest_command)
app.command("service")(_service_command)
