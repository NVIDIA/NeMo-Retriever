# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click
import typer
from typer.core import TyperGroup

from nemo_retriever.adapters.cli.ingest.graph import batch_command, local_command
from nemo_retriever.adapters.cli.ingest.service import service_command

_DEFAULT_COMMAND = "local"
_GROUP_OPTIONS = {"--help", "-h", "--install-completion", "--show-completion"}


class DefaultLocalIngestGroup(TyperGroup):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in _GROUP_OPTIONS:
            args = [_DEFAULT_COMMAND, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(
    cls=DefaultLocalIngestGroup,
    help="Ingest documents into Retriever indexes. Omitting a mode runs local ingest.",
    no_args_is_help=True,
)

app.command("local")(local_command)
app.command("batch")(batch_command)
app.command("service")(service_command)
