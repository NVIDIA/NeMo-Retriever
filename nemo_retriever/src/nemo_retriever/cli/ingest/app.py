# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click
import typer
from typer.core import TyperGroup

from nemo_retriever.cli.ingest.graph_commands import _graph_ingest_command
from nemo_retriever.cli.ingest.service import _service_command
from nemo_retriever.cli.ingest.options import DEFAULT_CAPTION_MODEL, DEFAULT_EMBED_MODEL


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
        "HTML, TXT, PDF, Office, image, audio, and video are input formats, not commands. "
        "CPU-only hosts use NVIDIA's hosted embedding endpoint when NVIDIA_API_KEY or NGC_API_KEY is set. "
        "Use batch or service --help for those explicit modes."
    ),
    no_args_is_help=True,
)

app.command(
    "local",
    help=(
        f"Run the default local ingest into a LanceDB index. Default embedding model: {DEFAULT_EMBED_MODEL}. "
        f"Default caption model when captioning: {DEFAULT_CAPTION_MODEL}. Use "
        "`retriever ingest batch --help` for Ray scale-out or `retriever ingest service --help` "
        "for a remote service."
    ),
)(_graph_ingest_command)
app.command(
    "batch",
    help=(
        f"Run Ray batch ingest into a LanceDB index. Default embedding model: {DEFAULT_EMBED_MODEL}. "
        f"Default caption model when captioning: {DEFAULT_CAPTION_MODEL}."
    ),
)(_graph_ingest_command)
app.command("service")(_service_command)
