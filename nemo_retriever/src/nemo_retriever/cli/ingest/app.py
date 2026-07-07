# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click
import typer
from typer.core import TyperCommand, TyperGroup

from nemo_retriever.cli.ingest.graph_commands import _graph_ingest_command
from nemo_retriever.cli.ingest.service import _service_command
from nemo_retriever.cli.ingest.options import DEFAULT_CAPTION_MODEL, DEFAULT_EMBED_MODEL

_DEFAULT_COMMAND = "local"
_GROUP_OPTIONS = {"-h"}


class DefaultLocalIngestGroup(TyperGroup):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in _GROUP_OPTIONS:
            args = [_DEFAULT_COMMAND, *args]
        return super().parse_args(ctx, args)


class PublicDefaultIngestContext(typer.Context):
    @property
    def command_path(self) -> str:
        return self.parent.command_path if self.parent is not None else super().command_path


class DefaultLocalIngestCommand(TyperCommand):
    context_class = PublicDefaultIngestContext


app = typer.Typer(
    cls=DefaultLocalIngestGroup,
    help=(
        "Ingest documents into Retriever indexes. Use retriever ingest DOCUMENTS for the default local workflow. "
        "HTML, TXT, PDF, Office, image, audio, and video are input formats, not commands. "
        "CPU-only hosts use NVIDIA's hosted embedding endpoint when NVIDIA_API_KEY or NGC_API_KEY is set. "
        "Use batch or service --help for those explicit modes."
    ),
    no_args_is_help=True,
)

app.command(
    "local",
    cls=DefaultLocalIngestCommand,
    hidden=True,
    help=(
        "Run the default local ingest into a LanceDB index.\n\n"
        "HTML, TXT, PDF, Office, image, audio, and video are input formats, not commands.\n\n"
        "CPU-only hosts use NVIDIA's hosted embedding endpoint when NVIDIA_API_KEY or NGC_API_KEY is set.\n\n"
        f"Default embedding model: {DEFAULT_EMBED_MODEL}.\n\n"
        f"Default caption model when captioning: {DEFAULT_CAPTION_MODEL}.\n\n"
        "For Ray scale-out options, use retriever ingest batch --help. "
        "For a remote service, use retriever ingest service --help."
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
