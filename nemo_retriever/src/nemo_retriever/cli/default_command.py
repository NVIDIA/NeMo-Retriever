# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Click helpers for a subcommand group with a first-class default command."""

from __future__ import annotations

import click
from typer.core import TyperCommand, TyperGroup

_DEFAULT_ROUTE_KEY = "nemo_retriever.cli.default_command.routed"


class DefaultCommand(TyperCommand):
    """Render routed command usage as the public parent command."""

    def format_usage(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        command_path = ctx.command_path
        if ctx.meta.get(_DEFAULT_ROUTE_KEY):
            command_path = command_path.rsplit(" ", 1)[0]
        formatter.write_usage(command_path, " ".join(self.collect_usage_pieces(ctx)))


class DefaultCommandGroup(TyperGroup):
    """Route non-subcommand arguments, including ``--help``, to a default."""

    default_command: str
    group_options = frozenset({"--install-completion", "--show-completion"})

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in self.group_options:
            ctx.meta[_DEFAULT_ROUTE_KEY] = True
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)
