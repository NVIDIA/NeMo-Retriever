# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base model with Rich pretty-printing for all service Pydantic objects."""

from __future__ import annotations

from typing import Any, Generator

from pydantic import BaseModel
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table


class RichModel(BaseModel):
    """Pydantic BaseModel with built-in Rich representations.

    - ``print(obj)``           → clean key/value table via __str__
    - ``Console().print(obj)`` → Rich pretty-print via __rich_repr__
    - ``logger.info(obj)``     → same clean table via __str__
    """

    def __rich_repr__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in self:
            yield name, value

    def __str__(self) -> str:
        console = Console(file=None, force_terminal=False, no_color=True, width=100)
        table = Table(
            title=type(self).__name__,
            title_style="bold",
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        table.add_column("field", style="dim")
        table.add_column("value")
        for name, value in self:
            table.add_row(name, repr(value))
        with console.capture() as capture:
            console.print(table)
        return capture.get().rstrip()

    def __rich__(self) -> Table:
        table = Table(
            title=type(self).__name__,
            title_style="bold",
            show_header=False,
            padding=(0, 2),
        )
        table.add_column("field", style="cyan")
        table.add_column("value", style="white")
        for name, value in self:
            table.add_row(name, Pretty(value))
        return table
