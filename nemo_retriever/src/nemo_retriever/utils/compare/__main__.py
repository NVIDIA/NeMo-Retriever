# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import import_module

import typer

app = typer.Typer(help="Comparison utilities")


def _register_optional_compare_commands() -> None:
    for module_name, command_name in (
        ("compare_json", "json"),
        ("compare_results", "results"),
    ):
        try:
            module = import_module(f"{__package__}.{module_name}")
        except ModuleNotFoundError:
            continue
        app.add_typer(module.app, name=command_name)


_register_optional_compare_commands()


def main() -> None:
    app()
