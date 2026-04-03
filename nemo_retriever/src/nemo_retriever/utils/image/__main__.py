# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from .render import app as render_app

app = typer.Typer(help="Utilities for working with images (visualization, inspection, conversions)")
app.add_typer(render_app, name="render")


def main() -> None:
    app()
