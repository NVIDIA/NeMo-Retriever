# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer


app = typer.Typer(help="Simple non-distributed pipeline for local development, debugging, and research.")


def main():
    app()
