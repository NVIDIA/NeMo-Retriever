# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from nemo_retriever.adapters.cli.ingest.graph import batch_command, local_command
from nemo_retriever.adapters.cli.ingest.service import service_command

app = typer.Typer(help="Ingest documents into Retriever indexes.", no_args_is_help=True)

app.command("local")(local_command)
app.command("batch")(batch_command)
app.command("service")(service_command)
