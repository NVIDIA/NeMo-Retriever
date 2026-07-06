# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typer

from nemo_retriever.cli.default_command import DefaultCommand, DefaultCommandGroup
from nemo_retriever.cli.ingest.graph_commands import _graph_ingest_command
from nemo_retriever.cli.ingest.service import _service_command
from nemo_retriever.cli.ingest.options import DEFAULT_CAPTION_MODEL, DEFAULT_EMBED_MODEL


class DefaultLocalIngestGroup(DefaultCommandGroup):
    default_command = "local"


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
    cls=DefaultCommand,
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
