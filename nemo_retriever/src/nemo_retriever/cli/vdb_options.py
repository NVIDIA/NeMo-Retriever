# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vector database selection options shared by ``retriever ingest`` and ``retriever query``."""

from __future__ import annotations

from typing import Annotated

import typer

from nemo_retriever.common.vdb.targets import DEFAULT_QDRANT_URL, VdbOpValue

VdbOpOption = Annotated[
    VdbOpValue,
    typer.Option("--vdb-op", help="Vector database backend: lancedb (embedded, default) or qdrant (server)."),
]
QdrantUrlOption = Annotated[
    str | None,
    typer.Option(
        "--qdrant-url",
        envvar="QDRANT_URL",
        help=f"Qdrant server URL for --vdb-op qdrant. Defaults to {DEFAULT_QDRANT_URL}.",
    ),
]
QdrantApiKeyOption = Annotated[
    str | None,
    typer.Option(
        "--qdrant-api-key",
        envvar="QDRANT_API_KEY",
        show_default=False,
        help="Qdrant API key for --vdb-op qdrant. Prefer setting the QDRANT_API_KEY environment variable.",
    ),
]
