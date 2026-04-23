# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward-compat shim for the graph ingestion pipeline.

The implementation was moved to :mod:`nemo_retriever.pipeline` and is exposed
as the ``retriever pipeline run`` CLI subcommand.

This module re-exports the same Typer :data:`app` and keeps the
``python -m nemo_retriever.examples.graph_pipeline <args>`` entry point
working so existing callers (notably
:mod:`nemo_retriever.harness.run`) do not need to change.

New code should invoke the pipeline via one of the following:

* ``retriever pipeline run <input> [OPTIONS]``
* ``python -m nemo_retriever.pipeline <input> [OPTIONS]``
* ``from nemo_retriever.pipeline import app`` (Typer app) or
  ``from nemo_retriever.pipeline import run`` (command callable)

Module-level names that used to live in the monolithic script (``GraphIngestor``,
``_resolve_file_patterns``, etc.) are re-exported for tests and harness code that
patches or imports them from this module.
"""

from __future__ import annotations

from nemo_retriever.audio import asr_params_from_env
from nemo_retriever.graph_ingestor import GraphIngestor
from nemo_retriever.pipeline import __main__ as _pipeline_main
from nemo_retriever.vector_store.lancedb_store import handle_lancedb

# Typer app and supporting hooks (same as :mod:`nemo_retriever.pipeline.__main__`).
app = _pipeline_main.app
_ensure_lancedb_table = _pipeline_main._ensure_lancedb_table
_resolve_file_patterns = _pipeline_main._resolve_file_patterns

__all__ = [
    "GraphIngestor",
    "_ensure_lancedb_table",
    "_resolve_file_patterns",
    "app",
    "asr_params_from_env",
    "handle_lancedb",
]

if __name__ == "__main__":
    app()
