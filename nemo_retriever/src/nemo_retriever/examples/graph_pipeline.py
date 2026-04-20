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
"""

from __future__ import annotations

from nemo_retriever.pipeline.__main__ import app

if __name__ == "__main__":
    app()
