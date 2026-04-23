# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ingestion pipeline subcommand for the ``retriever`` CLI.

This package wraps :class:`~nemo_retriever.graph_ingestor.GraphIngestor` with a
Typer application that exposes every knob needed to run a full PDF / doc /
txt / html / image / audio ingestion job, write results to LanceDB, and
optionally evaluate recall or BEIR.

It is registered on the ``retriever`` CLI as the ``pipeline`` subcommand::

    retriever pipeline run <input-dir-or-file> [OPTIONS]

The implementation historically lived in
``nemo_retriever/examples/graph_pipeline.py``; that module is now a thin
backward-compat shim that re-exports the same Typer app from
:mod:`nemo_retriever.pipeline.__main__`.

``app`` and ``run`` are exposed via lazy attribute access so that
``python -m nemo_retriever.pipeline`` can import the ``__main__`` module
cleanly (without a re-import warning).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["app", "run"]


if TYPE_CHECKING:
    from .__main__ import app, run  # noqa: F401


def __getattr__(name: str) -> Any:
    if name in {"app", "run"}:
        from . import __main__ as _main

        return getattr(_main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
