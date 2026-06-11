# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy end-to-end ingestion pipeline subcommand for the ``retriever`` CLI.

This package owns the ``retriever pipeline`` Typer application. Local pipeline
runs delegate ingest graph construction to the core ingest package; service
runs still build the service client path directly.

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
    from nemo_retriever.cli.pipeline.__main__ import app, run  # noqa: F401


def __getattr__(name: str) -> Any:
    if name in {"app", "run"}:
        from nemo_retriever.cli.pipeline import __main__ as _main

        return getattr(_main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
