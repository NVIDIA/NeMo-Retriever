# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import tempfile
from typing import cast

from pydantic import ValidationError
import typer

from nemo_retriever.query.options import QueryRetrievalMode, QueryRetrievalOptions

ROOT_CLI_ERRORS = (OSError, RuntimeError, ValueError, ValidationError, typer.BadParameter)

_RETRIEVAL_MODES: set[str] = {"auto", "dense", "hybrid", "sparse"}


def api_key_from_env_option(env_key: str | None) -> str | None:
    """Resolve the value of the environment variable named by ``env_key``."""
    key = (env_key or "").strip()
    if not key:
        return None
    value = os.environ.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is not set or is empty.")
    return value


def validate_retrieval_mode(retrieval_mode: str) -> QueryRetrievalMode:
    """Normalize and validate a ``--retrieval-mode`` value."""
    normalized = retrieval_mode.strip().lower()
    if normalized not in _RETRIEVAL_MODES:
        typer.echo(
            "Error: unknown --retrieval-mode " f"{retrieval_mode!r} (use 'auto', 'dense', 'hybrid', or 'sparse').",
            err=True,
        )
        raise typer.Exit(1)
    return cast(QueryRetrievalMode, normalized)


def resolve_retrieval_mode(ctx: typer.Context, retrieval_mode: str, hybrid: bool) -> QueryRetrievalMode:
    """Reconcile ``--retrieval-mode`` with the deprecated ``--hybrid`` alias."""
    resolved = validate_retrieval_mode(retrieval_mode)
    hybrid_source = ctx.get_parameter_source("hybrid")
    has_hybrid_alias = hybrid_source is not None and getattr(hybrid_source, "name", "") != "DEFAULT"
    retrieval_mode_source = ctx.get_parameter_source("retrieval_mode")
    has_retrieval_mode = retrieval_mode_source is not None and getattr(retrieval_mode_source, "name", "") != "DEFAULT"
    if has_hybrid_alias and has_retrieval_mode:
        typer.echo("Error: pass only one of --retrieval-mode or deprecated --hybrid.", err=True)
        raise typer.Exit(1)
    if has_hybrid_alias and hybrid:
        return "hybrid"
    return resolved


def build_retrieval_options(
    *,
    top_k: int,
    candidate_k: int | None,
    page_dedup: bool,
    content_types: str | None,
    retrieval_mode: QueryRetrievalMode = "auto",
) -> QueryRetrievalOptions:
    """Build retrieval options shared by the query and answer commands."""
    return QueryRetrievalOptions(
        top_k=top_k,
        candidate_k=candidate_k,
        page_dedup=page_dedup,
        content_types=content_types,
        retrieval_mode=retrieval_mode,
    )


def silence_noisy_libraries() -> None:
    # vLLM/transformers/HuggingFace otherwise emit dozens of INFO-level lines
    # and progress bars during local model startup.
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("vllm").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)


@contextlib.contextmanager
def quiet_capture():
    """Capture stdout and stderr inside the block and flush them only on errors."""
    try:
        stdout_fd, stderr_fd = sys.stdout.fileno(), sys.stderr.fileno()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        yield
        return

    saved_stdout = saved_stderr = buf = None
    try:
        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)
        buf = tempfile.TemporaryFile(mode="w+b")
        try:
            try:
                os.dup2(buf.fileno(), stdout_fd)
                os.dup2(buf.fileno(), stderr_fd)
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_stdout, stdout_fd)
                os.dup2(saved_stderr, stderr_fd)
        except BaseException:
            buf.seek(0)
            sys.stderr.buffer.write(buf.read())
            sys.stderr.flush()
            raise
    finally:
        if buf is not None:
            buf.close()
        if saved_stderr is not None:
            os.close(saved_stderr)
        if saved_stdout is not None:
            os.close(saved_stdout)
