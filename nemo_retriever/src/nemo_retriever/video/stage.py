# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Video extraction stage: frame extraction (+ optional OCR) sidecar CLI.

This module is a stub — full sidecar JSON output is not yet
implemented. ``retriever video extract`` and ``retriever video
discover`` exist so the CLI surface mirrors the audio module, but the
heavy lifting happens through the Ray Data graph in
:mod:`nemo_retriever.pipeline`.
"""

from __future__ import annotations

import glob as _glob
import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="video",
    help="Video extraction stage commands (frame extraction; OCR pending).",
    no_args_is_help=True,
)

DEFAULT_VIDEO_GLOBS = ["*.mp4", "*.mov", "*.mkv"]


@app.command("discover")
def discover(
    input_dir: str = typer.Argument(..., help="Directory to scan for video files."),
    glob: str = typer.Option(
        ",".join(DEFAULT_VIDEO_GLOBS),
        "--glob",
        help="Comma-separated glob(s) for discovery (default: *.mp4,*.mov,*.mkv).",
    ),
) -> None:
    """List video files under ``input_dir`` matching one of the globs."""
    patterns = [g.strip() for g in glob.split(",") if g.strip()] or DEFAULT_VIDEO_GLOBS
    root = Path(input_dir)
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(p) for p in _glob.glob(str(root / "**" / pattern), recursive=True))
    for p in sorted(set(found)):
        typer.echo(str(p))


@app.command("extract")
def extract(
    input_dir: str = typer.Argument(..., help="Directory of video files."),
    fps: float = typer.Option(1.0, "--fps", help="Frames per second to extract."),
) -> None:
    """Stub: full sidecar emission is not yet implemented.

    Use ``python -m nemo_retriever.pipeline --input-type video ...`` to
    run the full video ingestion pipeline (frames + ASR + scene fusion).
    """
    typer.echo(
        "retriever video extract is a stub. Run the full pipeline via:\n"
        "  python -m nemo_retriever.pipeline --input-type video --files <path>"
    )
    raise typer.Exit(code=0)
