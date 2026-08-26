# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embed failures that mean rows were lost.

Kept in their own module, with no third-party imports, so the local embedders
and the embed runtime can both import them on the endpoint-only path.
"""

from __future__ import annotations

import logging
from typing import Sequence

from nemo_retriever.models.nim.error_reporter import report_error

logger = logging.getLogger(__name__)


class LocalEmbedderReturnedNothingError(RuntimeError):
    """An in-process embedder returned no vectors for a non-empty batch."""


class LocalEmbedderRowsLostError(RuntimeError):
    """An in-process embedder produced no vector for some rows of a batch.

    Raised by :func:`report_lost_rows`. ``models/inference/runtime.py``
    classifies both errors in this module as fatal by name.

    Args:
        lost: Number of rows with no vector.
        total: Size of the batch.
        embedder: Class name of the embedder that lost them.
    """

    def __init__(self, *, lost: int, total: int, embedder: str) -> None:
        self.lost = int(lost)
        self.total = int(total)
        self.embedder = str(embedder)
        super().__init__(
            f"{embedder} returned no vector for {lost} of {total} row(s) in this batch. "
            "Continuing would pad or drop those rows, hide the loss, and allow an invalid or "
            "incomplete index to be published. This normally means the in-process engine failed "
            "for the batch - check the embed actor logs for engine initialization or out-of-memory "
            "errors."
        )


def _has_no_vector(vector: object) -> bool:
    """Return whether ``vector`` is empty or absent.

    Explicit rather than truthy: ``not vector`` raises on a multi-element numpy
    array.
    """
    if vector is None:
        return True
    if hasattr(vector, "__len__"):
        return len(vector) == 0
    return False


def report_lost_rows(vectors: Sequence[Sequence[float]], *, embedder: str) -> int:
    """Raise :class:`LocalEmbedderRowsLostError` if any row came back empty.

    Returns ``0`` when nothing was lost; never returns a non-zero count.

    Called from the local embedders' ``_finalize_vectors`` before they discard
    empty placeholders. Raising before padding is required because a padded row
    has the right width, so ``has_embedding`` would report ``True`` for a row
    carrying nothing.
    """
    lost = sum(1 for vector in vectors if _has_no_vector(vector))
    if not lost:
        return 0

    exc = LocalEmbedderRowsLostError(lost=lost, total=len(vectors), embedder=embedder)
    logger.error("%s", exc)
    report_error("embed", exc)
    raise exc
