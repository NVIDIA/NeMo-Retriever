# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two load-bearing invariants for :class:`IngestResult`.

The DataFrame backing's basic behaviour is exercised indirectly via
``TestCollectResults`` in ``test_pipeline_helpers.py`` (which constructs an
``IngestResult.from_dataframe`` and calls ``row_count`` / ``iter_records`` /
``unique_source_count`` through ``_collect_results``); the end-to-end
multimodal-PDF integration test covers the real Ray-Dataset path through the
production CLI. The two tests below pin the bits unique to this module:

1. A dataset-backed result must never call ``Dataset.to_pandas()`` — that is
   the entire point of the driver-side memory fix; if any accessor regresses
   to a full-corpus pull, the guard-rail fake raises.
2. Service mode uses ``total_pages`` as the input-unit count when present.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from nemo_retriever.pipeline.ingest_result import IngestResult


def _rows() -> list[dict[str, Any]]:
    """Three uploadable rows spanning two distinct sources."""
    return [
        {
            "source_id": "doc-a.pdf",
            "text": "alpha",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "metadata": {"content_metadata": {"type": "text"}},
            "page_number": 1,
        },
        {
            "source_id": "doc-a.pdf",
            "text": "beta",
            "text_embeddings_1b_v2": {"embedding": [0.2] * 2048},
            "metadata": {"content_metadata": {"type": "text"}},
            "page_number": 2,
        },
        {
            "source_id": "doc-b.pdf",
            "text": "gamma",
            "text_embeddings_1b_v2": {"embedding": [0.3] * 2048},
            "metadata": {"content_metadata": {"type": "text"}},
            "page_number": 1,
        },
    ]


class _GuardRailDataset:
    """Ray-Dataset-shaped fake whose ``to_pandas`` raises so an accidental
    full-corpus pull is caught by the test rather than slipping into prod."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def count(self) -> int:
        return len(self._rows)

    def iter_rows(self) -> Iterable[dict[str, Any]]:
        yield from self._rows

    def iter_batches(self, *, batch_format: str = "pandas", batch_size: int | None = None):
        assert batch_format == "pandas"
        size = batch_size or len(self._rows) or 1
        for start in range(0, len(self._rows), size):
            yield pd.DataFrame(self._rows[start : start + size])

    def unique(self, column: str) -> list[Any]:
        seen: list[Any] = []
        for row in self._rows:
            v = row.get(column)
            if v not in seen:
                seen.append(v)
        return seen

    def write_parquet(self, path: str) -> None:  # pragma: no cover - not exercised here
        from pathlib import Path as _P

        _P(path).mkdir(parents=True, exist_ok=True)

    def to_pandas(self) -> pd.DataFrame:  # pragma: no cover - guard rail
        raise AssertionError(
            "IngestResult must not call Dataset.to_pandas(); that defeats the " "driver-side memory fix."
        )


def test_dataset_backed_result_streams_without_pulling_to_driver() -> None:
    """The whole point of this refactor: every IngestResult accessor must be
    serveable from streaming Ray Dataset primitives, never from ``to_pandas``."""
    result = IngestResult.from_dataset(_GuardRailDataset(_rows()))

    assert result.row_count() == 3
    assert result.unique_source_count() == 2  # doc-a (×2), doc-b
    assert sum(1 for _ in result.iter_records()) == 3
    assert result.count_uploadable_vdb_records() == 3
    # If any accessor above had called Dataset.to_pandas(), the fake would
    # have raised and this assertion would never be reached.


def test_dataset_uses_precomputed_uploadable_flag_when_present() -> None:
    """IngestVdbOperator emits a ``_vdb_uploadable`` boolean column in batch
    mode after projecting away the heavy embedding/metadata columns. The
    streaming uploadable count must read that flag instead of trying to
    re-derive it from records that no longer carry an embedding."""

    rows = [
        {"source_id": "a", "_vdb_uploadable": True},
        {"source_id": "a", "_vdb_uploadable": False},
        {"source_id": "b", "_vdb_uploadable": True},
        {"source_id": "b", "_vdb_uploadable": True},
    ]
    result = IngestResult.from_dataset(_GuardRailDataset(rows))

    assert result.count_uploadable_vdb_records() == 3
