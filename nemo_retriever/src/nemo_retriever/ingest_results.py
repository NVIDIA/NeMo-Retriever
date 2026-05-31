# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Serialize and deserialize ingest pipeline DataFrames for service transport.

The retriever service returns per-document rows over HTTP; these helpers
keep the wire format aligned with the ``pandas.DataFrame`` produced by
:meth:`nemo_retriever.graph_ingestor.GraphIngestor.ingest` in
``inprocess`` and ``batch`` run modes (same column names and row shape,
with large/binary cell values replaced by compact placeholders).
"""

from __future__ import annotations

from typing import Any

import numpy as np

_MAX_STR_LEN = 500


def sanitize_cell_value(val: Any) -> Any:
    """Convert a single cell value to a JSON-safe, memory-friendly form."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return f"<ndarray shape={val.shape} dtype={val.dtype}>"
    if isinstance(val, (list, tuple)) and len(val) > 20:
        return f"<{type(val).__name__} len={len(val)}>"
    if isinstance(val, bytes):
        return f"<bytes len={len(val)}>"
    if isinstance(val, str) and len(val) > _MAX_STR_LEN:
        return val[:_MAX_STR_LEN] + f"…[{len(val)} chars total]"
    return val


def dataframe_to_transport_records(df: Any) -> list[dict[str, Any]]:
    """Serialize a pipeline DataFrame to JSON-safe row dicts.

    All columns are retained so the reconstructed frame matches the
    column layout of ``GraphIngestor.ingest()`` output; only cell values
    are sanitized to stay within service memory/transport limits.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected pandas.DataFrame, got {type(df).__name__}")
    records = df.to_dict(orient="records")
    return [{k: sanitize_cell_value(v) for k, v in row.items()} for row in records]


def dataframe_from_transport_records(records: list[dict[str, Any]]) -> Any:
    """Rebuild a pipeline DataFrame from transport row dicts."""
    import pandas as pd

    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


def concat_ingest_results(
    rows_by_document: dict[str, list[dict[str, Any]]],
    document_order: list[str],
) -> Any:
    """Concatenate per-document transport rows in upload order.

    Mirrors how :class:`~nemo_retriever.graph.executor.InprocessExecutor`
    processes a list of input paths as one combined result frame.
    """
    import pandas as pd

    frames: list[pd.DataFrame] = []
    for doc_id in document_order:
        rows = rows_by_document.get(doc_id)
        if rows:
            frames.append(dataframe_from_transport_records(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
