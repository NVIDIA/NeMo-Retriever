# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming round-trip for ``LanceDB.append`` + ``LanceDB.build_index``.

One direct test pins two invariants that together describe the streaming
contract used by ``IngestVdbOperator`` and ``GraphIngestor._finalize_vdb_upload``:

1. **Semantics** — first ``append(overwrite=True)`` creates the table,
   subsequent ``append(overwrite=False)`` calls add rows, ``build_index``
   then builds the vector index on the populated table.
2. **Connection caching** — every per-batch ``lancedb.connect()`` +
   ``open_table()`` re-scans Lance's manifest, which grows with every prior
   commit; the operator must cache the handle so repeated appends pay only
   the commit cost. (This was the throughput killer pre-cache.)

The end-to-end ingest+retrieve integration test covers the full pipeline; this
keeps a fast regression net around both invariants without the multi-minute
pipeline cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb

from nemo_retriever.vdb.lancedb import LanceDB


VECTOR_DIM = 8


def _records(*texts: str) -> list[list[dict[str, Any]]]:
    inner = [
        {
            "document_type": "text",
            "metadata": {
                "embedding": [0.1 + 0.01 * i] * VECTOR_DIM,
                "content": text,
                "content_metadata": {"type": "text", "page_number": i + 1},
                "source_metadata": {"source_id": f"/tmp/doc-{i}.pdf"},
            },
        }
        for i, text in enumerate(texts)
    ]
    return [inner]


def test_streaming_append_round_trip_with_cached_connection(tmp_path: Path, monkeypatch) -> None:
    from nemo_retriever.vdb import lancedb as lancedb_module

    connect_count = 0
    real_connect = lancedb_module.lancedb.connect

    def _counting_connect(*args, **kwargs):
        nonlocal connect_count
        connect_count += 1
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(lancedb_module.lancedb, "connect", _counting_connect)

    vdb = LanceDB(uri=str(tmp_path / "db"), table_name="stream", vector_dim=VECTOR_DIM, hybrid=False)

    vdb.append(_records("a", "b"), overwrite=True)
    connect_count_after_first_append = connect_count
    vdb.append(_records("c"), overwrite=False)
    vdb.append(_records("d"), overwrite=False)
    vdb.append(_records("e"), overwrite=False)

    # Three follow-up appends must reuse the cached connection — that's the
    # whole point of the cache, and a regression here would re-introduce the
    # per-batch manifest re-scan throughput collapse.
    assert connect_count == connect_count_after_first_append, (
        f"3 streaming appends opened " f"{connect_count - connect_count_after_first_append} extra connections"
    )

    vdb.build_index()

    table = lancedb.connect(uri=vdb.uri).open_table(vdb.table_name)
    assert table.count_rows() == 5
    indexed_columns = {idx.columns[0] if idx.columns else "" for idx in table.list_indices()}
    assert "vector" in indexed_columns
