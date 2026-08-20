# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IVF num_partitions clamping for small LanceDB tables."""

from __future__ import annotations

import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

lancedb = pytest.importorskip("lancedb")

from nemo_retriever.common.vdb.lancedb import (  # noqa: E402
    LanceDB,
    _effective_ivf_num_partitions,
    _is_ivf_vector_index,
)


def test_is_ivf_vector_index() -> None:
    assert _is_ivf_vector_index("IVF_HNSW_SQ") is True
    assert _is_ivf_vector_index("ivf_pq") is True
    assert _is_ivf_vector_index("FLAT") is False
    assert _is_ivf_vector_index("") is False


@pytest.mark.parametrize(
    "num_rows, requested, expected",
    [
        (0, 16, None),
        (1, 16, None),
        (2, 16, 1),
        (5, 16, 4),
        (100, 16, 16),
    ],
)
def test_effective_ivf_num_partitions(num_rows: int, requested: int, expected: int | None) -> None:
    assert _effective_ivf_num_partitions(num_rows, requested) == expected


def test_write_to_index_clamps_partitions_for_five_rows() -> None:
    """Regression: default num_partitions=16 must not fail K-means with ~5 vectors."""
    d = tempfile.mkdtemp()
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
        ]
    )
    rows = [{"vector": [float(i), 0.0], "text": f"t{i}", "metadata": "{}", "source": "{}"} for i in range(5)]
    db = lancedb.connect(d)
    db.create_table("small", rows, schema=schema, mode="overwrite")
    table = db.open_table("small")

    op = LanceDB(
        uri=d,
        table_name="small",
        overwrite=False,
        vector_dim=2,
        num_partitions=16,
        validate_vector_length=False,
    )
    op.write_to_index([], table=table, num_partitions=16, index_type="IVF_HNSW_SQ")
    assert table.list_indices()


def test_write_to_index_skips_vector_index_single_row() -> None:
    d = tempfile.mkdtemp()
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
        ]
    )
    rows = [{"vector": [1.0, 0.0], "text": "only", "metadata": "{}", "source": "{}"}]
    db = lancedb.connect(d)
    db.create_table("one", rows, schema=schema, mode="overwrite")
    table = db.open_table("one")

    op = LanceDB(
        uri=d,
        table_name="one",
        overwrite=False,
        vector_dim=2,
        num_partitions=16,
        validate_vector_length=False,
    )
    op.write_to_index([], table=table, num_partitions=16, hybrid=False)
    assert not table.list_indices()


def test_run_serializes_write_and_index_transactions(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Concurrent service callbacks must not submit competing CreateIndex commits."""
    op = LanceDB(uri=str(tmp_path), table_name="concurrent", vector_dim=2)
    first_write_started = threading.Event()
    allow_first_write_to_finish = threading.Event()
    create_calls: list[int] = []
    calls_lock = threading.Lock()

    def create_index(*, records, table_name):
        with calls_lock:
            create_calls.append(1)
            call_number = len(create_calls)
        if call_number == 1:
            first_write_started.set()
            assert allow_first_write_to_finish.wait(timeout=5)
        return object()

    monkeypatch.setattr(op, "create_index", create_index)
    monkeypatch.setattr(op, "write_to_index", lambda *args, **kwargs: None)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(op.run, [])
        assert first_write_started.wait(timeout=5)
        second = executor.submit(op.run, [])
        # The second run cannot call create_index until the first transaction
        # releases the backend write lock.
        assert create_calls == [1]
        allow_first_write_to_finish.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert create_calls == [1, 1]
