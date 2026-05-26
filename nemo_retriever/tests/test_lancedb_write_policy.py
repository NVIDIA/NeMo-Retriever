# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

import pytest

lancedb = pytest.importorskip("lancedb")

from nemo_retriever.vdb.lancedb import LanceDB


def _records(text: str = "hello", vector: list[float] | None = None) -> list[list[dict]]:
    return [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": vector or [1.0, 0.0],
                    "content": text,
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_name": "doc.pdf"},
                },
            }
        ]
    ]


def _count_rows(uri: Path, table_name: str = "t") -> int:
    return int(lancedb.connect(str(uri)).open_table(table_name).count_rows())


def test_overwrite_same_records_twice_keeps_row_count_stable(tmp_path: Path) -> None:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, create_index=False)

    op.run(_records())
    assert _count_rows(tmp_path) == 1

    op.run(_records())
    assert _count_rows(tmp_path) == 1


def test_append_to_missing_table_creates_it(tmp_path: Path) -> None:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, overwrite=False, create_index=False)

    op.run(_records())

    assert _count_rows(tmp_path) == 1


def test_append_same_records_twice_doubles_row_count(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, overwrite=False, create_index=False)

    op.run(_records())
    with caplog.at_level(logging.WARNING):
        op.run(_records())

    assert _count_rows(tmp_path) == 2
    assert "Append mode does not deduplicate" in caplog.text


def test_append_incompatible_schema_raises_clear_error(tmp_path: Path) -> None:
    LanceDB(uri=str(tmp_path), table_name="t", vector_dim=3, create_index=False).run(_records(vector=[1.0, 0.0, 0.0]))

    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, overwrite=False, create_index=False)

    with pytest.raises(ValueError, match="incompatible field 'vector'"):
        op.run(_records())


def test_create_index_kwarg_disables_index_build_without_shadowing_method(tmp_path: Path) -> None:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, create_index=False)
    assert callable(op.create_index)
    assert op.build_index is False

    def fail_if_called(*_args, **_kwargs) -> None:
        raise AssertionError("write_to_index should not be called when create_index=False")

    op.write_to_index = fail_if_called  # type: ignore[method-assign]
    op.run(_records())

    assert _count_rows(tmp_path) == 1
