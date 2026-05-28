# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_retriever.adapters.cli.sdk_workflow import _raise_for_empty_ingest


def test_empty_ingest_validation_accepts_rows_on_overwrite() -> None:
    _raise_for_empty_ingest(
        documents=["doc.pdf"],
        lancedb_uri="lancedb",
        table_name="nemo-retriever",
        n_rows=3,
        initial_n_rows=None,
    )


def test_empty_ingest_validation_accepts_new_rows_on_append() -> None:
    _raise_for_empty_ingest(
        documents=["doc.pdf"],
        lancedb_uri="lancedb",
        table_name="nemo-retriever",
        n_rows=4,
        initial_n_rows=3,
    )


def test_empty_ingest_validation_rejects_unknown_final_row_count() -> None:
    with pytest.raises(RuntimeError, match="could not verify rows"):
        _raise_for_empty_ingest(
            documents=["doc.pdf"],
            lancedb_uri="lancedb",
            table_name="nemo-retriever",
            n_rows=None,
            initial_n_rows=None,
        )


def test_empty_ingest_validation_rejects_unchanged_append_count() -> None:
    with pytest.raises(RuntimeError, match="did not add rows"):
        _raise_for_empty_ingest(
            documents=["doc.pdf"],
            lancedb_uri="lancedb",
            table_name="nemo-retriever",
            n_rows=3,
            initial_n_rows=3,
        )


def test_empty_ingest_validation_rejects_zero_rows_on_overwrite() -> None:
    with pytest.raises(RuntimeError, match="produced 0 rows"):
        _raise_for_empty_ingest(
            documents=["doc.pdf"],
            lancedb_uri="lancedb",
            table_name="nemo-retriever",
            n_rows=0,
            initial_n_rows=None,
        )
