# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_retriever.common.vdb.targets import VdbTarget
from nemo_retriever.ingest.execution import _raise_for_empty_ingest

_TARGET = VdbTarget(lancedb_uri="lancedb", table_name="nemo-retriever")


def test_empty_ingest_validation_accepts_rows_on_overwrite() -> None:
    _raise_for_empty_ingest(
        documents=["doc.pdf"],
        target=_TARGET,
        n_rows=3,
        result_n_rows=3,
        initial_n_rows=None,
    )


def test_empty_ingest_validation_accepts_new_rows_on_append() -> None:
    _raise_for_empty_ingest(
        documents=["doc.pdf"],
        target=_TARGET,
        n_rows=4,
        result_n_rows=1,
        initial_n_rows=3,
    )


def test_empty_ingest_validation_rejects_unknown_final_row_count() -> None:
    with pytest.raises(RuntimeError, match="could not verify rows"):
        _raise_for_empty_ingest(
            documents=["doc.pdf"],
            target=_TARGET,
            n_rows=None,
            result_n_rows=1,
            initial_n_rows=None,
        )


def test_empty_ingest_validation_rejects_unchanged_append_count() -> None:
    with pytest.raises(RuntimeError, match="did not add rows"):
        _raise_for_empty_ingest(
            documents=["doc.pdf"],
            target=_TARGET,
            n_rows=3,
            result_n_rows=1,
            initial_n_rows=3,
        )


def test_empty_ingest_validation_rejects_decreased_append_count() -> None:
    with pytest.raises(RuntimeError, match="row count decreased from 10 to 2"):
        _raise_for_empty_ingest(
            documents=["doc.pdf"],
            target=_TARGET,
            n_rows=2,
            result_n_rows=1,
            initial_n_rows=10,
        )


def test_empty_ingest_validation_rejects_zero_rows_on_overwrite() -> None:
    with pytest.raises(RuntimeError, match="produced 0 rows"):
        _raise_for_empty_ingest(
            documents=["doc.pdf"],
            target=_TARGET,
            n_rows=0,
            result_n_rows=None,
            initial_n_rows=None,
        )


def test_empty_ingest_validation_rejects_empty_current_result_before_stale_rows() -> None:
    with pytest.raises(RuntimeError, match="produced 0 rows before LanceDB write"):
        _raise_for_empty_ingest(
            documents=["doc.pdf"],
            target=_TARGET,
            n_rows=3,
            result_n_rows=0,
            initial_n_rows=None,
        )


def test_empty_ingest_validation_names_the_qdrant_target() -> None:
    target = VdbTarget(vdb_op="qdrant", table_name="docs", qdrant_url="http://qdrant:6333", qdrant_api_key="secret")
    with pytest.raises(RuntimeError) as excinfo:
        _raise_for_empty_ingest(
            documents=["doc.pdf"],
            target=target,
            n_rows=None,
            result_n_rows=1,
            initial_n_rows=None,
        )
    message = str(excinfo.value)
    assert "could not verify rows in Qdrant http://qdrant:6333/docs" in message
    assert "the Qdrant collection was not created" in message
    assert "secret" not in message
