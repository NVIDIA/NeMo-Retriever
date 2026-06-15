# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ingest DataFrame transport round-trip helpers."""

from __future__ import annotations

import pandas as pd

from nemo_retriever.ingestor.results import (
    concat_ingest_results,
    dataframe_from_transport_records,
    dataframe_to_transport_records,
)
from nemo_retriever.service.services.pipeline_executor import _sanitize_result_data


def test_transport_preserves_all_columns() -> None:
    df = pd.DataFrame(
        {
            "path": ["/a.pdf"],
            "page_number": [1],
            "text": ["hello"],
            "bytes": [b"pdf-bytes"],
            "page_image": [b"img"],
            "images": [[{"x": 1}]],
        }
    )
    records = dataframe_to_transport_records(df)
    assert set(records[0]) == set(df.columns)
    assert records[0]["bytes"] == "<bytes len=9>"
    assert records[0]["page_image"] == "<bytes len=3>"


def test_round_trip_matches_inprocess_column_layout() -> None:
    df = pd.DataFrame(
        {
            "path": ["/a.pdf", "/a.pdf"],
            "page_number": [1, 2],
            "text": ["a", "b"],
            "metadata": [{"type": "text"}, {"type": "text"}],
        }
    )
    rebuilt = dataframe_from_transport_records(dataframe_to_transport_records(df))
    assert list(rebuilt.columns) == list(df.columns)
    assert len(rebuilt) == len(df)
    assert rebuilt["text"].tolist() == df["text"].tolist()


def test_sanitize_result_data_delegates_to_shared_helper() -> None:
    df = pd.DataFrame({"path": ["/x.pdf"], "bytes": [b"x"]})
    assert _sanitize_result_data(df) == dataframe_to_transport_records(df)


def test_concat_ingest_results_follows_document_order() -> None:
    rows_a = [{"path": "/a.pdf", "page_number": 1, "text": "a"}]
    rows_b = [{"path": "/b.pdf", "page_number": 1, "text": "b"}]
    combined = concat_ingest_results(
        {"doc-b": rows_b, "doc-a": rows_a},
        ["doc-a", "doc-b"],
    )
    assert combined["path"].tolist() == ["/a.pdf", "/b.pdf"]
    assert list(combined.columns) == ["path", "page_number", "text"]
