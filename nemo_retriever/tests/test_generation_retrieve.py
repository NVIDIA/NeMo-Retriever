# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`nemo_retriever.generation.retrieve`.

These tests verify the bridge between ``Retriever.queries`` (which
returns ``list[list[dict]]``) and the DataFrame contract the rest of
the ``generation`` package consumes.  The ``Retriever`` is mocked so
the tests stay fast and hermetic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from nemo_retriever.generation import retrieve


def _make_hits(n: int, prefix: str = "p") -> list[dict]:
    """Build a synthetic list of ``n`` LanceDB-shaped hit dicts."""
    return [
        {
            "text": f"{prefix}{i}",
            "metadata": "{}",
            "source": "{}",
            "page_number": i,
            "_distance": float(0.5 + 0.01 * i),
        }
        for i in range(n)
    ]


class TestRetrieveHappyPath:
    """The happy path: list of queries -> DataFrame of hits."""

    def test_returns_dataframe_with_expected_columns(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(2), _make_hits(3)]

        df = retrieve(retriever, ["q1", "q2"])

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["query", "chunks", "metadata"]

    def test_one_row_per_input_query(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1), _make_hits(1), _make_hits(1)]

        df = retrieve(retriever, ["a", "b", "c"])

        assert len(df) == 3
        assert list(df["query"]) == ["a", "b", "c"]

    def test_chunks_are_text_only(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(3)]

        df = retrieve(retriever, ["q"])

        assert df.loc[0, "chunks"] == ["p0", "p1", "p2"]

    def test_metadata_is_hit_dict_without_text(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(2)]

        df = retrieve(retriever, ["q"])

        meta_list = df.loc[0, "metadata"]
        assert len(meta_list) == 2
        for item in meta_list:
            assert "text" not in item
            assert {"metadata", "source", "page_number", "_distance"}.issubset(item.keys())


class TestRetrieveStringBroadcast:
    """A single string query is broadcast to a list of length 1."""

    def test_single_string_becomes_one_row(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(2)]

        df = retrieve(retriever, "a single question")

        assert len(df) == 1
        assert df.loc[0, "query"] == "a single question"
        retriever.queries.assert_called_once()
        call_args = retriever.queries.call_args
        passed_queries = call_args.args[0] if call_args.args else call_args.kwargs.get("queries")
        assert passed_queries == ["a single question"]


class TestRetrieveEmptyInput:
    """Empty inputs return empty DataFrames without calling the retriever."""

    def test_empty_list_returns_empty_dataframe(self) -> None:
        retriever = MagicMock()

        df = retrieve(retriever, [])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["query", "chunks", "metadata"]
        retriever.queries.assert_not_called()


class TestRetrieveOverridesForwarding:
    """Per-call overrides must reach ``Retriever.queries`` verbatim."""

    def test_top_k_passed_through(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(2)]

        retrieve(retriever, ["q"], top_k=7)

        retriever.queries.assert_called_once_with(["q"], top_k=7, embedder=None, lancedb_uri=None, lancedb_table=None)

    def test_all_overrides_passed_through(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1)]

        retrieve(
            retriever,
            ["q"],
            top_k=3,
            embedder="bge",
            lancedb_uri="/tmp/db",
            lancedb_table="docs",
        )

        retriever.queries.assert_called_once_with(
            ["q"], top_k=3, embedder="bge", lancedb_uri="/tmp/db", lancedb_table="docs"
        )


class TestRetrieveNoMutation:
    """The retriever instance is not mutated -- only ``.queries()`` is invoked."""

    def test_retriever_top_k_attribute_unchanged(self) -> None:
        retriever = MagicMock()
        retriever.top_k = 5
        retriever.queries.return_value = [_make_hits(2)]

        retrieve(retriever, ["q"], top_k=99)

        assert retriever.top_k == 5
