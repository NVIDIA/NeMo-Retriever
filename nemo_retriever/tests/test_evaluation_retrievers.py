# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`nemo_retriever.evaluation.retrievers.FileRetriever`.

These tests pin the contract for both entry points -- the file-based
``__init__`` and the in-memory ``_from_dict`` -- and assert that both
produce instances with identical state.  That parity invariant is the
structural guard against the two construction paths silently diverging
when new instance fields are added in future.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from nemo_retriever.evaluation.retrievers import FileRetriever

_SAMPLE_QUERIES: dict[str, dict] = {
    "What is the range of the 767?": {
        "chunks": ["The 767 has a range of ~6,000 nmi.", "Variants differ."],
        "metadata": [{"source": "spec.pdf"}, {"source": "variants.pdf"}],
    },
    "How many seats does the 747 have?": {
        "chunks": ["Up to 524 passengers in 3-class config."],
        "metadata": [{"source": "747_brochure.pdf"}],
    },
}


def _write_retrieval_json(tmp_path: Path, queries: dict[str, dict]) -> Path:
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps({"queries": queries}), encoding="utf-8")
    return path


def test_init_roundtrip(tmp_path: Path) -> None:
    """Loading from JSON -> retrieve() returns the stored chunks/metadata.

    Under the :class:`~nemo_retriever.llm.types.RetrieverStrategy`
    DataFrame contract the call yields a single-row DataFrame with the
    columns ``[query, chunks, metadata]``.
    """
    path = _write_retrieval_json(tmp_path, _SAMPLE_QUERIES)

    retriever = FileRetriever(file_path=str(path))
    result = retriever.retrieve("What is the range of the 767?", top_k=2)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["query", "chunks", "metadata"]
    assert len(result) == 1
    row = result.iloc[0]
    assert row["query"] == "What is the range of the 767?"
    assert row["chunks"] == _SAMPLE_QUERIES["What is the range of the 767?"]["chunks"]
    assert row["metadata"] == _SAMPLE_QUERIES["What is the range of the 767?"]["metadata"]
    assert retriever.file_path == str(path)


def test_init_empty_raises(tmp_path: Path) -> None:
    """Empty ``queries`` dict raises ValueError with the file path in the message."""
    path = _write_retrieval_json(tmp_path, {})

    with pytest.raises(ValueError, match="no 'queries' key found") as exc_info:
        FileRetriever(file_path=str(path))

    assert str(path) in str(exc_info.value), "error must reference the offending file path"


def test_init_missing_chunks_raises(tmp_path: Path) -> None:
    """An entry without a ``chunks`` list raises ValueError."""
    path = _write_retrieval_json(tmp_path, {"a query": {"metadata": [{"source": "x"}]}})

    with pytest.raises(ValueError, match="missing a 'chunks' list"):
        FileRetriever(file_path=str(path))


def test_init_missing_file_raises(tmp_path: Path) -> None:
    """A non-existent file raises FileNotFoundError, not a cryptic IOError."""
    path = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError, match="retrieval results file not found"):
        FileRetriever(file_path=str(path))


def test_from_dict_roundtrip() -> None:
    """Loading from in-memory dict -> retrieve() returns the stored chunks."""
    retriever = FileRetriever._from_dict(_SAMPLE_QUERIES)
    result = retriever.retrieve("How many seats does the 747 have?", top_k=5)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0]["chunks"] == _SAMPLE_QUERIES["How many seats does the 747 have?"]["chunks"]
    assert retriever.file_path == "<in-memory>"


def test_from_dict_normalizes_keys() -> None:
    """Whitespace and case variations in the query still match the stored entry.

    Locks the contract with :func:`_normalize_query` -- if either
    construction path skips normalization the lookup would miss.
    """
    retriever = FileRetriever._from_dict(_SAMPLE_QUERIES)

    variants = [
        "what is the range of the 767?",
        "What is the range of the 767? ",
        "  What   is   the   range   of   the   767?  ",
    ]
    for variant in variants:
        result = retriever.retrieve(variant, top_k=2)
        assert len(result) == 1, f"normalized lookup returned no row for variant {variant!r}"
        assert result.iloc[0]["chunks"], f"normalized lookup returned empty chunks for variant {variant!r}"


def test_from_dict_empty_raises() -> None:
    """Empty dict raises ValueError whose message identifies the in-memory path."""
    with pytest.raises(ValueError, match="_from_dict: queries dict is empty"):
        FileRetriever._from_dict({})


def test_from_dict_missing_chunks_raises() -> None:
    """Entry without a ``chunks`` list raises a _from_dict-tagged ValueError."""
    with pytest.raises(ValueError, match="_from_dict: first entry is missing"):
        FileRetriever._from_dict({"a query": {"metadata": []}})


def test_init_and_from_dict_have_identical_state(tmp_path: Path) -> None:
    """Structural invariant: both construction paths produce the same instance shape.

    This is the guard against divergence that jperez flagged.  If a new
    instance field is ever added to one entry point but not the other,
    this test fails immediately -- no silent runtime bug.
    """
    path = _write_retrieval_json(tmp_path, _SAMPLE_QUERIES)

    via_init = FileRetriever(file_path=str(path))
    via_from_dict = FileRetriever._from_dict(_SAMPLE_QUERIES)

    # Both instances must expose the same set of public + private fields.
    assert set(vars(via_init).keys()) == set(vars(via_from_dict).keys())

    # And every field of the same type: catches e.g. one path forgetting
    # to initialise the lock or the miss counter.
    for field_name in vars(via_init):
        init_attr = getattr(via_init, field_name)
        from_dict_attr = getattr(via_from_dict, field_name)
        assert type(init_attr) is type(from_dict_attr), (
            f"field {field_name!r} has mismatched types: "
            f"{type(init_attr).__name__} via __init__ vs "
            f"{type(from_dict_attr).__name__} via _from_dict"
        )

    # The normalized index must contain the same keys and chunk payloads
    # regardless of entry point.
    assert via_init._norm_index.keys() == via_from_dict._norm_index.keys()
    for norm_key in via_init._norm_index:
        assert via_init._norm_index[norm_key] == via_from_dict._norm_index[norm_key]


def test_from_lancedb_save_path_sets_file_path(tmp_path: Path) -> None:
    """When ``save_path`` is provided, ``file_path`` reflects the saved path.

    Mocks :func:`query_lancedb` + :func:`write_retrieval_json` so the
    test does not depend on a live LanceDB directory.
    """
    save_path = tmp_path / "saved_retrieval.json"
    fake_meta = {"lancedb_uri": "mock"}

    with (
        patch(
            "nemo_retriever.export.query_lancedb",
            return_value=(_SAMPLE_QUERIES, fake_meta),
        ),
        patch("nemo_retriever.export.write_retrieval_json") as mock_write,
    ):
        retriever = FileRetriever.from_lancedb(
            qa_pairs=[{"query": "What is the range of the 767?"}],
            lancedb_uri="mock",
            save_path=str(save_path),
        )

    assert retriever.file_path == str(save_path)
    mock_write.assert_called_once()


def test_from_lancedb_no_save_path_keeps_memory_label() -> None:
    """Without ``save_path`` the instance reports the in-memory origin."""
    fake_meta = {"lancedb_uri": "mock"}

    with (
        patch(
            "nemo_retriever.export.query_lancedb",
            return_value=(_SAMPLE_QUERIES, fake_meta),
        ),
        patch("nemo_retriever.export.write_retrieval_json") as mock_write,
    ):
        retriever = FileRetriever.from_lancedb(
            qa_pairs=[{"query": "What is the range of the 767?"}],
            lancedb_uri="mock",
            save_path=None,
        )

    assert retriever.file_path == "<in-memory>"
    mock_write.assert_not_called()


def test_retrieve_many_returns_typed_columns(tmp_path: Path) -> None:
    """Empty and non-empty inputs both return the canonical ``[query, chunks, metadata]`` columns.

    Load-bearing invariant: the orchestrator's fast-path zips
    ``hit_df["chunks"]`` and ``hit_df["metadata"]`` positionally, so a
    DataFrame missing either column would fail with ``KeyError`` at
    runtime rather than degrading gracefully.
    """
    path = _write_retrieval_json(tmp_path, _SAMPLE_QUERIES)
    retriever = FileRetriever(file_path=str(path))

    empty_result = retriever.retrieve_many([], top_k=2)
    assert isinstance(empty_result, pd.DataFrame)
    assert list(empty_result.columns) == ["query", "chunks", "metadata"]
    assert len(empty_result) == 0

    one_result = retriever.retrieve_many(["What is the range of the 767?"], top_k=2)
    assert list(one_result.columns) == ["query", "chunks", "metadata"]
    assert len(one_result) == 1


def test_retrieve_many_matches_retrieve_per_row(tmp_path: Path) -> None:
    """Per-row delegation invariant: ``retrieve_many`` and ``retrieve`` cannot drift.

    Both methods route through ``_lookup``, so any row produced by
    ``retrieve_many`` must equal the corresponding row produced by
    ``retrieve``.  This is the structural guard that prevents the two
    surfaces from diverging if either implementation is touched.
    """
    queries = list(_SAMPLE_QUERIES.keys())
    extra_query = queries[0]
    inputs = queries + [extra_query]

    retriever = FileRetriever._from_dict(_SAMPLE_QUERIES)
    batch_df = retriever.retrieve_many(inputs, top_k=2)

    assert list(batch_df.columns) == ["query", "chunks", "metadata"]
    assert len(batch_df) == len(inputs)

    # Use a fresh retriever for the per-row comparison so the miss
    # counter on ``retriever`` isn't perturbed by these probe calls.
    probe = FileRetriever._from_dict(_SAMPLE_QUERIES)
    for i, query in enumerate(inputs):
        expected_row = probe.retrieve(query, top_k=2).iloc[0]
        actual_row = batch_df.iloc[i]
        assert actual_row["query"] == expected_row["query"]
        assert actual_row["chunks"] == expected_row["chunks"]
        assert actual_row["metadata"] == expected_row["metadata"]


def test_retrieve_many_all_misses_increments_counter(tmp_path: Path) -> None:
    """Misses are counted per-query, not per-batch.

    Confirms that a single ``retrieve_many`` call with N unknown queries
    bumps ``_miss_count`` by exactly N -- the same accounting a threaded
    loop of ``retrieve`` calls produced before this refactor.
    """
    retriever = FileRetriever._from_dict(_SAMPLE_QUERIES)
    assert retriever._miss_count == 0

    unknown = ["unknown query a", "unknown query b", "unknown query c"]
    result = retriever.retrieve_many(unknown, top_k=2)

    assert len(result) == 3
    for i, query in enumerate(unknown):
        row = result.iloc[i]
        assert row["query"] == query
        assert row["chunks"] == []
        assert row["metadata"] == []
    assert retriever._miss_count == 3


def test_retrieve_many_mixed_hits_and_misses(tmp_path: Path) -> None:
    """Row order matches input order; misses degrade in-place without perturbing hits.

    Two known queries interleaved with one unknown query -- the result
    must preserve input order, hit rows must carry their expected
    chunks, and the miss counter must increment exactly once.
    """
    retriever = FileRetriever._from_dict(_SAMPLE_QUERIES)

    known_a = "What is the range of the 767?"
    known_b = "How many seats does the 747 have?"
    unknown = "what is the wingspan of an A380?"
    inputs = [known_a, unknown, known_b]

    result = retriever.retrieve_many(inputs, top_k=2)

    assert list(result["query"]) == inputs
    assert result.iloc[0]["chunks"] == _SAMPLE_QUERIES[known_a]["chunks"]
    assert result.iloc[1]["chunks"] == []
    assert result.iloc[1]["metadata"] == []
    assert result.iloc[2]["chunks"] == _SAMPLE_QUERIES[known_b]["chunks"]
    assert retriever._miss_count == 1


def test_retrieve_top_k_zero_raises_value_error() -> None:
    """``top_k=0`` is a caller bug, not a "skip retrieval" signal.

    Without validation, ``top_k=0`` produces miss-shaped rows (empty
    ``chunks``/``metadata``) without bumping ``_miss_count`` -- a
    correctness divergence between the "real miss" and "zero-budget"
    states.  Both ``retrieve`` and ``retrieve_many`` must reject this
    consistently so the two surfaces stay behaviourally identical.
    """
    retriever = FileRetriever._from_dict(_SAMPLE_QUERIES)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("What is the range of the 767?", top_k=0)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve_many(["What is the range of the 767?"], top_k=0)


def test_retrieve_top_k_negative_raises_value_error() -> None:
    """Negative ``top_k`` triggers Python slice semantics (``list[:-1]``
    drops the last element) which silently returns an off-by-one result
    set instead of failing.  Reject it at the public boundary."""
    retriever = FileRetriever._from_dict(_SAMPLE_QUERIES)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("What is the range of the 767?", top_k=-1)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve_many(["What is the range of the 767?"], top_k=-1)
