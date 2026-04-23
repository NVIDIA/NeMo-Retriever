# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ``QAEvalPipeline._prepare_dataframe`` reference-answer
fallback.

Before the fix in this PR, line 268 of ``orchestrator.py`` used::

    reference = pair.get("reference_answer") or pair["answer"]

which raised ``KeyError('answer')`` in two situations:

1. Neither ``reference_answer`` nor ``answer`` key is present in the qa pair.
2. ``reference_answer`` is the empty string ``""`` (falsy), and ``answer``
   is missing.

Both cases were silently caught by the surrounding ``except Exception``
handler and surfaced to users as misleading ``"Retrieval for query [..]
failed"`` log lines, even though retrieval was never attempted.

The fix mirrors the identical fallback already used on line 294::

    reference = pair.get("reference_answer") or pair.get("answer", "")

so both cases now yield an empty-string reference and retrieval proceeds
normally. The row still participates in downstream scoring; the empty
reference simply produces an empty ``answer_in_context``/``token_f1``
value, which is the correct behaviour for a missing ground-truth pair.

These tests additionally guard against the two lines drifting apart again
by asserting identical behaviour between the happy-path and exception-path
on the same malformed input.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nemo_retriever.evaluation.orchestrator import QAEvalPipeline


def _empty_retrieval_df() -> pd.DataFrame:
    """Return a single-row DataFrame matching the RetrieverStrategy contract."""
    return pd.DataFrame([{"query": "", "chunks": [], "metadata": []}])


def _make_pipeline(*, retrieve_side_effect: Any = None) -> QAEvalPipeline:
    """Build a QAEvalPipeline with mocked retriever/llm/judge.

    The retriever returns a stable empty single-row DataFrame (matching
    the :class:`~nemo_retriever.llm.types.RetrieverStrategy` contract)
    unless a custom ``retrieve_side_effect`` is provided (e.g. to simulate
    a real retrieval failure for the exception-path test).
    """
    retriever = MagicMock()
    if retrieve_side_effect is None:
        retriever.retrieve.return_value = _empty_retrieval_df()
    else:
        retriever.retrieve.side_effect = retrieve_side_effect

    llm = MagicMock()
    judge = MagicMock()

    return QAEvalPipeline(
        retriever=retriever,
        llm_clients={"m": llm},
        judge=judge,
        top_k=3,
        max_workers=1,
    )


class TestReferenceAnswerFallback:
    """Line 268 must mirror line 294's fallback semantics exactly."""

    def test_missing_both_keys_does_not_raise(self) -> None:
        """No ``KeyError`` when neither ``reference_answer`` nor ``answer``
        is present; the row is still emitted with an empty reference."""
        pipeline = _make_pipeline()

        df = pipeline._prepare_dataframe([{"query": "what is foo?"}])

        assert len(df) == 1
        assert df.iloc[0]["query"] == "what is foo?"
        assert df.iloc[0]["reference_answer"] == ""
        pipeline.retriever.retrieve.assert_called_once_with("what is foo?", 3)

    def test_empty_reference_answer_falls_through_to_answer(self) -> None:
        """``reference_answer == ""`` is falsy in Python, so the fallback
        kicks in. ``answer`` wins when present."""
        pipeline = _make_pipeline()

        df = pipeline._prepare_dataframe([{"query": "q", "reference_answer": "", "answer": "alt"}])

        assert df.iloc[0]["reference_answer"] == "alt"

    def test_empty_reference_answer_without_answer_key_does_not_raise(self) -> None:
        """This is the subtle second crash path the reviewer identified:
        empty reference_answer + no answer key used to raise ``KeyError``
        because ``or`` treats "" as falsy and fell through to
        ``pair["answer"]``. It must now yield an empty string instead."""
        pipeline = _make_pipeline()

        df = pipeline._prepare_dataframe([{"query": "q", "reference_answer": ""}])

        assert df.iloc[0]["reference_answer"] == ""

    def test_reference_answer_present_is_preserved(self) -> None:
        """Happy path regression guard: an explicit reference wins."""
        pipeline = _make_pipeline()

        df = pipeline._prepare_dataframe([{"query": "q", "reference_answer": "truth", "answer": "other"}])

        assert df.iloc[0]["reference_answer"] == "truth"

    def test_only_answer_key_is_used_as_legacy_fallback(self) -> None:
        """Backward-compat path documented in ``evaluate.__doc__``: a legacy
        ground-truth CSV only has ``answer``, not ``reference_answer``."""
        pipeline = _make_pipeline()

        df = pipeline._prepare_dataframe([{"query": "q", "answer": "legacy"}])

        assert df.iloc[0]["reference_answer"] == "legacy"


class TestRetrievalExceptionPathMatchesHappyPath:
    """Structural invariant: the exception-branch fallback at line 294 and
    the happy-branch fallback at line 268 must produce the same reference
    string for the same input. This test locks them together so a future
    refactor cannot reintroduce the divergence Greptile flagged."""

    @pytest.mark.parametrize(
        "pair,expected_reference",
        [
            ({"query": "q"}, ""),
            ({"query": "q", "reference_answer": ""}, ""),
            ({"query": "q", "reference_answer": "r"}, "r"),
            ({"query": "q", "answer": "a"}, "a"),
            ({"query": "q", "reference_answer": "", "answer": "a"}, "a"),
        ],
    )
    def test_both_branches_agree(self, pair: dict, expected_reference: str) -> None:
        happy_pipeline = _make_pipeline()
        happy_df = happy_pipeline._prepare_dataframe([pair])
        assert happy_df.iloc[0]["reference_answer"] == expected_reference

        boom_pipeline = _make_pipeline(retrieve_side_effect=RuntimeError("lancedb down"))
        boom_df = boom_pipeline._prepare_dataframe([pair])
        assert boom_df.iloc[0]["reference_answer"] == expected_reference
        assert boom_df.iloc[0]["context"] == []
