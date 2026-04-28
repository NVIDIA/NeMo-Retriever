# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`nemo_retriever.generation.score`.

These are integration-style tests (no mocking of ``ScoringOperator``)
because ``score`` is a pure computation -- no LLM, no network -- and
exercising the real operator gives the strongest guarantee that the
DataFrame contract matches what downstream callers will see.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nemo_retriever.generation import score


def _make_scored_input(
    *,
    reference: str = "Paris",
    answer: str = "Paris",
    context: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal DataFrame with ``ScoringOperator``'s required columns."""
    return pd.DataFrame(
        {
            "query": ["What is the capital of France?"],
            "reference_answer": [reference],
            "answer": [answer],
            "context": [context if context is not None else ["Paris is the capital of France."]],
        }
    )


class TestScoreAdditiveColumns:
    """Scoring must append -- not replace -- the metric columns."""

    def test_returns_dataframe_with_four_new_columns(self) -> None:
        df = _make_scored_input()

        out = score(df)

        for col in ("answer_in_context", "token_f1", "exact_match", "failure_mode"):
            assert col in out.columns

    def test_existing_columns_preserved(self) -> None:
        df = _make_scored_input()
        df["my_extra_column"] = ["keep_me"]

        out = score(df)

        assert "my_extra_column" in out.columns
        assert out.loc[0, "my_extra_column"] == "keep_me"

    def test_input_dataframe_not_mutated(self) -> None:
        """``ScoringOperator.process`` calls ``df.copy()`` so the caller's frame is safe."""
        df = _make_scored_input()
        original_columns = list(df.columns)

        _ = score(df)

        assert list(df.columns) == original_columns


class TestScoreMetricValues:
    """Sanity-check a handful of metric outcomes for representative inputs."""

    def test_exact_match_on_identical_strings(self) -> None:
        df = _make_scored_input(reference="Paris", answer="Paris")

        out = score(df)

        assert bool(out.loc[0, "exact_match"]) is True
        assert out.loc[0, "token_f1"] == pytest.approx(1.0)

    def test_no_exact_match_on_different_strings(self) -> None:
        df = _make_scored_input(reference="Paris", answer="London")

        out = score(df)

        assert bool(out.loc[0, "exact_match"]) is False

    def test_answer_in_context_true_when_reference_keywords_present(self) -> None:
        df = _make_scored_input(
            reference="Paris",
            answer="Paris",
            context=["Paris is the capital of France."],
        )

        out = score(df)

        assert bool(out.loc[0, "answer_in_context"]) is True

    def test_answer_in_context_false_when_context_is_unrelated(self) -> None:
        df = _make_scored_input(
            reference="Paris",
            answer="Paris",
            context=["This document discusses quantum computing basics."],
        )

        out = score(df)

        assert bool(out.loc[0, "answer_in_context"]) is False


class TestScoreValidation:
    """Missing required columns must fail at preprocess with a clear error."""

    def test_missing_reference_answer_raises_value_error(self) -> None:
        df = pd.DataFrame(
            {
                "query": ["q"],
                "answer": ["a"],
                "context": [["c"]],
            }
        )

        with pytest.raises(ValueError, match="reference_answer"):
            score(df)

    def test_missing_answer_raises_value_error(self) -> None:
        df = pd.DataFrame(
            {
                "query": ["q"],
                "reference_answer": ["ref"],
                "context": [["c"]],
            }
        )

        with pytest.raises(ValueError, match="answer"):
            score(df)

    def test_missing_context_raises_value_error(self) -> None:
        df = pd.DataFrame(
            {
                "query": ["q"],
                "reference_answer": ["ref"],
                "answer": ["a"],
            }
        )

        with pytest.raises(ValueError, match="context"):
            score(df)
