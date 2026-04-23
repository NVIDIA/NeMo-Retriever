# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`nemo_retriever.generation.judge`.

``judge`` accepts a pre-built ``LLMJudge`` and constructs the
``JudgingOperator`` from the judge's ``_client.transport``.  We assert
the transport fields are plumbed correctly (including ``num_retries``,
which has historically dropped silently) and verify the DataFrame
contract via a stubbed operator so we don't require a live LLM.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_retriever.generation import judge


def _make_judge_stub(
    *,
    model: str = "test/judge",
    api_base: str = "https://judge",
    api_key: str = "sk-judge",
    num_retries: int = 5,
    timeout: float = 90.0,
    extra_params: dict | None = None,
) -> MagicMock:
    """Build a ``LLMJudge``-shaped MagicMock exposing just the used fields."""
    judge_stub = MagicMock()
    judge_stub._client = MagicMock()
    judge_stub._client.transport = MagicMock()
    judge_stub._client.transport.model = model
    judge_stub._client.transport.api_base = api_base
    judge_stub._client.transport.api_key = api_key
    judge_stub._client.transport.num_retries = num_retries
    judge_stub._client.transport.timeout = timeout
    judge_stub._client.transport.extra_params = extra_params or {}
    return judge_stub


def _stub_judging_operator(verdicts: list[dict]) -> MagicMock:
    """Build a ``JudgingOperator`` stub whose ``.run(df)`` appends *verdicts*.

    Each ``verdict`` is a dict with keys ``score``, ``reasoning``,
    ``error``.  Positional alignment with rows is caller-controlled.
    """

    def _run(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["judge_score"] = [v["score"] for v in verdicts[: len(df)]]
        out["judge_reasoning"] = [v["reasoning"] for v in verdicts[: len(df)]]
        out["judge_error"] = [v["error"] for v in verdicts[: len(df)]]
        return out

    instance = MagicMock()
    instance.run.side_effect = _run

    operator_cls = MagicMock(return_value=instance)
    return operator_cls


def _make_input_df() -> pd.DataFrame:
    """Minimal DataFrame satisfying JudgingOperator.required_columns."""
    return pd.DataFrame(
        {
            "query": ["What is RAG?"],
            "reference_answer": ["Retrieval-Augmented Generation."],
            "answer": ["RAG is Retrieval-Augmented Generation."],
        }
    )


class TestJudgeAdditiveColumns:
    """Judge must append three scoring columns without dropping existing ones."""

    def test_returns_dataframe_with_three_new_columns(self) -> None:
        df = _make_input_df()
        judge_stub = _make_judge_stub()
        op_cls = _stub_judging_operator([{"score": 5, "reasoning": "great", "error": None}])

        with patch(
            "nemo_retriever.evaluation.judging.JudgingOperator",
            op_cls,
        ):
            out = judge(df, judge=judge_stub)

        for col in ("judge_score", "judge_reasoning", "judge_error"):
            assert col in out.columns

    def test_existing_columns_preserved(self) -> None:
        df = _make_input_df()
        df["extra"] = ["keep"]
        judge_stub = _make_judge_stub()
        op_cls = _stub_judging_operator([{"score": 4, "reasoning": "ok", "error": None}])

        with patch(
            "nemo_retriever.evaluation.judging.JudgingOperator",
            op_cls,
        ):
            out = judge(df, judge=judge_stub)

        assert "extra" in out.columns
        assert out.loc[0, "extra"] == "keep"


class TestJudgeTransportPlumbing:
    """The JudgingOperator must be built from the judge's transport exactly."""

    def test_all_transport_fields_forwarded(self) -> None:
        """Regression guard: ``num_retries`` is the most-silently-dropped field."""
        df = _make_input_df()
        judge_stub = _make_judge_stub(
            model="nvidia_nim/judge-xl",
            api_base="https://nim.judge",
            api_key="sk-abc",
            num_retries=11,
            timeout=42.0,
            extra_params={"top_p": 0.5},
        )
        op_cls = _stub_judging_operator([{"score": 3, "reasoning": "meh", "error": None}])

        with patch(
            "nemo_retriever.evaluation.judging.JudgingOperator",
            op_cls,
        ):
            judge(df, judge=judge_stub)

        op_cls.assert_called_once()
        kwargs = op_cls.call_args.kwargs
        assert kwargs["model"] == "nvidia_nim/judge-xl"
        assert kwargs["api_base"] == "https://nim.judge"
        assert kwargs["api_key"] == "sk-abc"
        assert kwargs["num_retries"] == 11
        assert kwargs["timeout"] == 42.0
        assert kwargs["extra_params"] == {"top_p": 0.5}

    def test_empty_extra_params_becomes_none(self) -> None:
        df = _make_input_df()
        judge_stub = _make_judge_stub(extra_params={})
        op_cls = _stub_judging_operator([{"score": 5, "reasoning": "", "error": None}])

        with patch(
            "nemo_retriever.evaluation.judging.JudgingOperator",
            op_cls,
        ):
            judge(df, judge=judge_stub)

        assert op_cls.call_args.kwargs["extra_params"] is None


class TestJudgeValidation:
    """Required-column validation is delegated to ``EvalOperator.preprocess``."""

    @pytest.mark.parametrize(
        "missing_column",
        ["query", "reference_answer", "answer"],
    )
    def test_missing_required_column_raises_value_error(self, missing_column: str) -> None:
        df = _make_input_df().drop(columns=[missing_column])
        judge_stub = _make_judge_stub()

        with pytest.raises(ValueError, match=missing_column):
            judge(df, judge=judge_stub)
