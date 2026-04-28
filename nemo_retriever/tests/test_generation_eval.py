# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`nemo_retriever.generation.eval`.

``eval`` is the end-to-end ``retrieve -> answer -> [judge] -> score``
chain.  We stub ``QAGenerationOperator`` and ``JudgingOperator`` so no
live LLM calls happen, but we exercise the real ``ScoringOperator``
because it is pure computation and exercises the ``failure_mode``
classifier -- the main value-add of :func:`eval` over a manual chain.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from nemo_retriever.generation import eval as eval_fn


def _make_hits(n: int, prefix: str = "Paris") -> list[dict]:
    return [
        {"text": f"{prefix} is the capital of France. Fact {i}.", "metadata": "{}", "source": "{}", "page_number": i}
        for i in range(n)
    ]


def _make_llm() -> MagicMock:
    llm = MagicMock()
    llm.transport = MagicMock()
    llm.transport.model = "test/gen"
    llm.transport.api_base = "https://gen"
    llm.transport.api_key = "sk-gen"
    llm.transport.extra_params = {}
    llm.transport.num_retries = 3
    llm.transport.timeout = 60.0
    llm.sampling = MagicMock()
    llm.sampling.temperature = 0.0
    llm.sampling.top_p = None
    llm.sampling.max_tokens = 4096
    return llm


def _make_judge() -> MagicMock:
    j = MagicMock()
    j._client = MagicMock()
    j._client.transport = MagicMock()
    j._client.transport.model = "test/judge"
    j._client.transport.api_base = "https://judge"
    j._client.transport.api_key = "sk-judge"
    j._client.transport.num_retries = 3
    j._client.transport.timeout = 60.0
    j._client.transport.extra_params = {}
    return j


def _qa_op_stub(answers: list[str]) -> MagicMock:
    def _run(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["answer"] = answers[: len(df)]
        out["latency_s"] = [0.01] * len(df)
        out["model"] = ["test/gen"] * len(df)
        out["gen_error"] = [None] * len(df)
        return out

    instance = MagicMock()
    instance.run.side_effect = _run
    return MagicMock(return_value=instance)


def _judge_op_stub(scores: list[int]) -> MagicMock:
    def _run(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["judge_score"] = scores[: len(df)]
        out["judge_reasoning"] = ["ok"] * len(df)
        out["judge_error"] = [None] * len(df)
        return out

    instance = MagicMock()
    instance.run.side_effect = _run
    return MagicMock(return_value=instance)


class TestEvalWithoutJudge:
    """Default path: retrieve + answer + score, no Tier-3 LLM-as-judge."""

    def test_no_judge_columns_when_judge_is_none(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(2)]
        llm = _make_llm()

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            _qa_op_stub(["Paris"]),
        ):
            df = eval_fn(retriever, ["What is the capital of France?"], llm=llm, reference="Paris")

        for col in ("judge_score", "judge_reasoning", "judge_error"):
            assert col not in df.columns

    def test_score_columns_present_without_judge(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1)]
        llm = _make_llm()

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            _qa_op_stub(["Paris"]),
        ):
            df = eval_fn(retriever, ["q"], llm=llm, reference="Paris")

        for col in ("answer_in_context", "token_f1", "exact_match", "failure_mode"):
            assert col in df.columns


class TestEvalWithJudge:
    """Judge column attachment + ``failure_mode`` classification end-to-end."""

    def test_full_column_union_when_judge_supplied(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1)]
        llm = _make_llm()
        judge = _make_judge()

        with (
            patch(
                "nemo_retriever.evaluation.generation.QAGenerationOperator",
                _qa_op_stub(["Paris"]),
            ),
            patch(
                "nemo_retriever.evaluation.judging.JudgingOperator",
                _judge_op_stub([5]),
            ),
        ):
            df = eval_fn(retriever, ["q"], llm=llm, reference="Paris", judge=judge)

        expected = {
            "query",
            "chunks",
            "context",
            "metadata",
            "reference_answer",
            "answer",
            "latency_s",
            "model",
            "gen_error",
            "judge_score",
            "judge_reasoning",
            "judge_error",
            "answer_in_context",
            "token_f1",
            "exact_match",
            "failure_mode",
        }
        assert expected.issubset(df.columns)

    def test_failure_mode_reflects_judge_score_correct(self) -> None:
        """`classify_failure` marks judge_score>=4 as 'correct'."""
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1)]
        llm = _make_llm()
        judge = _make_judge()

        with (
            patch(
                "nemo_retriever.evaluation.generation.QAGenerationOperator",
                _qa_op_stub(["Paris"]),
            ),
            patch(
                "nemo_retriever.evaluation.judging.JudgingOperator",
                _judge_op_stub([5]),
            ),
        ):
            df = eval_fn(retriever, ["q"], llm=llm, reference="Paris", judge=judge)

        assert df.loc[0, "failure_mode"] == "correct"


class TestEvalReferenceBroadcast:
    """``reference`` broadcasts the same way as in :func:`answer`."""

    def test_scalar_reference_applied_to_all_rows(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1), _make_hits(1)]
        llm = _make_llm()

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            _qa_op_stub(["Paris", "Paris"]),
        ):
            df = eval_fn(retriever, ["q1", "q2"], llm=llm, reference="Paris")

        assert list(df["reference_answer"]) == ["Paris", "Paris"]

    def test_sequence_reference_aligns_one_to_one(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1), _make_hits(1)]
        llm = _make_llm()

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            _qa_op_stub(["A1", "A2"]),
        ):
            df = eval_fn(retriever, ["q1", "q2"], llm=llm, reference=["R1", "R2"])

        assert list(df["reference_answer"]) == ["R1", "R2"]


class TestEvalRowCount:
    """One row per input query, regardless of judge presence."""

    def test_row_count_matches_query_count_without_judge(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1) for _ in range(3)]
        llm = _make_llm()

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            _qa_op_stub(["a", "b", "c"]),
        ):
            df = eval_fn(retriever, ["q1", "q2", "q3"], llm=llm, reference="x")

        assert len(df) == 3

    def test_row_count_matches_query_count_with_judge(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1) for _ in range(3)]
        llm = _make_llm()
        judge = _make_judge()

        with (
            patch(
                "nemo_retriever.evaluation.generation.QAGenerationOperator",
                _qa_op_stub(["a", "b", "c"]),
            ),
            patch(
                "nemo_retriever.evaluation.judging.JudgingOperator",
                _judge_op_stub([5, 4, 3]),
            ),
        ):
            df = eval_fn(retriever, ["q1", "q2", "q3"], llm=llm, reference="x", judge=judge)

        assert len(df) == 3
