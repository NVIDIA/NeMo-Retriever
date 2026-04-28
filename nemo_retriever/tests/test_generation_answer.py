# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`nemo_retriever.generation.answer`.

These tests cover:
* the DataFrame column contract produced by the retrieve+generate chain
* the ``reference`` broadcast rules (scalar vs. sequence)
* that the :class:`QAGenerationOperator` is constructed from the
  caller-supplied ``LiteLLMClient``'s transport + sampling
* that the retriever's ``top_k`` attribute is not mutated by overrides
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_retriever.generation import answer


def _make_hits(n: int, prefix: str = "p") -> list[dict]:
    return [
        {"text": f"{prefix}{i}", "metadata": "{}", "source": "{}", "page_number": i, "_distance": 0.5} for i in range(n)
    ]


def _make_llm(
    *,
    model: str = "test/model",
    api_base: str = "https://test",
    api_key: str = "sk-test",
    temperature: float = 0.25,
    top_p: float = 0.9,
    max_tokens: int = 512,
    num_retries: int = 4,
    timeout: float = 60.0,
    extra_params: dict | None = None,
) -> MagicMock:
    """Build a ``LiteLLMClient``-shaped MagicMock.

    Only the ``.transport`` and ``.sampling`` attributes are exercised
    by :func:`answer`, so we only need to populate them.
    """
    llm = MagicMock()
    llm.transport = MagicMock()
    llm.transport.model = model
    llm.transport.api_base = api_base
    llm.transport.api_key = api_key
    llm.transport.extra_params = extra_params or {}
    llm.transport.num_retries = num_retries
    llm.transport.timeout = timeout
    llm.sampling = MagicMock()
    llm.sampling.temperature = temperature
    llm.sampling.top_p = top_p
    llm.sampling.max_tokens = max_tokens
    return llm


def _stub_qa_operator(generated: list[str]) -> MagicMock:
    """Build a ``QAGenerationOperator`` stub that appends deterministic answers.

    The returned MagicMock class, when called, produces an instance
    whose ``.run(df)`` returns ``df`` extended with the usual generation
    columns.  Used to decouple tests from live LLM calls.
    """

    def _run(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["answer"] = generated[: len(df)]
        out["latency_s"] = [0.01] * len(df)
        out["model"] = ["test/model"] * len(df)
        out["gen_error"] = [None] * len(df)
        return out

    instance = MagicMock()
    instance.run.side_effect = _run

    operator_cls = MagicMock(return_value=instance)
    return operator_cls


class TestAnswerHappyPath:
    """End-to-end shape of the DataFrame returned by :func:`answer`."""

    def test_returns_dataframe_with_expected_columns(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(2), _make_hits(2)]
        llm = _make_llm()
        op_cls = _stub_qa_operator(["A1", "A2"])

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            op_cls,
        ):
            df = answer(retriever, ["q1", "q2"], llm=llm)

        assert isinstance(df, pd.DataFrame)
        assert {"query", "chunks", "context", "metadata", "answer", "latency_s", "model", "gen_error"}.issubset(
            df.columns
        )
        assert list(df["answer"]) == ["A1", "A2"]

    def test_context_column_mirrors_chunks(self) -> None:
        """``context`` is the column name the generation operator consumes."""
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(3)]
        llm = _make_llm()
        op_cls = _stub_qa_operator(["only"])

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            op_cls,
        ):
            df = answer(retriever, ["q"], llm=llm)

        assert df.loc[0, "context"] == df.loc[0, "chunks"]

    def test_single_string_becomes_single_row(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1)]
        llm = _make_llm()
        op_cls = _stub_qa_operator(["one"])

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            op_cls,
        ):
            df = answer(retriever, "single q", llm=llm)

        assert len(df) == 1
        assert df.loc[0, "query"] == "single q"


class TestAnswerReferenceHandling:
    """``reference`` attaches a ``reference_answer`` column when supplied."""

    def test_no_reference_omits_column(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1)]
        llm = _make_llm()
        op_cls = _stub_qa_operator(["x"])

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            op_cls,
        ):
            df = answer(retriever, ["q"], llm=llm)

        assert "reference_answer" not in df.columns

    def test_scalar_reference_broadcasts_across_rows(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1), _make_hits(1), _make_hits(1)]
        llm = _make_llm()
        op_cls = _stub_qa_operator(["a", "b", "c"])

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            op_cls,
        ):
            df = answer(retriever, ["q1", "q2", "q3"], llm=llm, reference="ONE_TRUTH")

        assert list(df["reference_answer"]) == ["ONE_TRUTH", "ONE_TRUTH", "ONE_TRUTH"]

    def test_sequence_reference_aligns_one_to_one(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1), _make_hits(1)]
        llm = _make_llm()
        op_cls = _stub_qa_operator(["x", "y"])

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            op_cls,
        ):
            df = answer(retriever, ["q1", "q2"], llm=llm, reference=["r1", "r2"])

        assert list(df["reference_answer"]) == ["r1", "r2"]

    def test_length_mismatch_raises_value_error(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1), _make_hits(1)]
        llm = _make_llm()

        with pytest.raises(ValueError, match="reference length"):
            answer(retriever, ["q1", "q2"], llm=llm, reference=["only_one"])


class TestAnswerOperatorConstruction:
    """The ``QAGenerationOperator`` must be built from the LLM's transport/sampling."""

    def test_operator_receives_transport_and_sampling_values(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1)]
        llm = _make_llm(
            model="nvidia_nim/foo",
            api_base="https://nim.example",
            api_key="sk-xyz",
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
            num_retries=9,
            timeout=45.0,
            extra_params={"stop": ["\n\n"]},
        )
        op_cls = _stub_qa_operator(["answer"])

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            op_cls,
        ):
            answer(retriever, ["q"], llm=llm)

        op_cls.assert_called_once()
        kwargs = op_cls.call_args.kwargs
        assert kwargs["model"] == "nvidia_nim/foo"
        assert kwargs["api_base"] == "https://nim.example"
        assert kwargs["api_key"] == "sk-xyz"
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.95
        assert kwargs["max_tokens"] == 2048
        assert kwargs["num_retries"] == 9
        assert kwargs["timeout"] == 45.0
        assert kwargs["extra_params"] == {"stop": ["\n\n"]}


class TestAnswerOverrideForwarding:
    """Retrieval overrides on :func:`answer` must reach the retriever."""

    def test_top_k_forwarded_to_retriever_queries(self) -> None:
        retriever = MagicMock()
        retriever.queries.return_value = [_make_hits(1)]
        llm = _make_llm()
        op_cls = _stub_qa_operator(["x"])

        with patch(
            "nemo_retriever.evaluation.generation.QAGenerationOperator",
            op_cls,
        ):
            answer(retriever, ["q"], llm=llm, top_k=7, lancedb_uri="/tmp/db")

        retriever.queries.assert_called_once_with(
            ["q"], top_k=7, embedder=None, lancedb_uri="/tmp/db", lancedb_table=None
        )
