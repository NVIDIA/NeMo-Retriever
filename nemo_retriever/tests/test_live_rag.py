# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the live RAG SDK surface on ``Retriever``.

Covers:
    * Protocol compliance of ``Retriever`` against ``RetrieverStrategy``.
    * ``Retriever.retrieve`` shape (``RetrievalResult`` with aligned
      ``chunks`` / ``metadata`` from the raw ``.query()`` hits).
    * ``Retriever.answer`` for all four tiers:
        - no reference -> scoring and judge skipped.
        - reference without judge -> Tier 1+2 populated, Tier 3 None.
        - reference with judge -> all tiers populated, ``failure_mode`` set.
        - judge without reference -> ``ValueError``.
    * Generation error short-circuits scoring and judge.
    * Scoring runs concurrently with the judge (wall-clock proof).
    * ``RetrieverPipelineBuilder`` composition and skip-steps behaviour.
    * ``LiveRetrievalOperator.process`` populates ``context`` and
      ``context_metadata``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_retriever():
    """Build a bare ``Retriever`` instance with all defaults."""
    from nemo_retriever.retriever import Retriever

    return Retriever()


def _fake_hits() -> list[dict]:
    """Three fake LanceDB hits matching the shape Retriever.query() returns."""
    return [
        {"text": "Retrieval augmented generation combines context and LLMs.", "source": "doc-1.pdf", "page_number": 1},
        {
            "text": "RAG pipelines retrieve passages then feed them to a generator.",
            "source": "doc-1.pdf",
            "page_number": 2,
        },
        {"text": "Noisy unrelated chunk.", "source": "doc-9.pdf", "page_number": 7},
    ]


def _fake_generation(answer: str = "RAG retrieves context and uses an LLM.", error: str | None = None):
    """Build a GenerationResult with the given answer / error."""
    from nemo_retriever.llm.types import GenerationResult

    return GenerationResult(answer=answer, latency_s=0.12, model="fake-llm/test", error=error)


def _fake_judge_result(score: int | None = 5, reasoning: str = "correct and complete"):
    from nemo_retriever.llm.types import JudgeResult

    return JudgeResult(score=score, reasoning=reasoning, error=None)


class TestRetrieveProtocol:
    """Retriever as a RetrieverStrategy adapter over .query()."""

    def test_retriever_satisfies_protocol(self):
        """isinstance check must pass via @runtime_checkable."""
        from nemo_retriever.llm.types import RetrieverStrategy

        r = _make_retriever()
        assert isinstance(r, RetrieverStrategy)

    def test_retrieve_returns_result_shape(self):
        """``retrieve`` adapts ``.query()`` hits into a RetrievalResult."""
        from nemo_retriever.llm.types import RetrievalResult

        r = _make_retriever()
        with patch.object(r, "query", return_value=_fake_hits()) as mock_query:
            result = r.retrieve("What is RAG?", top_k=3)

        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) == 3
        assert result.chunks[0].startswith("Retrieval augmented")
        assert len(result.metadata) == 3
        assert result.metadata[0] == {"source": "doc-1.pdf", "page_number": 1}
        assert "text" not in result.metadata[0]
        mock_query.assert_called_once()

    def test_retrieve_top_k_override_is_scoped(self):
        """``top_k`` override applies only for the call, then restores."""
        r = _make_retriever()
        original_top_k = r.top_k
        with patch.object(r, "query", return_value=_fake_hits()):
            r.retrieve("q", top_k=3)
        assert r.top_k == original_top_k


class TestAnswer:
    """Retriever.answer -- retrieve -> generate -> optional scoring + judge."""

    def test_answer_without_reference(self):
        """No reference, no judge -> scoring / judge fields all None."""
        r = _make_retriever()
        llm = MagicMock()
        llm.generate.return_value = _fake_generation()

        with patch.object(r, "query", return_value=_fake_hits()):
            result = r.answer("q?", llm=llm)

        assert result.answer == "RAG retrieves context and uses an LLM."
        assert result.chunks and result.metadata
        assert result.model == "fake-llm/test"
        assert result.error is None
        assert result.token_f1 is None
        assert result.exact_match is None
        assert result.answer_in_context is None
        assert result.judge_score is None
        assert result.failure_mode is None

    def test_answer_with_reference_no_judge(self):
        """Reference supplied -> Tier 1+2 populated, Tier 3 left None."""
        r = _make_retriever()
        llm = MagicMock()
        llm.generate.return_value = _fake_generation(
            answer="RAG retrieves passages and feeds them to a generator.",
        )

        with patch.object(r, "query", return_value=_fake_hits()):
            result = r.answer(
                "What is RAG?",
                llm=llm,
                reference="RAG retrieves passages and feeds them to a generator.",
            )

        assert result.token_f1 == pytest.approx(1.0, abs=1e-6)
        assert result.exact_match is True
        assert result.answer_in_context is not None
        assert result.judge_score is None
        assert result.judge_reasoning is None

    def test_answer_with_reference_and_judge(self):
        """All tiers populated, ``failure_mode`` derived from combined signals."""
        r = _make_retriever()
        llm = MagicMock()
        llm.generate.return_value = _fake_generation()
        judge = MagicMock()
        judge.judge.return_value = _fake_judge_result(score=5)

        with patch.object(r, "query", return_value=_fake_hits()):
            result = r.answer(
                "What is RAG?",
                llm=llm,
                judge=judge,
                reference="RAG retrieves context and uses an LLM.",
            )

        assert result.judge_score == 5
        assert result.judge_reasoning == "correct and complete"
        assert result.token_f1 is not None
        assert result.exact_match is not None
        assert result.answer_in_context is not None
        assert result.failure_mode == "correct"
        judge.judge.assert_called_once()

    def test_answer_judge_requires_reference(self):
        """Passing a judge without a reference must raise ValueError."""
        r = _make_retriever()
        llm = MagicMock()
        judge = MagicMock()

        with pytest.raises(ValueError, match="judge requires reference"):
            r.answer("q", llm=llm, judge=judge)

    def test_answer_generation_error_short_circuits(self):
        """On generation error: result.error set, scoring and judge skipped."""
        r = _make_retriever()
        llm = MagicMock()
        llm.generate.return_value = _fake_generation(answer="", error="TimeoutError")
        judge = MagicMock()

        with patch.object(r, "query", return_value=_fake_hits()):
            result = r.answer(
                "q",
                llm=llm,
                judge=judge,
                reference="expected",
            )

        assert result.error == "TimeoutError"
        assert result.token_f1 is None
        assert result.judge_score is None
        assert result.failure_mode is None
        judge.judge.assert_not_called()

    def test_answer_concurrent_scoring_and_judge(self):
        """Scoring + judge must run concurrently, not serially.

        We make the judge sleep 400ms.  Scoring is sub-millisecond pure-CPU,
        so if scoring + judge run in parallel the total wall time is
        dominated by the judge.  A serial implementation would add scoring
        time on top; on modern CPUs scoring is <5ms so the margin is tight,
        but we validate the upper bound at judge_time + 200ms to keep the
        test robust under CI jitter.
        """
        r = _make_retriever()
        llm = MagicMock()
        llm.generate.return_value = _fake_generation()

        judge_latency = 0.4
        judge = MagicMock()

        def _slow_judge(query, reference, candidate):
            time.sleep(judge_latency)
            return _fake_judge_result(score=4)

        judge.judge.side_effect = _slow_judge

        with patch.object(r, "query", return_value=_fake_hits()):
            start = time.perf_counter()
            result = r.answer(
                "q",
                llm=llm,
                judge=judge,
                reference="RAG retrieves context and uses an LLM.",
            )
            elapsed = time.perf_counter() - start

        assert result.judge_score == 4
        assert result.token_f1 is not None
        assert (
            elapsed < judge_latency + 0.2
        ), f"Expected concurrent scoring+judge wall time < {judge_latency + 0.2:.2f}s, got {elapsed:.3f}s"


class TestLiveRetrievalOperator:
    """LiveRetrievalOperator adapts Retriever.retrieve_batch() into an EvalOperator."""

    def test_process_populates_context_columns(self):
        """Single ``retrieve_batch`` call covers the whole DataFrame.

        This is the batched contract -- one embed/LanceDB round trip for
        all rows.  The earlier per-row contract (N ``retrieve`` calls on
        an N-row frame) was retired because it scaled linearly with RTT
        to the embed NIM.  Asserting the call count here guards against a
        regression back to the quadratic path.
        """

        from nemo_retriever.evaluation.live_retrieval import LiveRetrievalOperator
        from nemo_retriever.llm.types import RetrievalResult

        mock_retriever = MagicMock()
        mock_retriever.retrieve_batch.return_value = [
            RetrievalResult(
                chunks=["a", "b"],
                metadata=[{"source": "s1"}, {"source": "s2"}],
            ),
            RetrievalResult(chunks=["c"], metadata=[{"source": "s3"}]),
        ]

        op = LiveRetrievalOperator(mock_retriever, top_k=5)
        df = pd.DataFrame({"query": ["q1", "q2"]})

        out = op.process(df)

        assert list(out.columns) == ["query", "context", "context_metadata"]
        assert out.loc[0, "context"] == ["a", "b"]
        assert out.loc[1, "context"] == ["c"]
        assert out.loc[0, "context_metadata"] == [{"source": "s1"}, {"source": "s2"}]

        # Exactly one batched call -- the whole point of the operator
        # rewrite.  ``retrieve`` must not be reached.
        assert mock_retriever.retrieve_batch.call_count == 1
        mock_retriever.retrieve.assert_not_called()

        call_args = mock_retriever.retrieve_batch.call_args
        queries_arg = call_args.args[0] if call_args.args else call_args.kwargs["queries"]
        assert list(queries_arg) == ["q1", "q2"]
        assert call_args.kwargs.get("top_k") == 5

    def test_process_scales_to_ten_rows_with_single_call(self):
        """A 10-row frame still triggers exactly one ``retrieve_batch`` call."""

        from nemo_retriever.evaluation.live_retrieval import LiveRetrievalOperator
        from nemo_retriever.llm.types import RetrievalResult

        mock_retriever = MagicMock()
        mock_retriever.retrieve_batch.return_value = [
            RetrievalResult(chunks=[f"chunk-{i}"], metadata=[{"row": i}]) for i in range(10)
        ]

        op = LiveRetrievalOperator(mock_retriever, top_k=3)
        df = pd.DataFrame({"query": [f"q{i}" for i in range(10)]})

        out = op.process(df)

        assert len(out) == 10
        assert mock_retriever.retrieve_batch.call_count == 1

    def test_process_rejects_mismatched_batch_length(self):
        """Guard against a retrieve_batch that drops or duplicates rows."""

        from nemo_retriever.evaluation.live_retrieval import LiveRetrievalOperator
        from nemo_retriever.llm.types import RetrievalResult

        mock_retriever = MagicMock()
        mock_retriever.retrieve_batch.return_value = [
            RetrievalResult(chunks=["a"], metadata=[{"source": "s1"}]),
        ]

        op = LiveRetrievalOperator(mock_retriever, top_k=3)
        df = pd.DataFrame({"query": ["q1", "q2"]})

        with pytest.raises(RuntimeError, match="retrieve_batch returned"):
            op.process(df)

    def test_process_requires_dataframe(self):
        from nemo_retriever.evaluation.live_retrieval import LiveRetrievalOperator

        op = LiveRetrievalOperator(MagicMock(), top_k=3)
        with pytest.raises(TypeError, match="requires a pandas.DataFrame"):
            op.process({"query": ["q"]})


class TestRetrieveBatch:
    """Batched analogue of ``retrieve`` -- one embed call for all rows."""

    def test_retrieve_batch_returns_aligned_results(self):
        """Length + order invariants hold across the batch."""

        r = _make_retriever()
        hits_per_query = [
            [
                {"text": "chunk-0-0", "source": "doc-A.pdf"},
                {"text": "chunk-0-1", "source": "doc-B.pdf"},
            ],
            [{"text": "chunk-1-0", "source": "doc-C.pdf"}],
            [],
        ]

        with patch.object(r, "queries", return_value=hits_per_query) as mock_queries:
            results = r.retrieve_batch(["q0", "q1", "q2"], top_k=4)

        assert mock_queries.call_count == 1
        assert len(results) == 3
        assert results[0].chunks == ["chunk-0-0", "chunk-0-1"]
        assert results[0].metadata == [{"source": "doc-A.pdf"}, {"source": "doc-B.pdf"}]
        assert results[1].chunks == ["chunk-1-0"]
        assert results[2].chunks == [] and results[2].metadata == []

    def test_retrieve_batch_top_k_is_scoped(self):
        """``top_k`` override must not persist on the instance."""

        r = _make_retriever()
        original_top_k = r.top_k
        with patch.object(r, "queries", return_value=[[]]):
            r.retrieve_batch(["q"], top_k=42)
        assert r.top_k == original_top_k

    def test_retrieve_batch_forwards_top_k_to_queries(self):
        """The per-call ``top_k`` must be forwarded as a kwarg, not via
        attribute mutation. This is the regression test for the
        Greptile P1 "thread-unsafe self.top_k mutation" finding: under
        the old try/finally pattern the value was visible to ``queries``
        only through ``self.top_k``, which was racy under concurrent use.
        """

        r = _make_retriever()
        original_top_k = r.top_k
        with patch.object(r, "queries", return_value=[[]]) as mock_queries:
            r.retrieve_batch(["q"], top_k=7)
        mock_queries.assert_called_once()
        call_kwargs = mock_queries.call_args.kwargs
        assert call_kwargs.get("top_k") == 7
        assert r.top_k == original_top_k

    def test_retrieve_batch_concurrent_distinct_top_k(self):
        """Concurrent ``retrieve_batch`` calls with different ``top_k``
        values must not clobber each other.

        Under the old ``previous_top_k = self.top_k; self.top_k = ...; try:
        self.queries(...); finally: self.top_k = previous_top_k`` dance
        two threads would race on ``self.top_k``: thread A could set
        ``top_k=3``, thread B could overwrite with ``top_k=10`` before
        thread A's ``queries()`` call read it, and thread A would run
        with the wrong k. The new implementation passes ``top_k``
        through as a local kwarg, so each call sees its own value.
        """

        from concurrent.futures import ThreadPoolExecutor

        r = _make_retriever()

        observed: list[int] = []
        lock = __import__("threading").Lock()

        def fake_queries(query_texts, *, top_k, **_kwargs):
            with lock:
                observed.append(int(top_k))
            return [[] for _ in query_texts]

        with patch.object(r, "queries", side_effect=fake_queries):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(r.retrieve_batch, ["q"], top_k=k) for k in (1, 3, 7, 15, 42)]
                for f in futures:
                    f.result()

        assert sorted(observed) == [1, 3, 7, 15, 42]

    def test_retrieve_batch_empty_input(self):
        """Empty input returns an empty list and does not call ``queries``."""

        r = _make_retriever()
        with patch.object(r, "queries") as mock_queries:
            assert r.retrieve_batch([]) == []
        mock_queries.assert_not_called()


class TestPipelineBuilder:
    """Retriever.pipeline() fluent builder composition."""

    def test_builder_composition_runs_expected_steps(self):
        """generate -> score -> judge builds and executes the full chain."""
        r = _make_retriever()
        hits = _fake_hits()

        # LiveRetrievalOperator uses retrieve_batch which delegates to queries().
        with patch.object(r, "queries", return_value=[hits]):
            # Mock out the three EvalOperator classes that the builder imports
            # lazily so we can assert which ones were appended and executed.
            with patch("nemo_retriever.evaluation.generation.QAGenerationOperator") as mock_gen_cls, patch(
                "nemo_retriever.evaluation.scoring_operator.ScoringOperator"
            ) as mock_score_cls, patch("nemo_retriever.evaluation.judging.JudgingOperator") as mock_judge_cls:
                # Configure each mocked operator to pass the DataFrame through
                # with a sentinel column so we can verify each step ran.
                def _gen_process(df, **_):
                    out = df.copy()
                    out["answer"] = ["gen-out"] * len(out)
                    return out

                def _score_process(df, **_):
                    out = df.copy()
                    out["token_f1"] = [1.0] * len(out)
                    return out

                def _judge_process(df, **_):
                    out = df.copy()
                    out["judge_score"] = [5] * len(out)
                    return out

                mock_gen_cls.return_value = _build_mock_operator("QAGenerationOperator", _gen_process)
                mock_score_cls.return_value = _build_mock_operator("ScoringOperator", _score_process)
                mock_judge_cls.return_value = _build_mock_operator("JudgingOperator", _judge_process)

                llm = _build_fake_llm_client()
                judge = _build_fake_judge()

                df_out = r.pipeline().generate(llm).score().judge(judge).run(queries=["q1"], reference=["r1"])

        assert isinstance(df_out, pd.DataFrame)
        assert "context" in df_out.columns
        assert "answer" in df_out.columns
        assert "token_f1" in df_out.columns
        assert "judge_score" in df_out.columns
        assert mock_gen_cls.called
        assert mock_score_cls.called
        assert mock_judge_cls.called

    def test_builder_skip_steps(self):
        """.pipeline().generate(llm).run([q]) skips score and judge."""
        r = _make_retriever()

        with patch.object(r, "queries", return_value=[_fake_hits()]):
            with patch("nemo_retriever.evaluation.generation.QAGenerationOperator") as mock_gen_cls, patch(
                "nemo_retriever.evaluation.scoring_operator.ScoringOperator"
            ) as mock_score_cls, patch("nemo_retriever.evaluation.judging.JudgingOperator") as mock_judge_cls:

                def _gen_process(df, **_):
                    out = df.copy()
                    out["answer"] = ["answer"] * len(out)
                    return out

                mock_gen_cls.return_value = _build_mock_operator("QAGenerationOperator", _gen_process)

                llm = _build_fake_llm_client()

                df_out = r.pipeline().generate(llm).run(queries=["q"])

        assert "context" in df_out.columns
        assert "answer" in df_out.columns
        assert "token_f1" not in df_out.columns
        assert "judge_score" not in df_out.columns
        assert mock_gen_cls.called
        mock_score_cls.assert_not_called()
        mock_judge_cls.assert_not_called()

    def test_builder_forwards_top_p_from_llm_client(self):
        """``.generate(llm)`` must forward ``llm.sampling.top_p`` to the
        operator.

        Regression test for the Greptile P1 finding that ``top_p`` was
        silently dropped when a caller passed a pre-built ``LiteLLMClient``
        with a non-default ``top_p``.
        """

        r = _make_retriever()

        with patch("nemo_retriever.evaluation.generation.QAGenerationOperator") as mock_gen_cls:
            mock_gen_cls.return_value = _build_mock_operator("QAGenerationOperator", lambda df, **_: df)
            llm = _build_fake_llm_client(top_p=0.7)
            r.pipeline().generate(llm)

        mock_gen_cls.assert_called_once()
        kwargs = mock_gen_cls.call_args.kwargs
        assert kwargs.get("top_p") == 0.7
        assert kwargs.get("temperature") == 0.0
        assert kwargs.get("max_tokens") == 512

    def test_builder_forwards_none_top_p_when_unset(self):
        """Default path (``top_p=None``) must forward ``None`` -- not raise
        and not silently substitute a non-default."""

        r = _make_retriever()

        with patch("nemo_retriever.evaluation.generation.QAGenerationOperator") as mock_gen_cls:
            mock_gen_cls.return_value = _build_mock_operator("QAGenerationOperator", lambda df, **_: df)
            llm = _build_fake_llm_client()
            r.pipeline().generate(llm)

        mock_gen_cls.assert_called_once()
        kwargs = mock_gen_cls.call_args.kwargs
        assert kwargs.get("top_p") is None

    def test_builder_generate_requires_llm_or_model(self):
        r = _make_retriever()
        with pytest.raises(ValueError, match="requires either llm= or model="):
            r.pipeline().generate()

    def test_builder_judge_requires_judge_or_model(self):
        r = _make_retriever()
        with pytest.raises(ValueError, match="requires either judge= or model="):
            r.pipeline().judge()

    def test_builder_reference_length_must_match(self):
        r = _make_retriever()
        llm = _build_fake_llm_client()
        with patch("nemo_retriever.evaluation.generation.QAGenerationOperator"):
            with pytest.raises(ValueError, match="reference length must match"):
                r.pipeline().generate(llm).run(queries=["q1", "q2"], reference=["r1"])

    def test_builder_surfaces_generation_failure_rate(self):
        """``gen_error`` column drives ``df.attrs['generation_failure_rate']``.

        Batch eval jobs that quietly skip scoring on generation failures
        would otherwise report misleading success rates: the fraction of
        rows with populated ``gen_error`` is attached as a DataFrame
        attribute so aggregators have a single authoritative field to
        read.  No row-level schema change needed.
        """

        r = _make_retriever()
        hits = _fake_hits()

        with patch.object(r, "queries", return_value=[hits, hits, hits]):
            with patch("nemo_retriever.evaluation.generation.QAGenerationOperator") as mock_gen_cls:

                def _gen_process(df, **_):
                    out = df.copy()
                    out["answer"] = ["answer", "", ""]
                    out["gen_error"] = [None, "TimeoutError", "RateLimitError"]
                    return out

                mock_gen_cls.return_value = _build_mock_operator("QAGenerationOperator", _gen_process)

                llm = _build_fake_llm_client()
                df_out = r.pipeline().generate(llm).run(queries=["q1", "q2", "q3"])

        assert "generation_failure_rate" in df_out.attrs
        assert df_out.attrs["generation_failure_rate"] == pytest.approx(2 / 3, abs=1e-6)

    def test_builder_skips_generation_failure_rate_without_gen_error(self):
        """No ``gen_error`` column -> no attrs pollution on retrieval-only runs."""

        r = _make_retriever()

        with patch.object(r, "queries", return_value=[_fake_hits()]):
            df_out = r.pipeline().run(queries=["q"])

        assert "generation_failure_rate" not in df_out.attrs


def _build_mock_operator(class_name: str, process_fn):
    """Build a mock operator that cooperates with the graph framework.

    The object must satisfy ``isinstance(op, AbstractOperator)`` so the
    pipeline_graph ``Node`` accepts it, and must expose ``.run(df)`` since
    ``Graph._execute_node`` invokes that.  We subclass the real
    ``EvalOperator`` so required-column validation does not fire, and
    simply override ``process``.
    """
    from nemo_retriever.evaluation.eval_operator import EvalOperator

    class _Mock(EvalOperator):
        required_columns = ()
        output_columns = ()

        def __init__(self):
            super().__init__()

        def process(self, data, **kwargs):
            return process_fn(data, **kwargs)

    op = _Mock()
    op.__class__.__name__ = class_name
    return op


def _build_fake_llm_client(*, top_p: float | None = None):
    """Build a fake LiteLLMClient-shaped object for the builder."""
    transport = SimpleNamespace(
        model="fake-llm/test",
        api_base=None,
        api_key=None,
        extra_params={},
        num_retries=3,
        timeout=120.0,
    )
    sampling = SimpleNamespace(temperature=0.0, top_p=top_p, max_tokens=512)
    return SimpleNamespace(transport=transport, sampling=sampling)


def _build_fake_judge():
    """Build a fake LLMJudge-shaped object for the builder."""
    transport = SimpleNamespace(
        model="fake-judge/test",
        api_base=None,
        api_key=None,
        extra_params={},
        num_retries=3,
        timeout=120.0,
    )
    client = SimpleNamespace(transport=transport)
    return SimpleNamespace(_client=client)
