# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for :class:`LiveRetrievalOperator`.

The operator is the live-LanceDB source for
:func:`nemo_retriever.generation.eval` -- it replaces per-row retrieval
iteration with a single batched :meth:`Retriever.queries
<nemo_retriever.retriever.Retriever.queries>` call and projects the raw
hits onto the ``[context, context_metadata]`` DataFrame contract that
downstream operators (``QAGenerationOperator``, ``ScoringOperator``,
``JudgingOperator``) consume.

End-to-end composition (retrieve -> generate -> score -> judge) is
covered by the dedicated ``tests/test_generation_*.py`` suite; this
module stays narrowly focused on the operator's input/output shape and
the "exactly one batched ``queries`` call" invariant that guards the
O(1) network-cost path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


def _fake_hits(tag: str, count: int = 2) -> list[dict]:
    """Build a list of fake LanceDB hits for *tag*.

    Every hit has a ``text`` field (the only key ``LiveRetrievalOperator``
    pulls into ``context``) plus a small set of metadata keys that
    should land in ``context_metadata``.
    """
    return [
        {
            "text": f"{tag}-chunk-{i}",
            "source": f"{tag}-doc-{i}.pdf",
            "page_number": i + 1,
        }
        for i in range(count)
    ]


class TestLiveRetrievalOperator:
    """``LiveRetrievalOperator.process`` contract."""

    def test_process_populates_context_columns_via_batched_queries(self):
        """A DataFrame of queries yields aligned ``context`` / ``context_metadata`` lists."""
        from nemo_retriever.evaluation.live_retrieval import LiveRetrievalOperator

        mock_retriever = MagicMock()
        mock_retriever.queries.return_value = [
            _fake_hits("q1", count=2),
            _fake_hits("q2", count=1),
        ]

        op = LiveRetrievalOperator(mock_retriever, top_k=5)
        df = pd.DataFrame({"query": ["q1-text", "q2-text"]})

        out = op.process(df)

        assert list(out.columns) == ["query", "context", "context_metadata"]
        assert out.loc[0, "context"] == ["q1-chunk-0", "q1-chunk-1"]
        assert out.loc[1, "context"] == ["q2-chunk-0"]
        assert out.loc[0, "context_metadata"] == [
            {"source": "q1-doc-0.pdf", "page_number": 1},
            {"source": "q1-doc-1.pdf", "page_number": 2},
        ]
        assert out.loc[1, "context_metadata"] == [
            {"source": "q2-doc-0.pdf", "page_number": 1},
        ]

    def test_process_uses_single_batched_queries_call(self):
        """Exactly one batched ``queries`` call regardless of row count.

        This invariant guards against a regression back to the O(N)
        per-row path.  A 10-row frame must still trigger a single
        embed + LanceDB round trip via :meth:`Retriever.queries`.
        """
        from nemo_retriever.evaluation.live_retrieval import LiveRetrievalOperator

        mock_retriever = MagicMock()
        mock_retriever.queries.return_value = [_fake_hits(f"q{i}", count=1) for i in range(10)]

        op = LiveRetrievalOperator(mock_retriever, top_k=3)
        df = pd.DataFrame({"query": [f"q{i}" for i in range(10)]})

        out = op.process(df)

        assert len(out) == 10
        assert mock_retriever.queries.call_count == 1

        call_args = mock_retriever.queries.call_args
        queries_arg = call_args.args[0] if call_args.args else call_args.kwargs["queries"]
        assert list(queries_arg) == [f"q{i}" for i in range(10)]
        assert call_args.kwargs.get("top_k") == 3

    def test_process_rejects_mismatched_batch_length(self):
        """A retriever that drops or duplicates rows must raise RuntimeError."""
        from nemo_retriever.evaluation.live_retrieval import LiveRetrievalOperator

        mock_retriever = MagicMock()
        mock_retriever.queries.return_value = [_fake_hits("q1", count=1)]

        op = LiveRetrievalOperator(mock_retriever, top_k=3)
        df = pd.DataFrame({"query": ["q1", "q2"]})

        with pytest.raises(RuntimeError, match="Retriever.queries returned"):
            op.process(df)

    def test_process_requires_dataframe(self):
        """Non-DataFrame input raises a descriptive TypeError."""
        from nemo_retriever.evaluation.live_retrieval import LiveRetrievalOperator

        op = LiveRetrievalOperator(MagicMock(), top_k=3)
        with pytest.raises(TypeError, match="requires a pandas.DataFrame"):
            op.process({"query": ["q"]})
