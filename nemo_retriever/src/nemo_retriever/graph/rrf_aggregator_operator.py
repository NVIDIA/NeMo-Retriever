# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator that fuses per-step retrieval results using Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator


class RRFAggregatorOperator(AbstractOperator, CPUOperator):
    """Fuse multiple per-step ranked lists into a single ranking per query using RRF.

    Implements the Reciprocal Rank Fusion formula
    ``score(d) = sum(1 / (rank_i + k))`` across all retrieval steps where
    document *d* appears.  This is the same formula used in
    ``retrieval_bench/nemo_agentic/utils.py:rrf_from_subquery_results``.

    Designed to consume the output of :class:`ReActAgentOperator` (or any
    operator that emits one row per ``(query_id, step_idx, doc_id)`` triple)
    and produce a single fused ranking per ``query_id`` suitable as input to
    :class:`SelectionAgentOperator`.

    Input DataFrame schema
    ----------------------
    query_id   : str  — unique query identifier
    query_text : str  — original query text (carried through)
    step_idx   : int  — which retrieval step produced this row (0, 1, 2 …)
    doc_id     : str  — retrieved document identifier
    text       : str  — document text content
    rank       : int  — 1-indexed rank within its step (1 = most relevant)
    (additional columns are ignored)

    Output DataFrame schema
    -----------------------
    query_id  : str   — same ``query_id`` as the input
    query_text: str   — original query text (first occurrence per query)
    doc_id    : str   — document identifier
    rrf_score : float — fused RRF score (higher = more relevant)
    text      : str   — document text (first occurrence per ``doc_id``)
    Rows are sorted by ``rrf_score`` descending within each ``query_id``.

    Parameters
    ----------
    k : int
        RRF damping factor.  The standard value is ``60`` (default).
        Larger values reduce the influence of top-ranked documents.

    Examples
    --------
    ::

        import pandas as pd
        from nemo_retriever.graph.rrf_aggregator_operator import RRFAggregatorOperator

        op = RRFAggregatorOperator(k=60)
        df = pd.DataFrame({
            "query_id":   ["q1", "q1", "q1", "q1"],
            "query_text": ["inflation causes"] * 4,
            "step_idx":   [0,    0,    1,    1   ],
            "doc_id":     ["d1", "d2", "d1", "d3"],
            "text":       ["t1", "t2", "t1", "t3"],
            "rank":       [1,    2,    1,    2   ],
        })
        result = op.run(df)
        # d1 appears in both steps at rank 1 → highest RRF score
    """

    def __init__(self, *, k: int = 60) -> None:
        super().__init__()
        self._k = k

    # ------------------------------------------------------------------
    # AbstractOperator interface
    # ------------------------------------------------------------------

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"RRFAggregatorOperator expects a pd.DataFrame, got {type(data).__name__!r}.")
        required = {"query_id", "query_text", "step_idx", "doc_id", "text", "rank"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required column(s): {sorted(missing)}. " f"Expected: {sorted(required)}."
            )
        return data.copy()

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Compute RRF scores, group by query_id, sort by score descending."""
        k = self._k
        rows: List[Dict[str, Any]] = []

        for query_id, qgroup in data.groupby("query_id", sort=False):
            query_text = str(qgroup["query_text"].iloc[0])

            rrf_scores: Dict[str, float] = defaultdict(float)
            first_text: Dict[str, str] = {}

            # Process each step's ranked list
            for _step_idx, sgroup in qgroup.groupby("step_idx", sort=True):
                # Sort by rank ascending so rank=1 is processed first
                for _, row in sgroup.sort_values("rank").iterrows():
                    doc_id = str(row["doc_id"])
                    rank = int(row["rank"])
                    rrf_scores[doc_id] += 1.0 / (rank + k)
                    if doc_id not in first_text:
                        first_text[doc_id] = str(row["text"])

            for doc_id, score in sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True):
                rows.append(
                    {
                        "query_id": query_id,
                        "query_text": query_text,
                        "doc_id": doc_id,
                        "rrf_score": score,
                        "text": first_text.get(doc_id, ""),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["query_id", "query_text", "doc_id", "rrf_score", "text"])

        return pd.DataFrame(rows)

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return data
