# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed rank-fusion policy for LanceDB hybrid retrieval."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from lancedb.rerankers.base import Reranker


@dataclass(frozen=True)
class HybridFusionPolicy:
    """Candidate depth and weighted-RRF parameters for one hybrid query."""

    candidate_depth: int
    dense_weight: float
    rrf_k: int

    def __post_init__(self) -> None:
        if self.candidate_depth <= 0:
            raise ValueError("candidate_depth must be greater than zero")
        if not 0.0 <= self.dense_weight <= 1.0:
            raise ValueError("dense_weight must be between zero and one")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")


DEFAULT_HYBRID_FUSION_POLICY = HybridFusionPolicy(candidate_depth=50, dense_weight=0.8, rrf_k=10)


class WeightedRRFReranker(Reranker):
    """Fuse dense and FTS ranks while preferring dense order for score ties."""

    def __init__(self, policy: HybridFusionPolicy) -> None:
        super().__init__(return_score="relevance")
        self.policy = policy

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ) -> pa.Table:
        del query
        vector_ids = vector_results["_rowid"].to_pylist() if len(vector_results) else []
        fts_ids = fts_results["_rowid"].to_pylist() if len(fts_results) else []
        scores: defaultdict[Any, float] = defaultdict(float)
        for weight, row_ids in (
            (self.policy.dense_weight, vector_ids),
            (1.0 - self.policy.dense_weight, fts_ids),
        ):
            for rank, row_id in enumerate(row_ids, start=1):
                scores[row_id] += weight / (self.policy.rrf_k + rank)

        combined = self.merge_results(vector_results, fts_results)
        row_ids = combined["_rowid"].to_pylist()
        combined = combined.append_column(
            "_relevance_score",
            pa.array([scores[row_id] for row_id in row_ids], type=pa.float64()),
        )
        combined = combined.append_column("_fusion_order", pa.array(range(len(combined)), type=pa.int64()))
        combined = combined.sort_by([("_relevance_score", "descending"), ("_fusion_order", "ascending")])
        combined = combined.drop_columns(["_fusion_order"])
        return self._keep_relevance_score(combined)
