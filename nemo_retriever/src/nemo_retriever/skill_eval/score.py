# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieval scoring for skill_eval v1: recall@k against a qrels-derived ground truth.

Pure-Python; no pytrec_eval dependency. Page references are (doc_id, page_number) tuples.
"""

from __future__ import annotations

from typing import Iterable


PageRef = tuple[str, int]


def recall_at_k(ranked: list[PageRef], relevant: Iterable[PageRef], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = set(ranked[:k])
    return len(top_k & relevant_set) / len(relevant_set)
