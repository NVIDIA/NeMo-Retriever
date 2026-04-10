# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Retriever strategy implementations for the QA evaluation pipeline.

FileRetriever: reads pre-computed retrieval results from a JSON file.

FileRetriever is the primary integration point. Any retrieval method -- vector
search, agentic retrieval, hybrid, reranked, BM25, or a completely custom
pipeline -- can plug into the QA eval harness by writing a single JSON file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import unicodedata
from nemo_retriever.evaluation.types import RetrievalResult

logger = logging.getLogger(__name__)


def _normalize_query(text: str) -> str:
    """Canonical form for query matching: NFKC unicode, stripped, case-folded,
    collapsed whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text


class FileRetriever:
    """Retriever that reads pre-computed results from a JSON file.

    This is the integration point for **any** retrieval method. Vector search,
    agentic retrieval, hybrid pipelines, BM25, rerankers, or a completely
    custom system -- as long as it produces a JSON file in the format below,
    the QA eval harness will generate answers and judge them identically.

    Minimal required JSON format::

        {
          "queries": {
            "What is the range of the 767?": {
              "chunks": ["First retrieved chunk text...", "Second chunk..."]
            }
          }
        }
    """

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FileRetriever: retrieval results file not found: {file_path}")

        self.file_path = file_path

        with open(file_path) as f:
            data = json.load(f)

        raw_index: dict[str, dict] = data.get("queries", {})
        if not raw_index:
            raise ValueError(
                f"FileRetriever: no 'queries' key found in {file_path}. "
                'Expected format: {"queries": {"query text": {"chunks": [...], "metadata": [...]}}}'
            )

        self._norm_index: dict[str, dict] = {}
        self._raw_keys: dict[str, str] = {}
        for raw_key, value in raw_index.items():
            norm = _normalize_query(raw_key)
            self._norm_index[norm] = value
            self._raw_keys[norm] = raw_key

        self._miss_count = 0
        self._miss_lock = threading.Lock()

    def check_coverage(self, qa_pairs: list[dict]) -> float:
        """Validate retrieval file covers the ground-truth queries."""
        total = len(qa_pairs)
        if total == 0:
            return 1.0

        misses: list[str] = []
        for pair in qa_pairs:
            norm = _normalize_query(pair.get("query", ""))
            if norm not in self._norm_index:
                misses.append(pair.get("query", "")[:80])

        coverage = (total - len(misses)) / total
        if misses:
            logger.warning(
                "FileRetriever coverage: %.1f%% (%d/%d queries matched)",
                coverage * 100,
                total - len(misses),
                total,
            )
            for q in misses[:10]:
                logger.warning("  MISS: %r", q)
            if len(misses) > 10:
                logger.warning("  ... and %d more", len(misses) - 10)
        else:
            logger.info("FileRetriever coverage: 100%% (%d/%d queries matched)", total, total)

        return coverage

    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        """Look up pre-computed chunks for a query string."""
        norm = _normalize_query(query)
        entry = self._norm_index.get(norm)

        if entry is None:
            with self._miss_lock:
                self._miss_count += 1
                count = self._miss_count
            if count <= 20:
                logger.warning("FileRetriever: query not found in retrieval file: %r", query)
            elif count == 21:
                logger.warning("FileRetriever: suppressing further miss warnings (>20)")
            return RetrievalResult(chunks=[], metadata=[])

        chunks = entry.get("chunks", [])[:top_k]
        metadata = entry.get("metadata", [])[:top_k]
        return RetrievalResult(chunks=chunks, metadata=metadata)
