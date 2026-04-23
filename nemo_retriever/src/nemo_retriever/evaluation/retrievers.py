# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Retriever strategy implementations for the QA evaluation pipeline.

FileRetriever: reads pre-computed retrieval results from a JSON file,
or queries LanceDB in-memory via ``from_lancedb()``.

FileRetriever is the primary integration point. Any retrieval method -- vector
search, agentic retrieval, hybrid, reranked, BM25, or a completely custom
pipeline -- can plug into the QA eval harness by writing a single JSON file
or by using ``FileRetriever.from_lancedb()`` to query a live vector DB.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

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

        with open(file_path) as f:
            data = json.load(f)

        raw_index: dict[str, dict] = data.get("queries", {})
        if not raw_index:
            raise ValueError(
                f"FileRetriever: no 'queries' key found in {file_path}. "
                'Expected format: {"queries": {"query text": {"chunks": [...], "metadata": [...]}}}'
            )

        sample = next(iter(raw_index.values()), {})
        if not isinstance(sample.get("chunks"), list):
            raise ValueError(
                f"FileRetriever: first entry in {file_path} is missing a 'chunks' list. "
                'Expected: {"queries": {"query": {"chunks": ["..."]}}}'
            )

        self._initialize_index(raw_index, source=file_path)

    def _initialize_index(self, raw_index: dict[str, dict], *, source: str) -> None:
        """Populate instance state from an already-validated queries mapping.

        Single source of truth for all :class:`FileRetriever` instance
        fields used by :meth:`retrieve` and :meth:`check_coverage`.
        Called by both :meth:`__init__` (file-based) and
        :meth:`_from_dict` (in-memory, used by :meth:`from_lancedb`) so
        that new instance fields only need to be added in one place and
        can never diverge between the two construction paths.

        Parameters
        ----------
        raw_index : dict[str, dict]
            ``{query_text: {"chunks": [...], "metadata": [...]}}`` --
            the same shape both entry points produce.  Must already be
            non-empty and contain a ``chunks`` list; validation is the
            caller's responsibility so error messages can reference the
            originating source (file path vs. in-memory dict).
        source : str
            Human-readable origin label stored on ``self.file_path``
            (e.g. a filesystem path or ``"<in-memory>"``).
        """
        self.file_path = source
        self._norm_index: dict[str, dict] = {}
        self._raw_keys: dict[str, str] = {}
        self._miss_count = 0
        self._miss_lock = threading.Lock()
        for raw_key, value in raw_index.items():
            norm = _normalize_query(raw_key)
            self._norm_index[norm] = value
            self._raw_keys[norm] = raw_key

    @classmethod
    def _from_dict(cls, queries: dict[str, dict]) -> "FileRetriever":
        """Build a FileRetriever from an in-memory queries dict.

        Bypasses file I/O while reusing the same normalized index that
        ``__init__`` builds from JSON.  All instance methods (``retrieve``,
        ``check_coverage``) work identically afterwards.

        Parameters
        ----------
        queries : dict
            ``{query_text: {"chunks": [...], "metadata": [...]}}`` --
            the same shape as the ``"queries"`` value in a retrieval JSON.
        """
        if not queries:
            raise ValueError("FileRetriever._from_dict: queries dict is empty")
        sample = next(iter(queries.values()), {})
        if not isinstance(sample.get("chunks"), list):
            raise ValueError(
                "FileRetriever._from_dict: first entry is missing a 'chunks' list. "
                'Expected: {"query": {"chunks": ["..."]}}'
            )

        instance = object.__new__(cls)
        instance._initialize_index(queries, source="<in-memory>")
        return instance

    @classmethod
    def from_lancedb(
        cls,
        qa_pairs: list[dict],
        lancedb_uri: str = "lancedb",
        lancedb_table: str = "nv-ingest",
        embedder: str = "nvidia/llama-nemotron-embed-1b-v2",
        top_k: int = 5,
        page_index: dict[str, dict[str, str]] | None = None,
        save_path: str | None = None,
    ) -> "FileRetriever":
        """Query LanceDB in-memory, optionally save, return a FileRetriever.

        Reuses :func:`~nemo_retriever.export.query_lancedb` for batched
        vector search and :func:`~nemo_retriever.export.write_retrieval_json`
        for optional disk persistence.

        Parameters
        ----------
        qa_pairs : list[dict]
            Ground-truth pairs; each must have a ``"query"`` key.
        lancedb_uri : str
            Path to the LanceDB directory.
        lancedb_table : str
            LanceDB table name.
        embedder : str
            Embedding model name for query encoding.
        top_k : int
            Number of chunks to retrieve per query.
        page_index : dict, optional
            ``{source_id: {page_str: markdown}}``.  Enables full-page
            markdown expansion when provided.
        save_path : str, optional
            If set, also writes the retrieval JSON to this path so it
            can be reloaded later via ``FileRetriever(file_path=...)``.
        """
        from nemo_retriever.export import query_lancedb, write_retrieval_json

        all_results, meta = query_lancedb(
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
            queries=qa_pairs,
            top_k=top_k,
            embedder=embedder,
            page_index=page_index,
        )

        if save_path:
            write_retrieval_json(all_results, save_path, meta)
            logger.info("Saved retrieval JSON to %s", save_path)

        instance = cls._from_dict(all_results)
        if save_path:
            instance.file_path = save_path
        return instance

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

    def retrieve(self, query: str, top_k: int) -> "pd.DataFrame":
        """Look up pre-computed chunks for a query string.

        Returns a single-row :class:`pandas.DataFrame` with columns
        ``[query, chunks, metadata]`` -- the same shape produced by
        :func:`nemo_retriever.generation.retrieve` -- so that the QA eval
        harness can consume both live and cached retrieval through the
        same DataFrame contract.  Missing queries are represented as a
        row with empty ``chunks`` / ``metadata`` lists rather than a
        raised exception, mirroring the previous "graceful miss"
        behaviour that counted misses on ``_miss_count``.
        """
        import pandas as pd

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
            return pd.DataFrame([{"query": query, "chunks": [], "metadata": []}])

        chunks = entry.get("chunks", [])[:top_k]
        metadata = entry.get("metadata", [])[:top_k]
        return pd.DataFrame([{"query": query, "chunks": chunks, "metadata": metadata}])
