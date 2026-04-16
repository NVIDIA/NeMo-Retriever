# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""text-to-sql extension of :class:`nemo_retriever.retriever.Retriever` with optional
LanceDB ``metadata``-based ``label`` filtering. Keeps upstream ``retriever.py`` unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from nemo_retriever.retriever import Retriever

logger = logging.getLogger(__name__)

_MAX_LOG_PRED_LEN = 6000


def _trunc_for_log(s: Optional[str], max_len: int = _MAX_LOG_PRED_LEN) -> str:
    if s is None:
        return "None"
    if len(s) <= max_len:
        return s
    return s[: max_len - 20] + f"...(truncated,len={len(s)})"


class TextToSqlRetriever(Retriever):
    """Same as :class:`~nemo_retriever.retriever.Retriever`, plus optional ``label_in``.

    Rows may store the semantic label in a top-level ``label`` column and/or inside the
    serialized ``metadata`` string. The filter is ``(label IN (...)) OR (<metadata LIKE …>)``
    when both exist, so vector search keeps rows whether the denormalized column or the blob
    carries the label (and Neo4j-style ``Column`` vs ``column`` is covered in ``LIKE`` patterns).
    """

    @staticmethod
    def _sql_in_literals(values: Sequence[str]) -> str:
        """Escape single-quoted strings for LanceDB ``IN (...)`` predicates."""
        return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)

    @staticmethod
    def _sql_string_literal(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    @classmethod
    def _build_label_where(
        cls,
        field_names: list[str],
        label_in: Optional[Sequence[str]],
    ) -> Optional[str]:
        """Return a DataFusion/Lance ``WHERE`` fragment, or ``None`` if no filter is needed."""
        if not label_in:
            return None
        labels = [str(x).strip() for x in label_in if x is not None and str(x).strip()]
        if not labels:
            return None

        have_label_col = "label" in field_names
        have_metadata = "metadata" in field_names
        if not have_label_col and not have_metadata:
            return None

        or_parts: list[str] = []

        if have_label_col:
            or_parts.append(f"label IN ({cls._sql_in_literals(labels)})")

        if have_metadata:

            def value_variants(canonical: str) -> list[str]:
                c = canonical.lower()
                if c == "table":
                    return ["table", "Table", "TABLE"]
                if c == "column":
                    return ["column", "Column", "COLUMN"]
                return [canonical]

            like_parts: list[str] = []
            for lab in labels:
                for v in value_variants(lab):
                    for inner in (f'"label": "{v}"', f'"label":"{v}"'):
                        pat = "%" + inner + "%"
                        like_parts.append(f"metadata LIKE {cls._sql_string_literal(pat)}")
                    # One escape pass via _sql_string_literal only (do not pre-escape inner quotes).
                    inner_repr = f"'label': '{v}'"
                    pat_repr = "%" + inner_repr + "%"
                    like_parts.append(f"metadata LIKE {cls._sql_string_literal(pat_repr)}")
            if like_parts:
                or_parts.append("(" + " OR ".join(like_parts) + ")")

        if not or_parts:
            return None
        if len(or_parts) == 1:
            return or_parts[0]
        return "(" + " OR ".join(or_parts) + ")"

    def _search_lancedb(
        self,
        *,
        lancedb_uri: str,
        lancedb_table: str,
        query_vectors: list[list[float]],
        query_texts: list[str],
        label_in: Optional[Sequence[str]] = None,
    ) -> list[list[dict[str, Any]]]:
        import lancedb  # type: ignore
        import numpy as np

        logger.info(
            "TextToSqlRetriever._search_lancedb: start uri=%r table=%r n_vectors=%d "
            "label_in=%s hybrid=%s reranker=%s top_k=%s refine_factor=%s nprobes(config)=%s vector_column=%r",
            lancedb_uri,
            lancedb_table,
            len(query_vectors),
            list(label_in) if label_in is not None else None,
            bool(self.hybrid),
            self.reranker,
            self.top_k,
            int(self.refine_factor),
            self.nprobes,
            self.vector_column_name,
        )

        try:
            logger.info("TextToSqlRetriever._search_lancedb: connecting lancedb.connect(%r)", lancedb_uri)
            db = lancedb.connect(lancedb_uri)
            logger.info("TextToSqlRetriever._search_lancedb: open_table(%r)", lancedb_table)
            table = db.open_table(lancedb_table)
        except Exception:
            logger.exception("TextToSqlRetriever._search_lancedb: connect/open_table failed")
            raise

        field_names = [f.name for f in table.schema]
        label_where = self._build_label_where(field_names, label_in)
        logger.info(
            "TextToSqlRetriever._search_lancedb: schema fields=%s label_where=%s",
            field_names,
            _trunc_for_log(label_where) if label_where else "None (no label filter)",
        )

        effective_nprobes = int(self.nprobes)
        if effective_nprobes <= 0:
            try:
                logger.info("TextToSqlRetriever._search_lancedb: resolving nprobes via table.list_indices()")
                for idx in table.list_indices():
                    num_parts = getattr(idx, "num_partitions", None)
                    if num_parts and int(num_parts) > 0:
                        effective_nprobes = int(num_parts)
                        break
            except Exception:
                logger.exception("TextToSqlRetriever._search_lancedb: list_indices failed; using fallback nprobes")
            if effective_nprobes <= 0:
                effective_nprobes = 16
        logger.info("TextToSqlRetriever._search_lancedb: effective_nprobes=%s", effective_nprobes)

        results: list[list[dict[str, Any]]] = []
        for i, vector in enumerate(query_vectors):
            q = np.asarray(vector, dtype="float32")
            top_k = self.top_k if not self.reranker else self.top_k * self.reranker_refine_factor
            qpreview = (
                (query_texts[i][:240] + "…")
                if i < len(query_texts) and len(query_texts[i]) > 240
                else (query_texts[i] if i < len(query_texts) else "")
            )
            logger.info(
                "TextToSqlRetriever._search_lancedb: query[%d] text_preview=%r top_k=%s where=%s",
                i,
                qpreview,
                int(top_k),
                _trunc_for_log(label_where) if label_where else "None",
            )
            if self.hybrid:
                from lancedb.rerankers import RRFReranker  # type: ignore

                chain = (
                    table.search(query_type="hybrid")
                    .vector(q)
                    .text(query_texts[i])
                    .nprobes(effective_nprobes)
                    .refine_factor(int(self.refine_factor))
                )
                if label_where:
                    chain = chain.where(label_where)
                hybrid_cols = [
                    "text",
                    "metadata",
                    "source",
                    "page_number",
                    "pdf_page",
                    "pdf_basename",
                    "source_id",
                    "path",
                ]
                if "label" in field_names:
                    hybrid_cols = ["label"] + hybrid_cols
                try:
                    hits = chain.select(hybrid_cols).limit(int(top_k)).rerank(RRFReranker()).to_list()
                except Exception:
                    logger.exception(
                        "TextToSqlRetriever._search_lancedb: hybrid search failed query[%d] where=%s",
                        i,
                        _trunc_for_log(label_where) if label_where else "None",
                    )
                    raise
            else:
                dense_select = [
                    "text",
                    "metadata",
                    "source",
                    "page_number",
                    "_distance",
                    "pdf_page",
                    "pdf_basename",
                    "source_id",
                    "path",
                ]
                if "label" in field_names:
                    dense_select = ["label"] + dense_select
                chain = (
                    table.search(q, vector_column_name=self.vector_column_name)
                    .nprobes(effective_nprobes)
                    .refine_factor(int(self.refine_factor))
                )
                if label_where:
                    chain = chain.where(label_where)
                try:
                    hits = chain.select(dense_select).limit(int(top_k)).to_list()
                except Exception:
                    logger.exception(
                        "TextToSqlRetriever._search_lancedb: dense vector search failed query[%d] "
                        "vector_column=%r where=%s",
                        i,
                        self.vector_column_name,
                        _trunc_for_log(label_where) if label_where else "None",
                    )
                    raise
            logger.info(
                "TextToSqlRetriever._search_lancedb: query[%d] ok n_hits=%d",
                i,
                len(hits),
            )
            results.append(hits)
        logger.info(
            "TextToSqlRetriever._search_lancedb: done total_query_batches=%d",
            len(results),
        )
        return results

    def query(
        self,
        query: str,
        *,
        embedder: Optional[str] = None,
        lancedb_uri: Optional[str] = None,
        lancedb_table: Optional[str] = None,
        label_in: Optional[Sequence[str]] = None,
    ) -> list[dict[str, Any]]:
        return self.queries(
            [query],
            embedder=embedder,
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
            label_in=label_in,
        )[0]

    def queries(
        self,
        queries: Sequence[str],
        *,
        embedder: Optional[str] = None,
        lancedb_uri: Optional[str] = None,
        lancedb_table: Optional[str] = None,
        label_in: Optional[Sequence[str]] = None,
    ) -> list[list[dict[str, Any]]]:
        query_texts = [str(q) for q in queries]
        if not query_texts:
            return []

        resolved_embedder = str(embedder or self.embedder)
        resolved_lancedb_uri = str(lancedb_uri or self.lancedb_uri)
        resolved_lancedb_table = str(lancedb_table or self.lancedb_table)

        endpoint = self._resolve_embedding_endpoint()
        if endpoint is not None:
            logger.info(
                "TextToSqlRetriever.queries: embedding via NIM endpoint=%r model=%r n_queries=%d",
                endpoint,
                resolved_embedder,
                len(query_texts),
            )
            vectors = self._embed_queries_nim(
                query_texts,
                endpoint=endpoint,
                model=resolved_embedder,
            )
        else:
            logger.info(
                "TextToSqlRetriever.queries: embedding via local HF model=%r n_queries=%d",
                resolved_embedder,
                len(query_texts),
            )
            vectors = self._embed_queries_local_hf(
                query_texts,
                model_name=resolved_embedder,
            )

        _vdim = len(vectors[0]) if vectors and vectors[0] is not None else 0
        logger.info(
            "TextToSqlRetriever.queries: embeddings computed dim=%s calling _search_lancedb label_in=%s",
            _vdim,
            list(label_in) if label_in is not None else None,
        )

        results = self._search_lancedb(
            lancedb_uri=resolved_lancedb_uri,
            lancedb_table=resolved_lancedb_table,
            query_vectors=vectors,
            query_texts=query_texts,
            label_in=label_in,
        )

        # If True: after LanceDB hits, re-score each (query, chunk) with a cross-encoder
        # (``reranker_model_name``), reorder by ``_rerank_score``, return top ``top_k``.
        # Cross-encoder = one model pass over query+passage together (finer than embedding cosine).
        if self.reranker:
            results = self._rerank_results(query_texts, results)

        return results
