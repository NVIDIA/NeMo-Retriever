# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-backed agentic retrieval mode.

The implementation is intentionally additive: it composes the existing graph
operators and wraps :class:`nemo_retriever.retriever.Retriever` without changing
the standard retrieval path.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.model import VL_EMBED_MODEL, VL_RERANK_MODEL
from nemo_retriever.recall.beir import compute_beir_metrics
from nemo_retriever.recall.core import (
    _hit_to_audio_segment_key,
    _normalize_pdf_name,
    _normalize_query_df,
)
from nemo_retriever.retriever import Retriever

logger = logging.getLogger(__name__)

AGENTIC_TOP_K = 10
AGENTIC_NUM_CONCURRENT = 1
AGENTIC_TEXT_TRUNCATION = 500
AGENTIC_PARALLEL_TOOL_CALLS = False
AGENTIC_RRF_K = 60
AGENTIC_REACT_MAX_STEPS = 10


class AgenticQueryInputOperator(AbstractOperator):
    """Adapt ``Retriever(graph=...)`` input DataFrames to agentic query schema."""

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        _ = kwargs
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"AgenticQueryInputOperator expects a pd.DataFrame, got {type(data).__name__}.")
        return data.copy()

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        _ = kwargs
        out = data.copy()
        if "query_text" not in out.columns:
            if "query" in out.columns:
                out["query_text"] = out["query"].astype(str)
            elif "text" in out.columns:
                out["query_text"] = out["text"].astype(str)
            else:
                raise ValueError("Agentic query graph input requires 'query_text', 'query', or 'text'.")
        if "query_id" not in out.columns:
            out["query_id"] = [str(idx) for idx in range(len(out.index))]
        return out[["query_id", "query_text"]]

    def postprocess(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        _ = kwargs
        return data


class AgenticSelectionOutputOperator(AbstractOperator):
    """Convert final agentic selection DataFrame to ``Retriever`` hit-list output."""

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        _ = kwargs
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"AgenticSelectionOutputOperator expects a pd.DataFrame, got {type(data).__name__}.")
        return data.copy()

    def process(self, data: pd.DataFrame, **kwargs: Any) -> list[list[dict[str, Any]]]:
        _ = kwargs
        if data.empty:
            return []
        required = {"query_id", "doc_id", "rank"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Agentic selection output missing required columns: {sorted(missing)}")

        hits: list[list[dict[str, Any]]] = []
        for _query_id, group in data.groupby("query_id", sort=False):
            query_hits: list[dict[str, Any]] = []
            for _, row in group.sort_values("rank").iterrows():
                hit = row.to_dict()
                doc_id = str(hit.get("doc_id", ""))
                if doc_id and not hit.get("pdf_page"):
                    hit["pdf_page"] = doc_id
                query_hits.append(hit)
            hits.append(query_hits)
        return hits

    def postprocess(self, data: list[list[dict[str, Any]]], **kwargs: Any) -> list[list[dict[str, Any]]]:
        _ = kwargs
        return data


@dataclass(frozen=True)
class AgenticRetrievalConfig:
    """Configuration for graph-backed agentic retrieval."""

    vdb_op: str = "lancedb"
    vdb_kwargs: dict[str, Any] = field(default_factory=dict)
    query_embedder: str = VL_EMBED_MODEL
    embedding_endpoint: Optional[str] = None
    embedding_api_key: str = ""
    embedding_use_grpc: Optional[bool] = None
    local_hf_batch_size: int = 32
    local_query_embed_backend: str = "hf"
    reranker: Optional[str] = None
    reranker_endpoint: Optional[str] = None
    reranker_api_key: str = ""
    local_reranker_backend: str = "vllm"
    embed_modality: str = "text"
    llm_model: str = ""
    invoke_url: Optional[str] = None
    api_key: Optional[str] = None
    react_max_steps: int = AGENTIC_REACT_MAX_STEPS

    def __post_init__(self) -> None:
        if not str(self.llm_model).strip():
            raise ValueError("Agentic retrieval requires a non-empty llm_model.")
        if int(self.react_max_steps) < 1:
            raise ValueError("react_max_steps must be >= 1.")


class AgenticRetriever:
    """Run graph-backed agentic retrieval over query IDs and query texts."""

    def __init__(self, cfg: AgenticRetrievalConfig, *, match_mode: str = "pdf_page") -> None:
        self._cfg = cfg
        self._match_mode = str(match_mode)
        self._retriever = Retriever(
            vdb_kwargs={
                "vdb_op": str(cfg.vdb_op),
                "vdb_kwargs": dict(cfg.vdb_kwargs or {}),
            },
            embed_kwargs={
                "model_name": str(cfg.query_embedder or VL_EMBED_MODEL),
                "embed_model_name": str(cfg.query_embedder or VL_EMBED_MODEL),
                "embedding_endpoint": cfg.embedding_endpoint,
                "api_key": cfg.embedding_api_key,
                "input_type": "query",
                "local_ingest_embed_backend": str(cfg.local_query_embed_backend),
                "inference_batch_size": int(cfg.local_hf_batch_size),
                "embed_inference_batch_size": int(cfg.local_hf_batch_size),
            },
            top_k=AGENTIC_TOP_K,
            rerank=bool(cfg.reranker),
            rerank_kwargs={
                "model_name": cfg.reranker or VL_RERANK_MODEL,
                "invoke_url": cfg.reranker_endpoint,
                "api_key": cfg.reranker_api_key,
                "local_reranker_backend": str(cfg.local_reranker_backend),
                "modality": str(cfg.embed_modality),
            },
        )
        self._lock = threading.Lock()

    def retrieve(self, query_ids: Sequence[str], query_texts: Sequence[str]) -> pd.DataFrame:
        """Return selected ranked documents for each query.

        The output schema matches ``SelectionAgentOperator``: ``query_id``,
        ``doc_id``, ``rank``, and ``message``.
        """

        if len(query_ids) != len(query_texts):
            raise ValueError("query_ids and query_texts must have the same length.")

        from nemo_retriever.graph.react_agent_operator import ReActAgentOperator
        from nemo_retriever.graph.rrf_aggregator_operator import RRFAggregatorOperator
        from nemo_retriever.graph.selection_agent_operator import SelectionAgentOperator

        pipeline = (
            AgenticQueryInputOperator()
            >> ReActAgentOperator(
                invoke_url=_none_if_empty(self._cfg.invoke_url),
                llm_model=str(self._cfg.llm_model),
                retriever_fn=self._retrieve_for_agent,
                retriever_top_k=AGENTIC_TOP_K,
                target_top_k=AGENTIC_TOP_K,
                user_msg_type="with_results",
                max_steps=int(self._cfg.react_max_steps),
                api_key=_none_if_empty(self._cfg.api_key),
                parallel_tool_calls=AGENTIC_PARALLEL_TOOL_CALLS,
                num_concurrent=AGENTIC_NUM_CONCURRENT,
            )
            >> RRFAggregatorOperator(k=AGENTIC_RRF_K)
            >> SelectionAgentOperator(
                invoke_url=_none_if_empty(self._cfg.invoke_url),
                llm_model=str(self._cfg.llm_model),
                top_k=AGENTIC_TOP_K,
                api_key=_none_if_empty(self._cfg.api_key),
                parallel_tool_calls=AGENTIC_PARALLEL_TOOL_CALLS,
            )
            >> AgenticSelectionOutputOperator()
        )
        graph_retriever = Retriever(
            graph=pipeline,
            top_k=AGENTIC_TOP_K,
            embed_kwargs={"text_column": "query_text"},
        )
        raw_hits = graph_retriever.queries([str(query_text) for query_text in query_texts], top_k=AGENTIC_TOP_K)
        return _raw_hits_to_agentic_result([str(query_id) for query_id in query_ids], raw_hits)

    def _retrieve_for_agent(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Retriever callback used by ``ReActAgentOperator``."""

        with self._lock:
            hits = self._retriever.query(str(query_text), top_k=int(top_k))

        docs: list[dict[str, Any]] = []
        for hit in hits:
            doc_id = _doc_id_for_match_mode(dict(hit), match_mode=self._match_mode)
            if not doc_id:
                continue
            docs.append(
                {
                    "doc_id": doc_id,
                    "text": str(hit.get("text", ""))[:AGENTIC_TEXT_TRUNCATION],
                    "score": _hit_score(hit),
                }
            )
            if len(docs) >= int(top_k):
                break
        return docs


def run_agentic_recall_evaluation(
    *,
    query_csv: Path,
    cfg: AgenticRetrievalConfig,
    match_mode: str,
    ks: Sequence[int] = (1, 5, 10),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]], dict[str, dict[str, float]], dict[str, float]]:
    """Run agentic retrieval for a recall query CSV and compute metrics."""

    df_query = _normalize_query_df(pd.read_csv(Path(query_csv)), match_mode=str(match_mode))
    query_ids = [str(idx) for idx in df_query.index]
    query_texts = df_query["query"].astype(str).tolist()
    qrels = build_qrels(query_ids, df_query["golden_answer"].astype(str).tolist())

    start = time.time()
    result = AgenticRetriever(cfg, match_mode=str(match_mode)).retrieve(query_ids, query_texts)
    elapsed = time.time() - start
    if elapsed > 0:
        logger.info(
            "Agentic retrieval time for %d queries: %.2f seconds (average %.2f queries/second)",
            len(query_ids),
            elapsed,
            len(query_ids) / elapsed,
        )

    run = build_beir_run_from_agentic_result(query_ids, result)
    metrics = compute_beir_metrics(qrels, run, ks=ks)
    return df_query, result, qrels, run, metrics


def build_qrels(query_ids: Sequence[str], gold_keys: Sequence[str]) -> dict[str, dict[str, int]]:
    """Build BEIR-style qrels from normalized recall gold keys."""

    if len(query_ids) != len(gold_keys):
        raise ValueError("query_ids and gold_keys must have the same length.")
    return {str(query_id): {str(gold_key): 1} for query_id, gold_key in zip(query_ids, gold_keys)}


def build_beir_run_from_agentic_result(
    query_ids: Sequence[str],
    result: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Convert ``SelectionAgentOperator`` output to BEIR run format."""

    run: dict[str, dict[str, float]] = {str(query_id): {} for query_id in query_ids}
    if result.empty:
        return run

    required = {"query_id", "doc_id", "rank"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Agentic result missing required columns: {sorted(missing)}")

    for query_id, group in result.groupby("query_id", sort=False):
        ordered = group.sort_values("rank")
        n = len(ordered.index)
        scores: dict[str, float] = {}
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            doc_id = str(row["doc_id"])
            if doc_id and doc_id not in scores:
                scores[doc_id] = float(n - rank + 1)
        run[str(query_id)] = scores
    return run


def _raw_hits_to_agentic_result(query_ids: Sequence[str], raw_hits: Sequence[Sequence[dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_id, hits in zip(query_ids, raw_hits):
        for rank, hit in enumerate(hits, start=1):
            rows.append(
                {
                    "query_id": str(query_id),
                    "doc_id": str(hit.get("doc_id") or hit.get("pdf_page") or ""),
                    "rank": int(hit.get("rank", rank)),
                    "message": str(hit.get("message", "")),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["query_id", "doc_id", "rank", "message"])
    return pd.DataFrame(rows)


def _doc_id_for_match_mode(hit: dict[str, Any], *, match_mode: str) -> str:
    if match_mode == "audio_segment":
        return _hit_to_audio_segment_key(hit) or ""
    if match_mode == "pdf_only":
        return _doc_id_from_hit(hit)
    return _pdf_page_from_hit(hit)


def _pdf_page_from_hit(hit: dict[str, Any]) -> str:
    pdf_page = hit.get("pdf_page")
    if isinstance(pdf_page, str) and pdf_page.strip():
        return pdf_page.strip()

    source = hit.get("source") or hit.get("source_id") or hit.get("path")
    page_number = hit.get("page_number")
    if source and page_number is not None:
        return f"{Path(str(source)).stem}_{page_number}"
    return _doc_id_from_hit(hit)


def _doc_id_from_hit(hit: dict[str, Any]) -> str:
    for key in ("pdf_basename", "source_id", "path", "source", "doc_id"):
        value = hit.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_pdf_name(Path(value).stem)
    return ""


def _hit_score(hit: dict[str, Any]) -> float:
    for key in ("_rerank_score", "_score", "score"):
        if key in hit:
            try:
                return float(hit[key])
            except (TypeError, ValueError):
                return 0.0
    if "_distance" in hit:
        try:
            return -float(hit["_distance"])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _none_if_empty(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped or stripped.lower() in {"none", "null"}:
        return None
    return stripped
