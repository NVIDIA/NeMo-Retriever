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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.models import VL_RERANK_MODEL
from nemo_retriever.query.agentic_options import (
    agentic_backend_top_k_error,
    agentic_int_min_error,
    agentic_int_value,
    agentic_temperature_error,
)
from nemo_retriever.tools.recall.beir import (
    BeirDataset,
    VALID_BEIR_DOC_ID_FIELDS,
    build_beir_run_from_ranked_doc_ids,
    compute_beir_metrics,
    load_beir_dataset,
)
from nemo_retriever.tools.recall.beir import _extract_doc_id_from_hit as _beir_doc_id_from_hit
from nemo_retriever.tools.recall.core import (
    _hit_to_audio_segment_key,
    _normalize_pdf_name,
    _normalize_query_df,
    _recall_at_k,
)
from nemo_retriever.graph.retriever import Retriever

logger = logging.getLogger(__name__)

AGENTIC_RETRIEVER_TOP_K = 10
AGENTIC_TARGET_TOP_K = 10
AGENTIC_BACKEND_TOP_K = 20  # backend retrieve-pool depth. show-count stays AGENTIC_TARGET_TOP_K=10
AGENTIC_SELECTION_TOP_K = 10
AGENTIC_NUM_CONCURRENT = 1
AGENTIC_TEXT_TRUNCATION = 0
AGENTIC_PARALLEL_TOOL_CALLS = False
AGENTIC_RRF_K = 60
AGENTIC_REACT_MAX_STEPS = 50
AGENTIC_TEMPERATURE = 0.0  # agent LLM sampling temperature (0.0 = greedy)


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
    query_embedder: Optional[str] = None
    query_embedder_provider_prefix: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    embedding_api_key: str = ""
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
    text_truncation: int = AGENTIC_TEXT_TRUNCATION
    num_concurrent: int = AGENTIC_NUM_CONCURRENT
    # Forwarded verbatim as the OpenAI `reasoning_effort` field on every LLM
    # call when explicitly configured.
    reasoning_effort: Optional[str] = None
    # Backend retrieve-pool depth, distinct from the final selected top_k.
    backend_top_k: int = AGENTIC_BACKEND_TOP_K
    # Sampling temperature sent on every agent LLM call (0.0 = greedy).
    temperature: float = AGENTIC_TEMPERATURE
    # Final number of documents the agent targets/selects and the pipeline returns.
    # Drives the ReAct target, the RRF/selection cut, and the per-hop fetch depth
    # (which is raised to at least this). Defaults to 10.
    top_k: int = AGENTIC_TARGET_TOP_K

    def __post_init__(self) -> None:
        if self.llm_model is None or not str(self.llm_model).strip():
            raise ValueError("Agentic retrieval requires a non-empty llm_model.")
        for field_name, value, min_value in (
            ("react_max_steps", self.react_max_steps, 1),
            ("text_truncation", self.text_truncation, 0),
            ("top_k", self.top_k, 1),
            ("num_concurrent", self.num_concurrent, 1),
        ):
            integer_error = agentic_int_min_error(value, field_name=field_name, min_value=min_value)
            if integer_error:
                raise ValueError(integer_error)
            object.__setattr__(self, field_name, agentic_int_value(value, field_name=field_name))

        backend_error = agentic_backend_top_k_error(
            self.backend_top_k,
            target_top_k=int(self.top_k),
            field_name="backend_top_k",
        )
        if backend_error:
            raise ValueError(backend_error)
        object.__setattr__(self, "backend_top_k", agentic_int_value(self.backend_top_k, field_name="backend_top_k"))

        temperature_error = agentic_temperature_error(
            self.temperature,
            invoke_url=self.invoke_url,
            field_name="temperature",
        )
        if temperature_error:
            raise ValueError(temperature_error)
        object.__setattr__(self, "temperature", float(self.temperature))


class AgenticRetriever:
    """Run graph-backed agentic retrieval over query IDs and query texts."""

    def __init__(
        self,
        cfg: AgenticRetrievalConfig,
        *,
        match_mode: str = "pdf_page",
        doc_id_field: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._match_mode = str(match_mode)
        self._doc_id_field = str(doc_id_field) if doc_id_field else None
        if self._doc_id_field is not None and self._doc_id_field not in VALID_BEIR_DOC_ID_FIELDS:
            raise ValueError(f"Unsupported doc_id_field: {self._doc_id_field}")
        embed_kwargs: dict[str, Any] = {
            "embed_model_provider_prefix": cfg.query_embedder_provider_prefix,
            "embedding_endpoint": cfg.embedding_endpoint,
            "api_key": cfg.embedding_api_key,
            "input_type": "query",
            "local_ingest_embed_backend": str(cfg.local_query_embed_backend),
            "inference_batch_size": int(cfg.local_hf_batch_size),
            "embed_inference_batch_size": int(cfg.local_hf_batch_size),
        }
        if cfg.query_embedder:
            embed_kwargs["model_name"] = str(cfg.query_embedder)
            embed_kwargs["embed_model_name"] = str(cfg.query_embedder)

        self._retriever = Retriever(
            vdb_kwargs={
                "vdb_op": str(cfg.vdb_op),
                "vdb_kwargs": dict(cfg.vdb_kwargs or {}),
            },
            embed_kwargs=embed_kwargs,
            top_k=AGENTIC_RETRIEVER_TOP_K,
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

        from nemo_retriever.operators.graph_ops.react_agent_operator import ReActAgentOperator
        from nemo_retriever.operators.graph_ops.rrf_aggregator_operator import RRFAggregatorOperator
        from nemo_retriever.operators.graph_ops.selection_agent_operator import SelectionAgentOperator

        # Honor the caller's requested top_k. The per-hop fetch is raised to at least
        # the default pool depth so the agent always sees a full working set even for
        # small top_k.
        target_top_k = int(self._cfg.top_k)
        per_hop_top_k = max(AGENTIC_RETRIEVER_TOP_K, target_top_k)

        pipeline = (
            AgenticQueryInputOperator()
            >> ReActAgentOperator(
                invoke_url=_none_if_empty(self._cfg.invoke_url),
                llm_model=str(self._cfg.llm_model),
                retriever_fn=self._retrieve_for_agent,
                retriever_top_k=per_hop_top_k,
                target_top_k=target_top_k,
                user_msg_type="with_results",
                max_steps=int(self._cfg.react_max_steps),
                extended_relevance=True,
                api_key=_none_if_empty(self._cfg.api_key),
                parallel_tool_calls=AGENTIC_PARALLEL_TOOL_CALLS,
                num_concurrent=int(self._cfg.num_concurrent),
                reasoning_effort=self._cfg.reasoning_effort,
                backend_top_k=self._cfg.backend_top_k,
                temperature=float(self._cfg.temperature),
            )
            >> RRFAggregatorOperator(k=AGENTIC_RRF_K)
            >> SelectionAgentOperator(
                invoke_url=_none_if_empty(self._cfg.invoke_url),
                llm_model=str(self._cfg.llm_model),
                top_k=target_top_k,
                api_key=_none_if_empty(self._cfg.api_key),
                parallel_tool_calls=AGENTIC_PARALLEL_TOOL_CALLS,
                extended_relevance=True,  # match Path A
                text_truncation=int(self._cfg.text_truncation),
                reasoning_effort=self._cfg.reasoning_effort,
                temperature=float(self._cfg.temperature),
            )
            >> AgenticSelectionOutputOperator()
        )
        graph_retriever = Retriever(
            graph=pipeline,
            top_k=target_top_k,
            embed_kwargs={"text_column": "query_text"},
        )
        raw_hits = graph_retriever.queries(
            [str(query_text) for query_text in query_texts],
            top_k=target_top_k,
        )
        return _raw_hits_to_agentic_result([str(query_id) for query_id in query_ids], raw_hits)

    def _retrieve_for_agent(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Retriever callback used by ``ReActAgentOperator``.

        Retrieval is serialized across concurrent ReAct workers via ``self._lock``
        because the shared ``Retriever`` is not assumed thread-safe. This caps the
        retrieval hop at single-threaded throughput; it is intentional and not the
        bottleneck, since per-query cost is dominated by the multi-step LLM calls,
        which still run concurrently under ``num_concurrent > 1``.
        """

        with self._lock:
            hits = self._retriever.query(str(query_text), top_k=int(top_k))

        docs: list[dict[str, Any]] = []
        doc_id_field = getattr(self, "_doc_id_field", None)
        for hit in hits:
            hit_dict = dict(hit)
            doc_id = (
                _beir_doc_id_from_hit(hit_dict, doc_id_field=doc_id_field)
                if doc_id_field is not None
                else _doc_id_for_match_mode(hit_dict, match_mode=self._match_mode)
            )
            if not doc_id:
                continue
            text = str(hit_dict.get("text", ""))
            if int(self._cfg.text_truncation) > 0:
                text = text[: int(self._cfg.text_truncation)]
            docs.append(
                {
                    "doc_id": doc_id,
                    "text": text,
                    "score": _hit_score(hit_dict),
                }
            )
            if len(docs) >= int(top_k):
                break
        return docs


def _raw_hits_to_agentic_result(query_ids: Sequence[str], raw_hits: Sequence[Sequence[dict[str, Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # The agentic graph (AgenticQueryInputOperator) assigns POSITIONAL query_ids
    # "0".."N-1" to its inputs, independent of the caller's real ids. Each hit
    # therefore carries its positional index; map that back through `query_ids`
    # to recover the caller's real id. This is robust to (a) ThreadPool
    # completion-order reordering at num_concurrent>1, (b) queries that produced
    # no rows (gaps don't shift anything), and (c) sharded offset ranges where the
    # positional index != the caller's real id (e.g. query_ids=["1000".."1049"]).
    # For a full sweep with sequential ids ("0".."N-1") this is a no-op vs the
    # old positional zip.
    n = len(query_ids)
    for pos, hits in enumerate(raw_hits):
        for rank, hit in enumerate(hits, start=1):
            raw_qid = hit.get("query_id")
            if raw_qid is not None and str(raw_qid).isdigit() and int(raw_qid) < n:
                qid = str(query_ids[int(raw_qid)])
            elif pos < n:
                qid = str(query_ids[pos])
            else:
                qid = str(raw_qid) if raw_qid is not None else ""
            rows.append(
                {
                    "query_id": qid,
                    "doc_id": str(hit.get("doc_id") or hit.get("pdf_page") or ""),
                    "rank": int(hit.get("rank", rank)),
                    "message": str(hit.get("message", "")),
                    "result_source": str(hit.get("result_source", "")),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["query_id", "doc_id", "rank", "message", "result_source"])
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


def _query_ids_from_normalized_query_df(df_query: pd.DataFrame) -> list[str]:
    if "query_id" in df_query.columns:
        return df_query["query_id"].astype(str).tolist()
    return [str(idx) for idx in range(len(df_query.index))]


def _agentic_result_to_ranked_doc_ids(query_ids: Sequence[str], result: pd.DataFrame) -> list[list[str]]:
    ranked_by_qid: dict[str, list[str]] = {str(query_id): [] for query_id in query_ids}
    seen_by_qid: dict[str, set[str]] = {str(query_id): set() for query_id in query_ids}
    if result.empty:
        return [ranked_by_qid[str(query_id)] for query_id in query_ids]

    required = {"query_id", "doc_id", "rank"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Agentic result missing required columns: {sorted(missing)}")

    for query_id, group in result.groupby("query_id", sort=False):
        qid = str(query_id)
        if qid not in ranked_by_qid:
            continue
        for _, row in group.sort_values("rank").iterrows():
            doc_id = str(row.get("doc_id", "")).strip()
            if not doc_id or doc_id in seen_by_qid[qid]:
                continue
            seen_by_qid[qid].add(doc_id)
            ranked_by_qid[qid].append(doc_id)

    return [ranked_by_qid[str(query_id)] for query_id in query_ids]


def run_agentic_audio_recall_evaluation(
    *,
    query_csv: Path,
    cfg: AgenticRetrievalConfig,
    ks: Sequence[int] = (1, 5, 10),
    audio_match_tolerance_secs: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[list[str]], dict[str, float]]:
    """Run agentic retrieval against an audio recall CSV and compute recall metrics."""

    df_query = _normalize_query_df(pd.read_csv(query_csv), match_mode="audio_segment")
    query_ids = _query_ids_from_normalized_query_df(df_query)
    queries = df_query["query"].astype(str).tolist()
    gold_doc_ids = df_query["golden_answer"].astype(str).tolist()

    result = AgenticRetriever(cfg, match_mode="audio_segment").retrieve(query_ids, queries)
    retrieved_doc_ids = _agentic_result_to_ranked_doc_ids(query_ids, result)
    ks_sorted = sorted({int(k) for k in ks if int(k) > 0})
    if not ks_sorted:
        raise ValueError("ks must contain at least one positive integer")
    metrics = {
        f"recall@{k}": _recall_at_k(
            gold_doc_ids,
            retrieved_doc_ids,
            k,
            match_mode="audio_segment",
            audio_match_tolerance_secs=float(audio_match_tolerance_secs),
        )
        for k in ks_sorted
    }
    return df_query, result, gold_doc_ids, retrieved_doc_ids, metrics


def run_agentic_beir_evaluation(
    *,
    loader: str,
    dataset_name: str,
    cfg: AgenticRetrievalConfig,
    split: str = "test",
    query_language: str | None = None,
    doc_id_field: str = "pdf_basename",
    ks: Sequence[int] = (1, 3, 5, 10),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]], dict[str, dict[str, float]], dict[str, float]]:
    """Load a BEIR-style dataset, run agentic retrieval, and compute metrics."""

    dataset = load_beir_dataset(
        loader,
        dataset_name=dataset_name,
        split=split,
        query_language=query_language,
        doc_id_field=doc_id_field,
    )
    result, ranked_doc_ids = agentic_beir_retrieve(dataset, cfg, doc_id_field=doc_id_field)
    run = build_beir_run_from_ranked_doc_ids(dataset.query_ids, ranked_doc_ids)
    metrics = compute_beir_metrics(dataset.qrels, run, ks=ks)
    df_query = pd.DataFrame({"query_id": dataset.query_ids, "query": dataset.queries})
    return df_query, result, dataset.qrels, run, metrics


def agentic_beir_retrieve(
    dataset: BeirDataset,
    cfg: AgenticRetrievalConfig,
    *,
    doc_id_field: str = "pdf_basename",
) -> tuple[pd.DataFrame, list[list[str]]]:
    """Run agentic (ReAct) retrieval over a pre-loaded BEIR dataset.

    Returns the raw agentic result DataFrame (``query_id``/``doc_id``/``rank``/…)
    and the per-query ranked doc-id lists aligned to ``dataset.query_ids``.
    Doc-id matching follows ``doc_id_field`` (BEIR semantics, applied inside the
    agent's retrieve hop), so the ranked ids align with the qrels keys produced by
    ``load_beir_dataset(..., doc_id_field=...)``. Splitting this out lets callers
    that already hold a loaded dataset (e.g. the harness) reuse the agent's
    retrieve+rank step without re-loading or re-implementing it.
    """
    result = AgenticRetriever(cfg, match_mode="pdf_page", doc_id_field=doc_id_field).retrieve(
        list(dataset.query_ids),
        list(dataset.queries),
    )
    ranked_doc_ids = _agentic_result_to_ranked_doc_ids(list(dataset.query_ids), result)
    return result, ranked_doc_ids
