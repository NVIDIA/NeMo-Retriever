"""BEIR-style evaluation helpers backed by LanceDB retrieval."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from nemo_retriever.retriever import Retriever

logger = logging.getLogger(__name__)

DEFAULT_BEIR_KS: tuple[int, ...] = (1, 3, 5, 10)
VALID_BEIR_DOC_ID_FIELDS: frozenset[str] = frozenset({"pdf_basename", "pdf_page", "source_id", "path"})


@dataclass(frozen=True)
class BeirDataset:
    dataset_name: str
    query_ids: list[str]
    queries: list[str]
    qrels: dict[str, dict[str, int]]


@dataclass(frozen=True)
class BeirConfig:
    lancedb_uri: str
    lancedb_table: str
    embedding_model: str
    loader: str
    dataset_name: str
    split: str = "test"
    query_language: str | None = None
    doc_id_field: str = "pdf_basename"
    ks: Sequence[int] = DEFAULT_BEIR_KS
    embedding_http_endpoint: str | None = None
    embedding_api_key: str = ""
    hybrid: bool = False
    nprobes: int = 0
    refine_factor: int = 10
    local_hf_device: str | None = None
    local_hf_cache_dir: str | None = None
    local_hf_batch_size: int = 64
    reranker: bool = False
    reranker_model_name: str = "nvidia/llama-nemotron-rerank-1b-v2"
    reranker_endpoint: str | None = None
    reranker_api_key: str = ""
    reranker_batch_size: int = 32
    #: Passed to :class:`~nemo_retriever.retriever.Retriever` for local query embedding.
    local_query_embed_backend: str = "auto"


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def build_queries_by_id(rows: Iterable[Any], *, query_language: str | None = None) -> tuple[list[str], list[str]]:
    """Normalize iterable rows into ordered ``(query_ids, queries)``."""
    normalized_language = str(query_language).strip().lower() if query_language is not None else None
    query_ids: list[str] = []
    queries: list[str] = []

    for idx, row in enumerate(rows):
        query_text = _row_get(row, "query")
        if not isinstance(query_text, str) or not query_text.strip():
            continue

        if normalized_language is not None:
            row_language = str(_row_get(row, "language", "") or "").strip().lower()
            if row_language != normalized_language:
                continue

        query_id = _row_get(row, "query_id", idx)
        query_ids.append(str(query_id))
        queries.append(query_text)

    return query_ids, queries


def build_qrels_by_query_id(
    rows: Iterable[Any], *, allowed_query_ids: set[str] | None = None
) -> dict[str, dict[str, int]]:
    """Normalize iterable qrel rows into ``{query_id: {doc_id: score}}``."""
    qrels: defaultdict[str, dict[str, int]] = defaultdict(dict)

    for row in rows:
        query_id = _row_get(row, "query_id")
        corpus_id = _row_get(row, "corpus_id")
        if query_id is None or corpus_id is None:
            continue

        query_id_str = str(query_id)
        if allowed_query_ids is not None and query_id_str not in allowed_query_ids:
            continue

        try:
            score = int(_row_get(row, "score", 1))
        except (TypeError, ValueError):
            score = 1
        qrels[query_id_str][str(corpus_id)] = score

    return dict(qrels)


def load_beir_dataset(
    loader: str, *, dataset_name: str, split: str = "test", query_language: str | None = None
) -> BeirDataset:
    """Load a BEIR-style dataset for evaluation."""
    loader_name = str(loader).strip().lower()
    if loader_name != "vidore_hf":
        raise ValueError(f"Unsupported BEIR loader: {loader}")

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in runtime environments
        raise ImportError("BEIR-style ViDoRe evaluation requires the 'datasets' package.") from exc

    queries_rows = load_dataset(f"vidore/{dataset_name}", data_dir="queries", split=split)
    qrels_rows = load_dataset(f"vidore/{dataset_name}", data_dir="qrels", split=split)

    query_ids, queries = build_queries_by_id(queries_rows, query_language=query_language)
    if not query_ids:
        raise ValueError(f"No queries loaded for dataset={dataset_name!r} split={split!r}")

    allowed_query_ids = set(query_ids)
    qrels = build_qrels_by_query_id(qrels_rows, allowed_query_ids=allowed_query_ids)
    if not qrels:
        raise ValueError(f"No qrels loaded for dataset={dataset_name!r} split={split!r}")

    filtered_pairs = [(qid, query) for qid, query in zip(query_ids, queries) if qid in qrels]
    if not filtered_pairs:
        raise ValueError(f"No query/qrels overlap for dataset={dataset_name!r} split={split!r}")

    filtered_query_ids = [qid for qid, _ in filtered_pairs]
    filtered_queries = [query for _, query in filtered_pairs]

    return BeirDataset(
        dataset_name=str(dataset_name),
        query_ids=filtered_query_ids,
        queries=filtered_queries,
        qrels={qid: qrels[qid] for qid in filtered_query_ids},
    )


def _extract_doc_id_from_hit(hit: dict[str, Any], *, doc_id_field: str) -> str | None:
    direct_value = hit.get(doc_id_field)
    if isinstance(direct_value, str) and direct_value.strip():
        return direct_value.strip()

    source_id = hit.get("source_id")
    if isinstance(source_id, str) and source_id.strip():
        source_path = source_id.strip()
    else:
        source_path = ""
        raw_source = hit.get("source")
        if isinstance(raw_source, str) and raw_source.strip():
            try:
                parsed_source = json.loads(raw_source)
            except json.JSONDecodeError:
                parsed_source = {}
            if isinstance(parsed_source, dict):
                source_path = str(parsed_source.get("source_id") or "").strip()

    if not source_path:
        return None

    path = Path(source_path)
    fallbacks = {
        "pdf_basename": path.stem,
        "source_id": source_path,
        "path": source_path,
        "pdf_page": hit.get("pdf_page") or path.stem,
    }
    fallback_value = fallbacks.get(doc_id_field)
    if isinstance(fallback_value, str) and fallback_value.strip():
        return fallback_value.strip()
    return None


def build_beir_run_from_hits(
    query_ids: Sequence[str],
    raw_hits: Sequence[Sequence[dict[str, Any]]],
    *,
    doc_id_field: str = "pdf_basename",
) -> dict[str, dict[str, float]]:
    """Convert ranked hit lists into BEIR/pytrec_eval run format."""
    if doc_id_field not in VALID_BEIR_DOC_ID_FIELDS:
        raise ValueError(f"Unsupported doc_id_field: {doc_id_field}")

    run: dict[str, dict[str, float]] = {}
    for query_id, hits in zip(query_ids, raw_hits):
        ordered_doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()

        for hit in hits:
            doc_id = _extract_doc_id_from_hit(dict(hit), doc_id_field=doc_id_field)
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            ordered_doc_ids.append(doc_id)

        ranked_scores = {doc_id: float(len(ordered_doc_ids) - rank) for rank, doc_id in enumerate(ordered_doc_ids)}
        run[str(query_id)] = ranked_scores

    return run


def _sorted_doc_ids_for_query(run_for_query: dict[str, float]) -> list[str]:
    return [doc_id for doc_id, _score in sorted(run_for_query.items(), key=lambda item: (-item[1], item[0]))]


def _dcg(relevances: Sequence[int], *, k: int) -> float:
    total = 0.0
    for rank, rel in enumerate(relevances[:k], start=1):
        if rel <= 0:
            continue
        total += float(rel) / math.log2(rank + 1)
    return total


def compute_beir_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    *,
    ks: Sequence[int] = DEFAULT_BEIR_KS,
) -> dict[str, float]:
    """Compute aggregate BEIR-style metrics at each requested cutoff."""
    ks_sorted = sorted({int(k) for k in ks if int(k) > 0})
    if not ks_sorted:
        raise ValueError("ks must contain at least one positive integer")

    query_ids = sorted(qrels.keys())
    if not query_ids:
        raise ValueError("qrels must contain at least one query")

    aggregates: dict[str, list[float]] = defaultdict(list)
    for query_id in query_ids:
        gold = qrels.get(query_id, {})
        ranked_doc_ids = _sorted_doc_ids_for_query(run.get(query_id, {}))
        relevant_scores = [int(score) for score in gold.values() if int(score) > 0]
        relevant_count = len(relevant_scores)

        for k in ks_sorted:
            top_doc_ids = ranked_doc_ids[:k]
            top_relevances = [int(gold.get(doc_id, 0)) for doc_id in top_doc_ids]
            top_binary = [1 if rel > 0 else 0 for rel in top_relevances]
            hits = sum(top_binary)

            recall_at_k = (hits / float(relevant_count)) if relevant_count else 0.0

            ideal_relevances = sorted(relevant_scores, reverse=True)
            idcg = _dcg(ideal_relevances, k=k)
            ndcg_at_k = (_dcg(top_relevances, k=k) / idcg) if idcg > 0 else 0.0

            aggregates[f"recall@{k}"].append(recall_at_k)
            aggregates[f"ndcg@{k}"].append(ndcg_at_k)

    return {metric_name: sum(values) / float(len(values)) for metric_name, values in sorted(aggregates.items())}


def evaluate_lancedb_beir(
    cfg: BeirConfig,
) -> tuple[BeirDataset, list[list[dict[str, Any]]], dict[str, dict[str, float]], dict[str, float]]:
    """Load a BEIR-style dataset, retrieve from LanceDB, and compute aggregate metrics."""
    dataset = load_beir_dataset(
        cfg.loader,
        dataset_name=cfg.dataset_name,
        split=cfg.split,
        query_language=cfg.query_language,
    )
    ks = tuple(sorted({int(k) for k in cfg.ks if int(k) > 0}))
    retriever = Retriever(
        lancedb_uri=str(cfg.lancedb_uri),
        lancedb_table=str(cfg.lancedb_table),
        embedder=str(cfg.embedding_model),
        embedding_http_endpoint=cfg.embedding_http_endpoint,
        embedding_api_key=(cfg.embedding_api_key or "").strip(),
        top_k=max(ks),
        nprobes=int(cfg.nprobes),
        refine_factor=int(cfg.refine_factor),
        hybrid=bool(cfg.hybrid),
        local_hf_device=cfg.local_hf_device,
        local_hf_cache_dir=Path(cfg.local_hf_cache_dir) if cfg.local_hf_cache_dir else None,
        local_hf_batch_size=int(cfg.local_hf_batch_size),
        reranker=bool(cfg.reranker),
        reranker_model_name=str(cfg.reranker_model_name),
        reranker_endpoint=cfg.reranker_endpoint,
        reranker_api_key=(cfg.reranker_api_key or "").strip(),
        reranker_batch_size=int(cfg.reranker_batch_size),
        local_query_embed_backend=str(cfg.local_query_embed_backend),
    )
    raw_hits = retriever.queries(dataset.queries)
    run = build_beir_run_from_hits(dataset.query_ids, raw_hits, doc_id_field=cfg.doc_id_field)
    metrics = compute_beir_metrics(dataset.qrels, run, ks=ks)
    logger.info(
        "Computed BEIR metrics for dataset=%s queries=%d ks=%s",
        dataset.dataset_name,
        len(dataset.query_ids),
        list(ks),
    )
    return dataset, raw_hits, run, metrics
