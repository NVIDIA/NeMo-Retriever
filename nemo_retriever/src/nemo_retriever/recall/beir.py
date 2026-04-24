"""BEIR-style evaluation helpers backed by nv-ingest-client VDB retrieval."""

from __future__ import annotations

import ast
from collections import defaultdict
import csv
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from nemo_retriever.retriever import Retriever

logger = logging.getLogger(__name__)

DEFAULT_BEIR_KS: tuple[int, ...] = (1, 3, 5, 10)
VALID_BEIR_LOADERS: frozenset[str] = frozenset(
    {"bo10k_csv", "bo767_csv", "earnings_csv", "financebench_json", "vidore_hf"}
)
VALID_BEIR_DOC_ID_FIELDS: frozenset[str] = frozenset(
    {"pdf_basename", "pdf_page", "pdf_page_modality", "source_id", "path"}
)
REPO_ROOT = Path(__file__).resolve().parents[4]
BO767_ANNOTATIONS_PATH = REPO_ROOT / "data" / "bo767_annotations.csv"
BO10K_ANNOTATIONS_PATH = REPO_ROOT / "data" / "digital_corpora_10k_annotations.csv"
EARNINGS_ANNOTATIONS_PATH = REPO_ROOT / "data" / "earnings_consulting_multimodal.csv"
FINANCEBENCH_ANNOTATIONS_PATH = REPO_ROOT / "data" / "financebench_train.json"
_ELEMENT_TYPE_ALIASES: dict[str, str] = {
    "caption": "image",
    "chart": "chart",
    "chart_caption": "chart",
    "figure": "image",
    "image": "image",
    "image_caption": "image",
    "infographic": "infographic",
    "infographic_caption": "infographic",
    "page_image": "image",
    "structured_image": "image",
    "table": "table",
    "table_caption": "table",
    "text": "text",
}


@dataclass(frozen=True)
class BeirDataset:
    dataset_name: str
    query_ids: list[str]
    queries: list[str]
    qrels: dict[str, dict[str, int]]


@dataclass(frozen=True)
class BeirConfig:
    loader: str
    dataset_name: str
    vdb_op: str = "lancedb"
    vdb_kwargs: dict[str, Any] = field(default_factory=dict)
    split: str = "test"
    query_language: str | None = None
    doc_id_field: str = "pdf_basename"
    ks: Sequence[int] = DEFAULT_BEIR_KS
    local_hf_device: str | None = None
    local_hf_cache_dir: str | None = None
    reranker: bool = False
    reranker_model_name: str = "nvidia/llama-nemotron-rerank-1b-v2"
    reranker_endpoint: str | None = None
    reranker_api_key: str = ""
    reranker_batch_size: int = 32


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _normalize_pdf_basename(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    basename = path.name if path.name else text
    return basename[:-4] if basename.lower().endswith(".pdf") else basename


def _parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}

    text = value.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    return {}


def _normalize_element_type(value: Any, *, subtype: Any = None) -> str | None:
    candidates = [value, subtype]
    for candidate in candidates:
        normalized = str(candidate or "").strip().lower()
        if not normalized:
            continue
        alias = _ELEMENT_TYPE_ALIASES.get(normalized)
        if alias:
            return alias
    return None


def _build_pdf_page_modality(pdf_basename: str, page_number: Any, element_type: str) -> str | None:
    basename = _normalize_pdf_basename(pdf_basename)
    if not basename:
        return None

    try:
        normalized_page = int(page_number)
    except (TypeError, ValueError):
        return None

    normalized_type = _normalize_element_type(element_type)
    if not normalized_type:
        return None

    return f"{basename}_{normalized_page}_{normalized_type}"


def _resolve_annotations_csv_path(dataset_name: str, *, loader_name: str) -> Path:
    dataset_str = str(dataset_name).strip()
    candidate = Path(dataset_str).expanduser()
    if candidate.suffix.lower() == ".csv":
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        return candidate
    if loader_name == "bo767_csv" and dataset_str.lower() == "bo767":
        return BO767_ANNOTATIONS_PATH
    if loader_name == "bo10k_csv" and dataset_str.lower() == "bo10k":
        return BO10K_ANNOTATIONS_PATH
    if loader_name == "earnings_csv" and dataset_str.lower() == "earnings":
        return EARNINGS_ANNOTATIONS_PATH
    raise ValueError(
        f"{loader_name} expects dataset_name='{dataset_str.lower()}' or a path to a CSV file, got {dataset_name!r}"
    )


def _resolve_annotations_json_path(dataset_name: str, *, loader_name: str) -> Path:
    dataset_str = str(dataset_name).strip()
    candidate = Path(dataset_str).expanduser()
    if candidate.suffix.lower() == ".json":
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        return candidate
    if loader_name == "financebench_json" and dataset_str.lower() == "financebench":
        return FINANCEBENCH_ANNOTATIONS_PATH
    raise ValueError(
        f"{loader_name} expects dataset_name='financebench' or a path to a JSON file, got {dataset_name!r}"
    )


def _build_csv_corpus_id(
    *,
    pdf_basename: str,
    page_number: int,
    modality: str,
    doc_id_field: str,
    loader_name: str,
) -> str:
    if doc_id_field == "pdf_page":
        return f"{pdf_basename}_{page_number}"
    if doc_id_field == "pdf_page_modality":
        return f"{pdf_basename}_{page_number}_{modality}"
    raise ValueError(
        f"{loader_name} only supports doc_id_field values " f"'pdf_page' or 'pdf_page_modality', got {doc_id_field!r}"
    )


def _load_annotations_csv_dataset(*, dataset_name: str, doc_id_field: str, loader_name: str) -> BeirDataset:
    dataset_path = _resolve_annotations_csv_path(dataset_name, loader_name=loader_name)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {dataset_path}")

    query_ids: list[str] = []
    queries: list[str] = []
    qrels: dict[str, dict[str, int]] = {}

    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            query_text = str(row.get("query") or "")
            pdf_basename = _normalize_pdf_basename(row.get("pdf"))
            modality = _normalize_element_type(row.get("modality"))
            raw_page = row.get("page")

            if not query_text.strip() or not pdf_basename:
                continue

            try:
                page_number = int(raw_page) + 1
            except (TypeError, ValueError):
                continue

            if doc_id_field == "pdf_page_modality" and modality is None:
                continue
            if modality is None:
                modality = "text"

            query_id = str(row.get("query_id") or idx)
            corpus_id = _build_csv_corpus_id(
                pdf_basename=pdf_basename,
                page_number=page_number,
                modality=modality,
                doc_id_field=doc_id_field,
                loader_name=loader_name,
            )
            query_ids.append(query_id)
            queries.append(query_text)
            qrels[query_id] = {corpus_id: 1}

    if not query_ids:
        raise ValueError(f"No queries loaded from {dataset_path}")

    return BeirDataset(
        dataset_name=str(dataset_name),
        query_ids=query_ids,
        queries=queries,
        qrels=qrels,
    )


def _load_financebench_json_dataset(*, dataset_name: str, doc_id_field: str) -> BeirDataset:
    if doc_id_field != "pdf_basename":
        raise ValueError(f"financebench_json only supports doc_id_field='pdf_basename', got {doc_id_field!r}")

    dataset_path = _resolve_annotations_json_path(dataset_name, loader_name="financebench_json")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Annotations JSON not found: {dataset_path}")

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"financebench_json expects a JSON list in {dataset_path}")

    query_ids: list[str] = []
    queries: list[str] = []
    qrels: dict[str, dict[str, int]] = {}

    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            continue

        query_text = item.get("question")
        if not isinstance(query_text, str) or not query_text.strip():
            continue

        contexts = item.get("contexts")
        if not isinstance(contexts, list) or not contexts:
            continue

        first_context = contexts[0]
        if not isinstance(first_context, dict):
            continue

        corpus_id = _normalize_pdf_basename(first_context.get("filename"))
        if not corpus_id:
            continue

        query_id = str(item.get("id") or idx)
        query_ids.append(query_id)
        queries.append(query_text)
        qrels[query_id] = {corpus_id: 1}

    if not query_ids:
        raise ValueError(f"No queries loaded from {dataset_path}")

    return BeirDataset(
        dataset_name=str(dataset_name),
        query_ids=query_ids,
        queries=queries,
        qrels=qrels,
    )


_LANG_ALIASES: dict[str, set[str]] = {
    "en": {"en", "eng", "english"},
    "fr": {"fr", "fra", "fre", "french", "français", "francais"},
    "de": {"de", "deu", "ger", "german", "deutsch"},
    "es": {"es", "spa", "spanish", "español", "espanol"},
    "it": {"it", "ita", "italian", "italiano"},
    "pt": {"pt", "por", "portuguese", "português", "portugues"},
    "zh": {"zh", "zho", "chi", "chinese"},
    "ja": {"ja", "jpn", "japanese"},
    "ko": {"ko", "kor", "korean"},
    "ar": {"ar", "ara", "arabic"},
    "ru": {"ru", "rus", "russian"},
    "nl": {"nl", "nld", "dut", "dutch"},
    "pl": {"pl", "pol", "polish"},
    "sv": {"sv", "swe", "swedish"},
    "tr": {"tr", "tur", "turkish"},
    "hi": {"hi", "hin", "hindi"},
}


def _languages_match(filter_lang: str, row_lang: str) -> bool:
    """Fuzzy language matching that accepts ISO codes and full names."""
    a = filter_lang.strip().lower()
    b = row_lang.strip().lower()
    if a == b:
        return True
    for aliases in _LANG_ALIASES.values():
        if a in aliases and b in aliases:
            return True
    return False


def build_queries_by_id(rows: Iterable[Any], *, query_language: str | None = None) -> tuple[list[str], list[str]]:
    """Normalize iterable rows into ordered ``(query_ids, queries)``."""
    normalized_language = str(query_language).strip().lower() if query_language is not None else None
    query_ids: list[str] = []
    queries: list[str] = []
    total_rows = 0
    skipped_empty = 0
    skipped_language = 0

    for idx, row in enumerate(rows):
        total_rows += 1
        query_text = _row_get(row, "query")
        if not isinstance(query_text, str) or not query_text.strip():
            skipped_empty += 1
            continue

        if normalized_language is not None:
            row_language = str(_row_get(row, "language", "") or "").strip().lower()
            if not _languages_match(normalized_language, row_language):
                skipped_language += 1
                continue

        query_id = _row_get(row, "query_id", idx)
        query_ids.append(str(query_id))
        queries.append(query_text)

    if not query_ids:
        logger.warning(
            "build_queries_by_id returned 0 queries: total_rows=%d, "
            "skipped_empty=%d, skipped_language=%d (filter=%r)",
            total_rows,
            skipped_empty,
            skipped_language,
            query_language,
        )

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
    loader: str,
    *,
    dataset_name: str,
    split: str = "test",
    query_language: str | None = None,
    doc_id_field: str = "pdf_basename",
) -> BeirDataset:
    """Load a BEIR-style dataset for evaluation."""
    loader_name = str(loader).strip().lower()
    if loader_name in {"bo767_csv", "bo10k_csv", "earnings_csv"}:
        return _load_annotations_csv_dataset(
            dataset_name=dataset_name,
            doc_id_field=str(doc_id_field),
            loader_name=loader_name,
        )
    if loader_name == "financebench_json":
        return _load_financebench_json_dataset(
            dataset_name=dataset_name,
            doc_id_field=str(doc_id_field),
        )
    if loader_name != "vidore_hf":
        raise ValueError(f"Unsupported BEIR loader: {loader}")

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in runtime environments
        raise ImportError("BEIR-style ViDoRe evaluation requires the 'datasets' package.") from exc

    ds_repo = f"vidore/{dataset_name}"
    try:
        queries_rows = load_dataset(ds_repo, "queries", split=split)
    except Exception as exc:
        logger.debug("load_dataset config='queries' failed (%s); retrying with data_dir", exc)
        queries_rows = load_dataset(ds_repo, data_dir="queries", split=split)
    try:
        qrels_rows = load_dataset(ds_repo, "qrels", split=split)
    except Exception as exc:
        logger.debug("load_dataset config='qrels' failed (%s); retrying with data_dir", exc)
        qrels_rows = load_dataset(ds_repo, data_dir="qrels", split=split)

    logger.info(
        "Loaded BEIR dataset %s split=%s: %d query rows, %d qrel rows",
        ds_repo,
        split,
        len(queries_rows),
        len(qrels_rows),
    )

    query_ids, queries = build_queries_by_id(queries_rows, query_language=query_language)
    if not query_ids:
        raise ValueError(
            f"No queries loaded for dataset={dataset_name!r} split={split!r} "
            f"query_language={query_language!r}. "
            f"Loaded {len(queries_rows)} raw rows from HuggingFace."
        )

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


def _extract_source_path_from_hit(hit: dict[str, Any]) -> str:
    source_id = hit.get("source_id")
    if isinstance(source_id, str) and source_id.strip():
        return source_id.strip()

    parsed_source = _parse_mapping(hit.get("source"))
    source_value = parsed_source.get("source_id")
    return str(source_value).strip() if isinstance(source_value, str) else ""


def _extract_page_number_from_hit(hit: dict[str, Any]) -> int | None:
    direct_page = hit.get("page_number")
    try:
        if direct_page is not None:
            return int(direct_page)
    except (TypeError, ValueError):
        pass

    metadata = _parse_mapping(hit.get("metadata"))
    try:
        if metadata.get("page_number") is not None:
            return int(metadata["page_number"])
    except (TypeError, ValueError):
        pass

    content_metadata = metadata.get("content_metadata")
    if isinstance(content_metadata, dict):
        try:
            if content_metadata.get("page_number") is not None:
                return int(content_metadata["page_number"])
        except (TypeError, ValueError):
            pass

    return None


def _extract_element_type_from_hit(hit: dict[str, Any]) -> str | None:
    direct_type = _normalize_element_type(
        hit.get("element_type") or hit.get("_content_type") or hit.get("content_type") or hit.get("document_type")
    )
    if direct_type is not None:
        return direct_type

    metadata = _parse_mapping(hit.get("metadata"))
    content_metadata = metadata.get("content_metadata") if isinstance(metadata.get("content_metadata"), dict) else {}
    normalized = _normalize_element_type(
        metadata.get("_content_type") or metadata.get("content_type") or metadata.get("document_type"),
        subtype=content_metadata.get("subtype") if isinstance(content_metadata, dict) else None,
    )
    if normalized is not None:
        return normalized

    normalized = _normalize_element_type(content_metadata.get("type") if isinstance(content_metadata, dict) else None)
    if normalized is not None:
        return normalized

    for metadata_key, fallback_type in (
        ("table_metadata", "table"),
        ("chart_metadata", "chart"),
        ("image_metadata", "image"),
        ("infographic_metadata", "infographic"),
    ):
        if metadata_key in metadata:
            return fallback_type

    return None


def _extract_doc_id_from_hit(hit: dict[str, Any], *, doc_id_field: str) -> str | None:
    direct_value = hit.get(doc_id_field)
    if isinstance(direct_value, str) and direct_value.strip():
        return direct_value.strip()

    source_path = _extract_source_path_from_hit(hit)
    if not source_path:
        return None

    path = Path(source_path)
    if doc_id_field == "pdf_page_modality":
        return _build_pdf_page_modality(
            path.stem,
            _extract_page_number_from_hit(hit),
            _extract_element_type_from_hit(hit) or "",
        )

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
    """Load a BEIR-style dataset, retrieve from a VDB, and compute aggregate metrics."""
    dataset = load_beir_dataset(
        cfg.loader,
        dataset_name=cfg.dataset_name,
        split=cfg.split,
        query_language=cfg.query_language,
        doc_id_field=cfg.doc_id_field,
    )
    ks = tuple(sorted({int(k) for k in cfg.ks if int(k) > 0}))
    retriever = Retriever(
        vdb_op=str(cfg.vdb_op),
        vdb_kwargs=dict(cfg.vdb_kwargs or {}),
        top_k=max(ks),
        local_hf_device=cfg.local_hf_device,
        local_hf_cache_dir=Path(cfg.local_hf_cache_dir) if cfg.local_hf_cache_dir else None,
        reranker=bool(cfg.reranker),
        reranker_model_name=str(cfg.reranker_model_name),
        reranker_endpoint=cfg.reranker_endpoint,
        reranker_api_key=(cfg.reranker_api_key or "").strip(),
        reranker_batch_size=int(cfg.reranker_batch_size),
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
