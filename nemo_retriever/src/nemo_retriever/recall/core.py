# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from nemo_retriever.retriever import Retriever

logger = logging.getLogger(__name__)
AUDIO_MATCH_TOLERANCE_SECS = 2.0

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecallConfig:
    lancedb_uri: str
    lancedb_table: str
    embedding_model: str
    # Embedding endpoints (optional).
    #
    # If neither HTTP nor gRPC endpoint is provided (and embedding_endpoint is empty),
    # stage7 will fall back to local HuggingFace embeddings via:
    #   nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder
    embedding_http_endpoint: Optional[str] = None
    embedding_grpc_endpoint: Optional[str] = None
    # Back-compat single endpoint string (http URL or host:port for gRPC).
    embedding_endpoint: Optional[str] = None
    embedding_api_key: str = ""
    top_k: int = 10
    ks: Sequence[int] = (1, 3, 5, 10)
    # ANN search tuning (LanceDB IVF_HNSW_SQ).
    # nprobes=0 means "search all partitions" (exhaustive); refine_factor re-ranks
    # top candidates with full-precision vectors to eliminate SQ quantization error.
    nprobes: int = 0
    refine_factor: int = 10
    hybrid: bool = False
    # Local HF knobs (only used when endpoints are missing).
    local_hf_device: Optional[str] = None
    local_hf_cache_dir: Optional[str] = None
    local_hf_batch_size: int = 64
    # Gold/retrieval comparison mode:
    # - pdf_page: compare on "{pdf}_{page}" keys
    # - pdf_only: compare on "{pdf}" document keys
    # - audio_segment: compare on "media_id<TAB>start<TAB>end" segment keys
    match_mode: str = "pdf_page"
    audio_match_tolerance_secs: float = AUDIO_MATCH_TOLERANCE_SECS
    reranker: Optional[str] = None
    reranker_endpoint: Optional[str] = None
    reranker_api_key: str = ""
    reranker_batch_size: int = 32


def _normalize_pdf_name(value: str) -> str:
    return str(value).replace(".pdf", "")


def _normalize_audio_media_id(value: object) -> str:
    basename = Path(str(value)).name
    return basename.split(".", 1)[0] if basename else ""


def _encode_audio_segment_key(media_id: str, start_time: float, end_time: float) -> str:
    return f"{media_id}\t{float(start_time):.6f}\t{float(end_time):.6f}"


def _parse_audio_segment_key(key: str) -> tuple[str, float, float]:
    parts = str(key).split("\t")
    if len(parts) != 3:
        raise ValueError(f"Invalid audio segment key: {key!r}")
    media_id, start_time, end_time = parts
    return media_id, float(start_time), float(end_time)


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
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


def _normalize_audio_segment_times(
    start_time: Any,
    end_time: Any,
    *,
    duration_hint_secs: Any = None,
) -> tuple[float, float] | None:
    try:
        start_val = float(start_time)
        end_val = float(end_time)
    except (TypeError, ValueError):
        return None

    duration_secs: float | None
    try:
        duration_secs = float(duration_hint_secs) if duration_hint_secs is not None else None
    except (TypeError, ValueError):
        duration_secs = None

    # Audio stage metadata currently stores segment times in milliseconds.
    # Normalize those to seconds when they obviously exceed the chunk duration.
    if duration_secs is not None and duration_secs > 0 and end_val > (duration_secs + 1.0):
        return start_val / 1000.0, end_val / 1000.0

    return start_val, end_val


def _normalize_audio_query_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "query" not in df.columns and "question" in df.columns:
        df = df.rename(columns={"question": "query"})
    if "expected_media_id" not in df.columns and "name" in df.columns:
        df["expected_media_id"] = df["name"].astype(str).apply(_normalize_audio_media_id)
    if "expected_start_time" not in df.columns and "start_time" in df.columns:
        df["expected_start_time"] = pd.to_numeric(df["start_time"], errors="raise").astype(float)
    if "expected_end_time" not in df.columns and "end_time" in df.columns:
        df["expected_end_time"] = pd.to_numeric(df["end_time"], errors="raise").astype(float)

    required = {"query", "expected_media_id", "expected_start_time", "expected_end_time"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            "For audio_segment mode, query data must contain "
            "['query','expected_media_id','expected_start_time','expected_end_time'] "
            "or ['question','name','start_time','end_time'] columns "
            f"(missing: {sorted(missing)})"
        )

    df["query"] = df["query"].astype(str)
    df["expected_media_id"] = df["expected_media_id"].astype(str).apply(_normalize_audio_media_id)
    df["expected_start_time"] = pd.to_numeric(df["expected_start_time"], errors="raise").astype(float)
    df["expected_end_time"] = pd.to_numeric(df["expected_end_time"], errors="raise").astype(float)
    df["golden_answer"] = df.apply(
        lambda row: _encode_audio_segment_key(
            row["expected_media_id"], row["expected_start_time"], row["expected_end_time"]
        ),
        axis=1,
    )
    return df


def _normalize_query_df(df: pd.DataFrame, *, match_mode: str) -> pd.DataFrame:
    """
    Normalize a query CSV into:
      - query (string)
      - golden_answer (string key that should match LanceDB `pdf_page`)

    Supported inputs by match mode:
      - pdf_page:
        - query,pdf_page
        - query,pdf,page (or query,pdf,gt_page)
      - pdf_only:
        - query,expected_pdf
        - query,pdf
      - audio_segment:
        - query,expected_media_id,expected_start_time,expected_end_time
        - question,name,start_time,end_time
    """
    if match_mode not in {"pdf_page", "pdf_only", "audio_segment"}:
        raise ValueError(f"Unsupported recall match mode: {match_mode}")

    if match_mode == "audio_segment":
        return _normalize_audio_query_df(df)

    df = df.copy()

    if "query" not in df.columns:
        raise KeyError("Query CSV must contain a 'query' column.")

    if match_mode == "pdf_only":
        if "expected_pdf" in df.columns:
            df["golden_answer"] = df["expected_pdf"].astype(str).apply(_normalize_pdf_name)
            return df
        if "pdf" in df.columns:
            df["golden_answer"] = df["pdf"].astype(str).apply(_normalize_pdf_name)
            return df
        raise KeyError(
            "For pdf_only mode, query data must contain ['query','expected_pdf'] or ['query','pdf'] columns."
        )

    if "gt_page" in df.columns and "page" not in df.columns:
        df = df.rename(columns={"gt_page": "page"})

    if "pdf_page" in df.columns:
        df["golden_answer"] = df["pdf_page"].astype(str)
        return df

    required = {"pdf", "page"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            "Query CSV must contain either columns ['query','pdf_page'] or ['query','pdf','page'] "
            f"(missing: {sorted(missing)})"
        )

    df["pdf"] = df["pdf"].astype(str).str.replace(".pdf", "", regex=False)
    df["page"] = df["page"].astype(str)
    df["golden_answer"] = df.apply(lambda x: f"{x.pdf}_{x.page}", axis=1)
    return df


def _resolve_embedding_endpoint(cfg: RecallConfig) -> Tuple[Optional[str], Optional[bool]]:
    """
    Resolve which embedding endpoint to use.

    Returns (endpoint, use_grpc) where:
      - endpoint is either an http(s) URL or a host:port string for gRPC
      - use_grpc is True for gRPC, False for HTTP, None when no endpoint is configured
    """
    http_ep = (cfg.embedding_http_endpoint or "").strip() if isinstance(cfg.embedding_http_endpoint, str) else None
    grpc_ep = (cfg.embedding_grpc_endpoint or "").strip() if isinstance(cfg.embedding_grpc_endpoint, str) else None
    single = (cfg.embedding_endpoint or "").strip() if isinstance(cfg.embedding_endpoint, str) else None

    if http_ep:
        return http_ep, False
    if grpc_ep:
        return grpc_ep, True
    if single:
        # Infer protocol: if a URL scheme is present, treat as HTTP; otherwise gRPC.
        return single, (not single.lower().startswith("http"))

    return None, None


def _embed_queries_nim(
    queries: List[str],
    *,
    endpoint: str,
    model: str,
    api_key: str,
    grpc: bool,
) -> List[List[float]]:
    from nv_ingest_api.util.nim import infer_microservice

    # `infer_microservice` returns a list of embeddings.
    embeddings = infer_microservice(
        queries,
        model_name=model,
        embedding_endpoint=endpoint,
        nvidia_api_key=(api_key or "").strip(),
        grpc=bool(grpc),
        input_type="query",
    )
    # Some backends return numpy arrays; normalize to list-of-list floats.
    out: List[List[float]] = []
    for e in embeddings:
        if isinstance(e, np.ndarray):
            out.append(e.astype("float32").tolist())
        else:
            out.append(list(e))
    return out


def _embed_queries_local_hf(
    queries: List[str],
    *,
    device: Optional[str],
    cache_dir: Optional[str],
    batch_size: int,
    model_name: Optional[str] = None,
) -> List[List[float]]:
    from nemo_retriever.model import create_local_embedder, is_vl_embed_model

    embedder = create_local_embedder(model_name, device=device, hf_cache_dir=cache_dir)

    if is_vl_embed_model(model_name):
        vecs = embedder.embed_queries(queries, batch_size=int(batch_size))
    else:
        vecs = embedder.embed(queries, batch_size=int(batch_size), prefix="query: ")
    return vecs.detach().to("cpu").tolist()


def _hits_to_keys(raw_hits: List[List[Dict[str, Any]]]) -> List[List[str]]:
    retrieved_keys: List[List[str]] = []
    for hits in raw_hits:
        keys: List[str] = []
        for h in hits:
            page_number = h["page_number"]
            source = h["source"]
            # Prefer explicit `pdf_page` column; fall back to derived form.
            if page_number is not None and source:
                filename = Path(source).stem
                keys.append(f"{filename}_{str(page_number)}")
            else:
                logger.warning(
                    "Skipping hit with missing page_number or source_id: metadata=%s source=%s",
                    h.get("metadata", ""),
                    h.get("source", ""),
                )
        retrieved_keys.append([k for k in keys if k])
    return retrieved_keys


def _hit_to_audio_segment_key(hit: Dict[str, Any]) -> str | None:
    metadata = _parse_mapping(hit.get("metadata"))
    source = _parse_mapping(hit.get("source"))

    source_id = source.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        source_id = hit.get("source_id") if isinstance(hit.get("source_id"), str) else hit.get("source")
    if not isinstance(source_id, str) or not source_id.strip():
        return None

    media_id = _normalize_audio_media_id(source_id)
    if not media_id:
        return None

    start_time = metadata.get("segment_start")
    end_time = metadata.get("segment_end")
    if start_time is not None and end_time is not None:
        normalized = _normalize_audio_segment_times(
            start_time,
            end_time,
            duration_hint_secs=metadata.get("duration"),
        )
        if normalized is None:
            return None
        start_secs, end_secs = normalized
        return _encode_audio_segment_key(media_id, start_secs, end_secs)

    content_metadata = metadata.get("content_metadata")
    if isinstance(content_metadata, dict):
        start_time = content_metadata.get("start_time")
        end_time = content_metadata.get("end_time")
        if start_time is not None and end_time is not None:
            normalized = _normalize_audio_segment_times(
                start_time,
                end_time,
                duration_hint_secs=metadata.get("duration"),
            )
            if normalized is None:
                return None
            start_secs, end_secs = normalized
            return _encode_audio_segment_key(media_id, start_secs, end_secs)

    return None


def _hits_to_audio_segment_keys(raw_hits: List[List[Dict[str, Any]]]) -> List[List[str]]:
    retrieved_keys: List[List[str]] = []
    for hits in raw_hits:
        keys: List[str] = []
        for hit in hits:
            encoded = _hit_to_audio_segment_key(hit)
            if encoded is not None:
                keys.append(encoded)
        retrieved_keys.append(keys)
    return retrieved_keys


def _extract_doc_from_pdf_page(key: str) -> str:
    parts = str(key).rsplit("_", 1)
    if len(parts) != 2:
        return str(key)
    return parts[0]


def _is_hit(
    golden_key: str,
    retrieved: List[str],
    k: int,
    *,
    match_mode: str,
    audio_match_tolerance_secs: float = AUDIO_MATCH_TOLERANCE_SECS,
) -> bool:
    """Check if a golden key is found in the top-k retrieved keys.

    Handles filenames with underscores via ``rsplit`` and also accepts
    whole-document keys (page ``-1``).
    """
    if match_mode == "audio_segment":
        gold_media, gold_start, gold_end = _parse_audio_segment_key(golden_key)
        for encoded_hit in retrieved[:k]:
            try:
                hit_media, hit_start, hit_end = _parse_audio_segment_key(encoded_hit)
            except ValueError:
                continue
            hit_midpoint = (hit_start + hit_end) / 2.0
            if (
                hit_media == gold_media
                and hit_midpoint > (gold_start - float(audio_match_tolerance_secs))
                and hit_midpoint < (gold_end + float(audio_match_tolerance_secs))
            ):
                return True
        return False

    if match_mode == "pdf_only":
        gold_doc = _normalize_pdf_name(str(golden_key))
        top_docs = [_extract_doc_from_pdf_page(r) for r in retrieved[:k]]
        return gold_doc in top_docs

    parts = golden_key.rsplit("_", 1)
    if len(parts) != 2:
        return golden_key in retrieved[:k]
    filename, page = parts
    specific_page = f"{filename}_{page}"
    entire_document = f"{filename}_-1"
    top = retrieved[:k]
    return specific_page in top or entire_document in top


def is_hit_at_k(
    golden_key: str,
    retrieved: Sequence[str],
    k: int,
    *,
    match_mode: str,
    audio_match_tolerance_secs: float = AUDIO_MATCH_TOLERANCE_SECS,
) -> bool:
    """Public wrapper for top-k hit checks across match modes."""
    return _is_hit(
        str(golden_key),
        list(retrieved),
        int(k),
        match_mode=str(match_mode),
        audio_match_tolerance_secs=float(audio_match_tolerance_secs),
    )


def gold_to_doc_page(golden_key: str) -> tuple[str, str]:
    """Split a golden key like ``"docname_page"`` into ``(doc, page)``."""
    s = str(golden_key)
    if "_" not in s:
        return s, ""
    doc, page = s.rsplit("_", 1)
    return doc, page


def hit_key_and_distance(hit: dict) -> tuple[str | None, float | None]:
    """Extract ``(pdf_page key, distance)`` from a single LanceDB hit dict.

    Supports both ``_distance`` and ``_score`` fields for compatibility across
    LanceDB query types (vector vs hybrid).
    """
    try:
        res = json.loads(hit.get("metadata", "{}"))
        source = json.loads(hit.get("source", "{}"))
    except Exception:
        return None, None

    source_id = source.get("source_id")
    page_number = res.get("page_number")
    if not source_id or page_number is None:
        return None, float(hit.get("_distance")) if "_distance" in hit else None

    key = f"{Path(str(source_id)).stem}_{page_number}"
    dist = float(hit["_distance"]) if "_distance" in hit else float(hit["_score"]) if "_score" in hit else None
    return key, dist


def _recall_at_k(
    gold: List[str],
    retrieved: List[List[str]],
    k: int,
    *,
    match_mode: str,
    audio_match_tolerance_secs: float = AUDIO_MATCH_TOLERANCE_SECS,
) -> float:
    hits = sum(
        is_hit_at_k(g, r, k, match_mode=match_mode, audio_match_tolerance_secs=audio_match_tolerance_secs)
        for g, r in zip(gold, retrieved)
    )
    return hits / max(1, len(gold))


def retrieve_and_score(
    query_csv: Path,
    *,
    cfg: RecallConfig,
    limit: Optional[int] = None,
    vector_column_name: str = "vector",
) -> Tuple[pd.DataFrame, List[str], List[List[Dict[str, Any]]], List[List[str]], Dict[str, float]]:
    """
    Run embeddings + LanceDB retrieval for a query CSV.

    Returns:
      - normalized query DataFrame
      - gold keys
      - raw LanceDB hits
      - retrieved keys (pdf_page-like or audio-segment-like)
      - metrics dict (recall@k)
    """
    df_query = _normalize_query_df(pd.read_csv(query_csv), match_mode=str(cfg.match_mode))
    if limit is not None:
        df_query = df_query.head(int(limit)).copy()

    queries = df_query["query"].astype(str).tolist()
    gold = df_query["golden_answer"].astype(str).tolist()
    retriever = Retriever(
        lancedb_uri=cfg.lancedb_uri,
        lancedb_table=cfg.lancedb_table,
        embedder=cfg.embedding_model or "nvidia/llama-nemotron-embed-1b-v2",
        embedding_http_endpoint=cfg.embedding_http_endpoint,
        embedding_api_key=cfg.embedding_api_key,
        top_k=cfg.top_k,
        nprobes=cfg.nprobes,
        refine_factor=cfg.refine_factor,
        hybrid=bool(cfg.hybrid),
        local_hf_device=cfg.local_hf_device,
        local_hf_cache_dir=cfg.local_hf_cache_dir,
        local_hf_batch_size=cfg.local_hf_batch_size,
        reranker=cfg.reranker,
        reranker_endpoint=cfg.reranker_endpoint,
        reranker_api_key=cfg.reranker_api_key,
        reranker_batch_size=cfg.reranker_batch_size,
    )
    start = time.time()
    raw_hits = retriever.queries(queries)
    end_queries = time.time() - start
    print(
        f"Retrieval time for {len(queries)} ",
        f"queries: {end_queries:.2f} seconds ",
        f"(average {len(queries)/end_queries:.2f} queries/second)",
    )

    if str(cfg.match_mode) == "audio_segment":
        retrieved_keys = _hits_to_audio_segment_keys(raw_hits)
    else:
        retrieved_keys = _hits_to_keys(raw_hits)
    metrics = {
        f"recall@{k}": _recall_at_k(
            gold,
            retrieved_keys,
            int(k),
            match_mode=str(cfg.match_mode),
            audio_match_tolerance_secs=float(cfg.audio_match_tolerance_secs),
        )
        for k in cfg.ks
    }
    return df_query, gold, raw_hits, retrieved_keys, metrics


def evaluate_recall(
    query_csv: Path,
    *,
    cfg: RecallConfig,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    df_query, gold, raw_hits, retrieved_keys, metrics = retrieve_and_score(
        query_csv,
        cfg=cfg,
        limit=None,
        vector_column_name="vector",
    )

    # Build per-query analysis DataFrame
    rows = []
    for i, (q, g, r) in enumerate(zip(df_query["query"].astype(str).tolist(), gold, retrieved_keys)):
        row = {"query_id": i, "query": q, "golden_answer": g, "top_retrieved": r[: cfg.top_k]}
        for k in cfg.ks:
            k = int(k)
            row[f"hit@{k}"] = is_hit_at_k(
                g,
                r,
                k,
                match_mode=str(cfg.match_mode),
                audio_match_tolerance_secs=float(cfg.audio_match_tolerance_secs),
            )
            if str(cfg.match_mode) == "audio_segment":
                rank = None
                for index, encoded_hit in enumerate(r[: cfg.top_k], start=1):
                    if is_hit_at_k(
                        g,
                        [encoded_hit],
                        1,
                        match_mode="audio_segment",
                        audio_match_tolerance_secs=float(cfg.audio_match_tolerance_secs),
                    ):
                        rank = index
                        break
                row[f"rank@{k}"] = rank
            elif str(cfg.match_mode) == "pdf_only":
                top_docs = [_extract_doc_from_pdf_page(key) for key in r[: cfg.top_k]]
                try:
                    row[f"rank@{k}"] = top_docs.index(_normalize_pdf_name(str(g))) + 1
                except ValueError:
                    row[f"rank@{k}"] = None
            else:
                row[f"rank@{k}"] = (r[: cfg.top_k].index(g) + 1) if (g in r[: cfg.top_k]) else None
        rows.append(row)
    results_df = pd.DataFrame(rows)

    saved: Dict[str, str] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = output_dir / f"recall_results_{ts}.csv"
        results_df.to_csv(out, index=False)
        saved["results_csv"] = str(out)

    return {
        "n_queries": int(len(df_query)),
        "top_k": int(cfg.top_k),
        "metrics": metrics,
        "saved": saved,
    }
