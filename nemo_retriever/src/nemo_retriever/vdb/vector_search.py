# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vector-search adapters for Retriever-owned query embeddings."""

from __future__ import annotations

import json
from typing import Any, Optional


def _maybe_parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)) or value is None:
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _lancedb_result_fields(table: Any, requested: Any = None) -> list[str]:
    if requested:
        return list(requested)

    fields = ["text", "metadata", "source"]
    try:
        columns = {field.name for field in table.schema}
    except Exception:
        return fields

    for name in (
        "page_number",
        "pdf_page",
        "pdf_basename",
        "source_id",
        "path",
        "stored_image_uri",
        "content_type",
        "bbox_xyxy_norm",
    ):
        if name in columns:
            fields.append(name)
    return fields


def _search_lancedb_vectors(
    *,
    query_vectors: list[list[float]],
    query_texts: list[str],
    top_k: int,
    vdb_kwargs: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    import numpy as np

    table = vdb_kwargs.get("table")
    if table is None:
        import lancedb  # type: ignore

        table_path = str(vdb_kwargs.get("table_path") or vdb_kwargs.get("uri") or "lancedb")
        table_name = str(vdb_kwargs.get("table_name") or "nv-ingest")
        table = lancedb.connect(uri=table_path).open_table(table_name)

    refine_factor = int(vdb_kwargs.get("refine_factor", 50))
    n_probe = int(vdb_kwargs.get("n_probe", vdb_kwargs.get("nprobes", 64)))
    vector_column_name = str(vdb_kwargs.get("vector_column_name") or "vector")
    hybrid = bool(vdb_kwargs.get("hybrid", False))
    result_fields = _lancedb_result_fields(table, vdb_kwargs.get("result_fields"))

    results: list[list[dict[str, Any]]] = []
    for query_text, query_vector in zip(query_texts, query_vectors):
        vector = np.asarray(query_vector, dtype="float32")
        if hybrid:
            from lancedb.rerankers import RRFReranker  # type: ignore

            hits = (
                table.search(query_type="hybrid")
                .vector(vector)
                .text(query_text)
                .select(result_fields)
                .limit(int(top_k))
                .refine_factor(refine_factor)
                .nprobes(n_probe)
                .rerank(RRFReranker())
                .to_list()
            )
        else:
            hits = (
                table.search(vector, vector_column_name=vector_column_name)
                .select(result_fields)
                .limit(int(top_k))
                .refine_factor(refine_factor)
                .nprobes(n_probe)
                .to_list()
            )

        formatted: list[dict[str, Any]] = []
        for hit in hits:
            source = _maybe_parse_json(hit.get("source"))
            content_metadata = _maybe_parse_json(hit.get("metadata"))
            if not isinstance(source, dict):
                source = {"source_id": "" if source is None else str(source)}
            if not isinstance(content_metadata, dict):
                content_metadata = {}

            formatted_hit = {
                "entity": {
                    "source": source,
                    "content_metadata": content_metadata,
                    "text": hit.get("text"),
                }
            }
            for key in (
                "stored_image_uri",
                "content_type",
                "bbox_xyxy_norm",
                "_distance",
                "_score",
            ):
                if key in hit:
                    formatted_hit[key] = hit[key]
            formatted.append(formatted_hit)
        results.append(formatted)
    return results


def _search_milvus_vectors(
    *,
    query_vectors: list[list[float]],
    query_texts: list[str],
    top_k: int,
    vdb_kwargs: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    from pymilvus import MilvusClient

    from nv_ingest_client.util.vdb.milvus import dense_retrieval, hybrid_retrieval

    collection_name = vdb_kwargs.get("collection_name")
    if not collection_name:
        raise ValueError("Milvus retrieval requires vdb_kwargs['collection_name']")

    milvus_uri = str(vdb_kwargs.get("milvus_uri") or vdb_kwargs.get("uri") or "http://localhost:19530")
    username = vdb_kwargs.get("username")
    password = vdb_kwargs.get("password")
    token = f"{username}:{password}" if username or password else None
    client = vdb_kwargs.get("client")
    if client is None:
        client_kwargs = {"uri": milvus_uri}
        if token:
            client_kwargs["token"] = token
        client = MilvusClient(**client_kwargs)

    dense_field = str(vdb_kwargs.get("dense_field") or "vector")
    sparse_field = str(vdb_kwargs.get("sparse_field") or "sparse")
    output_fields = list(vdb_kwargs.get("output_fields") or ["text", "source", "content_metadata"])
    gpu_search = bool(vdb_kwargs.get("gpu_search", False))
    hybrid = bool(vdb_kwargs.get("hybrid", False))
    local_index = milvus_uri.endswith(".db")
    filter_expr = str(vdb_kwargs.get("_filter") or "")
    ef_param = int(vdb_kwargs.get("ef_param", 200))

    vector_iter = iter(query_vectors)

    def dense_model(batch: list[str]) -> list[list[float]]:
        return [next(vector_iter) for _ in batch]

    common = dict(
        collection_name=str(collection_name),
        client=client,
        dense_model=dense_model,
        top_k=int(top_k),
        dense_field=dense_field,
        output_fields=output_fields,
        gpu_search=gpu_search,
        local_index=local_index,
        _filter=filter_expr,
        ef_param=ef_param,
    )

    if not hybrid:
        return dense_retrieval(query_texts, **common)

    sparse_model = None
    if local_index:
        from pymilvus.model.sparse import BM25EmbeddingFunction
        from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer

        sparse_model = BM25EmbeddingFunction(build_default_analyzer(language="en"))
        sparse_model.load(str(vdb_kwargs.get("sparse_model_filepath") or "bm25_model.json"))
    return hybrid_retrieval(
        query_texts,
        sparse_model=sparse_model,
        sparse_field=sparse_field,
        **common,
    )


def supports_vector_search_vdb(vdb_op: Optional[str]) -> bool:
    normalized = str(vdb_op or "").strip().lower().replace("_", "").replace("-", "")
    return normalized in {"lancedb", "lance", "milvus"}


def search_vdb_with_vectors(
    *,
    vdb_op: Optional[str],
    query_vectors: list[list[float]],
    query_texts: list[str],
    top_k: int,
    vdb_kwargs: dict[str, Any],
) -> list[list[dict[str, Any]]] | None:
    """Search a known VDB with already-computed query vectors.

    Returns ``None`` for unknown ops so callers can fall back to a custom VDB
    implementation if needed.
    """
    normalized = str(vdb_op or "").strip().lower().replace("_", "").replace("-", "")
    if normalized in {"lancedb", "lance"}:
        return _search_lancedb_vectors(
            query_vectors=query_vectors,
            query_texts=query_texts,
            top_k=top_k,
            vdb_kwargs=vdb_kwargs,
        )
    if normalized == "milvus":
        return _search_milvus_vectors(
            query_vectors=query_vectors,
            query_texts=query_texts,
            top_k=top_k,
            vdb_kwargs=vdb_kwargs,
        )
    return None
