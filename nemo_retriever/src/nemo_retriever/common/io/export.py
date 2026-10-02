# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export retrieval results from a NeMo Retriever index (LanceDB or Qdrant) to FileRetriever JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemo_retriever.common.vdb.records import hit_rank
from nemo_retriever.common.vdb.targets import VdbTarget


def parse_json_field(raw: Any) -> dict:
    """Safely parse a field that may be a JSON string or already a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _extract_source_id(hit: dict) -> str:
    """Extract source_id from a retriever hit.

    LanceDB stores ``source_id`` and ``source`` as top-level plain-string
    columns (e.g. ``/path/to/file.pdf``).  Prefer those over JSON-parsing
    the ``source`` field, which is not a JSON object.
    """
    sid = hit.get("source_id") or hit.get("source") or ""
    if isinstance(sid, str):
        return sid
    parsed = parse_json_field(sid)
    return parsed.get("source_id", "") if parsed else str(sid)


def _extract_page_number(hit: dict) -> int:
    """Extract page_number from a retriever hit.

    ``page_number`` is a top-level int column in LanceDB.  Fall back to
    the ``metadata`` dict only when the column is absent.
    """
    pn = hit.get("page_number")
    if pn is None:
        meta = parse_json_field(hit.get("metadata", "{}"))
        pn = meta.get("page_number", -1)
    try:
        return int(pn)
    except (TypeError, ValueError):
        return -1


def _hit_metadata(hit: dict, source_id: str, page_number: int) -> dict:
    metadata = {"source_id": source_id, "page_number": page_number, "distance": hit.get("_distance")}
    score = next((hit[key] for key in ("_relevance_score", "_score") if hit.get(key) is not None), None)
    if score is not None:
        metadata["score"] = score
    return metadata


def expand_hits_to_pages(
    hits: list[dict],
    page_index: dict[str, dict[str, str]],
) -> tuple[list[str], list[dict], int]:
    """Deduplicate hits by (source_id, page_number) and look up full-page markdown.

    Returns ``(chunks, metadata, miss_count)`` where *miss_count* is the number
    of ``(source_id, page)`` pairs that had no entry in the page index.
    """
    best: dict[tuple[str, int], dict] = {}
    ordered_pages: list[tuple[str, int]] = []

    for hit in hits:
        source_id = _extract_source_id(hit)
        page_number = _extract_page_number(hit)

        key = (source_id, page_number)
        if key not in best:
            best[key] = hit
            ordered_pages.append(key)
        elif hit_rank(hit) < hit_rank(best[key]):
            best[key] = hit

    chunks: list[str] = []
    metadata: list[dict] = []
    miss_count = 0
    for source_id, page_number in ordered_pages:
        page_str = str(page_number)
        doc_pages = page_index.get(source_id, {})
        md = doc_pages.get(page_str)
        if md is None:
            miss_count += 1
            continue
        chunks.append(md)
        metadata.append(_hit_metadata(best[(source_id, page_number)], source_id, page_number))

    return chunks, metadata, miss_count


def query_lancedb(
    lancedb_uri: str,
    lancedb_table: str,
    queries: list[dict],
    *,
    top_k: int = 5,
    embedder: str = "nvidia/llama-nemotron-embed-1b-v2",
    page_index: dict[str, dict[str, str]] | None = None,
    batch_size: int = 50,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """Query a LanceDB table with :func:`query_vdb`."""
    return query_vdb(
        VdbTarget(lancedb_uri=lancedb_uri, table_name=lancedb_table),
        queries,
        top_k=top_k,
        embedder=embedder,
        page_index=page_index,
        batch_size=batch_size,
    )


def query_vdb(
    target: VdbTarget,
    queries: list[dict],
    *,
    top_k: int = 5,
    embedder: str = "nvidia/llama-nemotron-embed-1b-v2",
    page_index: dict[str, dict[str, str]] | None = None,
    batch_size: int = 50,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """Query a LanceDB table or Qdrant collection and return results without writing to disk.

    Parameters
    ----------
    target : VdbTarget
        The index to query.
    queries : list[dict]
        Each dict must have a ``"query"`` key.
    top_k : int
        Number of chunks to retrieve per query.
    embedder : str
        Embedding model name for the Retriever.
    page_index : dict, optional
        ``{source_id: {page_str: markdown}}``.  When provided, chunk hits are
        expanded to full-page markdown.
    batch_size : int
        Number of queries per retrieval batch.

    Returns
    -------
    tuple[dict[str, dict], dict[str, Any]]
        ``(all_results, metadata)`` where *all_results* maps query text to
        ``{"chunks": [...], "metadata": [...]}`` and *metadata* is the
        envelope metadata dict.
    """
    from nemo_retriever.graph.retriever import Retriever

    retriever = Retriever(
        vdb_kwargs={"vdb_op": target.vdb_op, "vdb_kwargs": target.vdb_kwargs()},
        embed_kwargs={"model_name": embedder, "embed_model_name": embedder},
        top_k=top_k,
        rerank=False,
    )

    use_fullpage = page_index is not None
    query_strings = [q["query"] for q in queries]
    all_results: dict[str, dict] = {}
    total_page_misses = 0

    for batch_start in range(0, len(query_strings), batch_size):
        batch = query_strings[batch_start : batch_start + batch_size]
        batch_hits = retriever.queries(batch)

        for query, hits in zip(batch, batch_hits):
            if use_fullpage:
                chunks, metadata, misses = expand_hits_to_pages(hits, page_index)
                total_page_misses += misses
            else:
                chunks: list[str] = []
                metadata: list[dict] = []
                for hit in hits:
                    chunks.append(hit.get("text", ""))
                    metadata.append(_hit_metadata(hit, _extract_source_id(hit), _extract_page_number(hit)))
            all_results[query] = {"chunks": chunks, "metadata": metadata}

    chunk_mode = "full-page markdown" if use_fullpage else "sub-page chunks"
    meta: dict[str, Any] = {
        "vdb_backend": target.vdb_op,
        "collection_name": target.table_name,
        "top_k": top_k,
        "embedding_model": embedder,
        "chunk_mode": chunk_mode,
        "query_count": len(all_results),
    }
    if use_fullpage:
        meta["page_index_misses"] = total_page_misses

    return all_results, meta


def write_retrieval_json(
    all_results: dict[str, dict],
    output_path: str | Path,
    metadata: dict[str, Any],
) -> dict:
    """Write retrieval results to a FileRetriever-compatible JSON file.

    Parameters
    ----------
    all_results : dict
        Query text -> ``{"chunks": [...], "metadata": [...]}``.
    output_path : str or Path
        Destination file path.
    metadata : dict
        Envelope metadata (vdb_backend, top_k, etc.).

    Returns
    -------
    dict
        The full JSON structure written to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output: dict[str, Any] = {
        "metadata": metadata,
        "queries": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


def export_retrieval_json(
    lancedb_uri: str,
    lancedb_table: str,
    queries: list[dict],
    output_path: str | Path,
    *,
    top_k: int = 5,
    embedder: str = "nvidia/llama-nemotron-embed-1b-v2",
    page_index: dict[str, dict[str, str]] | None = None,
    batch_size: int = 50,
    target: VdbTarget | None = None,
) -> dict:
    """Query an index, optionally expand to full-page markdown, write FileRetriever JSON.

    Convenience wrapper around :func:`query_vdb` + :func:`write_retrieval_json`.
    Pass *target* to query a Qdrant collection. *lancedb_uri* and
    *lancedb_table* are then ignored.

    Parameters
    ----------
    lancedb_uri : str
        Path to LanceDB directory.
    lancedb_table : str
        LanceDB table name.
    queries : list[dict]
        Each dict must have a ``"query"`` key.
    output_path : str or Path
        Where to write the output JSON.
    top_k : int
        Number of chunks to retrieve per query.
    embedder : str
        Embedding model name for the Retriever.
    page_index : dict, optional
        ``{source_id: {page_str: markdown}}``.  When provided, chunk hits are
        expanded to full-page markdown.
    batch_size : int
        Number of queries per retrieval batch.

    Returns
    -------
    dict
        The output JSON structure (also written to *output_path*).
    """
    all_results, meta = query_vdb(
        target or VdbTarget(lancedb_uri=lancedb_uri, table_name=lancedb_table),
        queries=queries,
        top_k=top_k,
        embedder=embedder,
        page_index=page_index,
        batch_size=batch_size,
    )
    return write_retrieval_json(all_results, output_path, meta)
