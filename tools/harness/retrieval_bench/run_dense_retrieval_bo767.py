#!/usr/bin/env python3
"""
Run retrieval-bench dense retrieval on the bo767 corpus and output
a FileRetriever JSON compatible with the QA evaluation harness.

Architecture
------------
1.  Load **sub-page chunks** from extraction Parquet files produced by
    ``graph_pipeline --save-intermediate``.
2.  Index the chunks with a retrieval-bench embedding backend
    (e.g. ``llama-nv-embed-reasoning-3b``).
3.  For each ground-truth query, retrieve the top-K scoring chunks.
4.  **Deduplicate** chunk hits by ``(source_id, page_number)`` and
    **expand** each hit to full-page markdown using the page markdown
    index -- the same page-expansion logic used by the LanceDB export
    pipeline in ``nemo_retriever.export.expand_hits_to_pages``.

This gives an apples-to-apples comparison against the LanceDB pipeline:
both systems retrieve at sub-page chunk granularity and both output
full-page markdown for the QA generator.  The only independent variable
is the embedding model.

Usage::

    cd tools/harness
    python retrieval_bench/run_dense_retrieval_bo767.py \\
        --backend llama-nv-embed-reasoning-3b \\
        --top-k 5 \\
        --output data/test_retrieval/bo767_retrieval_dense.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_PARQUET_DIR = os.path.join(_HERE, "..", "data", "bo767_extracted")
_DEFAULT_PAGE_INDEX = os.path.join(_HERE, "..", "data", "bo767_page_markdown.json")
_DEFAULT_CSV = os.path.join(_HERE, "..", "..", "..", "data", "bo767_annotations.csv")


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _load_parquet_corpus(
    parquet_dir: str,
) -> tuple[list[str], list[str], dict[str, tuple[str, int]]]:
    """Load sub-page chunks from extraction Parquet files.

    Each non-empty ``text`` record becomes one corpus entry.  Corpus IDs
    encode ``{basename}:{page}:{chunk_idx}`` so provenance is recoverable
    from the ID alone.

    Only the columns needed for retrieval (``source_id``/``path``,
    ``page_number``, ``text``) are loaded -- this avoids the
    ``ArrowNotImplementedError`` that occurs when PyArrow tries to convert
    deeply nested struct/list columns in the full extraction Parquet.

    Returns
    -------
    corpus_ids : list[str]
        One deterministic string ID per chunk.
    corpus_texts : list[str]
        The text content of each chunk (what the embedding model sees).
    chunk_provenance : dict[str, tuple[str, int]]
        Maps each corpus_id to ``(source_id, page_number)`` for
        post-retrieval page expansion.
    """
    import pyarrow.parquet as pq

    parquet_path = Path(parquet_dir)
    if not parquet_path.is_dir():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_path}")

    single = parquet_path / "extraction.parquet"
    if single.is_file():
        parquet_files = [single]
    else:
        parquet_files = sorted(f for f in parquet_path.rglob("*.parquet") if f.name != "extraction.parquet")
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files in {parquet_path}")

    _NEEDED_COLUMNS = {"path", "source_id", "page_number", "text"}

    import pandas as pd

    dfs: list[pd.DataFrame] = []
    for pf_path in parquet_files:
        pf = pq.ParquetFile(pf_path)
        available = set(pf.schema_arrow.names)
        columns = sorted(_NEEDED_COLUMNS & available)
        table = pf.read(columns=columns)
        try:
            dfs.append(table.to_pandas(split_blocks=False))
        except Exception:
            dfs.append(pd.DataFrame(table.to_pydict()))
    df = pd.concat(dfs, ignore_index=True)

    path_col = "path" if "path" in df.columns else "source_id"

    corpus_ids: list[str] = []
    corpus_texts: list[str] = []
    chunk_provenance: dict[str, tuple[str, int]] = {}

    page_counters: dict[tuple[str, int], int] = {}

    for _, row in df.iterrows():
        source_id = str(row.get(path_col, ""))

        page_number = -1
        raw_pn = row.get("page_number")
        if raw_pn is not None:
            try:
                page_number = int(raw_pn)
            except (TypeError, ValueError):
                pass

        text = str(row.get("text", "") or "").strip()
        if not text:
            continue

        doc_basename = os.path.basename(source_id)
        key = (source_id, page_number)
        idx = page_counters.get(key, 0)
        page_counters[key] = idx + 1

        cid = f"{doc_basename}:{page_number}:{idx}"
        corpus_ids.append(cid)
        corpus_texts.append(text)
        chunk_provenance[cid] = (source_id, page_number)

    return corpus_ids, corpus_texts, chunk_provenance


def _load_page_index(path: str) -> tuple[dict[str, dict[str, str]], int, int]:
    """Load the page markdown index and return (index, doc_count, page_count)."""
    with open(path, encoding="utf-8") as f:
        page_index: dict[str, dict[str, str]] = json.load(f)
    total_pages = sum(len(pages) for pages in page_index.values())
    return page_index, len(page_index), total_pages


def _load_queries(csv_path: str) -> list[str]:
    """Load query strings from the annotations CSV."""
    queries: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = str(row.get("query", "")).strip()
            if q:
                queries.append(q)
    return queries


# ---------------------------------------------------------------------------
# Post-retrieval page expansion
# ---------------------------------------------------------------------------


def _expand_scored_chunks_to_pages(
    scored_chunks: list[tuple[str, float]],
    chunk_provenance: dict[str, tuple[str, int]],
    page_index: dict[str, dict[str, str]],
    top_k: int,
) -> tuple[list[str], list[dict[str, Any]], int]:
    """Deduplicate chunk hits by page and expand to full-page markdown.

    Follows the same algorithm as ``nemo_retriever.export.expand_hits_to_pages``
    but operates directly on ``(corpus_id, score)`` pairs from the retriever,
    preserving the original score (higher = better) in the output metadata.

    Parameters
    ----------
    scored_chunks : list[tuple[str, float]]
        ``(corpus_id, score)`` pairs **pre-sorted by score descending**.
    chunk_provenance : dict
        Maps corpus_id -> ``(source_id, page_number)``.
    page_index : dict
        ``{source_id: {page_str: markdown}}``.
    top_k : int
        Maximum number of deduplicated pages to return.

    Returns
    -------
    chunks : list[str]
        Full-page markdown strings.
    metadata : list[dict]
        Per-chunk provenance with ``source_id``, ``page_number``, ``score``.
    miss_count : int
        Pages not found in the page index.
    """
    seen: dict[tuple[str, int], float] = {}
    ordered_pages: list[tuple[str, int]] = []

    for doc_id, score in scored_chunks:
        provenance = chunk_provenance.get(doc_id)
        if provenance is None:
            continue
        source_id, page_number = provenance
        key = (source_id, page_number)
        if key not in seen:
            seen[key] = float(score)
            ordered_pages.append(key)

    chunks: list[str] = []
    metadata: list[dict[str, Any]] = []
    miss_count = 0

    for source_id, page_number in ordered_pages:
        if len(chunks) >= top_k:
            break
        doc_pages = page_index.get(source_id, {})
        md = doc_pages.get(str(page_number))
        if md is None:
            miss_count += 1
            continue
        chunks.append(md)
        metadata.append(
            {
                "source_id": source_id,
                "page_number": page_number,
                "score": seen[(source_id, page_number)],
            }
        )

    return chunks, metadata, miss_count


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_dense_retrieval(
    output: str,
    backend: str = "llama-nv-embed-reasoning-3b",
    top_k: int = 5,
    retriever_top_k: int = 100,
    parquet_dir: str | None = None,
    markdown_index: str | None = None,
    csv_path: str | None = None,
) -> dict:
    """Run dense retrieval on the bo767 corpus and write FileRetriever JSON.

    Retrieval operates on **sub-page chunks** from the extraction Parquet.
    After retrieval, chunk hits are deduplicated by ``(source_id, page)``
    and expanded to full-page markdown via the page index -- the same
    post-retrieval expansion the LanceDB export pipeline applies.

    Parameters
    ----------
    output : str
        Path to write the FileRetriever JSON.
    backend : str
        retrieval-bench backend name.
    top_k : int
        Number of full-page chunks per query in the output JSON.
    retriever_top_k : int
        Sub-page chunk candidates the retriever returns before page-level
        deduplication.  Should be >> ``top_k`` since multiple chunks from
        the same page collapse into a single entry.
    parquet_dir : str, optional
        Directory containing extraction Parquet from ``--save-intermediate``.
    markdown_index : str, optional
        Path to ``bo767_page_markdown.json`` for post-retrieval expansion.
    csv_path : str, optional
        Annotations CSV with a ``query`` column.

    Returns
    -------
    dict
        Run statistics.
    """
    parquet_dir = parquet_dir or _DEFAULT_PARQUET_DIR
    markdown_index = markdown_index or _DEFAULT_PAGE_INDEX
    csv_path = csv_path or _DEFAULT_CSV

    # -- 1. Corpus --------------------------------------------------------
    print("Loading sub-page chunks from Parquet ...")
    corpus_ids, corpus_texts, chunk_provenance = _load_parquet_corpus(parquet_dir)
    unique_sources = {sid for sid, _ in chunk_provenance.values()}
    unique_pages = len({v for v in chunk_provenance.values()})
    print(f"  {len(corpus_ids):,} chunks  |  " f"{unique_pages:,} pages  |  " f"{len(unique_sources):,} documents")

    # -- 2. Page index ----------------------------------------------------
    print(f"Loading page markdown index: {os.path.basename(markdown_index)}")
    page_index, idx_docs, idx_pages = _load_page_index(markdown_index)
    print(f"  {idx_docs:,} documents  |  {idx_pages:,} pages in index")

    # -- 3. Queries -------------------------------------------------------
    queries = _load_queries(os.path.normpath(csv_path))
    query_ids = [str(i) for i in range(len(queries))]
    print(f"  {len(queries):,} queries")

    # -- 4. Index corpus with retrieval-bench -----------------------------
    from retrieval_bench.pipelines.dense import DenseRetrievalPipeline

    pipeline = DenseRetrievalPipeline(backend=backend, top_k=retriever_top_k)
    corpus_images = [None] * len(corpus_ids)

    t0 = time.time()
    pipeline.index(
        corpus_ids=corpus_ids,
        corpus_images=corpus_images,
        corpus_texts=corpus_texts,
        dataset_name="bo767",
    )
    index_time = time.time() - t0

    # -- 5. Retrieve ------------------------------------------------------
    t0 = time.time()
    result = pipeline.search(query_ids=query_ids, queries=queries)
    run_dict = result[0] if isinstance(result, tuple) else result
    retrieval_time = time.time() - t0

    # -- 6. Expand chunk hits to full-page markdown -----------------------
    output_queries: dict[str, dict] = {}
    total_page_misses = 0

    for qid, query_text in zip(query_ids, queries):
        scores = run_dict.get(qid, {})
        ranked = sorted(scores.items(), key=lambda x: float(x[1]), reverse=True)

        chunks, meta, misses = _expand_scored_chunks_to_pages(ranked, chunk_provenance, page_index, top_k)
        total_page_misses += misses
        output_queries[query_text] = {"chunks": chunks, "metadata": meta}

    # -- 7. Write FileRetriever JSON --------------------------------------
    file_output: dict[str, Any] = {
        "metadata": {
            "retrieval_backend": backend,
            "corpus_source": "extraction_parquet",
            "chunk_mode": "sub-page chunks -> full-page expansion",
            "corpus_chunks": len(corpus_ids),
            "corpus_pages": unique_pages,
            "corpus_documents": len(unique_sources),
            "top_k": top_k,
            "retriever_top_k": retriever_top_k,
            "query_count": len(output_queries),
            "page_index_misses": total_page_misses,
        },
        "queries": output_queries,
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(file_output, f, indent=2)

    return {
        "queries_written": len(output_queries),
        "corpus_chunks": len(corpus_ids),
        "unique_pages": unique_pages,
        "documents": len(unique_sources),
        "backend": backend,
        "top_k": top_k,
        "retriever_top_k": retriever_top_k,
        "index_time_s": round(index_time, 1),
        "retrieval_time_s": round(retrieval_time, 1),
        "page_index_misses": total_page_misses,
        "output_path": output,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Dense retrieval on the bo767 corpus using sub-page chunks "
            "from extraction Parquet, with full-page expansion."
        ),
    )
    parser.add_argument(
        "--backend",
        default="llama-nv-embed-reasoning-3b",
        help="retrieval-bench backend (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Full-page chunks per query in output (default: %(default)s)",
    )
    parser.add_argument(
        "--retriever-top-k",
        type=int,
        default=100,
        help="Sub-page candidates before page dedup (default: %(default)s)",
    )
    parser.add_argument(
        "--parquet-dir",
        default=_DEFAULT_PARQUET_DIR,
        help="Extraction Parquet directory (default: data/bo767_extracted)",
    )
    parser.add_argument(
        "--markdown-index",
        default=_DEFAULT_PAGE_INDEX,
        help="Page markdown index for post-retrieval expansion",
    )
    parser.add_argument(
        "--csv",
        default=_DEFAULT_CSV,
        help="Annotations CSV with query column",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output FileRetriever JSON path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Dense Retrieval on bo767 (sub-page chunks)")
    print("=" * 60)
    print(f"Backend:           {args.backend}")
    print(f"Parquet dir:       {args.parquet_dir}")
    print(f"Page index:        {args.markdown_index}")
    print(f"CSV:               {args.csv}")
    print(f"Top-K (output):    {args.top_k}")
    print(f"Top-K (retriever): {args.retriever_top_k}")
    print("=" * 60)

    stats = run_dense_retrieval(
        output=args.output,
        backend=args.backend,
        top_k=args.top_k,
        retriever_top_k=args.retriever_top_k,
        parquet_dir=args.parquet_dir,
        markdown_index=args.markdown_index,
        csv_path=args.csv,
    )

    print(f"\nWrote {stats['queries_written']:,} queries to {stats['output_path']}")
    print(
        f"Corpus: {stats['corpus_chunks']:,} chunks, "
        f"{stats['unique_pages']:,} pages, "
        f"{stats['documents']:,} documents"
    )
    print(f"Index: {stats['index_time_s']}s  |  Retrieval: {stats['retrieval_time_s']}s")
    if stats["page_index_misses"] > 0:
        print(f"Page index misses: {stats['page_index_misses']}")
    print("Done.")


if __name__ == "__main__":
    main()
