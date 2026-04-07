#!/usr/bin/env python3
"""
Run retrieval-bench **agentic** retrieval on the bo767 corpus and output
a FileRetriever JSON compatible with the QA evaluation harness.

Architecture
------------
1.  Load **sub-page chunks** from extraction Parquet files produced by
    ``graph_pipeline --save-intermediate``.
2.  Index the chunks with a retrieval-bench embedding backend
    (e.g. ``llama-nv-embed-reasoning-3b``).
3.  For each ground-truth query, an **LLM agent** iteratively searches
    the corpus: it can call ``retrieve()`` multiple times with rewritten
    sub-queries, use a ``think`` tool for reasoning, and a
    ``selection_agent`` for re-ranking -- then emits a final ranked list
    via ``final_results``.
4.  **Deduplicate** chunk hits by ``(source_id, page_number)`` and
    **expand** each hit to full-page markdown using the page markdown
    index -- identical post-retrieval expansion as the dense script
    and the LanceDB export pipeline.

Compared to the dense script (``run_dense_retrieval_bo767.py``), the
retrieval phase is replaced by the agentic loop.  Everything before
(corpus loading, indexing) and after (page expansion, JSON output) is
shared.

Requirements
~~~~~~~~~~~~
- An OpenAI-compatible LLM endpoint for the agent (e.g. NVIDIA NIM,
  vLLM, or OpenAI).  Set ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` or
  pass ``--api-key`` / ``--base-url``.
- ``retrieval-bench`` installed (``pip install -e path/to/retrieval-bench``).

Usage::

    cd tools/harness

    export NVIDIA_API_KEY="nvapi-..."

    python retrieval_bench/run_agentic_retrieval_bo767.py \\
        --backend llama-nv-embed-reasoning-3b \\
        --llm-model nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5 \\
        --api-key "$NVIDIA_API_KEY" \\
        --top-k 5 \\
        --output data/test_retrieval/bo767_retrieval_agentic.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from run_dense_retrieval_bo767 import (
    _DEFAULT_CSV,
    _DEFAULT_PAGE_INDEX,
    _DEFAULT_PARQUET_DIR,
    _expand_scored_chunks_to_pages,
    _load_page_index,
    _load_parquet_corpus,
    _load_queries,
)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_agentic_retrieval(
    output: str,
    backend: str = "llama-nv-embed-reasoning-3b",
    llm_model: str = "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5",
    api_key: str | None = None,
    base_url: str | None = None,
    top_k: int = 5,
    retriever_top_k: int = 500,
    num_concurrent: int = 1,
    max_steps: int = 200,
    target_top_k: int = 10,
    parquet_dir: str | None = None,
    markdown_index: str | None = None,
    csv_path: str | None = None,
) -> dict:
    """Run agentic retrieval on the bo767 corpus and write FileRetriever JSON.

    The corpus is indexed exactly as in the dense script.  The difference
    is the retrieval phase: an LLM agent iteratively refines results using
    multi-angle search, query rewriting, and a selection agent.

    After retrieval, chunk hits are deduplicated by ``(source_id, page)``
    and expanded to full-page markdown -- identical to the dense script.

    Parameters
    ----------
    output : str
        Path to write the FileRetriever JSON.
    backend : str
        retrieval-bench embedding backend name.
    llm_model : str
        LiteLLM model string for the agent LLM.
    api_key : str, optional
        API key for the LLM endpoint.  Defaults to ``OPENAI_API_KEY`` or
        ``NVIDIA_API_KEY`` from the environment.
    base_url : str, optional
        Base URL for the LLM endpoint.  Defaults to ``OPENAI_BASE_URL``
        from the environment.
    top_k : int
        Number of full-page chunks per query in the output JSON.
    retriever_top_k : int
        Sub-page chunk candidates the embedding retriever returns per
        agent retrieve() call.
    num_concurrent : int
        Number of agent queries to run concurrently.
    max_steps : int
        Maximum agent loop iterations per query.
    target_top_k : int
        Number of document IDs the agent must select via final_results.
    parquet_dir : str, optional
        Directory containing extraction Parquet.
    markdown_index : str, optional
        Path to ``bo767_page_markdown.json``.
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

    api_key = api_key or os.environ.get("NVIDIA_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")

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
    from retrieval_bench.pipelines.agentic import AgenticRetrievalPipeline

    pipeline = AgenticRetrievalPipeline(
        backend=backend,
        retriever_top_k=retriever_top_k,
        llm_model=llm_model,
        api_key=api_key or "os.environ/OPENAI_API_KEY",
        base_url=base_url,
        target_top_k=target_top_k,
        max_steps=max_steps,
        num_concurrent=num_concurrent,
    )
    corpus_images = [None] * len(corpus_ids)

    t0 = time.time()
    pipeline.index(
        corpus_ids=corpus_ids,
        corpus_images=corpus_images,
        corpus_texts=corpus_texts,
        dataset_name="bo767",
    )
    index_time = time.time() - t0

    # -- 5. Agentic retrieve ----------------------------------------------
    print(f"\nStarting agentic retrieval " f"(agent LLM: {llm_model}, concurrency: {num_concurrent}) ...")
    t0 = time.time()
    result = pipeline.search(query_ids=query_ids, queries=queries)
    run_dict = result[0] if isinstance(result, tuple) else result
    infos = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
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
    per_query_trace = infos.get("per_query_trace", {})
    agentic_stats = {}
    if per_query_trace:
        retrieval_calls = [t.get("num_retrieval_calls", 0) for t in per_query_trace.values()]
        agentic_stats = {
            "total_retrieval_calls": sum(retrieval_calls),
            "mean_retrieval_calls_per_query": (
                round(sum(retrieval_calls) / len(retrieval_calls), 2) if retrieval_calls else 0
            ),
            "queries_with_fallback": sum(1 for t in per_query_trace.values() if t.get("fallback_used")),
        }

    file_output: dict[str, Any] = {
        "metadata": {
            "retrieval_method": "agentic",
            "retrieval_backend": backend,
            "agent_llm": llm_model,
            "corpus_source": "extraction_parquet",
            "chunk_mode": "sub-page chunks -> agentic retrieval -> full-page expansion",
            "corpus_chunks": len(corpus_ids),
            "corpus_pages": unique_pages,
            "corpus_documents": len(unique_sources),
            "top_k": top_k,
            "retriever_top_k": retriever_top_k,
            "target_top_k": target_top_k,
            "max_steps": max_steps,
            "num_concurrent": num_concurrent,
            "query_count": len(output_queries),
            "page_index_misses": total_page_misses,
            **agentic_stats,
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
        "agent_llm": llm_model,
        "top_k": top_k,
        "retriever_top_k": retriever_top_k,
        "index_time_s": round(index_time, 1),
        "retrieval_time_s": round(retrieval_time, 1),
        "page_index_misses": total_page_misses,
        "output_path": output,
        **agentic_stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Agentic retrieval on the bo767 corpus using sub-page chunks "
            "from extraction Parquet, with full-page expansion. "
            "An LLM agent iteratively refines retrieval via multi-angle "
            "search, query rewriting, and a selection agent."
        ),
    )
    parser.add_argument(
        "--backend",
        default="llama-nv-embed-reasoning-3b",
        help="retrieval-bench embedding backend (default: %(default)s)",
    )
    parser.add_argument(
        "--llm-model",
        required=True,
        help=("LiteLLM model string for the agent LLM " "(e.g. nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5)"),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM (default: NVIDIA_API_KEY or OPENAI_API_KEY env)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for the LLM endpoint (default: OPENAI_BASE_URL env)",
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
        default=500,
        help=("Sub-page candidates per agent retrieve() call " "(default: %(default)s)"),
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        default=1,
        help="Concurrent agent queries (default: %(default)s)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max agent loop iterations per query (default: %(default)s)",
    )
    parser.add_argument(
        "--target-top-k",
        type=int,
        default=10,
        help="Doc IDs the agent selects via final_results (default: %(default)s)",
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
    print("Agentic Retrieval on bo767 (sub-page chunks)")
    print("=" * 60)
    print(f"Backend:           {args.backend}")
    print(f"Agent LLM:         {args.llm_model}")
    print(f"Parquet dir:       {args.parquet_dir}")
    print(f"Page index:        {args.markdown_index}")
    print(f"CSV:               {args.csv}")
    print(f"Top-K (output):    {args.top_k}")
    print(f"Top-K (retriever): {args.retriever_top_k}")
    print(f"Target top-K:      {args.target_top_k}")
    print(f"Max steps:         {args.max_steps}")
    print(f"Concurrency:       {args.num_concurrent}")
    print("=" * 60)

    stats = run_agentic_retrieval(
        output=args.output,
        backend=args.backend,
        llm_model=args.llm_model,
        api_key=args.api_key,
        base_url=args.base_url,
        top_k=args.top_k,
        retriever_top_k=args.retriever_top_k,
        num_concurrent=args.num_concurrent,
        max_steps=args.max_steps,
        target_top_k=args.target_top_k,
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
    if stats.get("total_retrieval_calls"):
        print(
            f"Agent stats: {stats['total_retrieval_calls']} retrieve() calls total, "
            f"{stats['mean_retrieval_calls_per_query']} avg/query, "
            f"{stats.get('queries_with_fallback', 0)} fallbacks"
        )
    if stats["page_index_misses"] > 0:
        print(f"Page index misses: {stats['page_index_misses']}")
    print("Done.")


if __name__ == "__main__":
    main()
