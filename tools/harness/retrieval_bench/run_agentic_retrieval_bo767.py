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
from collections import defaultdict
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
# Latency instrumentation (monkeypatch-based, zero changes to retrieval-bench)
# ---------------------------------------------------------------------------


class _LatencyTracker:
    """Non-intrusive per-query phase timing via class-level monkeypatching.

    Wraps five methods in retrieval-bench's agent stack to record wall-clock
    latency for each sub-phase.  The pipeline runs identically -- same code
    paths, same parameters, same results -- only stopwatches are added.

    Intercept points
    ~~~~~~~~~~~~~~~~
    1. ``Agent.step``          -- main agent LLM API call
    2. ``Agent.call_one_tool`` -- tool execution (retrieve / think / final_results)
    3. ``Agent.conclude_task`` -- post-loop RRF + selection agent
    4. ``RetrieveTool._acall`` -- vector search (subset of call_one_tool)
    5. ``query_rewriter.rewrite_query`` -- rewriting LLM sub-call (subset of call_one_tool)
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "step_ms": [],
                "tool_calls": [],
                "rewrite_ms": [],
                "retrieve_ms": [],
                "conclude_ms": 0.0,
            }
        )
        self._originals: dict[str, tuple] = {}

    # -- install / uninstall ------------------------------------------------

    def install(self) -> None:
        """Apply timing wrappers.  Call once before ``pipeline.search()``."""
        from retrieval_bench.nemo_agentic import query_rewriter as _qr_mod
        from retrieval_bench.nemo_agentic.agent import Agent
        from retrieval_bench.pipelines.agentic import RetrieveTool, _CURRENT_QUERY_ID

        tracker = self

        orig_step = Agent.step

        async def _timed_step(self_agent, *a, **kw):
            t0 = time.perf_counter()
            result = await orig_step(self_agent, *a, **kw)
            elapsed = (time.perf_counter() - t0) * 1000.0
            qid = getattr(self_agent, "session_id", None)
            if qid is not None:
                tracker._data[str(qid)]["step_ms"].append(elapsed)
            return result

        Agent.step = _timed_step

        orig_call_tool = Agent.call_one_tool

        async def _timed_call_tool(self_agent, *args, **kw):
            fn_name = kw.get("fn_name") or (args[0] if args else "unknown")
            t0 = time.perf_counter()
            result = await orig_call_tool(self_agent, *args, **kw)
            elapsed = (time.perf_counter() - t0) * 1000.0
            qid = getattr(self_agent, "session_id", None)
            if qid is not None:
                tracker._data[str(qid)]["tool_calls"].append({"name": str(fn_name), "ms": elapsed})
            return result

        Agent.call_one_tool = _timed_call_tool

        orig_conclude = Agent.conclude_task

        async def _timed_conclude(self_agent, *a, **kw):
            t0 = time.perf_counter()
            result = await orig_conclude(self_agent, *a, **kw)
            elapsed = (time.perf_counter() - t0) * 1000.0
            qid = getattr(self_agent, "session_id", None)
            if qid is not None:
                tracker._data[str(qid)]["conclude_ms"] = elapsed
            return result

        Agent.conclude_task = _timed_conclude

        orig_retrieve = RetrieveTool._acall

        async def _timed_retrieve(self_tool, *a, **kw):
            t0 = time.perf_counter()
            result = await orig_retrieve(self_tool, *a, **kw)
            elapsed = (time.perf_counter() - t0) * 1000.0
            qid = _CURRENT_QUERY_ID.get()
            if qid is not None:
                tracker._data[str(qid)]["retrieve_ms"].append(elapsed)
            return result

        RetrieveTool._acall = _timed_retrieve

        orig_rewrite = _qr_mod.rewrite_query

        async def _timed_rewrite(*a, **kw):
            t0 = time.perf_counter()
            result = await orig_rewrite(*a, **kw)
            elapsed = (time.perf_counter() - t0) * 1000.0
            qid = _CURRENT_QUERY_ID.get()
            if qid is not None:
                tracker._data[str(qid)]["rewrite_ms"].append(elapsed)
            return result

        _qr_mod.rewrite_query = _timed_rewrite

        self._originals = {
            "Agent.step": (Agent, "step", orig_step),
            "Agent.call_one_tool": (Agent, "call_one_tool", orig_call_tool),
            "Agent.conclude_task": (Agent, "conclude_task", orig_conclude),
            "RetrieveTool._acall": (RetrieveTool, "_acall", orig_retrieve),
            "qr.rewrite_query": (_qr_mod, "rewrite_query", orig_rewrite),
        }

    def uninstall(self) -> None:
        """Remove timing wrappers, restoring original methods."""
        for _key, (cls_or_mod, attr, original) in self._originals.items():
            setattr(cls_or_mod, attr, original)
        self._originals.clear()

    # -- data access --------------------------------------------------------

    def build_phase_timing(self, qid: str) -> dict[str, Any]:
        """Return a timing breakdown dict for *qid*, ready to merge into trace."""
        raw = self._data.get(str(qid))
        if raw is None:
            return {}

        retrieve_tcs = [tc for tc in raw["tool_calls"] if tc["name"] == "retrieve"]
        think_tcs = [tc for tc in raw["tool_calls"] if tc["name"] == "think"]
        other_tcs = [tc for tc in raw["tool_calls"] if tc["name"] not in ("retrieve", "think")]

        return {
            "step_total_ms": round(sum(raw["step_ms"]), 2),
            "step_count": len(raw["step_ms"]),
            "step_latencies_ms": [round(x, 2) for x in raw["step_ms"]],
            "tool_retrieve_total_ms": round(sum(tc["ms"] for tc in retrieve_tcs), 2),
            "tool_retrieve_count": len(retrieve_tcs),
            "tool_think_total_ms": round(sum(tc["ms"] for tc in think_tcs), 2),
            "tool_think_count": len(think_tcs),
            "tool_other_total_ms": round(sum(tc["ms"] for tc in other_tcs), 2),
            "rewrite_total_ms": round(sum(raw["rewrite_ms"]), 2),
            "rewrite_count": len(raw["rewrite_ms"]),
            "rewrite_latencies_ms": [round(x, 2) for x in raw["rewrite_ms"]],
            "vector_search_total_ms": round(sum(raw["retrieve_ms"]), 2),
            "vector_search_count": len(raw["retrieve_ms"]),
            "vector_search_latencies_ms": [round(x, 2) for x in raw["retrieve_ms"]],
            "conclude_ms": round(raw["conclude_ms"], 2),
        }


def _percentile(data: list[float], p: float) -> float:
    """Linear-interpolation percentile (no numpy dependency)."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def _print_latency_summary(
    per_query_trace: dict[str, dict],
    tracker: _LatencyTracker,
    queries: list[str],
) -> dict[str, Any]:
    """Print a human-readable latency report and return summary dict."""
    n = len(per_query_trace)
    if n == 0:
        return {}

    elapsed_vals = [t["elapsed_ms"] for t in per_query_trace.values()]
    total_elapsed = sum(elapsed_vals)

    total_step = 0.0
    total_conclude = 0.0
    total_tool_retrieve = 0.0
    total_tool_think = 0.0
    total_rewrite = 0.0
    total_vector = 0.0

    for qid in per_query_trace:
        pt = tracker.build_phase_timing(qid)
        total_step += pt.get("step_total_ms", 0)
        total_conclude += pt.get("conclude_ms", 0)
        total_tool_retrieve += pt.get("tool_retrieve_total_ms", 0)
        total_tool_think += pt.get("tool_think_total_ms", 0)
        total_rewrite += pt.get("rewrite_total_ms", 0)
        total_vector += pt.get("vector_search_total_ms", 0)

    accounted = total_step + total_tool_retrieve + total_tool_think + total_conclude
    overhead = max(total_elapsed - accounted, 0)

    def pct(v: float) -> float:
        return (v / total_elapsed * 100) if total_elapsed > 0 else 0

    print("\n" + "=" * 70)
    print("LATENCY ANALYSIS")
    print("=" * 70)
    print(f"  {n} queries  |  total wall-clock: {total_elapsed / 1000:.1f}s")
    print(
        f"  p50: {_percentile(elapsed_vals, 50) / 1000:.1f}s  |  "
        f"p90: {_percentile(elapsed_vals, 90) / 1000:.1f}s  |  "
        f"p95: {_percentile(elapsed_vals, 95) / 1000:.1f}s  |  "
        f"max: {max(elapsed_vals) / 1000:.1f}s"
    )

    print("\n  Phase breakdown (summed across all queries):")
    print(f"    LLM turns (Agent.step):      {total_step / 1000:8.1f}s  ({pct(total_step):5.1f}%)")
    print(f"    Tool: retrieve calls:         {total_tool_retrieve / 1000:7.1f}s  ({pct(total_tool_retrieve):5.1f}%)")
    print(f"      - query rewriting (LLM):    {total_rewrite / 1000:7.1f}s  ({pct(total_rewrite):5.1f}%)")
    print(f"      - vector search (embed):    {total_vector / 1000:7.1f}s  ({pct(total_vector):5.1f}%)")
    print(f"    Tool: think calls:            {total_tool_think / 1000:7.1f}s  ({pct(total_tool_think):5.1f}%)")
    print(f"    Post-loop (RRF + selection):  {total_conclude / 1000:7.1f}s  ({pct(total_conclude):5.1f}%)")
    print(f"    Overhead / other:             {overhead / 1000:7.1f}s  ({pct(overhead):5.1f}%)")

    sorted_by_elapsed = sorted(
        per_query_trace.items(),
        key=lambda x: x[1].get("elapsed_ms", 0),
        reverse=True,
    )

    print("\n  Top 5 slowest queries:")
    for qid, trace in sorted_by_elapsed[:5]:
        pt = tracker.build_phase_timing(qid)
        idx = int(qid) if qid.isdigit() else -1
        query_text = queries[idx] if 0 <= idx < len(queries) else "?"
        truncated = (query_text[:55] + "...") if len(query_text) > 55 else query_text
        print(
            f"    #{qid:>3}  {trace['elapsed_ms'] / 1000:6.1f}s  "
            f"LLM:{pt.get('step_total_ms', 0) / 1000:5.1f}s({pt.get('step_count', 0)}t) "
            f"sel:{pt.get('conclude_ms', 0) / 1000:5.1f}s "
            f"retr:{pt.get('tool_retrieve_total_ms', 0) / 1000:5.1f}s"
        )
        print(f'          "{truncated}"')

    mean_turns = sum(t.get("llm_turns", 0) for t in per_query_trace.values()) / n
    mean_retr = sum(t.get("num_retrieval_calls", 0) for t in per_query_trace.values()) / n
    sel_count = sum(1 for t in per_query_trace.values() if t.get("selection_agent_ran"))
    rrf_count = sum(1 for t in per_query_trace.values() if t.get("rrf_used"))
    rw_count = sum(1 for t in per_query_trace.values() if t.get("query_rewriting_used"))

    print(f"\n  Means:  {mean_turns:.1f} LLM turns/query  |  " f"{mean_retr:.1f} retrieve calls/query")
    print(f"  Selection agent: {sel_count}/{n}  |  " f"RRF: {rrf_count}/{n}  |  " f"Rewriting: {rw_count}/{n}")
    print("=" * 70)

    return {
        "total_elapsed_s": round(total_elapsed / 1000, 2),
        "p50_query_s": round(_percentile(elapsed_vals, 50) / 1000, 2),
        "p90_query_s": round(_percentile(elapsed_vals, 90) / 1000, 2),
        "p95_query_s": round(_percentile(elapsed_vals, 95) / 1000, 2),
        "max_query_s": round(max(elapsed_vals) / 1000, 2),
        "pct_llm_turns": round(pct(total_step), 1),
        "pct_tool_retrieve": round(pct(total_tool_retrieve), 1),
        "pct_query_rewriting": round(pct(total_rewrite), 1),
        "pct_vector_search": round(pct(total_vector), 1),
        "pct_conclude_rrf_selection": round(pct(total_conclude), 1),
        "pct_overhead": round(pct(overhead), 1),
    }


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

    # -- 5. Agentic retrieve (with latency instrumentation) -----------------
    print(f"\nStarting agentic retrieval " f"(agent LLM: {llm_model}, concurrency: {num_concurrent}) ...")
    tracker = _LatencyTracker()
    tracker.install()
    try:
        t0 = time.time()
        result = pipeline.search(query_ids=query_ids, queries=queries)
        run_dict = result[0] if isinstance(result, tuple) else result
        infos = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
        retrieval_time = time.time() - t0
    finally:
        tracker.uninstall()

    # -- 6. Expand chunk hits to full-page markdown -----------------------
    output_queries: dict[str, dict] = {}
    total_page_misses = 0

    for qid, query_text in zip(query_ids, queries):
        scores = run_dict.get(qid, {})
        ranked = sorted(scores.items(), key=lambda x: float(x[1]), reverse=True)

        chunks, meta, misses = _expand_scored_chunks_to_pages(ranked, chunk_provenance, page_index, top_k)
        total_page_misses += misses
        output_queries[query_text] = {"chunks": chunks, "metadata": meta}

    # -- 7. Merge phase timing into per-query trace -------------------------
    per_query_trace = infos.get("per_query_trace", {})
    for qid in per_query_trace:
        per_query_trace[qid]["phase_timing"] = tracker.build_phase_timing(qid)

    agentic_stats: dict[str, Any] = {}
    if per_query_trace:
        retrieval_calls = [t.get("num_retrieval_calls", 0) for t in per_query_trace.values()]
        agentic_stats = {
            "total_retrieval_calls": sum(retrieval_calls),
            "mean_retrieval_calls_per_query": (
                round(sum(retrieval_calls) / len(retrieval_calls), 2) if retrieval_calls else 0
            ),
            "queries_with_fallback": sum(1 for t in per_query_trace.values() if t.get("fallback_used")),
        }

    # -- 8. Latency summary ------------------------------------------------
    latency_summary = _print_latency_summary(per_query_trace, tracker, queries)

    # -- 9. Write FileRetriever JSON ---------------------------------------
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
            "latency_summary": latency_summary,
        },
        "per_query_trace": per_query_trace,
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
        "latency_summary": latency_summary,
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
