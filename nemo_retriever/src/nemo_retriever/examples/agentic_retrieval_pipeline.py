# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-based agentic retrieval pipeline.

Chains four graph-operator stages to deliver an agentic, multi-step
retrieval experience:

1. **Ingestion** — graph-based document ingestion via
   :class:`~nemo_retriever.graph_ingestor.GraphIngestor`
   (files → extract → embed → LanceDB).  Skip with ``--skip-ingest`` if the
   table is already populated.

2. **Query expansion** *(agentic)* —
   :class:`~nemo_retriever.graph.subquery_operator.SubQueryGeneratorOperator`
   runs as a :class:`~nemo_retriever.graph.pipeline_graph.Graph` node via
   :class:`~nemo_retriever.graph.executor.InprocessExecutor`, expanding each
   input query into sub-queries via an LLM.

3. **Retrieval** — :class:`~nemo_retriever.retriever.Retriever` is invoked
   as-is (``retriever.queries()``) on the expanded sub-query texts, followed
   by Reciprocal Rank Fusion (RRF) to merge per-sub-query result lists into
   a single ranked list.

4. **Selection** *(agentic, optional)* —
   :class:`~nemo_retriever.graph.selection_agent_operator.SelectionAgentOperator`
   runs an LLM-based agentic loop (think → select) to re-rank and select
   the final top-k documents from the fused candidates.

Run with::

    python -m nemo_retriever.examples.agentic_retrieval_pipeline \\
        --input-path /data/pdfs \\
        --queries "What causes inflation?" \\
        --queries "How do vaccines work?" \\
        --llm-model meta/llama-3.1-70b-instruct \\
        --llm-api-key "os.environ/NVIDIA_API_KEY" \\
        --llm-base-url https://integrate.api.nvidia.com/v1 \\
        --embed-invoke-url http://localhost:8000/v1

    # Skip ingestion, run full agentic retrieval with selection agent:
    python -m nemo_retriever.examples.agentic_retrieval_pipeline \\
        --skip-ingest \\
        --queries "What causes inflation?" \\
        --llm-model meta/llama-3.1-70b-instruct \\
        --llm-api-key "os.environ/NVIDIA_API_KEY" \\
        --llm-base-url https://integrate.api.nvidia.com/v1 \\
        --embed-invoke-url http://localhost:8000/v1 \\
        --selection-top-k 5
"""

from __future__ import annotations

import glob as _glob
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import typer

logger = logging.getLogger(__name__)
app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


# ---------------------------------------------------------------------------
# RRF helpers
# ---------------------------------------------------------------------------


def _rrf_fuse(
    subquery_df: pd.DataFrame,
    per_subquery_hits: List[List[Dict[str, Any]]],
    *,
    rrf_k: int,
    top_n: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group per-sub-query hits by original query_id and apply RRF fusion.

    Returns a dict mapping ``query_id`` → list of hit dicts sorted by
    ``rrf_score`` descending (length ≤ *top_n*).  Each hit dict gains
    an ``"rrf_score"`` key.
    """
    df = subquery_df.copy()
    df["_hits"] = per_subquery_hits

    results: Dict[str, List[Dict[str, Any]]] = {}

    for query_id, group in df.groupby("query_id", sort=False):
        rrf_scores: Dict[str, float] = defaultdict(float)
        best_hit: Dict[str, Dict[str, Any]] = {}

        for _, row in group.iterrows():
            hit_list: List[Dict[str, Any]] = row["_hits"]
            # Sort best-first: rerank score (higher=better) or distance (lower=better)
            if hit_list and "_rerank_score" in hit_list[0]:
                sorted_hits = sorted(hit_list, key=lambda h: h.get("_rerank_score", 0.0), reverse=True)
            else:
                sorted_hits = sorted(hit_list, key=lambda h: h.get("_distance", 0.0))

            for rank, hit in enumerate(sorted_hits, 1):
                doc_id = str(hit.get("source_id") or hit.get("path") or rank)
                rrf_scores[doc_id] += 1.0 / (rank + rrf_k)
                if doc_id not in best_hit:
                    best_hit[doc_id] = hit

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        results[str(query_id)] = [{**best_hit[doc_id], "rrf_score": score} for doc_id, score in fused]

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Optional[Path] = typer.Option(
        None,
        "--input-path",
        help="Directory or file of documents to ingest. Ignored when --skip-ingest is set.",
        path_type=Path,
    ),
    input_type: str = typer.Option(
        "pdf",
        "--input-type",
        help="Document type: 'pdf', 'txt', 'html', or 'image'.",
    ),
    queries: List[str] = typer.Option(
        ...,
        "--queries",
        help="One or more query strings. Repeat the flag for multiple queries.",
    ),
    # LLM / sub-query expansion
    llm_model: str = typer.Option(..., "--llm-model", help="LLM model for sub-query generation (e.g. 'gpt-4o')."),
    llm_api_key: str = typer.Option("", "--llm-api-key", help="LLM API key or 'os.environ/VAR_NAME'."),
    llm_base_url: str = typer.Option("", "--llm-base-url", help="Custom LLM endpoint URL (NIM etc.)."),
    max_subqueries: int = typer.Option(4, "--max-subqueries", help="Max sub-queries to generate per query."),
    strategy: str = typer.Option(
        "decompose",
        "--strategy",
        help="Sub-query strategy: 'decompose', 'hyde', or 'multi_perspective'.",
    ),
    # Embedding
    embed_invoke_url: str = typer.Option("", "--embed-invoke-url", help="Embedding NIM endpoint URL."),
    embed_model_name: str = typer.Option(
        "nvidia/llama-nemotron-embed-1b-v2",
        "--embed-model-name",
        help="Embedding model name.",
    ),
    # LanceDB / retrieval
    lancedb_uri: str = typer.Option(LANCEDB_URI, "--lancedb-uri"),
    lancedb_table: str = typer.Option(LANCEDB_TABLE, "--lancedb-table"),
    top_k: int = typer.Option(10, "--top-k", help="Hits to retrieve per sub-query."),
    rrf_k: int = typer.Option(60, "--rrf-k", help="RRF rank constant (default 60)."),
    top_n: int = typer.Option(0, "--top-n", help="Final results per query after fusion (0 = same as top-k)."),
    hybrid: bool = typer.Option(False, "--hybrid/--no-hybrid", help="Enable hybrid (vector+BM25) search."),
    # Selection agent
    selection_top_k: int = typer.Option(
        0,
        "--selection-top-k",
        help=(
            "Run the LLM selection agent after RRF fusion, selecting this many documents. "
            "Set to 0 to skip the selection agent stage."
        ),
    ),
    # Ingestion control
    skip_ingest: bool = typer.Option(
        False,
        "--skip-ingest/--no-skip-ingest",
        help="Skip ingestion — assume LanceDB is already populated.",
    ),
    api_key: str = typer.Option("", "--api-key", help="API key for remote NIM extraction endpoints."),
    debug: bool = typer.Option(False, "--debug/--no-debug"),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
    effective_top_n = top_n if top_n > 0 else top_k

    # ------------------------------------------------------------------
    # Step 1 — Ingestion (graph-based, inprocess mode)
    # ------------------------------------------------------------------
    if not skip_ingest:
        if input_path is None:
            raise typer.BadParameter("--input-path is required when --skip-ingest is not set.")

        from nemo_retriever.graph_ingestor import GraphIngestor
        from nemo_retriever.params import EmbedParams, ExtractParams, TextChunkParams
        from nemo_retriever.utils.remote_auth import resolve_remote_api_key
        from nemo_retriever.vector_store.lancedb_store import handle_lancedb

        remote_api_key = resolve_remote_api_key(api_key or None)
        embed_api_key = remote_api_key if embed_invoke_url else None

        embed_kwargs: Dict[str, Any] = {"model_name": embed_model_name}
        if embed_invoke_url:
            embed_kwargs["embed_invoke_url"] = embed_invoke_url
        if embed_api_key:
            embed_kwargs["api_key"] = embed_api_key
        embed_params = EmbedParams(**embed_kwargs)

        input_path = Path(input_path).expanduser().resolve()
        if input_path.is_file():
            file_patterns = [str(input_path)]
        else:
            ext_map = {
                "pdf": ["*.pdf"],
                "txt": ["*.txt"],
                "html": ["*.html"],
                "image": ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.bmp"],
            }
            file_patterns = [
                str(input_path / ext)
                for ext in ext_map.get(input_type, ["*.pdf"])
                if _glob.glob(str(input_path / ext))
            ]
            if not file_patterns:
                raise typer.BadParameter(f"No files found for input_type={input_type!r} in {input_path}")

        logger.info("Step 1: Building ingestion graph (inprocess) for %s ...", input_path)

        ingestor = GraphIngestor(run_mode="inprocess")
        ingestor = ingestor.files(file_patterns)

        if input_type == "txt":
            ingestor = ingestor.extract_txt()
        elif input_type == "html":
            ingestor = ingestor.extract_html()
        elif input_type == "image":
            ingestor = ingestor.extract_image_files(ExtractParams())
        else:
            extract_kwargs: Dict[str, Any] = {}
            if remote_api_key:
                extract_kwargs["api_key"] = remote_api_key
            ingestor = ingestor.extract(ExtractParams(**extract_kwargs) if extract_kwargs else ExtractParams())

        ingestor = ingestor.embed(embed_params)

        logger.info("Step 1: Executing ingestion graph ...")
        result = ingestor.ingest()
        result_df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(list(result))
        logger.info("Step 1: Ingested %d rows; writing to LanceDB ...", len(result_df))
        handle_lancedb(result_df.to_dict("records"), lancedb_uri, lancedb_table, hybrid=hybrid, mode="overwrite")
        logger.info("Step 1 complete.")
    else:
        logger.info("Step 1: Skipped (--skip-ingest).")

    # ------------------------------------------------------------------
    # Step 2 — Query expansion via SubQueryGeneratorOperator (graph node)
    # ------------------------------------------------------------------
    from nemo_retriever.graph import InprocessExecutor
    from nemo_retriever.graph.pipeline_graph import Graph
    from nemo_retriever.graph.subquery_operator import SubQueryGeneratorOperator

    logger.info("Step 2: Expanding %d quer%s via LLM (%s) ...", len(queries), "y" if len(queries) == 1 else "ies", llm_model)

    subquery_kwargs: Dict[str, Any] = {
        "llm_model": llm_model,
        "max_subqueries": max_subqueries,
        "strategy": strategy,
    }
    if llm_api_key:
        subquery_kwargs["api_key"] = llm_api_key
    if llm_base_url:
        subquery_kwargs["base_url"] = llm_base_url


    subquery_op = SubQueryGeneratorOperator(**subquery_kwargs)

    # Build a single-node graph and run via InprocessExecutor
    query_graph = Graph()
    query_graph >> subquery_op

    executor = InprocessExecutor(query_graph, show_progress=False)

    queries_df = pd.DataFrame({
        "query_id": [f"q{i}" for i in range(len(queries))],
        "query_text": list(queries),
    })
    subquery_df = executor.ingest(queries_df)
    logger.info("Step 2 complete: %d sub-quer%s generated.", len(subquery_df), "y" if len(subquery_df) == 1 else "ies")

    if debug:
        for _, row in subquery_df.iterrows():
            logger.debug("  [%s] subq%d: %s", row["query_id"], row["subquery_idx"], row["subquery_text"])

    # ------------------------------------------------------------------
    # Step 3 — Retrieval via Retriever 
    # ------------------------------------------------------------------
    from nemo_retriever.retriever import Retriever

    logger.info("Step 3: Retrieving for %d sub-quer%s ...", len(subquery_df), "y" if len(subquery_df) == 1 else "ies")

    retriever_kwargs: Dict[str, Any] = {
        "lancedb_uri": lancedb_uri,
        "lancedb_table": lancedb_table,
        "embedder": embed_model_name,
        "top_k": top_k,
        "hybrid": hybrid,
    }
    if embed_invoke_url:
        retriever_kwargs["embedding_endpoint"] = embed_invoke_url


    retriever = Retriever(**retriever_kwargs)
    subquery_texts = subquery_df["subquery_text"].tolist()
    per_subquery_hits = retriever.queries(subquery_texts)
    logger.info("Step 3 complete.")

    # ------------------------------------------------------------------
    # Step 4 — RRF fusion (inline, no external deps)
    # ------------------------------------------------------------------
    logger.info("Step 4: Fusing results with RRF (k=%d, top_n=%d) ...", rrf_k, effective_top_n)

    fused = _rrf_fuse(subquery_df, per_subquery_hits, rrf_k=rrf_k, top_n=effective_top_n)

    # ------------------------------------------------------------------
    # Step 5 — Selection agent re-ranking (optional)
    # ------------------------------------------------------------------
    if selection_top_k > 0:
        from nemo_retriever.graph.selection_agent_operator import SelectionAgentOperator

        logger.info(
            "Step 5: Running LLM selection agent (top_k=%d, model=%s) ...",
            selection_top_k,
            llm_model,
        )

        # Build a flat DataFrame from the fused hits so SelectionAgentOperator
        # can group by query_id and process each query independently.
        selection_rows = []
        for i, query_text in enumerate(queries):
            qid = f"q{i}"
            for hit in fused.get(qid, []):
                selection_rows.append({
                    "query_id": qid,
                    "query_text": query_text,
                    "doc_id": str(hit.get("source_id") or hit.get("path") or ""),
                    "text": str(hit.get("text", "")),
                    "rrf_score": hit.get("rrf_score", 0.0),
                })
        selection_input_df = pd.DataFrame(selection_rows)

        selection_kwargs: Dict[str, Any] = {"llm_model": llm_model, "top_k": selection_top_k}
        if llm_api_key:
            selection_kwargs["api_key"] = llm_api_key
        if llm_base_url:
            selection_kwargs["base_url"] = llm_base_url

        selection_op = SelectionAgentOperator(**selection_kwargs)
        selection_graph = Graph()
        selection_graph >> selection_op
        selection_executor = InprocessExecutor(selection_graph, show_progress=False)
        selection_df = selection_executor.ingest(selection_input_df)

        logger.info("Step 5 complete.")

        # Print selection agent results
        print()
        for i, query_text in enumerate(queries):
            qid = f"q{i}"
            ranked = selection_df[selection_df["query_id"] == qid].sort_values("rank")
            print(f"Query [{qid}]: {query_text!r}")
            print(f"  {len(ranked)} result(s) after LLM selection (top {selection_top_k}):")
            for _, row in ranked.iterrows():
                # Retrieve text from fused hits since selection_df only has query_id/doc_id/rank/message
                hit_text = next(
                    (str(h.get("text", "")) for h in fused.get(qid, [])
                     if str(h.get("source_id") or h.get("path") or "") == row["doc_id"]),
                    "",
                )
                snippet = hit_text.replace("\n", " ")[:120]
                print(f"  {int(row['rank']):3d}. [{row['doc_id']}] {snippet}")
            if ranked["message"].iloc[0] if len(ranked) > 0 else "":
                print(f"  Rationale: {ranked['message'].iloc[0][:200]}")
            print()
    else:
        # ------------------------------------------------------------------
        # Print RRF results (no selection agent)
        # ------------------------------------------------------------------
        print()
        for i, query_text in enumerate(queries):
            qid = f"q{i}"
            hits = fused.get(qid, [])
            print(f"Query [{qid}]: {query_text!r}")
            print(f"  {len(hits)} result(s) after RRF fusion (top {effective_top_n}):")
            for rank, hit in enumerate(hits, 1):
                score = hit.get("rrf_score", 0.0)
                snippet = str(hit.get("text", "")).replace("\n", " ")[:120]
                print(f"  {rank:3d}. [{score:.4f}] {snippet}")
            print()


if __name__ == "__main__":
    app()
