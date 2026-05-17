# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end agentic retrieval pipeline example.

Chains ReActAgentOperator → RRFAggregatorOperator → SelectionAgentOperator
against a local LanceDB and the NVIDIA build endpoint.

Usage::

    export NVIDIA_API_KEY=<your-key>
    cd nemo_retriever
    uv run python src/nemo_retriever/examples/run_agentic_pipeline.py

To capture the debug trace::

    PYTHONUNBUFFERED=1 uv run python src/nemo_retriever/examples/run_agentic_pipeline.py \\
        2>&1 | tee pipeline.log
"""

from __future__ import annotations

import logging

import pandas as pd

# ---------------------------------------------------------------------------
# Logging — DEBUG for the agentic loop and LLM transport; quiet everything else
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("nemo_retriever.graph.react_agent_operator").setLevel(logging.DEBUG)
logging.getLogger("nemo_retriever.nim.chat_completions").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Configuration — edit these to match your setup
# ---------------------------------------------------------------------------

# --- Cloud (NVIDIA build) ---
# INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# LLM_MODEL  = "meta/llama-3.3-70b-instruct"
# API_KEY    = "os.environ/NVIDIA_API_KEY"

# --- Local NIM (Docker on this machine) ---
INVOKE_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL  = "meta/llama-3.3-70b-instruct"
API_KEY    = None  # local NIM needs no key

LANCEDB_URI = "/raid/mwason/lancedb/bo767"
LANCEDB_TABLE = "nv-ingest"
EMBEDDING_ENDPOINT = None  # None → local HF model; set to a NIM URL for remote embedding

RETRIEVER_CAP = 20    # docs fetched per retrieve call — keep low to stay within LLM context
TEXT_TRUNCATION = 500  # chars of doc text shown to the ReAct LLM per document
TARGET_TOP_K = 10     # docs the ReAct agent aims to return
SELECTION_TOP_K = 15   # docs SelectionAgentOperator keeps after re-ranking

QUERY_CSV = "/datasets/nv-ingest/ground_truths/bo767_query_gt.csv"
EVAL_KS = (1, 3, 5, 10)
N_QUERIES = 25  # set to None to run all 991 — expensive against cloud API

# ---------------------------------------------------------------------------
# retriever_fn — adapts Retriever output to the {doc_id, text} schema that
# ReActAgentOperator expects.  Retriever returns source_id, not doc_id.
# top_k is a constructor arg on Retriever, not per-call, so we build once
# with RETRIEVER_CAP and slice to the per-call top_k in the wrapper.
# ---------------------------------------------------------------------------

import threading  # noqa: E402

from nemo_retriever.retriever import Retriever  # noqa: E402

_retriever = Retriever(
    lancedb_uri=LANCEDB_URI,
    lancedb_table=LANCEDB_TABLE,
    embedder="nvidia/llama-nemotron-embed-1b-v2",
    local_hf_device="cuda",
    top_k=RETRIEVER_CAP,
)
_retriever_lock = threading.Lock()


def retriever_fn(query_text: str, top_k: int) -> list[dict]:
    with _retriever_lock:
        hits = _retriever.query(query_text)
    return [
        {
            "doc_id": h.get("source_id", h.get("path", "")),
            "text": h.get("text", "")[:TEXT_TRUNCATION],
        }
        for h in hits[:top_k]
    ]


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from nemo_retriever.graph.executor import InprocessExecutor  # noqa: E402
from nemo_retriever.graph.react_agent_operator import ReActAgentOperator  # noqa: E402
from nemo_retriever.graph.rrf_aggregator_operator import RRFAggregatorOperator  # noqa: E402
from nemo_retriever.graph.selection_agent_operator import SelectionAgentOperator  # noqa: E402
from nemo_retriever.recall.beir import compute_beir_metrics  # noqa: E402

pipeline = (
    ReActAgentOperator(
        invoke_url=INVOKE_URL,
        llm_model=LLM_MODEL,
        retriever_fn=retriever_fn,
        retriever_top_k=RETRIEVER_CAP,
        target_top_k=TARGET_TOP_K,
        user_msg_type="with_results",
        max_steps=10,
        api_key=API_KEY,
        parallel_tool_calls=False,  # required for local vLLM NIMs
    )
    >> RRFAggregatorOperator(k=60)
    >> SelectionAgentOperator(
        invoke_url=INVOKE_URL,
        llm_model=LLM_MODEL,
        top_k=SELECTION_TOP_K,
        api_key=API_KEY,
        parallel_tool_calls=False,  # required for local vLLM NIMs
    )
)

# ---------------------------------------------------------------------------
# Queries + ground truth — loaded from bo767 query CSV
# ---------------------------------------------------------------------------

_gt = pd.read_csv(QUERY_CSV)
if N_QUERIES is not None:
    _gt = _gt.head(N_QUERIES)

queries_df = pd.DataFrame(
    {
        "query_id": _gt.index.astype(str),
        "query_text": _gt["query"],
    }
)

# qrels: {query_id: {pdf_basename: 1}}  — one relevant doc per query
qrels = {str(i): {str(row["pdf"]): 1} for i, row in _gt.iterrows()}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = InprocessExecutor(pipeline).ingest(queries_df)
    print(result.to_string(index=False))

    # Build BEIR run: {query_id: {pdf_basename: score}} and compute metrics
    run: dict = {}
    for qid, group in result.groupby("query_id"):
        n = len(group)
        run[str(qid)] = {
            Path(str(row["doc_id"])).stem: float(n - int(row["rank"]) + 1)
            for _, row in group.iterrows()
        }

    metrics = compute_beir_metrics(qrels, run, ks=EVAL_KS)
    print("\n=== Eval Metrics ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
