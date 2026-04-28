# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-benchmark agentic retrieval eval.

Chains ReActAgentOperator -> RRFAggregatorOperator -> SelectionAgentOperator
across every corpus in the chosen benchmark, then prints per-corpus and
aggregate (mean) metrics.

Usage
-----
# ViDoRe v3 — all 8 corpora, 25 queries each
python -m nemo_retriever.examples.run_agentic_eval --benchmark vidore_v3

# ViDoRe v3 — two corpora only
python -m nemo_retriever.examples.run_agentic_eval \\
    --benchmark vidore_v3 \\
    --datasets vidore_v3_finance_en vidore_v3_physics

# BRIGHT — all 12 tasks (requires pre-ingested LanceDB)
python -m nemo_retriever.examples.run_agentic_eval \\
    --benchmark bright \\
    --lancedb-base /raid/mwason/lancedb/bright

# BRIGHT — two tasks, unlimited queries
python -m nemo_retriever.examples.run_agentic_eval \\
    --benchmark bright \\
    --datasets economics biology \\
    --n-queries 0
"""

from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("nemo_retriever.graph.react_agent_operator").setLevel(logging.DEBUG)
logging.getLogger("nemo_retriever.nim.chat_completions").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Fixed pipeline config — edit these to match your NIM / cloud endpoint
# ---------------------------------------------------------------------------

INVOKE_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "meta/llama-3.3-70b-instruct"
# LLM_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
API_KEY = None  # None = local NIM (no key); set to "os.environ/NVIDIA_API_KEY" for cloud

RETRIEVER_CAP = 50    # docs fetched per ReAct retrieve call
TARGET_TOP_K = 10     # docs ReAct agent aims to collect before stopping
SELECTION_TOP_K = 15   # docs SelectionAgent keeps after re-ranking
TEXT_TRUNCATION = 3000
EVAL_KS = (1, 3, 5, 10)

# ---------------------------------------------------------------------------
# Corpus / task lists
# ---------------------------------------------------------------------------

VIDORE_V3_CORPORA = [
    "vidore_v3_computer_science",
    "vidore_v3_energy",
    "vidore_v3_finance_en",
    "vidore_v3_finance_fr",
    "vidore_v3_hr",
    "vidore_v3_industrial",
    "vidore_v3_pharmaceuticals",
    "vidore_v3_physics",
]

BRIGHT_TASKS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_questions",
    "theoremqa_theorems",
]

# ---------------------------------------------------------------------------
# Loaders
# Each loader returns (query_ids, queries, qrels) for one corpus/task.
# qrels shape: {query_id: {doc_id: relevance_score}}
# ---------------------------------------------------------------------------


def _load_vidore(corpus_name: str, n_queries: Optional[int], split: str):
    """Load ViDoRe queries + ground truth, remapping qrel keys to pdf_page format.

    The vidore_hf loader returns qrels keyed by integer corpus_id (e.g. '157').
    LanceDB stores pages as pdf_page = '{pdf_basename}_{page_number_in_doc}'.
    We load the HF corpus once to build a corpus_id -> pdf_page mapping so the
    qrel keys match what the retriever returns via h.get('pdf_page').
    """
    from datasets import load_dataset  # type: ignore
    from nemo_retriever.recall.beir import load_beir_dataset

    beir = load_beir_dataset("vidore_hf", dataset_name=corpus_name, split=split)
    query_ids = beir.query_ids if not n_queries else beir.query_ids[:n_queries]
    queries = beir.queries if not n_queries else beir.queries[:n_queries]
    raw_qrels = {qid: beir.qrels[qid] for qid in query_ids if qid in beir.qrels}

    # Build corpus_id (int) -> pdf_page (str) mapping from the HF corpus split.
    corpus_ds = load_dataset(f"vidore/{corpus_name}", "corpus", split=split)
    id_to_pdf_page: dict[str, str] = {
        str(row["corpus_id"]): f"{row['doc_id']}_{row['page_number_in_doc']}"
        for row in corpus_ds
    }

    # Remap qrel doc-id keys from corpus_id integers to pdf_page strings.
    qrels = {
        qid: {
            id_to_pdf_page.get(str(cid), str(cid)): score
            for cid, score in doc_scores.items()
        }
        for qid, doc_scores in raw_qrels.items()
    }
    return query_ids, queries, qrels


def _load_bright(task_name: str, n_queries: Optional[int], split: str):
    """Load BRIGHT queries + gold_ids from HuggingFace directly.

    BRIGHT is not wired into the main load_beir_dataset yet, so we call the
    HuggingFace 'datasets' library directly.  The 'examples' config holds
    queries and gold_ids; we ignore the corpus (documents) here because the
    retriever already has the ingested LanceDB — we only need qrels for eval.
    """
    from datasets import load_dataset  # type: ignore

    examples_ds = None
    for repo in ("xlangai/bright", "xlangai/BRIGHT"):
        try:
            examples_ds = load_dataset(repo, "examples", split=split)[task_name]
            break
        except Exception:
            continue
    if examples_ds is None:
        raise RuntimeError(f"Could not load BRIGHT task '{task_name}' from HuggingFace")

    query_ids: list[str] = []
    queries: list[str] = []
    qrels: dict[str, dict[str, int]] = {}
    for row in examples_ds:
        qid = str(row["id"])
        query_ids.append(qid)
        queries.append(str(row["query"]))
        gold = row.get("gold_ids") or []
        qrels[qid] = {str(g): 1 for g in gold}

    if n_queries:
        query_ids = query_ids[:n_queries]
        queries = queries[:n_queries]
        qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}
    return query_ids, queries, qrels


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSpec:
    corpora: list[str]
    embedder: str
    lancedb_table: str
    default_lancedb_base: str
    load_fn: Callable
    # Extracts the doc_id string from a Retriever hit dict so it can be
    # compared against qrels keys. ViDoRe uses PDF basename; BRIGHT uses the
    # document's raw ID string stored in source_id.
    doc_id_fn: Callable[[dict], str]


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "vidore_v3": BenchmarkSpec(
        corpora=VIDORE_V3_CORPORA,
        # Must match the model used at ingestion time — we ingested with the
        # vision-language model so queries must be embedded with it too.
        embedder="nvidia/llama-nemotron-embed-vl-1b-v2",
        lancedb_table="nv-ingest",
        default_lancedb_base="/raid/mwason/lancedb",
        load_fn=_load_vidore,
        # ViDoRe qrels are keyed by pdf_page = '{pdf_basename}_{page_number_in_doc}'.
        # LanceDB stores this directly in the 'pdf_page' column.
        doc_id_fn=lambda h: str(h.get("pdf_page") or Path(str(h.get("source_id", ""))).stem),
    ),
    "bright": BenchmarkSpec(
        corpora=BRIGHT_TASKS,
        # BRIGHT is text-only — use the text embedding model.
        embedder="nvidia/llama-nemotron-embed-1b-v2",
        lancedb_table="nv-ingest",
        default_lancedb_base="/raid/mwason/lancedb/bright",
        load_fn=_load_bright,
        # BRIGHT qrels key is the raw corpus document ID stored in source_id.
        doc_id_fn=lambda h: str(h.get("source_id", h.get("doc_id", ""))),
    ),
}

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Multi-benchmark agentic retrieval eval")
parser.add_argument(
    "--benchmark", default="vidore_v3", choices=list(BENCHMARKS),
    help="Which benchmark to evaluate (default: vidore_v3)",
)
parser.add_argument(
    "--datasets", nargs="+", default=None,
    help="Run only these corpora/tasks (default: all for the benchmark)",
)
parser.add_argument(
    "--lancedb-base", default=None,
    help="Base directory containing one LanceDB sub-dir per corpus "
         "(default: benchmark's default_lancedb_base)",
)
parser.add_argument(
    "--n-queries", type=int, default=25,
    help="Queries per corpus/task; 0 = all (default: 25)",
)
parser.add_argument("--split", default="test", help="Dataset split (default: test)")
args = parser.parse_args()

spec = BENCHMARKS[args.benchmark]
corpora = args.datasets or spec.corpora
lancedb_base = args.lancedb_base or spec.default_lancedb_base
n_queries = args.n_queries or None  # 0 → None → no limit

# ---------------------------------------------------------------------------
# Pipeline imports (deferred so --help is fast)
# ---------------------------------------------------------------------------

from nemo_retriever.graph.executor import InprocessExecutor  # noqa: E402
from nemo_retriever.graph.react_agent_operator import ReActAgentOperator  # noqa: E402
from nemo_retriever.graph.rrf_aggregator_operator import RRFAggregatorOperator  # noqa: E402
from nemo_retriever.graph.selection_agent_operator import SelectionAgentOperator  # noqa: E402
from nemo_retriever.recall.beir import compute_beir_metrics  # noqa: E402
from nemo_retriever.retriever import Retriever  # noqa: E402

# ---------------------------------------------------------------------------
# Per-corpus eval loop
# ---------------------------------------------------------------------------

all_metrics: dict[str, dict[str, float]] = {}

for corpus in corpora:
    print(f"\n{'=' * 60}")
    print(f"Benchmark : {args.benchmark}   Corpus : {corpus}")
    print(f"LanceDB   : {lancedb_base}/{corpus}")

    # Build a Retriever bound to this corpus's LanceDB.
    # Rebuilt each iteration — each corpus lives in its own directory.
    _retriever = Retriever(
        lancedb_uri=f"{lancedb_base}/{corpus}",
        lancedb_table=spec.lancedb_table,
        embedder=spec.embedder,
        local_hf_device="cuda",
        top_k=RETRIEVER_CAP,
    )
    _lock = threading.Lock()

    # Capture retriever + lock by default arg so the closure is iteration-safe.
    def retriever_fn(
        query_text: str,
        top_k: int,
        _r: Retriever = _retriever,
        _l: threading.Lock = _lock,
    ) -> list[dict]:
        with _l:
            hits = _r.query(query_text)
        return [
            {
                "doc_id": spec.doc_id_fn(h),
                "text": h.get("text", "")[:TEXT_TRUNCATION],
            }
            for h in hits[:top_k]
        ]

    # Pipeline is rebuilt each corpus because retriever_fn is a closure over
    # this iteration's _retriever.
    pipeline = (
        ReActAgentOperator(
            invoke_url=INVOKE_URL,
            llm_model=LLM_MODEL,
            retriever_fn=retriever_fn,
            retriever_top_k=RETRIEVER_CAP,
            target_top_k=TARGET_TOP_K,
            user_msg_type="with_results",
            max_steps=20,
            api_key=API_KEY,
            parallel_tool_calls=False,
        )
        >> RRFAggregatorOperator(k=60)
        >> SelectionAgentOperator(
            invoke_url=INVOKE_URL,
            llm_model=LLM_MODEL,
            top_k=SELECTION_TOP_K,
            api_key=API_KEY,
            parallel_tool_calls=False,
        )
    )

    query_ids, queries, qrels = spec.load_fn(corpus, n_queries, args.split)
    print(f"Queries   : {len(query_ids)}")
    queries_df = pd.DataFrame({"query_id": query_ids, "query_text": queries})

    result = InprocessExecutor(pipeline).ingest(queries_df)

    # Build BEIR run: {query_id: {doc_id: score}} where score = rank inverted
    run: dict[str, dict[str, float]] = {}
    for qid, group in result.groupby("query_id"):
        n = len(group)
        run[str(qid)] = {
            str(row["doc_id"]): float(n - int(row["rank"]) + 1)
            for _, row in group.iterrows()
        }

    metrics = compute_beir_metrics(qrels, run, ks=EVAL_KS)
    all_metrics[corpus] = metrics
    for key, val in sorted(metrics.items()):
        print(f"  {key}: {val:.4f}")

# ---------------------------------------------------------------------------
# Aggregate metrics (mean across all corpora in this run)
# ---------------------------------------------------------------------------

print(f"\n{'=' * 60}")
print(f"AGGREGATE — mean across {len(all_metrics)} corpus/task(s)")
all_keys = sorted({k for m in all_metrics.values() for k in m})
for key in all_keys:
    vals = [m[key] for m in all_metrics.values() if key in m]
    print(f"  {key}: {sum(vals) / len(vals):.4f}")
