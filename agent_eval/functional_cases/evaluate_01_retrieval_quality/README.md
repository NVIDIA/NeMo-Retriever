# Functional test suite — EVALUATE: retrieval quality for customer datasets

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src`. This suite covers the **EVALUATE** job: given a labeled
set of `(query, gold_page)` pairs and an index, **measure how well retrieval is doing** and
surface the standard metrics — **Recall@k** and **nDCG@k** — plus a **per-query breakdown**
of which questions passed or failed at which `k`.

Each test is a self-contained triple: a prompt, a per-case `cases/<id>/data/` folder (the
PDFs to index + a labeled-set CSV), and an expected output naming the correct `retriever`
subcommands and flags.

---

## The user task under test

> **JTBD: EVALUATE — P1.** "Evaluate Retrieval quality for customer datasets."
> Seed queries: *"I have N labeled question-answer pairs and an index — tell me how well
> retrieval is doing, per-query."* / *"Run a retrieval benchmark on my corpus and tell me
> which queries are failing."*

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: the eval harness consumes a labeled `(query, gold_pdf_page)` / `(query, gold_source_ids)` set against an existing index and **surfaces Recall@k (and, via the BEIR path, nDCG@k) as numeric outputs**; a **per-query** pass/fail breakdown at each `k` is available; the result is **reproducible** across runs on the same labeled set + index. (This functional row asserts the metric *appears as a number* — it does **not** gate on a metric threshold; that is the separate performance-eval suite.) |
| Time | **slow — ≤ 10 min** per case (build/reuse a small index + embed N queries + search + score) |
| Trigger rate | ≥ 95% — an "evaluate / benchmark retrieval quality on my labeled set" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever recall vdb-recall run --query-csv … --table-name nemo-retriever` for Recall@k (after `retriever ingest`); the Recall@k + **nDCG@k** path via the `bo767_csv` BEIR loader / `retriever harness run`. **Not** a plain `retriever query` (that returns hits, not metrics). |
| Token usage | tracked, not gated |

---

## How the CLI behaves for this task (grounded in CLI source + `--help`)

Verified against `src/nemo_retriever/cli/main.py` (lazy sub-app registration) and the recall
/ evaluation modules. Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

### The `retriever recall` sub-app → Recall@k
`retriever recall` is a lazy sub-app (`cli/main.py` maps `recall → nemo_retriever.recall:app`).
Its tree:

```
retriever recall                       # "Recall utilities (query -> embed -> vector DB search)."
  └─ vdb-recall                        # "Embed query CSV rows, search LanceDB, print hits, compute recall@k."
       ├─ run                          # the primary command (used by this suite)
       └─ recall-with-main             # same engine, plainer text print loop
```

`retriever recall vdb-recall run` (source: `tools/recall/vdb_recall.py::run` +
`tools/recall/core.py::retrieve_and_score`):

- **Inputs:** `--query-csv <csv>` (the labeled set), `--table-name <table>`,
  `--lancedb-uri <dir>` (default `lancedb`), `--top-k N` (default 5),
  `--embedding-model` (default `nvidia/llama-nemotron-embed-1b-v2`), `--limit N`,
  `--print-hits/--no-print-hits` (default **on**), and embedding-endpoint / local-HF flags.
- **What it does:** reads the CSV, embeds each query, searches LanceDB, prints the top-k
  hits per query, and prints **Recall@1 / Recall@5 / Recall@10**.
- **Metric ks are FIXED to `(1, 5, 10)`** inside the command; `--top-k` only controls how
  many hits are *printed* (`search_k = max(top_k, 10)`).

**Output (stdout):** per-query `Query` / `Gold` / `Hits` lines (when `--print-hits`), then:

```
Recall metrics
  recall@1: 0.6667
  recall@5: 1.0000
  recall@10: 1.0000
```

### The `retriever eval` sub-app
`retriever eval` (`recall`/`eval` are both lazy sub-apps) exposes `run`
(`--config eval_sweep.yaml | --from-env`, a QA-generation sweep keyed off
`RETRIEVAL_FILE`/`QA_DATASET`/`GEN_MODEL`/`JUDGE_MODEL`), `export`, and `build-page-index`.
The **retrieval-quality** Recall/nDCG metrics for *this* row come from the **recall sub-app**
and the **BEIR evaluator**, not from `eval run` (which is the generation/RAGAS sweep).

### Recall@k **and** nDCG@k together → the BEIR evaluator
Both metrics are emitted by `tools/recall/beir.py::compute_beir_metrics`, which computes
`recall@k` **and** `ndcg@k` for `ks` (default `1,3,5,10`):
`ndcg@k = DCG(top-k relevances) / IDCG`. It is driven by `RecallEvaluatorActor`
(`tools/recall/recall_eval.py`, `evaluation_mode="beir"`) and reachable from the CLI via
`retriever harness run --dataset <name>`. The `bo767_csv` loader
(`load_beir_dataset` → `_load_annotations_csv_dataset`) reads an annotations CSV; passing a
**CSV path** as the dataset name loads a *local* labeled set. `print_run_summary` then prints
the BEIR `Recall@k` / `nDCG@k` block. (`ndcg@1 == recall@1` when there is one relevant doc
per query; a gold page at rank 2 contributes `1/log2(3) ≈ 0.63` to nDCG.)

### Per-query breakdown
`tools/recall/core.py::evaluate_recall` builds a per-query DataFrame:
`query_id, query, golden_answer, top_retrieved, hit@k (bool pass/fail), rank@k` for each `k`
— i.e. exactly "which queries passed/failed at which k". `vdb-recall run --print-hits`
(default on) additionally prints each query's `Gold` + top-k `Hits` so misses are visible
inline.

### Labeled-set format (the inputs the harness consumes)
- **Recall CSV** (`recall vdb-recall run`, `pdf_page` match mode): `query,pdf_page`
  (e.g. `woods_frost_2`) **or** `query,pdf,page` where `page` is **1-indexed**; the gold key
  is `{pdf}_{page}`, compared against retrieved keys `{source_stem}_{page_number}`
  (`page_number` 1-indexed; `-1` / `{pdf}_-1` matches the whole document). Extra columns are
  ignored. (`core.py::_normalize_query_df` + `_hits_to_keys` + `_is_hit`.)
- **BEIR annotations CSV** (`bo767_csv` loader, the nDCG path): `modality,query,answer,pdf,page`
  — the **same shape as the repo's `data/bo767_annotations.csv` and
  `data/digital_corpora_10k_annotations.csv`**. NOTE the loader does `page_number = int(page)+1`,
  so its `page` column is **0-indexed** (to target 1-indexed page `P`, store `P-1`). One
  relevant doc id per query (graded relevance 1).

### Key caveat — default table mismatch
`retriever ingest` writes table **`nemo-retriever`** by default, but
`retriever recall vdb-recall run` defaults to **`nv-ingest`**. **Every** recall/eval command
in this suite must pass `--table-name nemo-retriever` (or `lancedb_table=nemo-retriever`) or
it reads an empty/absent table and reports `Recall@k = 0` for every query. The query embedder
must also match the embedder used at ingest.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `eval-rq-001` | **Baseline.** Tiny labeled set (3 queries) over a 2-PDF index → **Recall@k surfaced as a number**. The EVALUATE floor. | `ingest`, `recall vdb-recall run` |
| 2 | `eval-rq-002` | **Add nDCG@k.** Same labeled set scored through the BEIR evaluator → Recall@k **and** nDCG@k. | `ingest`, `recall vdb-recall run`, `harness run` (BEIR) |
| 3 | `eval-rq-003` | **Per-query breakdown.** Which queries passed/failed at which k (hit@k / rank@k + inline Gold vs hits). | `ingest`, `recall vdb-recall run --print-hits` |
| 4 | `eval-rq-004` | **Vary k / failing query.** 5-query set with a hard query → Recall@1 < Recall@5; pinpoint the query that only passes once k widens. | `ingest`, `recall vdb-recall run --print-hits` |
| 5 | `eval-rq-005` | **Acceptance gate.** Customer-style labeled set (bo767 annotations shape) end-to-end → Recall@k + nDCG@k + per-query breakdown, **reproducible** across runs. | `ingest`, `recall vdb-recall run`, `harness run` (BEIR) |

The ladder adds exactly one dimension per rung: T1 surfaces Recall@k as a number; T2 adds the
ranking metric nDCG@k; T3 adds the per-query pass/fail view; T4 adds k-sensitivity (a failing
query at k=1); T5 composes everything over a customer-style labeled set and adds
reproducibility.

---

### T1 — `eval-rq-001` · baseline Recall@k  *(complexity 1)*
- **Satisfies:** EVALUATE operational-pass core — consume a labeled set + index, surface
  Recall@k as a number.
- **Data:** `woods_frost.pdf`, `table_test.pdf`, `labeled_set.csv` (3 queries, `query,pdf,page`).
- **Expected:** `ingest` the 2 PDFs (table `nemo-retriever`), then
  `recall vdb-recall run --query-csv data/labeled_set.csv --table-name nemo-retriever --top-k 5`
  → `Recall@1/@5/@10` print as floats. Ground truth: woods_frost p2 = "New Hampshire / 1923";
  table_test p1: James 2019 = **978**; woods_frost p1 = the "miles to go before I sleep" poem.

### T2 — `eval-rq-002` · add nDCG@k  *(complexity 2)*
- **Satisfies:** the "nDCG@k surfaced" half of the validation path.
- **Data:** as T1, plus `labeled_set_annotations.csv` (`modality,query,answer,pdf,page`, 0-indexed pages).
- **Adds:** the BEIR evaluator (`bo767_csv` loader → `compute_beir_metrics`) so **both**
  `recall@k` and `ndcg@k` print for ks `(1,3,5,10)`. nDCG reflects *where* in the top-k the
  gold page landed.

### T3 — `eval-rq-003` · per-query breakdown  *(complexity 3)*
- **Satisfies:** the "per-query breakdown (which queries pass/fail at which k)" clause.
- **Data:** as T1.
- **Adds:** `evaluate_recall`'s per-query `hit@k`/`rank@k` table + `--print-hits` inline
  Gold-vs-hits, so a reader can see exactly which query missed at k=1 / k=5.

### T4 — `eval-rq-004` · vary k, surface a failing query  *(complexity 4)*
- **Satisfies:** the "vary k (Recall@1 vs @5) / show a failing query" dimension.
- **Data:** `woods_frost.pdf`, `table_test.pdf`, a 5-query `labeled_set.csv` including a hard
  query ("List Robert Frost poetry collections and their publication years" → gold
  `woods_frost_2`) whose gold page is expected at rank > 1.
- **Adds:** Recall@1 < Recall@5 as distinct numbers, and identification of the query that
  only succeeds once k widens (hit@1=false, hit@5=true).

### T5 — `eval-rq-005` · acceptance gate  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the EVALUATE row.
- **Data:** `woods_frost.pdf`, `table_test.pdf`, `multimodal_test.pdf`,
  `customer_annotations.csv` (bo767 shape, 5 queries), and `bo767_annotations_slice.csv`
  (a real 12-row slice of `data/bo767_annotations.csv` documenting the production labeled-set
  format).
- **Expected:** `ingest … --use-table-structure`, then **both** the recall path (Recall@k) and
  the BEIR path (Recall@k + nDCG@k) over the customer labeled set, the per-query breakdown,
  and a **second** recall run proving identical Recall@k (reproducibility).
- **Adds (the gate):** a customer-style labeled set evaluated end-to-end with all three
  outputs (Recall@k + nDCG@k + per-query) and reproducibility.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** the agent uses the **recall sub-app** (and, for nDCG, the BEIR path) over the labeled
set — **not** a plain `retriever query`; **(b)** **Recall@k** is surfaced as a number;
**(c)** **nDCG@k** is surfaced (rungs 2, 5); **(d)** a **per-query** pass/fail breakdown is
available (rungs 3–5); **(e)** `--table-name nemo-retriever` matches the ingest table (else
all-zero); **(f)** rung 5 shows the result is **reproducible** across two runs.

**Note on live runs.** Expected outputs are grounded in the CLI **source** (`cli/main.py`,
`tools/recall/vdb_recall.py`, `core.py`, `beir.py`, `recall_eval.py`) and the
`--help` surfaces of `retriever recall` / `retriever eval`; the suite has **not** been run
live yet. A live run requires building the LanceDB index (`retriever ingest`, which on a
no-GPU host calls hosted extraction/embedding endpoints needing `NVIDIA_API_KEY`, or runs on
a local GPU) and an embedder reachable for query embedding. Running live would capture the
real per-query ranks, the actual Recall@1/@5/@10 and nDCG@k floats, the exact row counts of
the built index, latencies, and token baselines — and would confirm the reproducibility
assertion in rung 5 numerically.
