# Synopsis — EVALUATE: retrieval quality for customer datasets

**What user task this covers.** A customer has a set of labeled questions — each tagged with
the document and page that *should* answer it — plus a search index over their documents.
They want NeMo Retriever to tell them **how good the retrieval is**: the standard scores
(**Recall@k** — did the right page show up in the top-k — and **nDCG@k** — how *high* it
ranked), a **per-question breakdown** of which queries passed or failed, and confidence that
the numbers are **stable** if they run it again. This is measurement, not generation — no LLM
judge, just retrieval metrics over a labeled set.

**How we test it.** Five agent prompts, each handing the agent a few small real PDFs and a
labeled-set CSV, asking it to benchmark retrieval. We check the agent drives the real
`retriever` CLI correctly: it builds an index (`retriever ingest`), then evaluates with the
**recall sub-app** — `retriever recall vdb-recall run --query-csv <labeled.csv>
--table-name nemo-retriever` — which embeds each query, searches the index, and prints
**Recall@1 / @5 / @10**. For the ranking metric it uses the BEIR evaluator (the `bo767_csv`
loader behind `retriever harness run`), which prints **Recall@k and nDCG@k together**. The
key trap we guard against: `ingest` writes to table `nemo-retriever` but the recall command
*defaults* to a different table (`nv-ingest`), so the agent must pass `--table-name` or every
score comes back zero. Plain `retriever query` is **wrong** here — it returns hits, not
metrics.

**The five tests, simplest to hardest:**

1. **Baseline Recall@k** — a 3-question labeled set over a 2-PDF index; prove Recall@1/@5/@10
   print as numbers. (Ground truth: Frost's "New Hampshire" collection → 1923 on page 2;
   James's 2019 table value → 978.)
2. **Add nDCG@k** — score the same set through the BEIR evaluator so ranking quality
   (nDCG@k) shows up alongside Recall@k.
3. **Per-query breakdown** — show, question by question, whether the gold page landed in the
   top-1 / top-5 and at what rank — i.e. exactly which queries passed and which failed.
4. **Vary k / failing query** — a 5-question set with one deliberately hard query whose gold
   page is not rank 1, so Recall@1 < Recall@5 and we can point at the query that only
   succeeds once k widens.
5. **Acceptance gate** — a customer-style labeled set (the same shape as the repo's bo767 /
   digital-corpora annotation files) evaluated end-to-end: Recall@k + nDCG@k + the per-query
   breakdown, plus a second run proving the numbers are reproducible. This is the test the
   others build up to.

**Why this order.** Each rung adds exactly one new thing: first "can we surface Recall@k as a
number at all," then the ranking metric (nDCG@k), then the per-query view, then k-sensitivity
(a query that fails at k=1), then everything composed over a real customer-style labeled set
with reproducibility.

**Status.** Tests are authored and grounded in the real CLI — `retriever recall` /
`retriever eval` `--help` and the source modules `tools/recall/vdb_recall.py`, `core.py`,
`beir.py`, `recall_eval.py`; **not yet run live**. Live runs need a built LanceDB index
(`retriever ingest`, hosted endpoints + `NVIDIA_API_KEY` on a no-GPU host, or a local GPU)
and a reachable query embedder, and would capture the real Recall@1/@5/@10 and nDCG@k floats,
per-query ranks, and the numeric reproducibility check. See `README.md` for the full spec and
`cases.json` for the machine-gradable definitions.
