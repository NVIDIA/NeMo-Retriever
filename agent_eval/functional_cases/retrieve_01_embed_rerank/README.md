# Functional test suite — RETRIEVE: embed then rerank

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and flags.

This suite covers the **RETRIEVE** job: pull candidate chunks with the platform's
**embedder**, then **reorder them with the reranker** so the most relevant passages come
first — and prove the reranker actually changed the ordering and still returned the right
grounded answer.

This is an **operational-pass** suite, **not** a RAGAS-graded one. RAGAS judge scoring
(≥80% of queries ≥0.75) lives in the separate **performance-eval** suites. Here we assert
the *mechanics*: the embed→rerank chain fired, the reranked order differs from dense-only,
and the gold row is in the top-k with a correct citation.

---

## The user task under test

> **JTBD: RETRIEVE — P0.** "Retrieve chunks via the platform's embedder (text or VL), then
> reorder with the reranker."

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational, NOT RAGAS) | binary pass: rerank-enabled query runs **dense-retrieve → `nvidia/llama-nemotron-rerank-1b-v2` → top-k**; each hit carries **both** a vector distance (dense stage) **and** a rerank score (rerank stage); the **reranked order differs** from the dense-only order on the same query; and the **top-k reranked hits contain the gold row** (correct grounded answer) |
| Time | **RETRIEVE — ≤ 1 min** per case (small corpus already ingested + one or two queries) |
| Trigger rate | ≥ 95% — a "retrieve and put the most relevant passages first / reorder by relevance" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest <paths>` then `retriever query "<q>" --rerank --top-k K`. **Rerank is OFF by default**; `--rerank` (or any reranker flag) turns it on. To prove rerank changed the order, compare `--no-rerank` vs `--rerank` on the same query. **No `--input-type` flag** (it does not exist). |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased into the prompts):
- *"What does &lt;doc&gt; say about X? Make sure the most relevant passages come first."*
- *"What was &lt;metric&gt; in &lt;year&gt;?"*
- *"Across all docs, what's the most commonly mentioned &lt;topic&gt;?"*

---

## How the CLI behaves for this task (grounded in `query --help` + the CLI source)

- **Rerank is off by default.** `retriever query` defaults to `--no-rerank` (dense-only,
  ordered by vector distance). Passing **`--rerank`** — or any reranker flag
  (`--reranker-model-name` / `--reranker-invoke-url` / `--reranker-backend`) — turns it on.
- **The chain when rerank is on.** Dense vector retrieve a candidate pool (`--candidate-k`,
  which **defaults to `--top-k`**) → rerank that pool with **`nvidia/llama-nemotron-rerank-1b-v2`**
  → return the top **`--top-k`**. Each surviving hit has been scored by **both** stages
  (a vector distance from the dense retrieve and a rerank score from the reranker).
- **`--candidate-k` ≥ `--top-k`.** `--candidate-k` is the dense pool fetched *before*
  reranking and must be at least `--top-k`; if unset it equals `--top-k`. Setting it
  **larger** than `--top-k` lets the reranker promote a passage that dense ranking placed
  below the top-k cutoff into the final top-k — the behavior rung 4 exercises.
- **Proving rerank did something.** Run the *same* query `--no-rerank` then `--rerank`; the
  result ordering must change. That A/B pair is the operational proof (rungs 2 and 5).
- **Where embed/rerank run is host-dependent (and out of scope here).** Embedder + reranker
  run hosted (`integrate.api.nvidia.com` / `build.nvidia.com`, key from `NVIDIA_API_KEY`) on
  a no-GPU box, or on a local GPU via `--query-embed-backend hf --reranker-backend hf` (see
  the two SETUP suites). This suite is **host-agnostic**: the assertion is that the
  embed→rerank chain **fired and reordered the hits**, not where it ran. The commands below
  omit host flags; a live runner adds the host-appropriate backend/endpoint flags.
- **Ingest success line:** `Ingested N file(s) -> M row(s) in LanceDB lancedb/<table>.`
- **Query hit shape:** a JSON array; each hit has exactly `{source, page_number, text}`;
  `page_number` is a 1-indexed int. The vector distance and rerank score are produced
  internally by the two stages; at the CLI layer the **ordering of the array** reflects them
  — dense order with `--no-rerank`, reranked order with `--rerank`.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

Models referenced in the validation path: embedder `nvidia/llama-nemotron-embed-1b-v2`,
reranker `nvidia/llama-nemotron-rerank-1b-v2` (VL variant `…-rerank-vl-1b-v2`).

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `retrieve-er-001` | **Baseline dense.** One doc, `--no-rerank` query → relevant hit returned. Establishes the dense-only ordering rung 2 reorders. | `ingest`, `query --no-rerank` |
| 2 | `retrieve-er-002` | **Turn on rerank.** Same corpus + query, add `--rerank` → gold hit promoted, order ≠ dense-only (A/B proof). | `ingest`, `query --no-rerank` (baseline), `query --rerank` |
| 3 | `retrieve-er-003` | **Multi-doc.** Rerank a candidate pool drawn from 3 competing docs → best passage across docs wins. | `ingest` (folder), `query --rerank` |
| 4 | `retrieve-er-004` | **Candidate-pool sizing.** `--candidate-k` > `--top-k` → reranker promotes the gold cell from a wide pool into a narrow top-k. | `ingest --use-table-structure`, `query --rerank --candidate-k --top-k` |
| 5 | `retrieve-er-005` | **Acceptance gate.** A small *labeled set* (one query per gold doc) over the corpus; each proves the full dense→rerank chain returns gold in top-k with reordering + citation. | `ingest --use-table-structure`, 3× `query --rerank`, 1× `query --no-rerank` (A/B) |

The ladder: T1 proves dense retrieval alone returns the right chunk (the baseline order);
T2 adds the reranker and shows the order changes; T3 changes the corpus to multiple docs so
the reranker must win across documents; T4 widens the candidate pool so the reranker
*promotes* a passage dense ranking would have cut; T5 composes it all into a small labeled
mini-eval that gates the row.

---

### T1 — `retrieve-er-001` · baseline dense retrieve  *(complexity 1)*
- **Satisfies:** RETRIEVE operational-pass core, simplest form (dense-only).
- **Data:** `cases/retrieve-er-001/data/woods_frost.pdf` (2pg; p1 Frost poem).
- **Expected:** `RETRIEVER ingest data/woods_frost.pdf` → non-zero rows; `RETRIEVER query
  "Whose woods are these and where does the owner live?" --top-k 5 --no-rerank` → dense-only
  hits; the gold passage says the owner's **"house is in the village"** (owner lives in the
  village), cite `woods_frost.pdf` p1. No reranker invoked — this is the baseline order.

### T2 — `retrieve-er-002` · turn on rerank, prove the order changes  *(complexity 2)*
- **Satisfies:** "the reranked order differs from dense-only" + "make the most relevant
  passage come first".
- **Data:** `cases/retrieve-er-002/data/woods_frost.pdf` (same doc).
- **Adds:** `--rerank` (nemotron-rerank-1b-v2) on the same corpus, plus a `--no-rerank` vs
  `--rerank` A/B pair on the identical query as the proof the ordering changed. **Ground
  truth: "miles to go before I sleep"** (with "promises to keep"), cite `woods_frost.pdf` p1.
  Each top-k hit now reflects both a vector distance and a rerank score.

### T3 — `retrieve-er-003` · multi-doc, best passage across docs  *(complexity 3)*
- **Satisfies:** the "across all docs" seed query.
- **Data:** `cases/retrieve-er-003/data/{woods_frost,table_test,multimodal_test}.pdf`.
- **Adds:** a 3-document corpus. Dense retrieval pulls candidates from competing sources
  (table_test and multimodal_test also contain years and tables), and the reranker must
  surface the single best passage across docs. **Ground truth: New Hampshire (1923)**, cite
  `woods_frost.pdf` p2 (the collection × year table). The gold doc/page must beat the
  competitors after rerank.

### T4 — `retrieve-er-004` · `--candidate-k` / `--top-k` interplay  *(complexity 4)*
- **Satisfies:** "What was &lt;metric&gt; in &lt;year&gt;?" + candidate-pool mechanics.
- **Data:** `cases/retrieve-er-004/data/table_test.pdf` (Year × {Bill,Amy,James,Ted,Susan}).
- **Adds:** an explicit `--candidate-k 20` larger than `--top-k 3`, so the reranker reorders
  a wide dense pool and **promotes** the gold cell into a narrow final top-k. Forces a
  precise cell where many near-duplicate numeric rows compete. **Ground truth: James 2019 =
  978**, cite `table_test.pdf` p1. Ingested with `--use-table-structure` so the cell exists
  as a discrete `content_type=table` row; query filtered with `--content-types text,table`.

### T5 — `retrieve-er-005` · acceptance gate, small labeled set  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the RETRIEVE embed→rerank row.
- **Data:** `cases/retrieve-er-005/data/{woods_frost,table_test,multimodal_test}.pdf`.
- **Expected:** one `ingest data/ --use-table-structure`, then a **labeled set** of three
  reranked queries (one per gold doc) plus a `--no-rerank` A/B spot-check:
  - **owner location** — "house is in the village" (`woods_frost.pdf` p1);
  - **James 2019** — 978, `content_type=table` (`table_test.pdf` p1);
  - **testing-doc counts** — 2 tables, 2 charts (and 3 bullet points) (`multimodal_test.pdf` p3).
- **Adds (the gate):** every labeled query's top-k must contain its gold row with the right
  citation, the dense→rerank chain must have fired on each (vector distance + rerank score),
  and the reranked order must differ from dense-only (shown by the A/B spot-check). This is
  an **operational pass**, **not** RAGAS grading.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** the right subcommand/flags — `--no-rerank` for the baseline, `--rerank` to enable the
reranker, `--candidate-k > --top-k` for the pool rung;
**(b)** the embed→rerank chain fired — each top-k hit reflects a dense vector distance **and**
a rerank score;
**(c)** the reranked order **differs** from dense-only on the same query (the `--no-rerank`
vs `--rerank` A/B proof);
**(d)** the gold row is in the top-k with the correct grounded answer + page citation
(village/p1; "miles to go"/p1; New Hampshire-1923/p2; James-2019=978/p1; 2 tables+2 charts/p3);
**(e)** no `--input-type` flag is used (it does not exist).
This is **operational pass**, not RAGAS — no judge score is computed here.

**Note on live runs.** Expected outputs are grounded in the CLI **source** and
`retriever query --help` (which confirms `--rerank/--no-rerank`, `--candidate-k`, `--top-k`,
`--content-types`, and the reranker flags), plus the fixtures' verified text; the suite has
**not** been run live yet. A live run requires a reachable **embedder + reranker** backend —
either the **hosted** `integrate.api.nvidia.com` / `build.nvidia.com` endpoints (needs
`NVIDIA_API_KEY`, makes small billable calls) or a **local GPU** with
`--query-embed-backend hf --reranker-backend hf` per the SETUP-GPU suite. Running live would
capture the real dense-only vs reranked orderings (the rank deltas), the per-hit vector
distances and rerank scores, the row counts by content type, retrieval latencies, and token
baselines.
