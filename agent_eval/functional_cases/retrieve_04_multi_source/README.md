# Functional test suite — RETRIEVE: multi-source synthesis

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and flags.

This suite covers the **RETRIEVE** job: **multi-source synthesis** — an answer that is
**composed from multiple retrieved chunks across different documents**, where no single
document can answer the question on its own.

---

## The user task under test

> **JTBD: RETRIEVE — P0.** "Multi-source synthesis: answer composed from multiple
> retrieved chunks across different documents."

**Success criteria for the row (OPERATIONAL pass — not RAGAS):**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: (1) retrieved hits span **≥ 2 distinct `source` values** (different documents), not all from one doc; (2) the final synthesized answer **cites EACH contributing source** by name; (3) the answer **covers ALL expected sources** — an expected, loaded, relevant source missing from the answer is a **partial fail** counted against pass. Multi-source coverage is the defining gate. |
| Time | **RETRIEVE ≤ 1 min** per case (one multi-doc ingest into a single table + one cross-doc query; non-agentic) |
| Trigger rate | ≥ 95% — an "across all my docs / what do A's, B's, C's docs say" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — ingest the corpus into **ONE table** (`retriever ingest <fileA> <fileB> [<fileC>]`), then `retriever query "<cross-doc q>" --top-k 8` (top-k large enough to pull hits from multiple docs; optionally `--rerank`). Same `--table-name` on ingest and query. **No `--input-type` flag** (it does not exist). |
| Token usage | tracked, not gated |

**Functional RETRIEVE uses OPERATIONAL pass, not RAGAS.** Pass = correct grounded answer +
right subcommand/flags + the multi-source gates. The graded (RAGAS ≥0.75) flavor lives only
in the separate performance-eval suites, not here.

Seed queries this suite is derived from (paraphrased into the prompts):
- *"What do <A>'s, <B>'s, and <C>'s docs say about <topic>?"*
- *"Across all the docs I loaded, what's the most commonly mentioned <thing>?"*
- *"Across the memos in this folder, list every <item> and the total per project."*

**What makes this distinct from single-source RETRIEVE:** the answer **REQUIRES multiple
`source_id` values**. Every question here is built so the contributing facts live in
*different* PDFs, so a correct answer is impossible from any one document.

---

## How the CLI behaves for this task (grounded in `--dry-run` + skill references)

Verified with `retriever ingest <fileA> <fileB> <fileC> --dry-run` (offline, no network):

- **Multi-doc → one table.** `retriever ingest a.pdf b.pdf c.pdf` ingests all the files
  into a **single** LanceDB table (default `nemo-retriever`). The dry-run of the 3-PDF
  corpus reports `branch_summary: pdf:3` with one branch `(family=pdf, extraction_mode=pdf,
  count=3)` and all three under `documents` — i.e. one query can then retrieve across all of
  them. Keep the same `--table-name` on ingest and query.
- **`--top-k 8`.** Default top-k is 10; this suite pins `--top-k 8` explicitly to make the
  intent visible: pull enough hits that multiple documents are represented in the result set
  (a small top-k can collapse onto a single doc and silently miss a source).
- **`--rerank`** (rung 5). Off by default; turning it on (or setting any reranker flag)
  orders the cross-doc candidate pool with `llama-nemotron-rerank-1b-v2` so the best hit
  from each document surfaces inside top-k — useful when several docs compete for the same
  slots.
- **`--content-types text,table`** (rungs 3–5). Several contributing facts live in tables
  (Frost's Collections table; the numeric grid; multimodal Table 1), so the table-bearing
  rungs filter to typed hits to make table-derived rows eligible.
- **Format auto-detection.** `.pdf` is resolved from the extension; there is **no
  `--input-type` flag**.
- **Ingest success line:** `Ingested N file(s) -> M row(s) in LanceDB lancedb/<table>.`
- **Query hit shape:** a JSON array; each hit has exactly `{source, page_number, text}`;
  `page_number` is a 1-indexed int. **`source` is the per-hit document identifier used to
  count DISTINCT sources** and to check that each contributing doc is cited.
- **Host-agnostic.** This RETRIEVE suite asserts the *cross-document synthesis behavior*
  (≥2 distinct sources, each cited, full coverage), not where embedding/extraction ran. On a
  CPU/hosted box those default to hosted endpoints (`NVIDIA_API_KEY`); on a local GPU use the
  `[local]` backends (see the two SETUP suites).

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## The corpus and its ground truth

A small, deterministic 2–3 PDF corpus from `/raid/nemo_retriever/data/`, chosen so the
answer to each question is **spread across documents**:

- **`woods_frost.pdf`** (2pg) — p1: Robert Frost poem *"Stopping by Woods on a Snowy
  Evening"* ("His house is in the village though"; "And miles to go before I sleep").
  p2: table **"Frost's Collections"** — *A Boy's Will = 1913*, North of Boston = 1914,
  Mountain Interval = 1916, **New Hampshire = 1923**, **West Running Brook = 1928** (most recent).
- **`table_test.pdf`** (1pg) — numeric grid Year × {Bill, Amy, James, Ted, Susan}.
  **James 2019 = 978**; **Susan 2023 = 970**; Bill 2023 = 919.
- **`multimodal_test.pdf`** (3pg "Testing Document") — **Table 1** (Animal/Activity/Place):
  **Giraffe → Driving a car → At the beach**. Contains 2 tables, 2 charts, 3 bullet points.

**Shared cross-document theme (verified):** *every one of the three PDFs contains a table* —
`woods_frost` p2 ("Frost's Collections"), `table_test` (the numeric grid itself), and
`multimodal_test` (Table 1 and Table 2). "A table" is therefore a real, checkable
common-theme answer for the most-commonly-mentioned / common-theme rung.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `retr-ms-001` | **Multi-source floor.** 2-PDF corpus; one query whose answer needs a fact from EACH doc → hits span ≥2 distinct `source` values. The dimension that distinguishes this task from single-source RETRIEVE. | `ingest`, `query --top-k 8` |
| 2 | `retr-ms-002` | **Citation gate.** Same 2-PDF corpus; grade now turns on **each contributing source being named** next to its fact (not just that hits spanned two docs). | `ingest`, `query --top-k 8` |
| 3 | `retr-ms-003` | **3-doc common-theme synthesis.** Adds a third doc and a corpus-wide aggregation: the element common to ALL three docs (a table), answerable only by retrieving from all three. | `ingest`, `query --top-k 8 --content-types text,table` |
| 4 | `retr-ms-004` | **Coverage gate.** One distinct fact per named doc; the case is a **partial fail if any expected source is missing**, even when present facts are correct and span ≥2 sources. The ≥2-source floor alone no longer passes. | `ingest`, `query --top-k 8 --content-types text,table` |
| 5 | `retr-ms-005` | **Acceptance gate.** One synthesis answer that exercises all three gates at once — ≥2 distinct sources, each cited, full coverage — as a per-doc roll-up; `--rerank` so every doc surfaces in top-k. | `ingest`, `query --top-k 8 --rerank --content-types text,table` |

The ladder: T1 proves a cross-doc answer pulls from ≥2 sources at all; T2 adds explicit
per-source citation; T3 adds a third document and shifts from fact-per-doc to corpus-wide
common-theme synthesis; T4 adds the full-coverage requirement (a missing expected source is
a partial fail); T5 composes all three gates into one acceptance answer.

---

### T1 — `retr-ms-001` · 2-doc corpus, a fact from each  *(complexity 1)*
- **Satisfies:** RETRIEVE multi-source-synthesis core, simplest form.
- **Data:** `cases/retr-ms-001/data/{woods_frost.pdf, table_test.pdf}` (one table).
- **Expected:** ingest both into one table; `query "… New Hampshire year; James 2019 …"
  --top-k 8` → hits span **woods_frost.pdf AND table_test.pdf**. Ground truth: **New
  Hampshire = 1923** (woods_frost p2) and **James 2019 = 978** (table_test p1), each
  attributed to its source.

### T2 — `retr-ms-002` · citation gate  *(complexity 2)*
- **Satisfies:** "the final answer cites EACH contributing source."
- **Data:** same 2-PDF corpus.
- **Adds:** the prompt demands a source label per fact. Ground truth: a verbatim Frost line
  (e.g. *"And miles to go before I sleep"*) [woods_frost.pdf] and **Susan 2023 = 970**
  [table_test.pdf]. **Fails** if both facts are correct but neither is attributed to its doc.

### T3 — `retr-ms-003` · 3-doc common-theme synthesis  *(complexity 3)*
- **Satisfies:** the "what's the most commonly mentioned thing across all the docs" seed query.
- **Data:** `cases/retr-ms-003/data/{woods_frost, table_test, multimodal_test}.pdf`.
- **Adds:** a third doc and corpus-wide aggregation. Ground truth: **a table appears in all
  three** — Frost's Collections (woods_frost p2), the numeric grid (table_test p1), Table
  1/Table 2 (multimodal_test). Answer must point to the table in each doc.

### T4 — `retr-ms-004` · coverage gate  *(complexity 4)*
- **Satisfies:** "covers ALL expected sources (missing one = partial fail)."
- **Data:** same 3-PDF corpus.
- **Adds:** one distinct fact from EACH named doc: **earliest Frost collection = 1913**
  ("A Boy's Will", woods_frost p2), **James 2019 = 978** (table_test), **Giraffe = "Driving a
  car"** (multimodal_test Table 1). **Partial fail** if any of the three is absent from the
  answer — the ≥2-source floor alone does not pass this rung.

### T5 — `retr-ms-005` · acceptance gate, all three gates at once  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the multi-source-synthesis row.
- **Data:** same 3-PDF corpus.
- **Expected:** one `ingest`, then `query … --top-k 8 --rerank --content-types text,table`
  → a roll-up listing every document with one distinct cited fact each: **West Running Brook
  = 1928** [woods_frost.pdf p2]; **James 2019 = 978** [table_test.pdf p1]; **Giraffe "At the
  beach"** [multimodal_test.pdf Table 1].
- **Adds (the gate):** distinct-sources **and** each-cited **and** full-coverage must ALL
  hold in one synthesized answer; `--rerank` so the best hit from each doc surfaces in top-k.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** all of a case's PDFs are ingested into **one** table (same `--table-name` on the
query); **(b)** retrieved hits span **≥ 2 distinct `source` values**; **(c)** the answer
**cites each contributing source** next to its fact; **(d)** the answer **covers all expected
sources** (a missing expected source is a partial fail); **(e)** facts match the grounded
values (New Hampshire = 1923; James 2019 = 978; Susan 2023 = 970; A Boy's Will = 1913; West
Running Brook = 1928; Giraffe "Driving a car" / "At the beach"; a table in all three docs)
with the right page citation; **(f)** no `--input-type` flag is used (it does not exist).

**Note on live runs.** Expected outputs are grounded in the CLI **source + `--dry-run`**
(the 3-PDF dry-run reports `branch_summary: pdf:3`, one pdf branch, three documents → one
table) and in the verbatim text of the three fixtures; the suite has **not** been run live
yet. A live run requires a reachable embedding backend (hosted
`https://integrate.api.nvidia.com/v1` with `NVIDIA_API_KEY`, makes small billable calls — or
a local GPU embedder per the SETUP-GPU suite) and, for the table-typed rungs, the visual
extraction stack (page-elements → OCR → table-structure) reachable. Running live would
capture the real per-file row counts, the actual distinct-`source` distribution within
top-k (the load-bearing multi-source signal), rerank ordering, retrieval latencies, and
token baselines.
