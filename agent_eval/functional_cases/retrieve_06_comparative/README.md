# Functional test suite — RETRIEVE: comparative (compare entities or sections across chunks)

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and flags.

This suite covers the **RETRIEVE → comparative** job: take retrieved chunks and produce a
**structured side-by-side comparison** of two (or more) entities or sections, with **one
citation per side**. The defining behavior is not a single fact lookup — it is the agent
**composing a two-sided structure from cited hits**.

---

## The user task under test

> **JTBD: RETRIEVE — P0.** "Comparative: compare entities or sections across retrieved
> chunks."

**Success criteria for the row (operational — NOT RAGAS):**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: the response is a **structured side-by-side** with **one citation per side**; each citation has a **distinct `source_id`** (comparing across docs) **OR** resolves to **distinct rows/cells** (comparing entities within one doc); the **per-side facts each match ground truth**. |
| Time | **RETRIEVE ≤ 1 min** per case (corpus already ingested; 1–2 small queries + compose the two-sided answer) |
| Trigger rate | ≥ 95% — a "compare A vs B" / "side-by-side" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest <paths>` once to build the corpus, then `retriever query` to pull the comparable chunks (one query, or one per side), then compose the side-by-side from the **cited** hits. **No `--input-type` flag** (it does not exist). |
| Token usage | tracked, not gated |

**Why operational, not RAGAS.** Per `CONVENTIONS.md`, functional RETRIEVE tests use the
**operational** quality flavor (correct grounded answer + right subcommand + right
structure), binary pass/fail. The graded RAGAS path (≥80% of queries scoring ≥0.75) lives
only in the separate **performance-eval** suites, not here.

Seed queries this suite is derived from (paraphrased into the prompts):
- *"Compare `<A>`'s and `<B>`'s `<metric>` in `<year>`."*
- *"Side-by-side: `<A>`'s and `<B>`'s `<attribute>`."*
- *"Compare the `<section>` in `<docA>.pdf` and `<docB>.pdf`."*

---

## How the CLI behaves for this task (grounded in `--dry-run` + skill references)

- **Two subcommands only.** `retriever ingest <paths…>` builds the LanceDB corpus;
  `retriever query "<text>"` returns hits. There is **no `--input-type` flag** (format is
  auto-detected from the extension).
- **Query hit shape.** A JSON array; each hit has exactly `{source, page_number, text}`;
  `page_number` is a 1-indexed int. The agent uses **`source` as the per-side source_id**
  and **`text` as the per-side fact** (for a table-structure ingest, `text` carries the
  row/cell content).
- **Comparative answers are composed, not retrieved.** A comparison is the agent assembling
  a two-sided structure from cited hits. Two valid retrieval shapes — both graded the same:
  - **(a) one query** whose `--top-k` spans both sides, then the agent splits the hits per
    side; or
  - **(b) two queries**, one per entity/side, each contributing its own cited hit.
- **Table cells → `--use-table-structure`.** The within-doc entity comparisons pull numbers
  out of a table, so those cases **ingest with `--use-table-structure`** (flips
  `use_table_structure` true, `table_output_format` → `markdown`, materializes
  **nemotron-table-structure-v1**) and **query with `--content-types text,table`** so each
  compared cell returns as a discrete `content_type=table` row. Without it the table can
  flatten to one plain-text chunk and the two entity values collapse together — breaking the
  "distinct rows per side" half of the rule.
- **Two distinct-citation forms** (the heart of the validation rule):
  - **Across docs →** the two citations have **distinct `source_id`s** (e.g.
    `woods_frost.pdf` vs `multimodal_test.pdf`).
  - **Within one doc →** the two citations share a `source_id` but resolve to **distinct
    rows/cells** (e.g. James's row vs Susan's row in `table_test.pdf`).
- **Ingest success line:** `Ingested N file(s) -> M row(s) in LanceDB lancedb/<table>.`
  (`--dry-run` shows `branch_summary pdf:K` for K PDFs; offline, no network.)

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## Ground truth (verified by reading the fixtures)

- **`table_test.pdf`** — 1-page table, `Year × {Bill, Amy, James, Ted, Susan}`:
  - **2019 row:** Bill **665**, Amy **600**, **James 978**, Ted **707**, **Susan 922**.
  - **2023 row:** Bill **919**, Amy **656**, **James 539**, Ted **870**, **Susan 970**.
- **`woods_frost.pdf`** — 2 pages; p2 table `Collection × Year`: **"New Hampshire" = 1923**
  (entry #4).
- **`multimodal_test.pdf`** — 3 pages "TestingDocument"; p1 **Table 1** (Animal/Activity/
  Place): **Giraffe → "Driving a car" → "At the beach"**.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `retrieve-cmp-001` | **Baseline.** Two ENTITIES within one doc (James vs Susan, 2019) → side-by-side, distinct rows cited. The comparative floor. | `ingest --use-table-structure`, `query --content-types text,table` |
| 2 | `retrieve-cmp-002` | **New metric.** Same shape, a different year (2023) where the ranking FLIPS (Susan leads). Per-side accuracy on a second slice. | `ingest --use-table-structure`, `query --content-types text,table` |
| 3 | `retrieve-cmp-003` | **Across docs.** Compare a SECTION in two different docs → one citation per source, **distinct source_ids**. | `ingest --use-table-structure`, 2× `query --content-types text,table` |
| 4 | `retrieve-cmp-004` | **Per-side accuracy at scale.** All FIVE people for 2019 → five distinct cells, each fact correct, no value bleed. | `ingest --use-table-structure`, `query --content-types text,table` |
| 5 | `retrieve-cmp-005` | **Acceptance gate.** Mixed 3-doc corpus: one within-doc (distinct rows) AND one across-doc (distinct source_ids) comparison in one structured answer. | `ingest --use-table-structure`, multi-`query` |

The ladder: T1 establishes the structured two-sided answer with distinct rows; T2 changes
only the metric (and flips the verdict); T3 changes the comparison axis to **across-doc**
(distinct source_ids); T4 stresses **per-side accuracy** by widening to five entities; T5
composes both citation forms (distinct rows + distinct source_ids) in one answer.

---

### T1 — `retrieve-cmp-001` · two entities within one doc  *(complexity 1)*
- **Satisfies:** the comparative core (structured two-sided answer) in its simplest form.
- **Data:** `cases/retrieve-cmp-001/data/table_test.pdf`.
- **Expected:** `ingest … --use-table-structure`; `query "James and Susan values in 2019"
  --content-types text,table`. Side-by-side: **James 2019 = 978**, **Susan 2019 = 922**.
  One citation per side, both `table_test.pdf` p1, **distinct rows** (James vs Susan).

### T2 — `retrieve-cmp-002` · different year, verdict flips  *(complexity 2)*
- **Satisfies:** the same within-doc two-entity shape on a **different attribute slice**.
- **Data:** `cases/retrieve-cmp-002/data/table_test.pdf`.
- **Adds:** year **2023** (vs 2019). Side-by-side: **James 2023 = 539**, **Susan 2023 = 970**;
  verdict **Susan is higher** (970 > 539) — the ranking flips relative to 2019. One citation
  per side, distinct rows.

### T3 — `retrieve-cmp-003` · section across two docs  *(complexity 3)*
- **Satisfies:** the **across-doc** comparison and the **distinct-source_id** half of the rule.
- **Data:** `cases/retrieve-cmp-003/data/{woods_frost.pdf, multimodal_test.pdf}`.
- **Adds:** cross-document retrieval (top-k spans both docs, or one query per doc). Side-by-
  side: **woods_frost "New Hampshire" collection = 1923** vs **multimodal Giraffe = "Driving a
  car" (At the beach)**. The two citations carry **distinct source_ids**.

### T4 — `retrieve-cmp-004` · all five entities, per-side accuracy  *(complexity 4)*
- **Satisfies:** the **per-side-accuracy** clause directly.
- **Data:** `cases/retrieve-cmp-004/data/table_test.pdf`.
- **Adds:** five sides instead of two for year 2019: **Bill 665, Amy 600, James 978, Ted 707,
  Susan 922**; verdict **James (978) highest**. Each of five sides must independently match
  ground truth from a distinct cell — guards against value bleed between adjacent columns.

### T5 — `retrieve-cmp-005` · acceptance gate, both citation forms  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the comparative row.
- **Data:** `cases/retrieve-cmp-005/data/{table_test.pdf, woods_frost.pdf, multimodal_test.pdf}`.
- **Expected:** one ingest of the 3-doc corpus, then queries covering both comparisons in
  **one structured answer**:
  - **within-doc (distinct rows):** James 2019 = **978** vs Susan 2019 = **922**, both cited
    to `table_test.pdf` p1;
  - **across-doc (distinct source_ids):** woods_frost "New Hampshire" = **1923** vs
    multimodal Giraffe = **"Driving a car" (At the beach)**.
- **Adds (the gate):** every side has exactly one citation; both citation forms are exercised;
  facts on both sides of **both** comparisons match ground truth; **no source
  cross-contamination** (e.g. a Frost fact mis-cited to `multimodal_test.pdf`).

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** the response is a **structured side-by-side** (per-side rows/columns), not loose
prose; **(b)** **one citation per side**; **(c)** the right citation form — **distinct
source_ids** across docs, **distinct rows/cells** within a doc; **(d)** the **per-side facts
match ground truth** (978 / 922; 539 / 970; 1923 / "Driving a car"; the full 2019 row); and
**(e)** no `--input-type` flag is used (it does not exist).

**Note on live runs.** Expected outputs are grounded in the CLI **source + `--dry-run`** and
the skill's ingest/query references; the suite has **not** been run live yet. A live run
requires a reachable embedding backend (hosted `integrate.api.nvidia.com` with
`NVIDIA_API_KEY`, making small billable calls, or a local GPU per the SETUP-GPU suite) and,
for the table cases, the table-structure backend behind `--use-table-structure`. Running live
would capture the real per-doc/per-case row counts, the cell-level table-structure fidelity
(that James's and Susan's cells return as distinct rows), retrieval latencies, and token
baselines.
