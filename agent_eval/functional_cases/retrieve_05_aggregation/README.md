# Functional test suite — RETRIEVE: aggregation (count / sum / list) across the retrieved corpus

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`) plus the canonical
LanceDB aggregate one-liners from the skill's `references/query.md`. Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and the explicit aggregation operator.

This suite covers the **RETRIEVE** job's **aggregation** task: answer **count / sum /
list** questions *across* the retrieved corpus — and prove the aggregate was **computed by
the agent over rows**, not hallucinated from an LLM summary.

---

## The user task under test

> **JTBD: RETRIEVE — P0.** "Aggregation: count / sum / list operations across the
> retrieved corpus."

**Success criteria for the row (operational, NOT RAGAS):**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: **NRL (the retriever) is TRIGGERED** to build/serve the index (`retriever ingest`, optionally `query`), but the **aggregation itself is performed BY THE AGENT deterministically over rows** (LanceDB rows or retrieved hits) — **not** guessed from a summary. An aggregation **operator must be visible in the trace** (count = `len`/`Counter`, list-distinct = `sorted(set(...))`, `sum` over numeric cells), and the numeric/list answer must match the ground truth **EXACTLY** |
| Time | **RETRIEVE ≤ 1 min** per case (small ingest + one deterministic aggregate one-liner; no agentic multi-hop) |
| Trigger rate | ≥ 95% — a "how many documents / count / sum / list-distinct across my corpus" prompt must fire the skill and drive `retriever ingest`/`query`, **not** native `ls`/`find`/`wc` |
| Subcommand accuracy | ≥ 90% — `retriever ingest <paths>` to build the index, then the canonical `references/query.md` corpus-level aggregate one-liner over the LanceDB table (and/or `retriever query`). **No `--input-type` flag** (it does not exist). |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased into the prompts):
- *"How many documents are in my knowledge base?"*
- *"How many `<X>` docs are in my pile?"*
- *"Sum `<metric>` across all `<groups>`."*

**The operational-pass nuance that defines this suite.** For an *aggregation* RETRIEVE
task, a correct-looking number is **not** enough. The retriever must be **triggered** (it
builds/serves the index), but the count/sum/list must be **agent-computed over the actual
rows** — an LLM "there look like about three documents" guess **fails** even if it lands on
the right number. The grader looks for an explicit aggregation operator in the trace
operating on LanceDB rows or retrieved hits, **and** an exact match to ground truth.

---

## How the CLI behaves for this task (grounded in `--dry-run` + `references/query.md`)

Verified with `retriever ingest … --dry-run` (offline, no network) and the skill's
`references/query.md` ("Non-semantic operations → Corpus-level aggregate"):

- **Format auto-detection / folder ingest.** `retriever ingest data/` resolves the PDFs in
  the folder from their extensions. A 3-PDF folder `--dry-run` reports
  `branch_summary: pdf:3` (one `family=pdf` branch over the three paths). There is **no
  `--input-type` flag**.
- **Ingest builds a LanceDB table the agent can aggregate over.** Default
  `--lancedb-uri lancedb`, `--table-name nemo-retriever`. Ingest success line:
  `Ingested N file(s) -> M row(s) in LanceDB lancedb/<table>.`
- **The canonical corpus-level aggregate is a LanceDB one-liner**, lifted verbatim from
  `references/query.md`:
  ```bash
  <RETRIEVER_VENV>/bin/python -c "import lancedb,json; from collections import Counter; \
    df=lancedb.connect('./lancedb').open_table('nemo-retriever').to_pandas(); \
    names=[json.loads(s)['source_name'] for s in df['source']]; \
    print(sorted(set(names))); print(dict(Counter(names)))"
  ```
  `references/query.md` says explicitly: for **"list distinct sources"** / **"count chunks
  per source"**, use this — **no `ls`/`grep`/`find`**. Each row's `source` is a JSON string;
  `json.loads(source)['source_name']` is the file basename. So:
  - **COUNT documents** → `len(set(source_name))`
  - **LIST distinct sources** → `sorted(set(source_name))`
  - **COUNT chunks per source** → `Counter(source_name)`
  - **SUM a numeric column** → extract the table column's cells (table-structure rows, or the
    pdfium `page-elements` text extract) and `sum()` them in Python.
- **Table-structure for the SUM rung.** `--use-table-structure` flips `use_table_structure`
  true and `table_output_format` to `markdown`, materializing **nemotron-table-structure-v1**
  so a table column's cells are queryable as `content_type=table` rows. If the grid is
  flattened, the agent can recover the column from
  `retriever pdf stage page-elements … --method pdfium` and still sum deterministically.
- **Query hit shape:** a JSON array; each hit has exactly `{source, page_number, text}`;
  `page_number` is a 1-indexed int.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## Ground truth (computed from the real fixtures, before authoring)

**3-PDF corpus** (`woods_frost.pdf` + `table_test.pdf` + `multimodal_test.pdf`):
`--dry-run` → `branch_summary pdf:3`. Distinct `source_name` set =
`{multimodal_test.pdf, table_test.pdf, woods_frost.pdf}` → **COUNT distinct documents = 3**.

**`table_test.pdf`** — 1 page, header `Year | Bill | Amy | James | Ted | Susan`, then
**20 data rows** for years **2023 … 2004** (descending). Some cells are `N/A`. Read off the
page and computed deterministically (cross-checks vs the fixture catalog: **James 2019 =
978**, **Susan 2023 = 970** — both confirmed):

| Column | present cells | SUM of present cells |
|---|---|---|
| Bill | 15 | 11217 |
| Amy | 12 | 9211 |
| James | 18 | 13237 |
| **Ted** | **20 (full column, no N/A)** | **14806** |
| Susan | 18 | 14985 |

- Total data rows (years) = **20**.
- **Ted is the SUM target** because its column has a value in **all 20** year rows (no
  `N/A`), so the exact total **14806** is unambiguous — no missing-value handling needed.
  Ted's cells, years 2023→2004:
  `870, 912, 520, 502, 707, 942, 509, 943, 582, 507, 971, 761, 944, 936, 753, 786, 845, 684, 631, 501`.
- (Grand total of all 83 numeric cells across the five name columns = **63456** — available
  as an alternate aggregate, not used as the headline target.)

---

## The five tests (one aggregation ladder)

| # | id | What it adds over the previous rung | Operator |
|---|---|---|---|
| 1 | `retrieve-agg-001` | **COUNT documents.** "How many documents are in my KB?" → ingest the folder, count distinct `source_name` over rows. The aggregation floor: one scalar count operator. | `len(set(source_name))` |
| 2 | `retrieve-agg-002` | **LIST distinct.** Operator changes from a scalar COUNT to an enumerated LIST-DISTINCT (exact membership). Same corpus. | `sorted(set(source_name))` |
| 3 | `retrieve-agg-003` | **GROUP-BY COUNT.** Per-source chunk counts — grouping + counting rows within each group. | `Counter(source_name)` |
| 4 | `retrieve-agg-004` | **Numeric SUM.** A numeric reduction over a table column's cells (exact integer), not a count of rows. | `sum(ted_cells)` |
| 5 | `retrieve-agg-005` | **Acceptance gate.** End-to-end: NRL triggered + TWO chained aggregates (distinct-doc COUNT and column SUM), both agent-computed over rows, both exact. | `len(set(...))` + `sum(...)` |

The ladder: T1 proves a corpus-level aggregate can be computed at all (a count over rows);
T2 changes the operator to list-distinct; T3 adds grouping (per-source counts); T4 switches
to a numeric SUM over extracted table cells (exact arithmetic); T5 composes a count **and** a
sum end to end, gating that **both** are computed-not-hallucinated.

Every rung's defining check is the same operational pass: **NRL triggered + aggregation
operator visible in the trace over rows + exact match to ground truth**.

---

### T1 — `retrieve-agg-001` · COUNT documents  *(complexity 1)*
- **Satisfies:** the aggregation core, simplest COUNT form ("how many documents are in my KB?").
- **Data:** `cases/retrieve-agg-001/data/` (`woods_frost.pdf` + `table_test.pdf` + `multimodal_test.pdf`).
- **Expected:** `RETRIEVER ingest data/` (`branch_summary pdf:3`, non-zero rows); then
  `len(set(source_name))` over the LanceDB table → **3**. Fails if the number is guessed from
  a summary or via native `ls`/`find`.

### T2 — `retrieve-agg-002` · LIST distinct sources  *(complexity 2)*
- **Satisfies:** the LIST flavor ("list distinct sources").
- **Data:** the same 3-PDF folder.
- **Adds:** the operator becomes `sorted(set(source_name))` — exact membership, not just the
  count. Ground truth: `['multimodal_test.pdf', 'table_test.pdf', 'woods_frost.pdf']`.

### T3 — `retrieve-agg-003` · COUNT chunks per source  *(complexity 3)*
- **Satisfies:** the per-group COUNT flavor (the canonical `references/query.md` "count chunks
  per source").
- **Data:** the same 3-PDF folder.
- **Adds:** a GROUP-BY (`Counter(source_name)`). Per-source chunk counts depend on the live
  chunking profile, so they are **not** hard-coded; the deterministic, gradable invariant is
  the **shape**: exactly **3** source keys, each count ≥ 1, and the per-source counts **sum to
  the total ingested row count M** from the ingest line.

### T4 — `retrieve-agg-004` · SUM a numeric column  *(complexity 4)*
- **Satisfies:** the SUM flavor ("sum `<metric>` across all `<groups>`") with exact arithmetic.
- **Data:** `cases/retrieve-agg-004/data/table_test.pdf`.
- **Adds:** a numeric reduction over a table column. `RETRIEVER ingest … --use-table-structure`,
  recover **Ted's** 20-cell column (table rows, or the pdfium `page-elements` text extract),
  then `sum()` → **14806**. Ted is chosen because it has **no `N/A`** cells. Fails if the total
  is asserted without a visible summation operator over the extracted cells.

### T5 — `retrieve-agg-005` · acceptance gate, count + sum, end-to-end  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the RETRIEVE-aggregation row.
- **Data:** the 3-PDF folder (count over the corpus; sum over `table_test.pdf`).
- **Expected:** one `ingest data/ --use-table-structure`, then two aggregate one-liners:
  distinct-document **COUNT = 3** (`len(set(source_name))`) and Ted-column **SUM = 14806**
  (`sum(...)` over the extracted cells).
- **Adds (the gate):** BOTH aggregates must be agent-computed over rows with the operator
  **visible in the trace**, and **both** must match ground truth exactly — proving the
  count/sum is computed-not-hallucinated end to end.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** NRL is **triggered** — `retriever ingest` builds the index with the expected
`branch_summary` (`pdf:3` for the corpus rungs, `pdf:1` for the table-only rung) and a
non-zero row count;
**(b)** the aggregate is **agent-computed over rows** — an explicit operator (`len`/`set`,
`Counter`, `sum`) over the LanceDB table or retrieved hits is **visible in the trace**, not an
LLM guess from a summary and not native `ls`/`find`/`wc`;
**(c)** the answer matches ground truth **EXACTLY** — COUNT = 3; LIST =
{multimodal_test.pdf, table_test.pdf, woods_frost.pdf}; per-source counts = 3 keys summing to
M; Ted-column SUM = **14806**;
**(d)** no `--input-type` flag is used (it does not exist).

**Note on live runs.** Expected outputs are grounded in the CLI **source + `--dry-run`**
(`branch_summary pdf:3` / `pdf:1` confirmed offline) and the skill's `references/query.md`
corpus-aggregate one-liners; the suite has **not** been run live yet. The exact-match
aggregates that **do not** depend on live extraction — the **distinct-document COUNT = 3**,
the **distinct-source LIST**, and the **Ted-column SUM = 14806** (read from the fixture and
computed deterministically) — are fixed ground truth. The only value left to a live run is
the per-source **chunk count** in T3 (and the total row count M), which depends on the live
chunking/extraction profile; T3 is graded on the group-by **shape** (3 keys summing to M),
not hard-coded counts. A live run would capture the real per-file row counts by content type,
table-structure cell fidelity for the SUM column, latencies, and token baselines.
