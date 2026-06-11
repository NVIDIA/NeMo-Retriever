# Functional test suite — RETRIEVE: multi-hop retrieval (single source, multiple chunks)

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/src` (`retriever ingest` / `retriever query`). Each test is a
self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and flags.

This suite covers the **RETRIEVE** job's **multi-hop** row: a question over **one** ingested
document whose answer requires **chaining two retrievals** — hop-1 finds an intermediate
value (a year, a title, an argmax), and that value is reused to phrase **hop-2**, which
returns the final answer. The operational signature under test is: **one ingest, then
≥ 2 chained `retriever query` calls** where hop-1's result feeds hop-2.

---

## The user task under test

> **JTBD: RETRIEVE — P0.** "Multi-hop retrieval (single source - multiple chunks)."
> Time budget: **≤ 1 min**.

Seed queries this suite paraphrases (from the RETRIEVE seed-query tab):
- *"Which entity had the largest \<X\>, and what was that entity's \<Y\>?"*
- *"What was \<metric\> in the year of \<some derived condition\>?"*
- *"For the \<superlative\> item, what's its \<other attribute\>?"*

**Success criteria for the row (operational, NOT RAGAS):**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: the trace shows **≥ 2 chained `retriever query` calls** against a single ingested source; hop-1's retrieved value is extracted and reused to build hop-2; the **final answer matches the ground truth** computed from the real fixture. A single combined query that answers in one shot **FAILS** — the task is to demonstrate chaining over multiple chunks. |
| Time | **≤ 1 min** per case (one small single-file ingest + 2+ fast text/table queries against the built index) |
| Trigger rate | ≥ 95% — a "find X, then use it to answer Y from this one document" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — one `retriever ingest <file>` (table rungs add `--use-table-structure`) then **≥ 2 sequential** `retriever query` calls; hop-2 built from hop-1's result. **No `--input-type` flag** (it does not exist). |
| Token usage | tracked, not gated |

This is **functional RETRIEVE → operational pass**, not the graded RAGAS path (that lives in
the separate performance-eval suites). We grade the *chained-query signature* + the *exact
grounded final value*, not a judge score.

---

## How the CLI behaves for this task (grounded in `--dry-run` + the CLI source)

- **Two subcommands only:** `retriever ingest <paths…>` and `retriever query "<text>"`.
  Format is auto-detected from the extension; there is **no `--input-type` flag**.
- **Table cells need `--use-table-structure`.** All numeric rungs (mh-001/002/003/005) read
  table cells, so ingest passes `--use-table-structure` (flips `use_table_structure` true,
  `table_output_format` → `markdown`, materializing **nemotron-table-structure-v1** so cells
  are queryable as discrete `content_type=table` rows). Queries pass
  `--content-types text,table`. The cross-page rung (mh-004) spans the p1 poem and the p2
  table, so it also ingests with `--use-table-structure` and queries `text,table`.
- **Chaining is at the agent layer, not a CLI feature.** The CLI has no "multi-hop" flag.
  Multi-hop = the agent issues query #1, parses the returned hit JSON to extract an
  intermediate (a year / title / computed argmax), then issues query #2 whose **query string
  is built from that intermediate**. The test asserts this two-call pattern in the trace.
- **Superlative hops need a generous pool.** When hop-1 is an argmax over a column (mh-003,
  mh-005), the query must retrieve **all** the relevant year rows — use a generous `--top-k`
  (and `--candidate-k` if needed) so the true maximum row is in the pool; an under-retrieved
  pool can yield the wrong superlative year.
- **Host-agnostic.** Extraction/embedding run on hosted `ai.api.nvidia.com` /
  `integrate.api.nvidia.com` (key from `NVIDIA_API_KEY`) on a no-GPU box, or on a local GPU
  when configured (see the two SETUP suites). The assertion here is the chained-query pattern
  + the correct value, not where it ran.
- **Ingest success line:** `Ingested N file(s) -> M row(s) in LanceDB lancedb/<table>.`
- **Query hit shape:** a JSON array; each hit has exactly `{source, page_number, text}`;
  `page_number` is a 1-indexed int.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`.

---

## Fixtures and the VERIFIED hop chains

Both fixtures were read directly off disk and the chains computed exactly.

**`table_test.pdf`** — 1-page grid, Year × {Bill, Amy, James, Ted, Susan}, years 2004–2023
(some early cells are `N/A`). Verified column maxima and chains:

| Person | Max value | Year of max | Linked value used in a chain |
|---|---|---|---|
| James | **987** | **2015** | Susan in 2015 = **854** |
| Susan | **994** | **2020** | James in 2020 = **922** |
| Amy | 938 | 2022 | (Ted in 2022 = 912) |

Note the trap baked into rung 3: **James scored 978 in 2019, but that is NOT his maximum —
his maximum is 987 in 2015.** A naive agent that conflates "James = 978" (the extract suite's
famous cell) with "James's max" will return the wrong year.

**`woods_frost.pdf`** — 2-page single source. **p1** = the Robert Frost poem *"Stopping by
Woods on a Snowy Evening"* (closing stanza repeats *"And miles to go before I sleep"*). **p2**
= a *"Frost's Collections"* table (`# / Collection / Year`): 1 A Boy's Will 1913; 2 North of
Boston 1914; 3 Mountain Interval 1916; 4 New Hampshire 1923; 5 West Running Brook 1928;
6 A Further Range 1937; 7 A Witness Tree 1942; 8 In the Clearing 1962; 9 Steeple Bush 1947;
10 An Afterword unknown.

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Fixture / chain |
|---|---|---|---|
| 1 | `retrieve-mh-001` | **Baseline explicit 2-hop.** The user states both hops; the agent just executes two queries, hop-1 value feeds hop-2. | table_test: James scored 978 → **year 2019** → Susan 2019 = **922** |
| 2 | `retrieve-mh-002` | **Intermediate must be retrieved, not given.** The linking literal is not in the prompt — hop-1 must run to obtain it. | woods_frost: 'North of Boston' year → **1914** → next collection by year = **Mountain Interval (1916)** |
| 3 | `retrieve-mh-003` | **Hop-1 is a SUPERLATIVE (argmax over a retrieved column).** | table_test: James's max **987 in 2015** → Susan 2015 = **854** |
| 4 | `retrieve-mh-004` | **Two DISTINCT query calls, cross-page/cross-modality.** Single-query shortcut impossible; trace must show two calls. | woods_frost: poem title (p1) → **'Stopping by Woods on a Snowy Evening'** → closing repeated line (p1) = **'And miles to go before I sleep'** |
| 5 | `retrieve-mh-005` | **Acceptance gate.** Superlative hop-1 + cross-cell hop-2, graded purely on the chained-query signature + exact value. | table_test: Susan's max **994 in 2020** → James 2020 = **922** |

The ladder: T1 establishes the chained-query pattern with both hops handed to the agent; T2
removes the linking literal so hop-1 must be *retrieved*; T3 makes hop-1 a *computed
argmax*; T4 forces the two hops onto different pages/modalities so two distinct query calls
are unavoidable and explicitly checked; T5 composes argmax-hop-1 + cross-cell-hop-2 as the
operational acceptance gate.

---

### T1 — `retrieve-mh-001` · explicit 2-hop  *(complexity 1)*
- **Satisfies:** the multi-hop floor — two stated hops executed as two queries.
- **Data:** `cases/retrieve-mh-001/data/table_test.pdf`.
- **Chain:** hop-1 "year James scored 978" → **2019**; hop-2 "Susan in 2019" → **922**.
- **Expected:** `ingest … --use-table-structure`; then two `query … --content-types
  text,table`. Hop-2's string must contain `2019` (from hop-1), not a user-given literal.

### T2 — `retrieve-mh-002` · intermediate retrieved, not given  *(complexity 2)*
- **Satisfies:** faithful extract-and-reuse of an intermediate hit.
- **Data:** `cases/retrieve-mh-002/data/woods_frost.pdf`.
- **Chain:** hop-1 "year 'North of Boston' published" → **1914**; hop-2 "next collection by
  year after 1914" → **Mountain Interval (1916)**. The user never states 1914.
- **Caveat:** the collections table is not contiguous by year (no 1915 entry); "the year
  right after" resolves to the next *published* collection = Mountain Interval (1916).

### T3 — `retrieve-mh-003` · superlative hop-1  *(complexity 3)*
- **Satisfies:** the "for the superlative item, what's its other attribute?" seed query.
- **Data:** `cases/retrieve-mh-003/data/table_test.pdf`.
- **Chain:** hop-1 argmax over James's column → **max 987 in 2015**; hop-2 "Susan in 2015" →
  **854**. Use a generous `--top-k` so all of James's year rows are pooled.
- **Trap:** returning **2019/978** (James's 2019 value) instead of his true max **987/2015**
  fails this rung.

### T4 — `retrieve-mh-004` · two distinct calls, cross-page  *(complexity 4)*
- **Satisfies:** verifiable "≥ 2 distinct query calls (not one)".
- **Data:** `cases/retrieve-mh-004/data/woods_frost.pdf`.
- **Chain:** hop-1 poem title (p1) → **'Stopping by Woods on a Snowy Evening'**; hop-2 uses
  that title to fetch the closing repeated line (p1) → **'And miles to go before I sleep'**.
- **Adds:** the two hops land on the same source but require two separate retrievals; grading
  asserts exactly two `retriever query` invocations in the trace.

### T5 — `retrieve-mh-005` · acceptance gate  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the multi-hop RETRIEVE row.
- **Data:** `cases/retrieve-mh-005/data/table_test.pdf`.
- **Chain:** hop-1 argmax over Susan's column → **max 994 in 2020**; hop-2 "James in 2020" →
  **922**.
- **Gate:** one ingest, then ≥ 2 **chained** `retriever query` calls (hop-2 string contains
  `2020` derived from hop-1's argmax), ending in the exact grounded value. Operational pass,
  not RAGAS.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** the trace contains **≥ 2 sequential `retriever query` calls** against the single
ingested source (a one-shot combined query fails); **(b)** hop-1's retrieved value is
extracted and **embedded in hop-2's query string** (the year/title/argmax is not supplied by
the user as a literal in rungs 2/3/4/5); **(c)** the **final answer matches the verified
ground truth** with the right page citation — 2019/922 (T1), 1914/Mountain Interval (T2),
2015/854 (T3), the poem title + 'miles to go' line (T4), 2020/922 (T5); **(d)** numeric
rungs ingest with `--use-table-structure` and query `--content-types text,table`; **(e)** no
`--input-type` flag is used (it does not exist).

**Note on live runs.** Expected outputs are grounded in the CLI **source + `--dry-run`** and
in the fixture content read directly off disk; the suite has **not** been run live yet. A
live run requires a reachable retrieval backend — either the **hosted**
`ai.api.nvidia.com` / `integrate.api.nvidia.com` endpoints (needs `NVIDIA_API_KEY`, makes
small billable calls) or a **local GPU** configured per the SETUP-GPU suite. Running live
would capture the real per-file row counts (by content type), the actual hit JSON for each
hop, table-structure cell fidelity, end-to-end latency against the ≤ 1 min budget, and token
baselines for the two-query chain.
