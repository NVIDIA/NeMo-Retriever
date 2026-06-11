# Functional test suite — RETRIEVE: every result carries citation metadata

Agent-driven functional suite for the **NeMo Retriever Library skill**, built against the
real CLI in `nemo_retriever/nemo_retriever/src` (`retriever ingest` / `retriever query`).

Each test is a self-contained triple — a prompt, a per-case `data/` folder, and an expected
output naming the correct `retriever` subcommand(s) and flags. The variable under test in
this suite is a single **operational invariant**: *does every retrieved result come back
with citation metadata (source + page)?*

---

## The user task under test

> **JTBD: RETRIEVE.** "Every retrieved result includes citation metadata (source +
> page/offset)." — **P0**

This row is **OPERATIONAL-ONLY**: there is **no accuracy / RAGAS gate** (per the spec).
The pass/fail question is purely *"is the citation metadata present and resolvable on every
returned hit?"* — not *"is the answer good?"*

**Success criteria for the row:**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: **every** hit returned by `retriever query` carries a non-null `source` and a valid 1-indexed integer `page_number`, and each `source` resolves to a real ingested file. **A single null citation field on ANY returned hit fails the run.** Answer correctness is *not* graded. |
| Time | RETRIEVE non-agentic — per query **≤ 1 min** |
| Trigger rate | ≥ 95% — a "give me page references / quoted excerpts for every claim" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — `retriever ingest` then `retriever query` (with `--top-k`); query table-name must match the ingest table-name |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased):
- *"What does \<doc\>.pdf say about \<topic\>? I want exact page references and quoted excerpts for every claim."*
- *"Quote me what \<doc\> says about \<topic\> — exact wording and page numbers."*
- *"Get the full extracted contents of \<doc\>.pdf."*

---

## How the CLI emits citations (verified against `cli/main.py` + `references/query.md`)

- `retriever query "<text>"` prints a **JSON array of hits**; each hit has **exactly three
  keys**: `{source, page_number, text}`.
- `source` is the ingested file (path/name); `page_number` is an **int, 1-indexed**.
- There is **no** `metadata` / `_distance` key at the CLI layer.

**`pdf_page` vs the CLI hit (important).** The spec models a citation as
`<source_id, page_number, pdf_page, filename>`, where `pdf_page` looks like
`WALMART_2017_10K_42`. The shipped CLI hit does **not** expose `pdf_page` (nor `source_id`
nor `filename`) as its own field — it exposes only `{source, page_number, text}`. Therefore:

- `source_id` / `filename` collapse onto **`source`**.
- `pdf_page` is **derivable** as `<basename(source)>_<page_number>` — e.g.
  `source = woods_frost.pdf`, `page_number = 2` → `woods_frost_2`.

This suite asserts the **two primitives the CLI actually returns** (`source` +
`page_number`) and **derives** `pdf_page` from them; it deliberately does **not** assert a
literal `pdf_page` field, because the CLI does not return one. The grader should compute
`pdf_page` from `source` + `page_number` and confirm it is well-formed.

Convention in every command: `RETRIEVER=$RETRIEVER_VENV/bin/retriever`. This suite is
**host-agnostic** — the citation invariant lives at the CLI output layer and is independent
of whether embedding/extraction ran hosted (build.nvidia.com) or on a local GPU; pick either
install. Defaults: lancedb uri `lancedb`, table `nemo-retriever`.

---

## Relationship to `retrieve_02_nl_citations`

Both suites touch citations, but the **distinct focus** here is the **strict 100%-coverage
invariant**: where `retrieve_02` checks that a natural-language answer is grounded with *a*
citation, this suite checks that **every single returned hit** — across many hits, multiple
pages, multiple documents, and multiple content types — carries non-null, resolvable
citation metadata, with **a single missing citation failing the whole run**. It is
operational-only (no accuracy gate).

---

## The five tests (increasing complexity)

| # | id | What it adds over the previous rung | Subcommands |
|---|---|---|---|
| 1 | `retrieve-cm-001` | **Baseline.** One query, one hit; that single hit carries non-null `source` + valid 1-indexed `page_number`. | `ingest`, `query --top-k 1` |
| 2 | `retrieve-cm-002` | **100% coverage at scale (1 doc).** `--top-k 10` over a 3-page doc → **all 10** hits carry source + page; page numbers vary. | `ingest`, `query --top-k 10` |
| 3 | `retrieve-cm-003` | **Resolvability (multi-doc).** 3-PDF corpus → every hit's `source` resolves to a real ingested file (no dangling citations). | `ingest data/`, `query --top-k 10` |
| 4 | `retrieve-cm-004` | **Page correctness.** A page-3-only fact must come back with `page_number == 3` — the offset points to the real source page. | `ingest`, `query --top-k 5` |
| 5 | `retrieve-cm-005` | **Acceptance gate.** One strict `--top-k 10` run over multi-doc / multi-page / multi-content-type into a **named** index; every hit resolvable; any null = FAIL. | `ingest --table-name`, `query --table-name --content-types` |

The ladder: T1 proves a single hit is cited; T2 enforces the invariant on **all** of
top-k=10 over one multi-page doc; T3 adds **resolvability** across a multi-doc corpus; T4
adds **page-offset correctness** (the right page, not just any page); T5 composes
everything — multi-doc, multi-page, multi-content-type, named index — into the row's real
operational-pass gate where a single null citation fails the run.

---

### T1 — `retrieve-cm-001` · single query, single hit  *(complexity 1)*
- **Satisfies:** citation-metadata operational pass, simplest form.
- **Data:** `data/woods_frost.pdf` (2 pages).
- **Expected:** `RETRIEVER ingest data/woods_frost.pdf` → `RETRIEVER query "What does
  woods_frost.pdf say about the woods?" --top-k 1`. The one hit has `source ==
  woods_frost.pdf` (non-null) and `page_number ∈ {1,2}`; derived `pdf_page =
  woods_frost_<page_number>`. Excerpt correctness is **not** graded.

### T2 — `retrieve-cm-002` · top-k=10 over a multi-page doc  *(complexity 2)*
- **Satisfies:** the 100%-coverage clause on a single document.
- **Data:** `data/multimodal_test.pdf` (3 pages).
- **Adds:** breadth — **all** returned hits (up to 10), spanning pages 1–3, must each carry
  non-null `source` + valid 1-indexed `page_number`. Page numbers are expected to vary.
  **Any single null on any of the hits fails the run.**

### T3 — `retrieve-cm-003` · multi-doc corpus, resolvable sources  *(complexity 3)*
- **Satisfies:** the **resolvability** clause ("citations resolve to a real row in the index").
- **Data:** `data/` = `woods_frost.pdf` + `multimodal_test.pdf` + `table_test.pdf`.
- **Expected:** `RETRIEVER ingest data/` → `RETRIEVER query "…" --top-k 10`. Every hit's
  `source` resolves to exactly one of the three ingested files, and `page_number` is within
  that file's page range (woods_frost ≤ 2, multimodal_test ≤ 3, table_test == 1). A source
  not among the ingested files = FAIL.

### T4 — `retrieve-cm-004` · page-offset correctness  *(complexity 4)*
- **Satisfies:** that `page_number` is the **right** page, not a constant.
- **Data:** `data/multimodal_test.pdf` (3 pages).
- **Expected:** `RETRIEVER ingest data/multimodal_test.pdf` → `RETRIEVER query "What is the
  conclusion of the document?" --top-k 5`. The hit carrying the page-3-only **"Conclusion"**
  / **"Chart 2 … average frequency ranges for speaker drivers"** content must come back with
  `page_number == 3` (derived `pdf_page = multimodal_test_3`).
- **Caveat (built into the test):** page mapping depends on extraction chunking; the page-3
  "Conclusion" text is unique to p3, so a returned `page_number` of 1 or 2 for that passage
  would indicate a page-offset bug.

### T5 — `retrieve-cm-005` · full acceptance gate  *(complexity 5)*
- **Satisfies:** the complete operational-pass row — strict, composed.
- **Data:** `data/` (3 PDFs).
- **Expected:** `RETRIEVER ingest data/ --table-name cite_audit` → `RETRIEVER query "…"
  --table-name cite_audit --top-k 10 --content-types text,table,chart`. Hits span multiple
  files, pages, and content types; **every** hit carries non-null, in-range `source` +
  `page_number` resolving to a real ingested row; derived `pdf_page` well-formed for each.
- **Adds (the gate):** custom `--table-name` aligned across both commands, multi-content-type
  breadth, and the strict rule that **a single null citation field on a single hit fails the
  entire run**. Operational-only — no answer-accuracy assertion.

---

## Running / grading

Mount each case's `data/` folder into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining check unique to this suite: parse the
`query` JSON output and assert, **on 100% of returned hits**, that `source` is non-null,
`page_number` is an integer ≥ 1 (and within the cited file's page range), and `source`
resolves to a file that was actually ingested into the queried table. Compute the derived
`pdf_page = <basename(source)>_<page_number>` and confirm it is well-formed. **A single hit
missing either citation field fails the whole run.** Answer correctness is not graded.

**Note on live runs:** the expected outputs are grounded in the CLI source + dry-run and the
skill's query reference; the suite has **not** been run live yet (a live ingest/query may hit
billable hosted endpoints or need a GPU). A live run would capture real row counts, the
actual per-hit `source`/`page_number` values, latencies, and token baselines — and would
confirm the page-offset assertion in T4 against the real extraction chunking.

**Note on `pdf_page` vs the CLI hit:** the CLI returns only `{source, page_number, text}`;
it does **not** emit the spec's `pdf_page` (or `source_id` / `filename`) as a field.
`source_id`/`filename` collapse onto `source`, and `pdf_page` is derived as
`<basename(source)>_<page_number>` (e.g. `woods_frost.pdf` + page 2 → `woods_frost_2`). This
suite asserts the two primitives the CLI actually returns and derives `pdf_page` from them.
