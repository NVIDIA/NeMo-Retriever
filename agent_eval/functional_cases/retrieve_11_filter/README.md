# Functional test suite — RETRIEVE: filter results by document attributes

Agent-driven functional tests for the **NeMo Retriever Library skill**, built against the
real CLI (`retriever ingest` / `retriever query`) and the skill's `references/query.md`.
Each test is a self-contained triple — a prompt, a per-case `cases/<id>/data/` folder, and
an expected output naming the correct mechanism (native flag **or** LanceDB predicate) and
the rows it must return.

This suite covers the **RETRIEVE** job: take a question and return **only** the chunks that
match a document attribute — by **content type**, by **source / file path / filename**, or
by **page number** — with the filter applied **at retrieval** as a real predicate, not by
retrieving everything and asking the LLM to keep the matching ones.

---

## The user task under test

> **JTBD: RETRIEVE — P0.** "Filter results by document attributes (source, file path,
> filename, page number, content type)." Time budget: **≤ 1 min**.

**Success criteria for the row (operational — this is a FUNCTIONAL RETRIEVE suite, so
operational pass, NOT RAGAS):**

| Dimension | Target for this row |
|---|---|
| Quality (operational) | binary pass: the filter is applied **at retrieval via a predicate**. On the labeled filter set, **every returned row matches the filter predicate**; any off-predicate row in the result = a **miss**. Content-type filtering uses the **native `--content-types`** flag; source/filename/page filtering uses the **LanceDB predicate one-liner** (no `--where` CLI flag exists). **Not** post-hoc LLM filtering. |
| Time | **RETRIEVE — ≤ 1 min** per case (tiny-corpus ingest + one filtered query/predicate). Non-agentic. |
| Trigger rate | ≥ 95% — a "list / get / show **only** the rows matching `<attribute>`" prompt must fire the skill |
| Subcommand accuracy | ≥ 90% — content-type filter MUST use native `retriever query --content-types …`; source/page filter MUST use the LanceDB predicate one-liner. Inventing a `--source` / `--page` / `--where` query flag, or substituting post-hoc LLM filtering, is a miss. |
| Token usage | tracked, not gated |

Seed queries this suite is derived from (paraphrased into the prompts):
- *"List all `<docs matching attribute>` in my knowledge base."* (→ content-type filter)
- *"Get the full extracted contents of `<doc>.pdf`."* (→ source filter)
- *"What's on page 14 of `<doc>.pdf`?"* (→ source + page_number filter)

---

## How the filtering works (grounded in the CLI + `references/query.md`)

There are **two distinct mechanisms**, and this suite deliberately keeps them separate. Do
not conflate them, and **do not invent flags** — reflect what the shipped CLI actually has.

### 1. Content-type filter — NATIVE CLI flag
`retriever query "<q>" --content-types text,table,chart,image,infographic`
(comma-separated). Query-time values normalize to the canonical hit metadata `type`;
**untyped hits are excluded**. This is a real retrieval-time filter applied at the CLI layer
— no Python, no predicate string. There is also `--page-dedup` (collapse to unique pages),
which is orthogonal to attribute filtering.

### 2. Source / file path / filename / page-number filter — LanceDB PREDICATE one-liner
**`retriever query` has NO `--where` / `--source` / `--filename` / `--page` flag.** The
canonical mechanism (from `references/query.md`) is a LanceDB predicate evaluated against the
table the retriever already built — a real SQL-style `.where(clause)` predicate (or the
equivalent pandas row predicate) applied **before** any top-k limit:

```bash
# "Get the full extracted contents of <doc>.pdf" — filter on source_name:
$RETRIEVER_VENV/bin/python -c "import lancedb,json; df=lancedb.connect('./lancedb').open_table('nemo-retriever').to_pandas(); [print(r['text']) for _,r in df.iterrows() if json.loads(r['source'])['source_name'].endswith('woods_frost.pdf')]"

# "What's on page 2 of <doc>.pdf" — filter on metadata.page_number:
$RETRIEVER_VENV/bin/python -c "import lancedb,json; df=lancedb.connect('./lancedb').open_table('nemo-retriever').to_pandas(); print('\n'.join(r['text'] for _,r in df.iterrows() if json.loads(r['metadata']).get('page_number')==2))"
```

An alternative source filter is to **ingest a single doc into its own `--table-name`**, so
a query against that table is implicitly source-scoped. Either is a real retrieval-time
filter; retrieving everything and LLM-trimming afterward is **not**.

**Row schema** (each row written by `retriever ingest`): `text`, `source` (JSON string →
`source_name` = full path; basename = filename), `page_number` (1-indexed int), `metadata`
(JSON string → `type` = content type, `page_number`, `bbox`, …), and the embedding vector.
Source/page predicates read `source` / `metadata`; the native `--content-types` flag reads
`metadata.type`. (Verified against `references/query.md` and `cli/{ingest,query}.md`; the
`.pdf_extraction.json` of each fixture was inspected to ground the per-page content.)

`RETRIEVER=$RETRIEVER_VENV/bin/retriever` in every command.

---

## Fixtures (deterministic, verified per page)

- **`multimodal_test.pdf`** (3pg "TestingDocument") — p1: Introduction + **Table 1**
  (animals/activities, *Giraffe / Driving a car / At the beach*); p2: Section One/Two,
  the **3 bullet points**, Chart 1 region; p3: Chart 2 ("average frequency ranges for
  speaker drivers") + Conclusion ("2 tables, 2 charts, and … 3 bullet points"). Carries
  **text + table + chart** content types.
- **`table_test.pdf`** (1pg) — Year × {Bill, Amy, James, Ted, Susan} grid; **table** content.
- **`woods_frost.pdf`** (2pg) — **p1**: the Robert Frost poem "Stopping by Woods on a Snowy
  Evening" ("His house is in the village", "miles to go before I sleep"); **p2**: *Frost's
  Collections* table (A Boy's Will 1913 … **New Hampshire 1923** … In the Clearing 1962).
  Page mapping verified by `retriever pdf stage page-elements --method pdfium`.

---

## The five tests (one complexity ladder)

| # | id | What it adds over the previous rung | Mechanism |
|---|---|---|---|
| 1 | `retrieve-filter-001` | **Baseline content-type filter.** `--content-types table` → only table-typed hits. | NATIVE `--content-types` |
| 2 | `retrieve-filter-002` | **Content-type is selective.** Same doc, `text` vs `chart` → two disjoint sets. | NATIVE `--content-types` |
| 3 | `retrieve-filter-003` | **SOURCE attribute.** Multi-doc corpus → only one doc's rows. | LanceDB **predicate** (`source_name`) |
| 4 | `retrieve-filter-004` | **PAGE_NUMBER attribute.** "What's on page 2 of `<doc>`" → only that page's rows. | LanceDB **predicate** (`page_number`) |
| 5 | `retrieve-filter-005` | **Acceptance gate.** Compound `source AND page` over 3 docs → ONLY matching rows. | Compound LanceDB **predicate** |

The ladder: T1 introduces filtering with the simplest native flag; T2 proves that flag is
selective (disjoint text vs chart sets), still native; T3 switches mechanism to the LanceDB
predicate for the **source** attribute (the seed query "get the full contents of `<doc>.pdf`");
T4 adds the **page_number** attribute (the "what's on page N" seed query); T5 composes
**source AND page** over a 3-doc corpus — the operational-pass gate where a single
off-predicate row fails the case.

---

### T1 — `retrieve-filter-001` · content-type filter to `table`  *(complexity 1)*
- **Satisfies:** the content-type leg, simplest form (one native flag, one type).
- **Data:** `multimodal_test.pdf` + `table_test.pdf`.
- **Expected:** `RETRIEVER query "…" --content-types table` → **every** hit is a table-typed
  row (Table 1 from multimodal p1; the Year-grid from table_test p1); no prose row.
- **Mechanism:** NATIVE `--content-types`. Off-type rows in the result = miss.

### T2 — `retrieve-filter-002` · content-type `text` vs `chart`  *(complexity 2)*
- **Satisfies:** the content-type filter is selective, not cosmetic.
- **Data:** `multimodal_test.pdf`.
- **Adds:** two contrasting native-flag values on one corpus: `--content-types text`
  (prose only: Intro p1, bullets p2, Conclusion p3) and `--content-types chart` (chart rows
  only: Chart 1 p2-region, Chart 2 p3). The two result sets must be **disjoint**.

### T3 — `retrieve-filter-003` · filter by SOURCE  *(complexity 3)*
- **Satisfies:** the **source/filename** attribute — "get the full extracted contents of
  `woods_frost.pdf`".
- **Data:** `multimodal_test.pdf` + `woods_frost.pdf`.
- **Adds:** mechanism switch to the **LanceDB predicate** (`source_name` endswith
  `woods_frost.pdf`) over a multi-doc corpus. Returns the poem (p1) + collections table (p2)
  of woods_frost **only**; zero `multimodal_test.pdf` rows. There is **no** `--source`/`--where`
  CLI flag; post-hoc LLM filtering fails this rung.

### T4 — `retrieve-filter-004` · filter by PAGE_NUMBER  *(complexity 4)*
- **Satisfies:** the **page_number** attribute — "what's on page 2 of `woods_frost.pdf`".
- **Data:** `woods_frost.pdf`.
- **Adds:** the predicate keys on `metadata.page_number == 2` (1-indexed). Returns **only**
  page-2 content — *Frost's Collections* (New Hampshire 1923, …) — not the page-1 poem.
  No `--page` CLI flag.

### T5 — `retrieve-filter-005` · acceptance: combined SOURCE + PAGE  *(complexity 5)*
- **Satisfies:** the composed operational-pass for the filter row.
- **Data:** `multimodal_test.pdf` + `woods_frost.pdf` + `table_test.pdf`.
- **Expected:** one compound predicate — `source_name endswith woods_frost.pdf AND
  page_number == 1` — over the 3-doc corpus, returning **only** woods_frost page-1 rows (the
  poem). Excludes woods_frost p2 (collections table) and **every** row of the other two docs.
- **Adds (the gate):** two attributes satisfied simultaneously, applied as a real predicate
  **before** the limit. **Any** single off-predicate row (wrong doc OR wrong page) fails.

---

## Running / grading

Mount each case's `cases/<id>/data/` into the agent workdir, give it the `prompt`, and grade
against `pass_when` in `cases.json`. The defining checks for this suite:
**(a)** content-type filtering uses the **native `--content-types`** flag and returns only
the requested type(s); **(b)** source/page filtering uses the **LanceDB predicate one-liner**
(no invented `--source`/`--page`/`--where` flag) and is applied **before** the limit;
**(c)** on every case, **every returned row matches the filter predicate** and there are
**zero off-predicate rows**; **(d)** the filter is at-retrieval, **not** post-hoc LLM
filtering; **(e)** answers match the grounded per-page content (Frost poem p1 / collections
table + New Hampshire 1923 p2; Table 1 vs chart rows; James-grid table).

**Note on filtering mechanism.** Content-type filtering = the **native `retriever query
--content-types`** flag. Source / file-path / filename / page-number filtering = the
**LanceDB predicate one-liner** from `references/query.md` (a real `.where`-style predicate
over `source_name` / `metadata.page_number`, applied before the limit) or ingesting a doc
into its own `--table-name`. The CLI has **no** `--where` / `--source` / `--page` flag — do
not invent one, and do not substitute post-hoc LLM filtering.

**Note on live runs.** Expected outputs are grounded in the CLI **source + `--dry-run`**, the
skill's `references/query.md`, and direct inspection of each fixture's
`pdf stage page-elements` extraction (per-page content verified) — the suite has **not** been
run live yet. A live run requires a reachable embedding backend for the `ingest`/`query`
steps (hosted `integrate.api.nvidia.com` with `NVIDIA_API_KEY`, makes small billable calls,
or a local GPU per the SETUP-GPU suite); the LanceDB predicate steps then run fully offline
against the table that ingest built. Running live would capture the real per-content-type and
per-(source, page) row counts, confirm zero off-predicate rows on the labeled filter sets,
and record latencies and token baselines.
