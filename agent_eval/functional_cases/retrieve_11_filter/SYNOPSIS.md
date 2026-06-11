# Synopsis — RETRIEVE: filter results by document attributes

**What user task this covers.** A user has a knowledge base of ingested documents and wants
retrieval to return **only** the chunks matching a document attribute — by **content type**
(just the tables, just the charts), by **source / filename** ("get the full contents of
`woods_frost.pdf`"), or by **page number** ("what's on page 2 of `<doc>`?"). Success means
the filter is applied **at retrieval as a real predicate** — every returned row matches the
filter, with no off-predicate rows — not "fetch everything, then let the model throw out the
ones that don't fit." This is a **functional** RETRIEVE row, so the bar is an **operational
pass** (correct mechanism + correct filtered rows), not a RAGAS judge score.

**How we test it.** Five agent prompts, each handing the agent a tiny real corpus and asking
for results filtered to one attribute. We check the agent drives the right mechanism — and
the two mechanisms are deliberately **different**, because the shipped CLI is:
- **Content-type filtering is a native flag:** `retriever query … --content-types table`
  (or `text`, `chart`, `image`, `infographic`) returns only rows of those types; untyped
  hits are dropped at the CLI layer.
- **Source / filename / page filtering has NO CLI flag.** There is no `--where`, `--source`,
  or `--page`. The canonical mechanism is a small **LanceDB predicate** (from the skill's
  `references/query.md`) that filters the table the retriever already built — on
  `source_name` or `metadata.page_number` — **before** any top-k limit. Inventing a flag, or
  retrieving everything and LLM-filtering afterward, fails.

**The five tests, simplest to hardest:**

1. **Content-type filter** — `--content-types table` over two PDFs → only the table rows come
   back (Table 1 from the multimodal doc + the Year-grid from `table_test.pdf`), no prose. The
   floor: one native flag, one content type.
2. **Content-type is selective** — same document, `text` vs `chart`: the text query returns
   only prose, the chart query returns only chart rows, and the two sets don't overlap. Proves
   the native filter actually discriminates between modalities.
3. **Filter by source** — a two-document corpus; "get the full contents of `woods_frost.pdf`"
   → the LanceDB predicate returns only that file's rows (the Frost poem on p1, the
   collections table on p2), nothing from the other doc. Switches mechanism to the predicate.
4. **Filter by page number** — "what's on page 2 of `woods_frost.pdf`?" → the predicate keys
   on `page_number == 2` (1-indexed) and returns only the collections table (New Hampshire
   1923, …), not the page-1 poem.
5. **Acceptance gate** — a three-document corpus and a compound predicate (`source == woods_frost.pdf`
   **and** `page_number == 1`) → only the poem on page 1 of that one file; every row from the
   other two docs and from page 2 is excluded. A single off-predicate row fails the case.

**Why this order.** Each rung adds exactly one thing: first filtering at all (native
content-type flag), then proof that flag is selective, then a new attribute that forces the
predicate mechanism (source), then another attribute on the same mechanism (page number),
then the two composed over a larger corpus as the operational-pass gate.

**Status.** Tests are authored and grounded in the real CLI, the skill's `references/query.md`,
and direct per-page inspection of each fixture; **not yet run live**. Live runs need a
reachable embedding backend for ingest/query (hosted `integrate.api.nvidia.com` with an API
key, or a local GPU); the LanceDB predicate steps then run offline against the built table.
See `README.md` for the full spec and `cases.json` for the machine-gradable definitions.
