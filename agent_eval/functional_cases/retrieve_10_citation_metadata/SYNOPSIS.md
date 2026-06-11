# Synopsis — RETRIEVE: every retrieved result carries citation metadata

**What user task this covers.** When a user asks for evidence — "quote me what this document
says, with exact page references for every claim" — they need **provenance on every single
result**: which source file it came from and which page. This suite checks the P0 RETRIEVE
row *"Every retrieved result includes citation metadata (source + page/offset)."* It is an
**operational-only** check: the question is purely *"is the citation present and resolvable
on every returned hit?"* — **not** whether the answer is good. There is no accuracy/RAGAS
gate on this row.

**How we test it.** Five agent prompts, each handing the agent a small set of PDFs and
checking that the agent drives the `retriever` CLI (`ingest` then `query`) and that the
returned hits all carry citation metadata. At the CLI layer each hit is `{source,
page_number, text}`; we assert that **`source` is non-null and `page_number` is a valid
1-indexed integer on every hit**, and that the source resolves to a file that was actually
ingested. The single defining rule: **a missing citation on any one returned result fails
the whole run.**

**The five tests, simplest to hardest:**

1. **Single hit** — one query, one result; that result carries source + page.
2. **All ten** — `--top-k 10` over a 3-page document; **every** one of the returned hits
   carries source + page, with page numbers varying across the doc.
3. **Multi-doc, resolvable** — a 3-PDF corpus; every hit's source resolves to a real
   ingested file (no fabricated or dangling citations).
4. **Right page** — a fact that lives only on page 3 must come back citing page 3, proving
   the page offset points to the real source page rather than a constant.
5. **Acceptance gate** — one strict run spanning all docs, all pages, and multiple content
   types (text, tables, charts) into a named index, where every returned result must carry
   resolvable citation metadata; a single null fails the run.

**Why this order.** Each rung adds exactly one dimension to the citation invariant: first a
single hit is cited; then the invariant holds across **all** of top-k (coverage); then
across **multiple documents** (resolvability); then the page number must be the **correct**
one (offset accuracy); then everything is composed into the real pass/fail gate.

**Relationship to `retrieve_02_nl_citations`.** That suite checks that an answer is grounded
with *a* citation; this one enforces the strict **100%-coverage** invariant — every hit, no
exceptions — and is operational-only.

**Note on `pdf_page`.** The spec models a citation as `<source_id, page_number, pdf_page,
filename>`; the shipped CLI returns only `{source, page_number, text}`. So `source_id` /
`filename` collapse onto `source`, and `pdf_page` (e.g. `woods_frost_2`) is **derived** as
`<basename(source)>_<page_number>` — the CLI does not emit it directly. This suite asserts
the two primitives the CLI actually returns and derives `pdf_page` from them.

**Status.** Tests are authored and grounded in the real CLI + the skill's query reference;
**not yet run live** (live ingest/query may hit billable hosted endpoints or need a GPU).
See `README.md` for the full spec and `cases.json` for the machine-gradable definitions.
