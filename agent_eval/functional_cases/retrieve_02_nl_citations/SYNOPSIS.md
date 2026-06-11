# Synopsis — RETRIEVE: NL question → grounding context with structured citations

**What user task this covers.** A user asks a question in plain English and wants NeMo
Retriever to answer it **and show its work** — every answer comes back with a **structured
citation**: which document it came from (a source identifier resolvable to a filename and
full path), which **page**, and which chunk. Success means: ask a natural-language question,
get a grounded answer, and have **every** supporting passage carry a **non-null** source +
page citation that **resolves to a real passage** and whose **text actually supports** the
answer. This is judged as an **operational pass** (correct grounded answer + a resolvable
citation for every hit), **not** a RAGAS judge score — RAGAS lives in the separate
performance-eval suites.

**How we test it.** Five agent prompts, each a natural-language question over a small,
deterministic corpus of real PDFs. We check the agent drives the `retriever` CLI correctly:
`retriever ingest <paths>` (format auto-detected — there is **no** `--input-type` flag), then
`retriever query "<NL question>" --top-k 5`. At the CLI layer each hit is exactly
`{source, page_number, text}` with `page_number` **1-indexed**, so the assertions are:
every hit has a **non-null source and page_number**, the page is **correct**, and the cited
`text` **supports the answer**. (The deeper "what's on page 14?" page-filtering query is
intentionally left to the `retrieve_11_filter` suite — this one stays on citations.)

**The five tests, simplest to hardest:**

1. **Baseline** — a NL question gets a grounded answer plus **one** structured citation
   (non-null source + page). Ground truth: the woods' owner's *house is in the village*,
   cited from `woods_frost.pdf` p1.
2. **Correct page** — ask for a fact that lives **only on page 2** (which Frost collection
   was published in **1923** → **New Hampshire**); the cited page must be **2, not 1**.
   Catches a citation that is present but points at the wrong page.
3. **No nulls** — over a 2-document folder, return the top passages and require that **every
   single one** has a source and a page — no blank citations anywhere, across both files.
4. **Grounding** — answer a specific value (James, 2019 → **978**) and require the cited
   passage's **text to literally contain `978`**, so the citation genuinely supports the
   answer rather than pointing at an unrelated row.
5. **Acceptance gate** — a NL question over a **3-document** corpus where the wrong document
   and wrong page are available distractors: the answer (*"Stopping by Woods on a Snowy
   Evening"* by Robert Frost, p1) must cite the **right doc and right page**, every hit must
   be cited (no nulls), and every citation must resolve to a real grounded passage.

**Why this order.** Each rung adds exactly one new thing: first "does a citation come back at
all," then page-number **correctness**, then **completeness** (no nulls across multiple
sources), then **grounding** (the cited text contains the answer), then everything
**composed** over a multi-doc corpus with distractors.

**Status.** Tests are authored and grounded in the real CLI (`--dry-run` + the
`{source, page_number, text}` hit shape) and in reading the fixtures; **not yet run live**.
A live run needs a reachable embedding backend (bundled HF model, hosted endpoint with an
API key, or a local GPU); text extraction for these PDFs is local pdfium. See `README.md`
for the full spec and `cases.json` for the machine-gradable definitions.
